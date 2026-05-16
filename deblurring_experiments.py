from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import convolve as nd_convolve
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

IMG_NAMES = [
    "kodim01.png",
    "kodim03.png",
    "kodim05.png",
    "kodim13.png",
    "kodim15.png",
    "kodim23.png",
]

NOISE_LIST = [0.0, 0.01, 0.03, 0.05]
FIXED_K = 0.005
ADAPTIVE_ALPHA = 1.0
RNG_SEED = 0


def read_img(path: Path) -> np.ndarray:
    img = Image.open(path).convert("L")
    return np.asarray(img, dtype=np.float64) / 255.0


def save_img(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.clip(img, 0, 1)
    Image.fromarray((x * 255).round().astype(np.uint8)).save(path)


def make_motion_psf(size: int, length: int, angle_deg: float) -> np.ndarray:
    psf = np.zeros((size, size), dtype=np.float64)
    c = size // 2
    angle = np.deg2rad(angle_deg)
    half = length // 2
    for t in range(-half, half + 1):
        x = int(round(c + t * np.cos(angle)))
        y = int(round(c + t * np.sin(angle)))
        if 0 <= x < size and 0 <= y < size:
            psf[y, x] = 1.0
    if psf.sum() <= 0:
        raise ValueError("Motion PSF sum is zero.")
    return psf / psf.sum()


def make_defocus_psf(size: int, radius: int) -> np.ndarray:
    psf = np.zeros((size, size), dtype=np.float64)
    c = size // 2
    yy, xx = np.indices((size, size))
    psf[(yy - c) ** 2 + (xx - c) ** 2 <= radius ** 2] = 1.0
    if psf.sum() <= 0:
        raise ValueError("Defocus PSF sum is zero.")
    return psf / psf.sum()


def psf2otf(psf: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=np.float64)
    h, w = psf.shape
    out[:h, :w] = psf
    out = np.roll(out, -(h // 2), axis=0)
    out = np.roll(out, -(w // 2), axis=1)
    return np.fft.fft2(out)


def apply_blur(img: np.ndarray, psf: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    H = psf2otf(psf, img.shape)
    blurred = np.real(np.fft.ifft2(np.fft.fft2(img) * H))
    return blurred, H


def add_noise(img: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    # Keep the degradation model linear. Clip only when saving/displaying or computing final metrics.
    return img + rng.normal(0.0, sigma, img.shape)


def inverse_filter(g: np.ndarray, H: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    G = np.fft.fft2(g)
    H_safe = H.copy()
    mask = np.abs(H_safe) < eps
    H_safe[mask] = eps * np.exp(1j * np.angle(H_safe[mask]))
    rec = np.real(np.fft.ifft2(G / H_safe))
    return np.clip(rec, 0, 1)


def wiener_filter(g: np.ndarray, H: np.ndarray, k: float) -> np.ndarray:
    G = np.fft.fft2(g)
    rec = np.real(np.fft.ifft2((np.conj(H) / (np.abs(H) ** 2 + k)) * G))
    return np.clip(rec, 0, 1)


def estimate_noise_sigma_mad(g: np.ndarray) -> float:
    hp_kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)
    hp = nd_convolve(g, hp_kernel, mode="reflect")
    # 6.0 = sqrt(sum(hp_kernel^2)); 0.6745 converts MAD to Gaussian sigma.
    return float(np.median(np.abs(hp)) / (6.0 * 0.6745))


def adaptive_k_improved(
    g: np.ndarray,
    alpha: float = ADAPTIVE_ALPHA,
    k_min: float = 1e-6,
    k_max: float = 0.2,
) -> float:
    sigma_hat = estimate_noise_sigma_mad(g)
    signal_var_hat = max(float(np.var(g)) - sigma_hat**2, 1e-8)
    k = alpha * (sigma_hat**2) / signal_var_hat
    return float(np.clip(k, k_min, k_max))


def adaptive_k_spectral_old(
    g: np.ndarray,
    alpha: float = 1.0,
    low_radius_ratio: float = 0.18,
    high_radius_ratio: float = 0.42,
    k_min: float = 1e-6,
    k_max: float = 0.2,
) -> float:

    F = np.fft.fftshift(np.fft.fft2(g))
    power = np.abs(F) ** 2
    h, w = g.shape
    yy, xx = np.indices((h, w))
    cy, cx = h // 2, w // 2
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    rmax = rr.max()
    low_mask = rr <= low_radius_ratio * rmax
    high_mask = rr >= high_radius_ratio * rmax
    low_e = float(np.mean(power[low_mask])) + 1e-12
    high_e = float(np.mean(power[high_mask]))
    k = alpha * high_e / low_e
    return float(np.clip(k, k_min, k_max))


def calc_metrics(ref: np.ndarray, restored: np.ndarray) -> Tuple[float, float]:
    restored_clip = np.clip(restored, 0, 1)
    return (
        float(peak_signal_noise_ratio(ref, restored_clip, data_range=1.0)),
        float(structural_similarity(ref, restored_clip, data_range=1.0)),
    )


def get_blur_configs(include_angle45: bool = False) -> Dict[str, dict]:
    cfgs: Dict[str, dict] = {}
    for length in [9, 15, 21]:
        cfgs[f"motion_len{length}_angle0"] = {
            "family": "motion",
            "strength": length,
            "angle": 0,
            "psf": make_motion_psf(max(21, length + 6), length, 0),
        }
    if include_angle45:
        for length in [9, 15, 21]:
            cfgs[f"motion_len{length}_angle45"] = {
                "family": "motion",
                "strength": length,
                "angle": 45,
                "psf": make_motion_psf(max(31, length + 8), length, 45),
            }
    for radius in [4, 7, 10]:
        cfgs[f"defocus_radius{radius}"] = {
            "family": "defocus",
            "strength": radius,
            "angle": np.nan,
            "psf": make_defocus_psf(max(21, 2 * radius + 5), radius),
        }
    return cfgs


def safe_tag(value: object) -> str:
    return str(value).replace(".", "_").replace("/", "_").replace("=", "")


def save_panel(path: Path, imgs: List[np.ndarray], titles: List[str], ncols: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(imgs)
    if ncols is None:
        ncols = n
    nrows = int(math.ceil(n / ncols))
    plt.figure(figsize=(3.0 * ncols, 3.1 * nrows))
    for i, (img, title) in enumerate(zip(imgs, titles), start=1):
        plt.subplot(nrows, ncols, i)
        plt.imshow(np.clip(img, 0, 1), cmap="gray", vmin=0, vmax=1)
        plt.title(title, fontsize=9)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_spectrum_panel(path: Path, imgs: List[np.ndarray], titles: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(3.0 * len(imgs), 3.1))
    for i, (img, title) in enumerate(zip(imgs, titles), start=1):
        spec = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(np.clip(img, 0, 1)))))
        plt.subplot(1, len(imgs), i)
        plt.imshow(spec, cmap="gray")
        plt.title(title, fontsize=9)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def make_methods(noisy: np.ndarray, H: np.ndarray) -> Dict[str, Tuple[np.ndarray, float]]:
    k_old = adaptive_k_spectral_old(noisy)
    k_adapt = adaptive_k_improved(noisy)
    return {
        "inverse": (inverse_filter(noisy, H), np.nan),
        f"wiener_fixed_K={FIXED_K:g}": (wiener_filter(noisy, H, FIXED_K), FIXED_K),
        "adaptive_wiener_spectral_old": (wiener_filter(noisy, H, k_old), k_old),
        "adaptive_wiener_MAD": (wiener_filter(noisy, H, k_adapt), k_adapt),
    }


def run_experiment(
    input_dir: Path,
    output_dir: Path,
    include_angle45: bool = False,
    save_all_panels: bool = False,
    save_single_images: bool = False,
    save_spectra: bool = True,
) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparisons").mkdir(exist_ok=True)
    if save_single_images:
        (output_dir / "single_images").mkdir(exist_ok=True)
    if save_spectra:
        (output_dir / "spectra").mkdir(exist_ok=True)

    blur_cfgs = get_blur_configs(include_angle45=include_angle45)
    representative_cases = {
        ("kodim05.png", "motion_len15_angle0", 0.03),
        ("kodim01.png", "defocus_radius7", 0.05),
        ("kodim23.png", "motion_len21_angle0", 0.03),
        ("kodim13.png", "defocus_radius10", 0.03),
        ("kodim03.png", "motion_len9_angle0", 0.01),
        ("kodim15.png", "defocus_radius4", 0.01),
    }

    rows = []
    condition_count = 0
    for name in IMG_NAMES:
        img = read_img(input_dir / name)
        for blur_name, cfg in blur_cfgs.items():
            blurred, H = apply_blur(img, cfg["psf"])
            for sigma in NOISE_LIST:
                condition_count += 1
                noisy = add_noise(blurred, sigma, rng)
                methods = make_methods(noisy, H)

                for method, (rec, param) in methods.items():
                    psnr, ssim = calc_metrics(img, rec)
                    rows.append(
                        {
                            "image": name,
                            "blur_name": blur_name,
                            "blur_family": cfg["family"],
                            "blur_strength": cfg["strength"],
                            "motion_angle": cfg["angle"],
                            "noise_sigma": sigma,
                            "method": method,
                            "PSNR": psnr,
                            "SSIM": ssim,
                            "parameter": param,
                        }
                    )

                tag = f"{name.replace('.png','')}_{blur_name}_sigma{safe_tag(sigma)}"
                should_save_panel = save_all_panels or (name, blur_name, sigma) in representative_cases
                if should_save_panel:
                    panel_imgs = [
                        img,
                        blurred,
                        noisy,
                        methods["inverse"][0],
                        methods[f"wiener_fixed_K={FIXED_K:g}"][0],
                        methods["adaptive_wiener_spectral_old"][0],
                        methods["adaptive_wiener_MAD"][0],
                    ]
                    panel_titles = [
                        "original",
                        "blurred",
                        "blurred+noise",
                        "inverse",
                        "fixed Wiener",
                        "old adaptive",
                        "adaptive MAD",
                    ]
                    save_panel(output_dir / "comparisons" / f"{tag}_comparison.png", panel_imgs, panel_titles, ncols=4)

                    if save_spectra and (name, blur_name, sigma) in representative_cases:
                        save_spectrum_panel(output_dir / "spectra" / f"{tag}_spectrum.png", panel_imgs, panel_titles)

                if save_single_images:
                    single_dir = output_dir / "single_images" / tag
                    save_img(single_dir / "original.png", img)
                    save_img(single_dir / "blurred.png", blurred)
                    save_img(single_dir / "blurred_noise.png", noisy)
                    for method, (rec, _) in methods.items():
                        save_img(single_dir / f"{safe_tag(method)}.png", rec)

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "metrics_v2_full.csv", index=False)
    save_summaries_and_plots(df, output_dir)
    return df


def save_summaries_and_plots(df: pd.DataFrame, output_dir: Path) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Summary tables.
    summary_method = df.groupby("method", as_index=False)[["PSNR", "SSIM"]].mean().sort_values("PSNR", ascending=False)
    summary_noise = df.groupby(["noise_sigma", "method"], as_index=False)[["PSNR", "SSIM"]].mean()
    summary_blur_family = df.groupby(["blur_family", "method"], as_index=False)[["PSNR", "SSIM"]].mean()
    summary_blur_strength = df.groupby(["blur_family", "blur_strength", "method"], as_index=False)[["PSNR", "SSIM"]].mean()
    summary_image = df.groupby(["image", "method"], as_index=False)[["PSNR", "SSIM"]].mean()

    summary_method.to_csv(output_dir / "summary_by_method.csv", index=False)
    summary_noise.to_csv(output_dir / "summary_by_noise.csv", index=False)
    summary_blur_family.to_csv(output_dir / "summary_by_blur_family.csv", index=False)
    summary_blur_strength.to_csv(output_dir / "summary_by_blur_strength.csv", index=False)
    summary_image.to_csv(output_dir / "summary_by_image.csv", index=False)

    # Adaptive K statistics.
    k_df = df[df["method"] == "adaptive_wiener_MAD"].copy()
    k_stats_noise = k_df.groupby("noise_sigma")["parameter"].agg(["mean", "median", "min", "max"]).reset_index()
    k_stats_blur = k_df.groupby(["blur_family", "blur_strength", "noise_sigma"])["parameter"].agg(["mean", "median", "min", "max"]).reset_index()
    k_stats_noise.to_csv(output_dir / "adaptive_k_stats_by_noise.csv", index=False)
    k_stats_blur.to_csv(output_dir / "adaptive_k_stats_by_blur_and_noise.csv", index=False)

    # Win-count tables.
    idx_cols = ["image", "blur_name", "blur_family", "blur_strength", "noise_sigma"]
    win_psnr = df.loc[df.groupby(idx_cols)["PSNR"].idxmax()].copy()
    win_ssim = df.loc[df.groupby(idx_cols)["SSIM"].idxmax()].copy()
    win_psnr["criterion"] = "PSNR"
    win_ssim["criterion"] = "SSIM"
    wins = pd.concat([win_psnr, win_ssim], ignore_index=True)
    win_counts = wins.groupby(["criterion", "method"], as_index=False).size().rename(columns={"size": "win_count"})
    win_counts.to_csv(output_dir / "method_win_counts.csv", index=False)

    # Adaptive vs fixed gain table.
    fixed_name = f"wiener_fixed_K={FIXED_K:g}"
    fixed = df[df["method"] == fixed_name]
    adapt = df[df["method"] == "adaptive_wiener_MAD"]
    merge_cols = ["image", "blur_name", "blur_family", "blur_strength", "noise_sigma"]
    gain = adapt.merge(fixed, on=merge_cols, suffixes=("_adaptive", "_fixed"))
    gain["PSNR_gain_adaptive_minus_fixed"] = gain["PSNR_adaptive"] - gain["PSNR_fixed"]
    gain["SSIM_gain_adaptive_minus_fixed"] = gain["SSIM_adaptive"] - gain["SSIM_fixed"]
    gain.to_csv(output_dir / "adaptive_vs_fixed_per_condition.csv", index=False)
    gain_summary = gain.groupby(["blur_family", "blur_strength", "noise_sigma"], as_index=False)[[
        "PSNR_gain_adaptive_minus_fixed", "SSIM_gain_adaptive_minus_fixed"
    ]].mean()
    gain_summary.to_csv(output_dir / "adaptive_vs_fixed_gain_summary.csv", index=False)

    # New adaptive versus old spectral adaptive ablation.
    old_name = "adaptive_wiener_spectral_old"
    old_adapt = df[df["method"] == old_name]
    if not old_adapt.empty:
        old_gain = adapt.merge(old_adapt, on=merge_cols, suffixes=("_MAD", "_old"))
        old_gain["PSNR_gain_MAD_minus_old"] = old_gain["PSNR_MAD"] - old_gain["PSNR_old"]
        old_gain["SSIM_gain_MAD_minus_old"] = old_gain["SSIM_MAD"] - old_gain["SSIM_old"]
        old_gain.to_csv(output_dir / "MAD_adaptive_vs_old_adaptive_per_condition.csv", index=False)
        old_gain_summary = old_gain.groupby(["blur_family", "blur_strength", "noise_sigma"], as_index=False)[[
            "PSNR_gain_MAD_minus_old", "SSIM_gain_MAD_minus_old"
        ]].mean()
        old_gain_summary.to_csv(output_dir / "MAD_adaptive_vs_old_adaptive_gain_summary.csv", index=False)

    # Plots.
    plot_overall_bars(summary_method, plot_dir)
    plot_vs_noise(summary_noise, plot_dir)
    plot_vs_blur_strength(summary_blur_strength, plot_dir)
    plot_adaptive_k(k_stats_noise, k_stats_blur, plot_dir)
    plot_gain_heatmaps(gain_summary, plot_dir)


def plot_overall_bars(summary_method: pd.DataFrame, plot_dir: Path) -> None:
    for metric in ["PSNR", "SSIM"]:
        data = summary_method.sort_values(metric, ascending=False)
        plt.figure(figsize=(8, 4.5))
        plt.bar(data["method"], data[metric])
        plt.xticks(rotation=20, ha="right")
        plt.ylabel(metric if metric == "SSIM" else "PSNR (dB)")
        plt.title(f"Overall average {metric} by method")
        plt.tight_layout()
        plt.savefig(plot_dir / f"overall_{metric}_by_method.png", dpi=160)
        plt.close()


def plot_vs_noise(summary_noise: pd.DataFrame, plot_dir: Path) -> None:
    for metric in ["PSNR", "SSIM"]:
        plt.figure(figsize=(7.5, 4.8))
        for method, sub in summary_noise.groupby("method"):
            sub = sub.sort_values("noise_sigma")
            plt.plot(sub["noise_sigma"], sub[metric], marker="o", label=method)
        plt.xlabel("Noise standard deviation")
        plt.ylabel(metric if metric == "SSIM" else "PSNR (dB)")
        plt.title(f"{metric} versus noise level")
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(plot_dir / f"{metric}_vs_noise.png", dpi=160)
        plt.close()


def plot_vs_blur_strength(summary_strength: pd.DataFrame, plot_dir: Path) -> None:
    for family in sorted(summary_strength["blur_family"].unique()):
        fam = summary_strength[summary_strength["blur_family"] == family]
        for metric in ["PSNR", "SSIM"]:
            plt.figure(figsize=(7.5, 4.8))
            for method, sub in fam.groupby("method"):
                sub = sub.sort_values("blur_strength")
                plt.plot(sub["blur_strength"], sub[metric], marker="o", label=method)
            plt.xlabel("Motion length" if family == "motion" else "Defocus radius")
            plt.ylabel(metric if metric == "SSIM" else "PSNR (dB)")
            plt.title(f"{metric} versus {family} blur strength")
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=8)
            plt.tight_layout()
            plt.savefig(plot_dir / f"{metric}_vs_{family}_strength.png", dpi=160)
            plt.close()


def plot_adaptive_k(k_stats_noise: pd.DataFrame, k_stats_blur: pd.DataFrame, plot_dir: Path) -> None:
    plt.figure(figsize=(7.0, 4.5))
    plt.plot(k_stats_noise["noise_sigma"], k_stats_noise["mean"], marker="o", label="mean")
    plt.plot(k_stats_noise["noise_sigma"], k_stats_noise["median"], marker="o", label="median")
    plt.xlabel("Noise standard deviation")
    plt.ylabel("Estimated adaptive K")
    plt.title("Adaptive K increases with estimated noise level")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "adaptive_K_vs_noise.png", dpi=160)
    plt.close()

    for family in sorted(k_stats_blur["blur_family"].unique()):
        fam = k_stats_blur[k_stats_blur["blur_family"] == family]
        plt.figure(figsize=(7.5, 4.8))
        for strength, sub in fam.groupby("blur_strength"):
            sub = sub.sort_values("noise_sigma")
            label = f"strength={strength:g}"
            plt.plot(sub["noise_sigma"], sub["mean"], marker="o", label=label)
        plt.xlabel("Noise standard deviation")
        plt.ylabel("Mean estimated adaptive K")
        plt.title(f"Adaptive K by noise and {family} blur strength")
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(plot_dir / f"adaptive_K_{family}_strength_noise.png", dpi=160)
        plt.close()


def plot_gain_heatmaps(gain_summary: pd.DataFrame, plot_dir: Path) -> None:
    for family in sorted(gain_summary["blur_family"].unique()):
        fam = gain_summary[gain_summary["blur_family"] == family]
        for metric_col, title_metric in [
            ("PSNR_gain_adaptive_minus_fixed", "PSNR gain"),
            ("SSIM_gain_adaptive_minus_fixed", "SSIM gain"),
        ]:
            pivot = fam.pivot(index="blur_strength", columns="noise_sigma", values=metric_col).sort_index()
            plt.figure(figsize=(6.2, 4.4))
            plt.imshow(pivot.values, aspect="auto")
            plt.colorbar(label=f"adaptive - fixed {title_metric}")
            plt.xticks(range(len(pivot.columns)), [str(c) for c in pivot.columns])
            plt.yticks(range(len(pivot.index)), [str(i) for i in pivot.index])
            plt.xlabel("Noise sigma")
            plt.ylabel("Motion length" if family == "motion" else "Defocus radius")
            plt.title(f"Adaptive vs fixed Wiener: {family} {title_metric}")
            plt.tight_layout()
            plt.savefig(plot_dir / f"heatmap_{family}_{metric_col}.png", dpi=160)
            plt.close()



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default=".")
    parser.add_argument("--output_dir", type=str, default="v2_results")
    parser.add_argument("--include_angle45", action="store_true", help="Also include motion blur at 45 degrees.")
    parser.add_argument("--save_all_panels", action="store_true", help="Save comparison panels for every degraded condition.")
    parser.add_argument("--save_single_images", action="store_true", help="Save every restored image separately.")
    parser.add_argument("--no_spectra", action="store_true", help="Do not save representative spectrum panels.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    missing = [name for name in IMG_NAMES if not (input_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing input images in {input_dir}: {missing}")

    df = run_experiment(
        input_dir=input_dir,
        output_dir=output_dir,
        include_angle45=args.include_angle45,
        save_all_panels=args.save_all_panels,
        save_single_images=args.save_single_images,
        save_spectra=not args.no_spectra,
    )
    print("Finished V2 experiment.")
    print("Output folder:", output_dir)
    print(df.groupby("method")[["PSNR", "SSIM"]].mean().sort_values("PSNR", ascending=False))


if __name__ == "__main__":
    main()
