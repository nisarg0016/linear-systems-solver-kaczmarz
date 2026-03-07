"""
Qualitative & Quantitative Evaluation of CT Reconstruction

Loads the original phantom (sim_512.dat) and the reconstruction
(recon.npy or .bin), computes SSIM and MAE on the central slices
in all three orthogonal planes (XY / Axial, YZ / Sagittal,
XZ / Coronal), and displays a side-by-side comparison.

Usage:
    python examples/visualize_results.py
    python examples/visualize_results.py --recon recon.npy --save results.png
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_phantom(filepath: str, npix: int = 512) -> np.ndarray:
    """Load the original phantom volume from a .dat text file."""
    print(f"Loading phantom from: {filepath} ...")
    with open(filepath, "r") as f:
        data = []
        for line in f:
            data.extend(float(x) for x in line.split())
    if len(data) != npix ** 3:
        raise ValueError(f"Expected {npix**3} values, got {len(data)}")
    volume = np.array(data, dtype=np.float64).reshape((npix, npix, npix))
    print(f"  Shape: {volume.shape}  Range: [{volume.min():.4f}, {volume.max():.4f}]")
    return volume


def compute_metrics(original: np.ndarray, reconstructed: np.ndarray):
    """Compute SSIM and MAE between two 2-D slices."""
    combined_min = min(original.min(), reconstructed.min())
    combined_max = max(original.max(), reconstructed.max())
    data_range = combined_max - combined_min

    orig_norm = (original - combined_min) / data_range if data_range > 0 else original
    recon_norm = (reconstructed - combined_min) / data_range if data_range > 0 else reconstructed

    ssim_val = ssim(orig_norm, recon_norm, data_range=1.0)
    mae_val = np.mean(np.abs(orig_norm - recon_norm))
    return ssim_val, mae_val


def extract_slices(volume: np.ndarray, mid: int):
    """Extract the three central orthogonal slices from a 3-D volume.

    Returns
    -------
    dict  with keys 'XY (Axial)', 'YZ (Sagittal)', 'XZ (Coronal)'
    """
    return {
        "XY (Axial)":    volume[mid, :, :],   # z = mid
        "YZ (Sagittal)": volume[:, :, mid],    # x = mid
        "XZ (Coronal)":  volume[:, mid, :],    # y = mid
    }


def visualize_all_planes(orig_vol, recon_vol, mid, save_path=None):
    """3-row × 3-col figure: one row per plane, cols = Original | Reconstructed | Difference."""
    orig_slices  = extract_slices(orig_vol, mid)
    recon_slices = extract_slices(recon_vol, mid)
    plane_names  = list(orig_slices.keys())

    fig, axes = plt.subplots(3, 3, figsize=(18, 18))

    metrics_text_parts = []

    for row, plane in enumerate(plane_names):
        o = orig_slices[plane]
        r = recon_slices[plane]
        diff = np.abs(o - r)

        ssim_val, mae_val = compute_metrics(o, r)
        metrics_text_parts.append(f"{plane}:  SSIM = {ssim_val:.4f}  |  MAE = {mae_val:.6f}")

        vmin = min(o.min(), r.min())
        vmax = max(o.max(), r.max())

        # Original
        im0 = axes[row, 0].imshow(o, cmap="gray", vmin=vmin, vmax=vmax)
        axes[row, 0].set_title(f"Original – {plane}", fontsize=13)
        axes[row, 0].axis("off")
        plt.colorbar(im0, ax=axes[row, 0], fraction=0.046, pad=0.04)

        # Reconstructed
        im1 = axes[row, 1].imshow(r, cmap="gray", vmin=vmin, vmax=vmax)
        axes[row, 1].set_title(f"Reconstructed – {plane}", fontsize=13)
        axes[row, 1].axis("off")
        plt.colorbar(im1, ax=axes[row, 1], fraction=0.046, pad=0.04)

        # Absolute difference
        im2 = axes[row, 2].imshow(diff, cmap="hot")
        axes[row, 2].set_title(f"Difference – {plane}", fontsize=13)
        axes[row, 2].axis("off")
        plt.colorbar(im2, ax=axes[row, 2], fraction=0.046, pad=0.04)

    # Print per-plane metrics to console
    print(f"\n{'='*60}")
    for line in metrics_text_parts:
        print(f"  {line}")
    print(f"{'='*60}")

    fig.suptitle(
        "Central Slice Comparison — All Planes\n" + "\n".join(metrics_text_parts),
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"\nFigure saved to: {save_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize CT reconstruction quality (SSIM & MAE) in XY, YZ, XZ planes"
    )
    parser.add_argument(
        "--input", "-i", default="sim_512.dat",
        help="Path to original phantom .dat file (default: sim_512.dat)",
    )
    parser.add_argument(
        "--recon", "-r", default="recon.npy",
        help="Path to reconstructed volume (.npy or .bin) (default: recon.npy)",
    )
    parser.add_argument(
        "--npix", type=int, default=512, help="Volume size (default: 512)"
    )
    parser.add_argument(
        "--save", "-s", default=None,
        help="Save the figure to this path (e.g. results.png)",
    )
    args = parser.parse_args()

    # Resolve paths relative to workspace root
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = args.input if os.path.isabs(args.input) else os.path.join(root, args.input)
    recon_path = args.recon if os.path.isabs(args.recon) else os.path.join(root, args.recon)

    # --- Load data ---
    original_vol = load_phantom(input_path, args.npix)

    print(f"Loading reconstruction from: {recon_path} ...")
    if recon_path.endswith(".bin"):
        recon_vol = np.fromfile(recon_path, dtype=np.float64).reshape(
            (args.npix, args.npix, args.npix)
        )
    else:
        recon_vol = np.load(recon_path).astype(np.float64)
    print(f"  Shape: {recon_vol.shape}  Range: [{recon_vol.min():.4f}, {recon_vol.max():.4f}]")

    if original_vol.shape != recon_vol.shape:
        raise ValueError(
            f"Shape mismatch: original {original_vol.shape} vs reconstructed {recon_vol.shape}"
        )

    mid = args.npix // 2
    print(f"\nCentral slice index: {mid}")

    # --- Visualize all three planes ---
    save_path = args.save
    if save_path and not os.path.isabs(save_path):
        save_path = os.path.join(root, save_path)
    visualize_all_planes(original_vol, recon_vol, mid, save_path=save_path)


if __name__ == "__main__":
    main()
