"""
2D Filtered Back Projection (FBP) Reconstruction — Per-Slice

Reconstructs a 3D volume slice-by-slice using scikit-image's iradon()
(2D parallel-beam FBP with hann filter, 473 angles over [0, 180) degrees).

This demonstrates the theoretical best achievable quality for FBP with
this number of projections. It serves as a baseline comparison for the
iterative Kaczmarz/SART algorithms from the cone-beam pipeline.

Usage:
    python examples/reconstruct_2d_fbp.py              # default: 473 angles, hann
    python examples/reconstruct_2d_fbp.py --nangles 720 --filter hamming
    python examples/reconstruct_2d_fbp.py --slice-only 256   # single slice (fast)
"""

import numpy as np
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def reconstruct_slice(orig_slice, theta, filter_name="hann"):
    """Reconstruct a single 2D slice using radon → filter → iradon."""
    from skimage.transform import radon, iradon

    sinogram = radon(orig_slice, theta=theta, circle=False)
    recon = iradon(sinogram, theta=theta, circle=False, filter_name=filter_name)

    # Match output shape to input
    if recon.shape != orig_slice.shape:
        dy = (recon.shape[0] - orig_slice.shape[0]) // 2
        dx = (recon.shape[1] - orig_slice.shape[1]) // 2
        if dy > 0 or dx > 0:
            recon = recon[dy : dy + orig_slice.shape[0], dx : dx + orig_slice.shape[1]]

    recon = np.clip(recon, 0.0, 1.0)
    return recon


def main():
    parser = argparse.ArgumentParser(
        description="2D FBP slice-by-slice reconstruction"
    )
    parser.add_argument(
        "--input", "-i", default="sim_512.dat",
        help="Path to original phantom .dat file",
    )
    parser.add_argument(
        "--output", "-o", default="recon_fbp_2d.npy",
        help="Output .npy file (default: recon_fbp_2d.npy)",
    )
    parser.add_argument("--npix", type=int, default=512, help="Volume size")
    parser.add_argument(
        "--nangles", type=int, default=473, help="Number of projection angles"
    )
    parser.add_argument(
        "--filter", choices=["ramp", "shepp-logan", "cosine", "hamming", "hann"],
        default="hann", help="FBP filter (default: hann)",
    )
    parser.add_argument(
        "--slice-only", type=int, default=None,
        help="Reconstruct only this slice index (for quick test)",
    )
    args = parser.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = (
        args.input if os.path.isabs(args.input)
        else os.path.join(root, args.input)
    )
    output_path = (
        args.output if os.path.isabs(args.output)
        else os.path.join(root, args.output)
    )

    # Load phantom
    print(f"Loading phantom from: {input_path}")
    phantom = np.loadtxt(input_path).reshape((args.npix,) * 3)
    print(f"  Shape: {phantom.shape}  Range: [{phantom.min():.4f}, {phantom.max():.4f}]")

    # Projection angles — covering [0, 180) degrees for proper FBP
    theta = np.linspace(0, 180, args.nangles, endpoint=False)
    print(f"\nFBP settings: {args.nangles} angles, filter={args.filter}")

    if args.slice_only is not None:
        # Single slice mode
        sl = args.slice_only
        print(f"\nReconstructing single slice {sl}...")
        t0 = time.time()
        recon_slice = reconstruct_slice(phantom[sl], theta, args.filter)
        dt = time.time() - t0
        print(f"  Done in {dt:.1f}s")

        # Metrics
        from skimage.metrics import structural_similarity as ssim

        data_range = max(phantom[sl].max(), recon_slice.max()) - min(
            phantom[sl].min(), recon_slice.min()
        )
        s = ssim(phantom[sl], recon_slice, data_range=data_range)
        m = np.mean(np.abs(phantom[sl] - recon_slice))
        c = np.corrcoef(phantom[sl].ravel(), recon_slice.ravel())[0, 1]
        print(f"  SSIM: {s:.4f}  MAE: {m:.6f}  Correlation: {c:.4f}")

        # Save just this slice for quick inspection
        np.save(output_path, recon_slice)
        print(f"  Saved slice to: {output_path}")
        return

    # Full volume reconstruction
    recon = np.zeros_like(phantom)
    t0 = time.time()

    for sl in range(args.npix):
        recon[sl] = reconstruct_slice(phantom[sl], theta, args.filter)
        if (sl + 1) % 50 == 0 or sl == 0 or sl == args.npix - 1:
            elapsed = time.time() - t0
            remaining = elapsed / (sl + 1) * (args.npix - sl - 1)
            print(
                f"  Slice {sl+1:3d}/{args.npix}  "
                f"elapsed: {elapsed:.0f}s  "
                f"remaining: ~{remaining:.0f}s"
            )

    total = time.time() - t0
    print(f"\nReconstruction completed in {total:.1f}s")

    # Central slice metrics
    from skimage.metrics import structural_similarity as ssim

    mid = args.npix // 2
    orig_mid = phantom[mid]
    recon_mid = recon[mid]
    data_range = max(orig_mid.max(), recon_mid.max()) - min(
        orig_mid.min(), recon_mid.min()
    )
    s = ssim(orig_mid, recon_mid, data_range=data_range)
    m = np.mean(np.abs(orig_mid - recon_mid))
    c = np.corrcoef(orig_mid.ravel(), recon_mid.ravel())[0, 1]
    print(f"\nCentral slice (z={mid}) metrics:")
    print(f"  SSIM:        {s:.4f}")
    print(f"  MAE:         {m:.6f}")
    print(f"  Correlation: {c:.4f}")

    # Save
    np.save(output_path, recon)
    print(f"\nSaved reconstruction to: {output_path}  ({recon.nbytes/1e9:.2f} GB)")


if __name__ == "__main__":
    main()
