# TV-Regularised Kaczmarz (SIRT) CT Reconstruction

Reconstructs a 512³ cone-beam CT phantom from `sim_512.dat` using
**TV-regularised SIRT** (a Kaczmarz-family iterative method) with
FBP warm-start, achieving **SSIM = 0.9119** on the central slice.

## Quick Start

```bash
# Single-slice test (~3 min)
python examples/reconstruct_kaczmarz.py --slice-only 256

# Full volume, all CPU cores
python examples/reconstruct_kaczmarz.py -o recon_kacz.npy

# Visualise results
python examples/visualize_results.py --recon recon_kacz.npy --save results.png
```

## Algorithm

Each 2-D slice is reconstructed independently via SIRT with
scikit-image's `radon()`/`iradon()` for model-consistent
forward/back projection:

1. **FBP warm-start** — `iradon(..., filter_name='hann')`
2. **SIRT loop** (25 sweeps) — `x += λ · (A^T R⁻¹ (b − Ax)) / C`
3. **TV regularisation** — `denoise_tv_chambolle(x, w)` with
   exponentially decaying weight (0.003 → 0.0005)
4. **Non-negativity + [0, 1] clamping** after every update

Key parameters: λ = 1.9, 473 angles, 25 sweeps.

## Project Structure

```
AA Project/
├── README.md
├── progress_report.tex / .pdf
├── sim_512.dat                         # Input phantom (512³)
├── sim_512.dat.npy                     # Cached phantom (.npy)
├── recon_kacz.npy                      # Reconstruction output
├── results.png                         # Visualisation output
└── examples/
    ├── reconstruct_kaczmarz.py         # SIRT + TV reconstruction
    └── visualize_results.py            # 3×3 multi-plane comparison
```

## Command-Line Options

| Flag | Default | Description |
|------|---------|-------------|
| `-i / --input` | `sim_512.dat` | Input phantom file |
| `-o / --output` | `recon_kacz.npy` | Output `.npy` file |
| `--sweeps` | 25 | SIRT sweeps per slice |
| `--relax` | 1.9 | Relaxation parameter λ |
| `--tv` | 0.003 | Initial TV weight |
| `--tv-end` | 0.0005 | Final TV weight (exp. decay) |
| `--no-tv` | — | Disable TV regularisation |
| `--no-warmstart` | — | Zero initialisation instead of FBP |
| `--slice-only Z` | — | Reconstruct only slice Z |
| `-j / --workers` | CPU count | Parallel workers |

## Results

| Method | SSIM | MAE | Correlation |
|--------|------|-----|-------------|
| Row-action Kaczmarz (500K iter) | 0.2269 | 0.1031 | 0.032 |
| 2-D FBP (Hann, 473 angles) | 0.8604 | 0.0356 | 0.946 |
| **SIRT + TV (25 sweeps, Hann)** | **0.9119** | **0.0302** | **0.952** |

## Requirements

- Python 3.8+
- NumPy, scikit-image, SciPy, Matplotlib
- ~3 GB RAM (phantom + reconstruction volume)

```bash
pip install numpy scikit-image scipy matplotlib
```

## References

1. Strohmer, T., & Vershynin, R. (2009). A randomized Kaczmarz algorithm with exponential convergence.
2. Kaczmarz, S. (1937). Angenäherte Auflösung von Systemen linearer Gleichungen.

---
*Advanced Algorithms Project — March 2026*
