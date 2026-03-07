"""
CT Reconstruction using Weighted Randomized Kaczmarz

This script adapts the mergecode.c logic to work with the Kaczmarz solver.
It creates an implicit projection matrix for cone-beam CT reconstruction.

The matrix A represents the CT projection geometry where:
- Each row = one detector pixel measurement
- Each column = one voxel in the 3D volume
- A[i,j] = 1 if ray i passes through voxel j

Input: sim_512.dat (512x512x512 phantom volume)
Output: Reconstructed volume
"""

import numpy as np
import sys
import os
import time
from typing import List, Tuple


def format_time(seconds: float) -> str:
    """Format seconds into a human-readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"{m}m {s:02d}s"
    else:
        h, rem = divmod(int(seconds), 3600)
        m, s = divmod(rem, 60)
        return f"{h}h {m:02d}m {s:02d}s"


class ProgressBar:
    """
    Dynamic single-line progress bar that updates in-place.
    Throttles output so it only redraws when there's a visible change.
    """
    
    def __init__(self, total: int, label: str = "Progress", bar_len: int = 40):
        self.total = total
        self.label = label
        self.bar_len = bar_len
        self.start_time = time.time()
        self._last_pct_int = -1       # Last displayed integer percentage
        self._last_update_time = 0.0  # Last time we drew
        self._min_interval = 0.05     # Minimum 50ms between redraws
    
    def update(self, current: int):
        """Update the progress bar. Only redraws when percentage changes or 50ms has passed."""
        pct = current / self.total * 100 if self.total > 0 else 0
        pct_int = int(pct * 10)  # Tenths of a percent resolution
        now = time.time()
        
        # Only redraw if percentage changed or enough time passed or we're done
        if (pct_int == self._last_pct_int 
                and (now - self._last_update_time) < self._min_interval
                and current < self.total):
            return
        
        self._last_pct_int = pct_int
        self._last_update_time = now
        
        filled = int(self.bar_len * current // self.total) if self.total > 0 else 0
        bar = "█" * filled + "░" * (self.bar_len - filled)
        
        elapsed = now - self.start_time
        rate = current / elapsed if elapsed > 0 else 0
        eta = (self.total - current) / rate if rate > 0 else 0
        
        sys.stdout.write(
            f"\r  {self.label}: |{bar}| {pct:5.1f}%  "
            f"[{current:,}/{self.total:,}]  "
            f"{rate:,.0f} it/s  ETA {eta:.0f}s   "
        )
        sys.stdout.flush()
        
        if current >= self.total:
            sys.stdout.write("\n")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sparse_matrix import ImplicitBitMatrix, ExplicitSparseRowMatrix
from src.kaczmarz import WeightedRandomizedKaczmarz, solve_kaczmarz


# =============================================================================
# CT GEOMETRY PARAMETERS (from mergecode.c)
# =============================================================================

class CTGeometry:
    """CT scanner geometry parameters."""
    
    def __init__(
        self,
        npix: int = 512,        # Volume size (npix x npix x npix)
        Npa: int = 473,         # Number of projection angles
        Ndet: int = 512,        # Detector pixels per row
        Nslice: int = 512,      # Detector rows (slices)
        span: int = 640,        # Angular span in degrees
        d: float = 0.8,         # Detector pixel size
        scob: float = 166.641,  # Source to center of rotation
        scd: float = 1238.157,  # Source to detector distance
        shift: float = 0.136,   # Axial shift per rotation
        dia: float = 80.86      # Phantom diameter
    ):
        self.npix = npix
        self.Npa = Npa
        self.Ndet = Ndet
        self.Nslice = Nslice
        self.span = span
        self.d = d
        self.scob = scob
        self.scd = scd
        self.shift = shift
        self.dia = dia
        self.pixlen = dia / npix
        
        # Derived values
        self.Nvox = npix ** 3           # Total voxels
        self.Nrays = Npa * Ndet * Nslice  # Total measurements
        
    def __repr__(self):
        return (f"CTGeometry(volume={self.npix}³, projections={self.Npa}, "
                f"detector={self.Ndet}×{self.Nslice})")


# =============================================================================
# RAY TRACING (Python version of weightgen)
# =============================================================================

def compute_ray_voxels(geom: CTGeometry, proj_idx: int, det_j: int, det_k: int) -> np.ndarray:
    """
    Compute which voxels a ray passes through (vectorized).
    
    This is a NumPy-vectorized equivalent of the weightgen() function in mergecode.c.
    
    Args:
        geom: CT geometry parameters
        proj_idx: Projection angle index (0 to Npa-1)
        det_j: Detector slice index (0 to Nslice-1)
        det_k: Detector column index (0 to Ndet-1)
        
    Returns:
        1D numpy array of voxel indices that the ray intersects
    """
    PI = 3.1428571
    theta = (geom.span * PI) / (geom.Npa * 180) * proj_idx
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    
    # Source position
    x1 = geom.scob * sin_t
    y1 = -geom.scob * cos_t
    z1 = 51.63 - proj_idx * geom.shift / geom.Npa
    
    # Detector pixel position
    p1 = ((geom.Ndet / 2) - (0.5 + det_k)) * geom.d
    p3 = z1 + ((geom.Nslice / 2) - (det_j + 0.5)) * geom.d
    p2 = geom.scd - geom.scob
    
    x2 = p1 * cos_t - p2 * sin_t
    y2 = p1 * sin_t + p2 * cos_t
    z2 = p3
    
    dy = y2 - y1
    if abs(dy) < 1e-10:
        return np.array([], dtype=np.int64)
    
    half_dia = geom.dia / 2
    npix = geom.npix
    pixlen = geom.pixlen
    
    # Vectorised: compute all intersections at once
    l = np.arange(npix, dtype=np.float64)
    y21 = -half_dia + l * pixlen
    
    t_param = (y21 - y1) / dy          # Parametric position along ray
    x21 = x1 + (x2 - x1) * t_param
    z21 = z2 * t_param
    
    # Boolean mask for intersections inside the volume
    mask = (
        (y21 >= -half_dia) & (y21 < half_dia) &
        (x21 > -half_dia) & (x21 <= half_dia) &
        (z21 > -half_dia) & (z21 <= half_dia)
    )
    
    if not np.any(mask):
        return np.array([], dtype=np.int64)
    
    x21 = x21[mask]
    y21_m = y21[mask]
    z21 = z21[mask]
    
    m1 = np.clip(np.floor((half_dia - x21) / pixlen).astype(np.int64), 0, npix - 1)
    m2 = np.clip(np.floor((y21_m + half_dia) / pixlen).astype(np.int64), 0, npix - 1)
    m3 = np.clip(np.floor((half_dia - z21) / pixlen).astype(np.int64), 0, npix - 1)
    
    return m1 * npix * npix + m3 * npix + m2


def ray_index_to_params(geom: CTGeometry, ray_idx: int) -> Tuple[int, int, int]:
    """Convert linear ray index to (projection, slice, detector) indices."""
    proj_idx = ray_idx // (geom.Ndet * geom.Nslice)
    remainder = ray_idx % (geom.Ndet * geom.Nslice)
    det_j = remainder // geom.Ndet
    det_k = remainder % geom.Ndet
    return proj_idx, det_j, det_k


# =============================================================================
# IMPLICIT CT MATRIX
# =============================================================================

class ImplicitCTMatrix:
    """
    Implicit representation of the CT projection matrix.
    
    The matrix A has shape (Nrays, Nvox) where:
    - Nrays = Npa × Ndet × Nslice (total detector measurements)
    - Nvox = npix³ (total voxels)
    
    A[i,j] = 1 if ray i passes through voxel j
    
    This matrix is WAY too large to store:
    - For 512³ volume with 473 projections of 512×512:
    - Nrays = 473 × 512 × 512 ≈ 124 million
    - Nvox = 512³ ≈ 134 million
    - Full matrix would need ~16 petabytes!
    """
    
    def __init__(self, geom: CTGeometry, cache_size: int = 10000):
        self.geom = geom
        self._cache = {}
        self._cache_size = cache_size
        self._row_weights = None
        
    def get_row_indices(self, ray_idx: int) -> np.ndarray:
        """Get voxel indices for a given ray."""
        if ray_idx in self._cache:
            return self._cache[ray_idx]
        
        proj_idx, det_j, det_k = ray_index_to_params(self.geom, ray_idx)
        result = compute_ray_voxels(self.geom, proj_idx, det_j, det_k)
        # compute_ray_voxels already returns np.ndarray
        
        # Simple LRU cache management
        if len(self._cache) >= self._cache_size:
            # Remove oldest half
            keys_to_remove = list(self._cache.keys())[:self._cache_size // 2]
            for k in keys_to_remove:
                del self._cache[k]
        
        self._cache[ray_idx] = result
        return result
    
    def get_row_weight(self, ray_idx: int) -> float:
        """Get the weight (number of voxels) for a ray."""
        return float(len(self.get_row_indices(ray_idx)))
    
    @property
    def shape(self) -> Tuple[int, int]:
        return (self.geom.Nrays, self.geom.Nvox)
    
    @property
    def row_weights(self) -> np.ndarray:
        """
        Get all row weights. 
        WARNING: This is slow for large matrices - only use for small tests!
        """
        if self._row_weights is None:
            print("Computing row weights (this may take a while)...")
            self._row_weights = np.zeros(self.geom.Nrays, dtype=np.float64)
            progress = ProgressBar(self.geom.Nrays, label="Weights")
            for i in range(self.geom.Nrays):
                self._row_weights[i] = self.get_row_weight(i)
                progress.update(i + 1)
        return self._row_weights
    
    @property
    def total_weight(self) -> float:
        return np.sum(self.row_weights)
    
    @property
    def m(self) -> int:
        return self.geom.Nrays
    
    @property
    def n(self) -> int:
        return self.geom.Nvox
    
    def sparse_dot(self, ray_idx: int, x: np.ndarray) -> float:
        """Compute ray projection: sum of voxel values along ray."""
        indices = self.get_row_indices(ray_idx)
        if len(indices) == 0:
            return 0.0
        return np.sum(x[indices])
    
    def sparse_axpy(self, ray_idx: int, alpha: float, x: np.ndarray) -> None:
        """x += alpha * a_i (backprojection step)."""
        indices = self.get_row_indices(ray_idx)
        if len(indices) > 0:
            x[indices] += alpha


# =============================================================================
# FILE I/O
# =============================================================================

def load_phantom(filepath: str, npix: int = 512) -> np.ndarray:
    """
    Load phantom volume from .dat file.
    
    Expected format: space-separated float values, npix³ total values
    stored as [i][j][k] with i,j,k each from 0 to npix-1
    """
    print(f"Loading phantom from: {filepath}")
    print(f"Expected size: {npix}³ = {npix**3:,} voxels")
    
    # Read all values
    with open(filepath, 'r') as f:
        data = []
        for line in f:
            data.extend([float(x) for x in line.split()])
    
    if len(data) != npix ** 3:
        raise ValueError(f"Expected {npix**3} values, got {len(data)}")
    
    volume = np.array(data, dtype=np.float32).reshape((npix, npix, npix))
    print(f"Loaded volume: shape={volume.shape}, range=[{volume.min():.2f}, {volume.max():.2f}]")
    
    return volume


def save_volume(filepath: str, volume: np.ndarray):
    """Save reconstructed volume to file."""
    print(f"Saving volume to: {filepath}")
    np.save(filepath, volume)


# =============================================================================
# FORWARD PROJECTION (Generate sinogram from volume)
# =============================================================================

def forward_project(geom: CTGeometry, volume: np.ndarray, 
                    num_projections: int = None) -> np.ndarray:
    """
    Generate projection data (sinogram) from volume.
    
    This simulates the CT measurement process: b = Ax
    """
    if num_projections is None:
        num_projections = geom.Npa
    
    # Flatten volume
    x = volume.ravel().astype(np.float64)
    
    # Create projection matrix
    ct_matrix = ImplicitCTMatrix(geom)
    
    # Compute projections
    total_rays = num_projections * geom.Ndet * geom.Nslice
    b = np.zeros(total_rays, dtype=np.float64)
    
    print(f"Forward projecting {num_projections} angles, {total_rays:,} rays...")
    progress = ProgressBar(total_rays, label="Projecting")
    for i in range(total_rays):
        indices = ct_matrix.get_row_indices(i)
        if len(indices) > 0:
            b[i] = np.sum(x[indices])
        progress.update(i + 1)
    
    return b


# =============================================================================
# FAST DIRECT KACZMARZ (bypasses generic solver for speed)
# =============================================================================

def kaczmarz_direct(
    ct_matrix: ImplicitCTMatrix,
    b: np.ndarray,
    n_voxels: int,
    tolerance: float = 1e-4,
    max_iterations: int = 1_000_000,
    seed: int = 42,
    relaxation: float = 0.25,
    enforce_nonnegativity: bool = True
) -> Tuple[np.ndarray, dict]:
    """
    Fast direct Kaczmarz loop optimised for CT reconstruction.
    
    Avoids the overhead of the generic solver infrastructure by
    inlining the iteration, sampling, and convergence check.
    Provides a live progress bar with ETA in the terminal.
    
    Key improvements for CT quality:
    - Under-relaxation (lambda < 1) prevents overshooting
    - Non-negativity constraint acts as a powerful regulariser
    
    Args:
        ct_matrix: ImplicitCTMatrix with the projection geometry
        b: Sinogram vector (measurements)
        n_voxels: Number of unknowns (npix³)
        tolerance: Stop when RMS residual drops below this
        max_iterations: Hard iteration cap
        seed: Random seed
        relaxation: Relaxation parameter lambda in (0, 2). Values < 1
                    under-relax (more stable), > 1 over-relax (faster
                    but may diverge). Recommended: 0.25 for CT.
        enforce_nonnegativity: If True, clamp voxel values >= 0 after
                               each update (CT physical constraint).
    
    Returns:
        (x, info_dict) where x is the solution vector and info_dict
        contains convergence statistics.
    """
    m = len(b)
    x = np.zeros(n_voxels, dtype=np.float64)
    rng = np.random.RandomState(seed)
    
    tol_sq = tolerance ** 2
    window_size = min(2000, max(500, m // 100))
    residuals = np.zeros(window_size, dtype=np.float64)
    ptr = 0             # Circular-buffer pointer
    window_sum = 0.0
    
    start_time = time.time()
    bar_len = 40
    last_draw = 0.0
    
    converged = False
    final_iter = max_iterations
    
    # Initial calibration: run 1000 iterations to estimate speed
    print(f"  Max iterations : {max_iterations:,}")
    print(f"  Tolerance      : {tolerance:.1e}")
    print(f"  Relaxation     : {relaxation}")
    print(f"  Non-negativity : {enforce_nonnegativity}")
    print(f"  Window size    : {window_size:,}")
    print()
    
    for it in range(1, max_iterations + 1):
        # --- Kaczmarz step ---
        i = rng.randint(0, m)
        indices = ct_matrix.get_row_indices(i)
        weight = float(len(indices))
        if weight == 0:
            continue
        
        dot = np.sum(x[indices])
        residual = b[i] - dot
        alpha = relaxation * residual / weight
        x[indices] += alpha
        
        # Non-negativity constraint (physical: CT values >= 0)
        if enforce_nonnegativity:
            x[indices] = np.maximum(x[indices], 0.0)
        
        r_sq = residual * residual
        
        # --- Moving-window update ---
        old = residuals[ptr]
        residuals[ptr] = r_sq
        window_sum += r_sq - old
        ptr = (ptr + 1) % window_size
        
        # --- Progress + convergence check (every window_size iters) ---
        if it % window_size == 0:
            mean_r_sq = window_sum / window_size
            rms = np.sqrt(max(mean_r_sq, 0.0))
            
            now = time.time()
            elapsed = now - start_time
            rate = it / elapsed if elapsed > 0 else 0
            
            pct = it / max_iterations * 100
            filled = int(bar_len * it // max_iterations)
            bar = "\u2588" * filled + "\u2591" * (bar_len - filled)
            eta = (max_iterations - it) / rate if rate > 0 else 0
            
            sys.stdout.write(
                f"\r  Kaczmarz: |{bar}| {pct:5.1f}%  "
                f"iter {it:>12,}/{max_iterations:,}  "
                f"RMS {rms:.2e}  "
                f"{rate:,.0f} it/s  "
                f"Elapsed {format_time(elapsed)}  ETA {format_time(eta)}   "
            )
            sys.stdout.flush()
            
            if mean_r_sq < tol_sq:
                converged = True
                final_iter = it
                break
    
    # Final line
    elapsed = time.time() - start_time
    rms = np.sqrt(max(window_sum / window_size, 0.0))
    status = "CONVERGED" if converged else "MAX ITERATIONS"
    full_bar = "\u2588" * bar_len
    sys.stdout.write(
        f"\r  Kaczmarz: |{full_bar}| 100.0%  "
        f"iter {final_iter:>12,}/{max_iterations:,}  "
        f"RMS {rms:.2e}  "
        f"Done in {format_time(elapsed)}  [{status}]          \n"
    )
    sys.stdout.flush()
    
    info = {
        "converged": converged,
        "iterations": final_iter,
        "elapsed": elapsed,
        "rms_residual": rms,
        "iterations_per_second": final_iter / elapsed if elapsed > 0 else 0,
    }
    return x, info


# =============================================================================
# RECONSTRUCTION (Inverse problem: find x given b)
# =============================================================================

def reconstruct_kaczmarz(geom: CTGeometry, sinogram: np.ndarray,
                         tolerance: float = 1e-4,
                         max_iterations: int = 1_000_000,
                         verbose: bool = True,
                         relaxation: float = 0.25,
                         enforce_nonnegativity: bool = True) -> np.ndarray:
    """
    Reconstruct volume from sinogram using Kaczmarz iteration.
    
    Solves: Ax = b where
    - A = projection matrix
    - b = sinogram (measurements)
    - x = volume to reconstruct
    
    Uses a fast direct Kaczmarz loop with live progress and ETA.
    Applies under-relaxation and non-negativity for CT-quality output.
    """
    print("\n" + "="*60)
    print("CT RECONSTRUCTION via Kaczmarz")
    print("="*60)
    print(f"  Volume size    : {geom.npix}\u00b3 = {geom.Nvox:,} voxels")
    print(f"  Measurements   : {len(sinogram):,} rays")
    print()
    
    # Create implicit matrix
    ct_matrix = ImplicitCTMatrix(geom)
    
    # Run the fast direct solver
    x, info = kaczmarz_direct(
        ct_matrix, sinogram,
        n_voxels=geom.Nvox,
        tolerance=tolerance,
        max_iterations=max_iterations,
        seed=42,
        relaxation=relaxation,
        enforce_nonnegativity=enforce_nonnegativity
    )
    
    # Reshape to volume
    volume = x.reshape((geom.npix, geom.npix, geom.npix))
    
    # Post-processing: clip to physical range [0, 1]
    volume = np.clip(volume, 0.0, 1.0)
    
    print(f"\nReconstruction complete!")
    print(f"  Status     : {'CONVERGED' if info['converged'] else 'MAX ITERATIONS'}")
    print(f"  Iterations : {info['iterations']:,}")
    print(f"  Time       : {format_time(info['elapsed'])}")
    print(f"  Throughput : {info['iterations_per_second']:,.0f} iter/s")
    print(f"  RMS resid  : {info['rms_residual']:.2e}")
    print(f"  Volume     : [{volume.min():.4f}, {volume.max():.4f}]")
    
    return volume


# =============================================================================
# DEMO WITH SMALLER PROBLEM
# =============================================================================

def demo_small_ct():
    """Demo with a smaller CT problem for quick testing."""
    print("\n" + "="*60)
    print("DEMO: Small CT Reconstruction")
    print("="*60)
    
    # Use smaller dimensions for demo
    geom = CTGeometry(
        npix=32,      # 32³ volume instead of 512³
        Npa=20,       # 20 projections instead of 473
        Ndet=32,
        Nslice=32
    )
    
    print(f"Geometry: {geom}")
    print(f"Total voxels: {geom.Nvox:,}")
    print(f"Total rays: {geom.Nrays:,}")
    
    # Create a simple phantom (sphere)
    print("\nCreating phantom...")
    phantom = np.zeros((geom.npix, geom.npix, geom.npix), dtype=np.float32)
    center = geom.npix // 2
    radius = geom.npix // 4
    for i in range(geom.npix):
        for j in range(geom.npix):
            for k in range(geom.npix):
                if (i-center)**2 + (j-center)**2 + (k-center)**2 < radius**2:
                    phantom[i,j,k] = 1.0
    
    print(f"Phantom shape: {phantom.shape}")
    print(f"Non-zero voxels: {np.sum(phantom > 0):,}")
    
    # Forward project
    print("\nForward projection...")
    sinogram = forward_project(geom, phantom)
    print(f"Sinogram shape: {sinogram.shape}")
    
    # Reconstruct
    print("\nReconstruction...")
    recon = reconstruct_kaczmarz(
        geom, sinogram,
        tolerance=1e-3,
        max_iterations=100_000,
        verbose=True,
        relaxation=0.25,
        enforce_nonnegativity=True
    )
    
    # Compare
    error = np.linalg.norm(recon - phantom) / np.linalg.norm(phantom)
    print(f"\nRelative reconstruction error: {error:.2%}")
    
    return phantom, recon


# =============================================================================
# MAIN: Using sim_512.dat
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="CT Reconstruction with Kaczmarz")
    parser.add_argument('--input', '-i', help='Input phantom file (e.g., sim_512.dat)')
    parser.add_argument('--output', '-o', help='Output reconstructed volume')
    parser.add_argument('--demo', action='store_true', help='Run small demo')
    parser.add_argument('--npix', type=int, default=512, help='Volume size')
    parser.add_argument('--nproj', type=int, default=200, help='Number of projections to use')
    parser.add_argument('--tol', type=float, default=1e-4, help='Tolerance')
    parser.add_argument('--maxiter', type=int, default=2_000_000, help='Max iterations')
    parser.add_argument('--relax', type=float, default=0.25,
                        help='Relaxation parameter (0<\u03bb<2, default 0.25)')
    parser.add_argument('--no-nonneg', action='store_true',
                        help='Disable non-negativity constraint')
    parser.add_argument('--sinogram', default=None,
                        help='Path to save/load sinogram (.npy). Saves after '
                             'forward projection; loads if file exists to skip it.')
    parser.add_argument('--rerun', action='store_true',
                        help='Skip forward projection, load sinogram from --sinogram '
                             'and only re-run the Kaczmarz solver.')
    
    args = parser.parse_args()
    
    if args.demo:
        demo_small_ct()
        return
    
    if not args.input:
        print("Usage:")
        print("  python ct_reconstruction.py --demo                    # Run small demo")
        print("  python ct_reconstruction.py -i sim_512.dat -o recon.npy  # Full reconstruction")
        print("\nFor sim_512.dat (512³ volume), this will take significant time and memory.")
        print("Consider using --nproj to limit the number of projections.")
        return
    
    # Setup geometry
    geom = CTGeometry(npix=args.npix, Npa=args.nproj)
    print(f"\nUsing {args.nproj} projections")
    
    # --- Sinogram: load from cache or compute via forward projection ---
    sinogram_path = args.sinogram
    if args.rerun and sinogram_path and os.path.exists(sinogram_path):
        # Rerun mode: skip phantom loading and forward projection entirely
        print(f"\nLoading cached sinogram from: {sinogram_path}")
        sinogram = np.load(sinogram_path)
        print(f"  Sinogram length: {len(sinogram):,}")
    elif sinogram_path and os.path.exists(sinogram_path):
        # Cached sinogram exists – load it
        print(f"\nLoading cached sinogram from: {sinogram_path}")
        sinogram = np.load(sinogram_path)
        print(f"  Sinogram length: {len(sinogram):,}")
    else:
        if not args.input:
            print("ERROR: --input is required for forward projection.")
            print("  Use --sinogram <path> with --rerun to skip forward projection.")
            return
        # Load phantom and forward project
        phantom = load_phantom(args.input, args.npix)
        sinogram = forward_project(geom, phantom, args.nproj)
        # Save sinogram for future reuse
        if sinogram_path:
            np.save(sinogram_path, sinogram)
            print(f"  Sinogram cached to: {sinogram_path}")
    
    # --- Reconstruct ---
    recon = reconstruct_kaczmarz(
        geom, sinogram,
        tolerance=args.tol,
        max_iterations=args.maxiter,
        relaxation=args.relax,
        enforce_nonnegativity=not args.no_nonneg
    )
    
    # Save
    if args.output:
        save_volume(args.output, recon)
    
    return recon


if __name__ == "__main__":
    main()
