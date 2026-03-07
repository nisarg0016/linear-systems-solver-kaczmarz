"""
Weighted Randomized Kaczmarz Algorithm Implementation

This module implements the core Kaczmarz algorithm with:
- Weighted row sampling (probability ∝ row sparsity)
- Minimum-norm initialization (x₀ = 0)
- Adaptive convergence detection
- Inconsistency detection with graceful fallback

The algorithm solves Ax = b for extremely large sparse binary matrices
that cannot be explicitly stored in memory.

Mathematical Reference:
    Strohmer, T., & Vershynin, R. (2009). A randomized Kaczmarz algorithm 
    with exponential convergence. Journal of Fourier Analysis and Applications.

Date: January 2026
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Union, Callable
import time

from .sparse_matrix import ImplicitBitMatrix, ExplicitSparseRowMatrix
from .sampling import WeightedSampler
from .stopping import AdaptiveStoppingCriterion, StoppingReason, ConvergenceInfo, ResidualHistory


# Type alias for matrix types we support
SparseMatrix = Union[ImplicitBitMatrix, ExplicitSparseRowMatrix]


@dataclass
class SolverResult:
    """
    Complete result from the Kaczmarz solver.
    
    Attributes:
        x: Solution vector
        convergence: Detailed convergence information
        elapsed_time: Total solve time in seconds
        iterations_per_second: Throughput metric
    """
    x: np.ndarray
    convergence: ConvergenceInfo
    elapsed_time: float
    iterations_per_second: float
    
    @property
    def converged(self) -> bool:
        """Whether the solver converged to tolerance."""
        return self.convergence.reason == StoppingReason.CONVERGED
    
    @property 
    def is_consistent(self) -> bool:
        """Whether the system appears to have an exact solution."""
        return self.convergence.is_consistent()
    
    def __repr__(self) -> str:
        status = "✓" if self.converged else "✗"
        return (
            f"SolverResult({status} {self.convergence.reason.value}, "
            f"iter={self.convergence.iterations:,}, "
            f"time={self.elapsed_time:.2f}s, "
            f"residual={np.sqrt(self.convergence.final_residual_mean):.2e})"
        )


class WeightedRandomizedKaczmarz:
    """
    Weighted Randomized Kaczmarz Solver for Large Sparse Systems.
    
    This solver implements Algorithm 1 from the project specification:
    
    1. Initialize x = 0 (minimum-norm bias)
    2. Sample row i with P(i) ∝ ||a_i||² = nnz(row_i) for binary matrices
    3. Update: x ← x + (b_i - ⟨a_i, x⟩)/||a_i||² · a_i
    4. Check convergence using moving-average residuals
    5. Detect stagnation for inconsistent systems
    
    Convergence Guarantee:
        For consistent systems, E[||x_k - x*||²] ≤ (1 - σ²_min/||A||²_F)^k ||x_0 - x*||²
        
    Minimum-Norm Property:
        Starting from x₀ = 0, iterates converge to x* = A†b, the minimum-norm solution.
    """
    
    def __init__(
        self,
        tolerance: float = 1e-6,
        max_iterations: int = 10_000_000,
        window_size: int = 1000,
        stagnation_threshold: float = 0.1,
        seed: Optional[int] = None,
        verbose: bool = False,
        track_history: bool = False,
        history_interval: int = 100
    ):
        """
        Initialize the Kaczmarz solver.
        
        Args:
            tolerance: Stop when RMS residual < tolerance
            max_iterations: Maximum number of iterations
            window_size: Size of moving window for convergence detection
            stagnation_threshold: CV threshold for detecting inconsistency
            seed: Random seed for reproducibility
            verbose: Print progress during solving
            track_history: Store full residual history (memory intensive)
            history_interval: If tracking, store every k-th residual
        """
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.window_size = window_size
        self.stagnation_threshold = stagnation_threshold
        self.seed = seed
        self.verbose = verbose
        self.track_history = track_history
        self.history_interval = history_interval
        
        # These are initialized during solve()
        self._sampler: Optional[WeightedSampler] = None
        self._stopping: Optional[AdaptiveStoppingCriterion] = None
        self._history: Optional[ResidualHistory] = None
    
    def solve(
        self,
        A: SparseMatrix,
        b: np.ndarray,
        x0: Optional[np.ndarray] = None
    ) -> SolverResult:
        """
        Solve the system Ax = b using randomized Kaczmarz.
        
        Args:
            A: Sparse matrix (implicit or explicit)
            b: Right-hand side vector of length m
            x0: Initial guess (default: zero vector for minimum-norm)
            
        Returns:
            SolverResult containing solution and convergence info
            
        Raises:
            ValueError: If dimensions don't match
        """
        # Validate inputs
        m, n = A.shape
        b = np.asarray(b, dtype=np.float64)
        
        if len(b) != m:
            raise ValueError(f"RHS length {len(b)} doesn't match matrix rows {m}")
        
        # Initialize solution vector
        if x0 is None:
            # Zero initialization for minimum-norm solution
            x = np.zeros(n, dtype=np.float64)
        else:
            x = np.asarray(x0, dtype=np.float64).copy()
            if len(x) != n:
                raise ValueError(f"Initial guess length {len(x)} doesn't match columns {n}")
        
        # Initialize weighted sampler
        self._sampler = WeightedSampler(A.row_weights, seed=self.seed)
        
        # Initialize stopping criterion
        self._stopping = AdaptiveStoppingCriterion(
            tolerance=self.tolerance,
            window_size=self.window_size,
            max_iterations=self.max_iterations,
            stagnation_cv_threshold=self.stagnation_threshold,
            verbose=self.verbose
        )
        
        # Optional history tracking
        if self.track_history:
            self._history = ResidualHistory(sample_interval=self.history_interval)
        
        # Main iteration loop
        start_time = time.perf_counter()
        conv_info = self._kaczmarz_loop(A, b, x)
        elapsed = time.perf_counter() - start_time
        
        # Compute throughput
        iters_per_sec = conv_info.iterations / elapsed if elapsed > 0 else float('inf')
        
        return SolverResult(
            x=x,
            convergence=conv_info,
            elapsed_time=elapsed,
            iterations_per_second=iters_per_sec
        )
    
    def _kaczmarz_loop(
        self, 
        A: SparseMatrix, 
        b: np.ndarray, 
        x: np.ndarray
    ) -> ConvergenceInfo:
        """
        Core Kaczmarz iteration loop.
        
        This is the performance-critical inner loop. For each iteration:
        1. Sample a row index proportional to row weight
        2. Compute sparse dot product ⟨a_i, x⟩
        3. Compute residual r_i = b_i - ⟨a_i, x⟩
        4. Update x += (r_i / ||a_i||²) * a_i
        5. Check stopping criterion
        
        Time per iteration: O(s) where s is average row sparsity
        """
        sampler = self._sampler
        stopping = self._stopping
        history = self._history
        
        while True:
            # Step 1: Sample row index
            i = sampler.sample()
            
            # Step 2: Get row data
            indices = A.get_row_indices(i)
            row_weight = A.get_row_weight(i)
            
            # Handle empty rows (shouldn't happen but be safe)
            if row_weight == 0:
                continue
            
            # Step 3: Sparse dot product ⟨a_i, x⟩
            # For binary matrix: sum of x[j] for j in indices
            dot_product = np.sum(x[indices])
            
            # Step 4: Compute residual
            residual = b[i] - dot_product
            residual_squared = residual * residual
            
            # Step 5: Kaczmarz update
            # x += (residual / ||a_i||²) * a_i
            # For binary matrix: x[indices] += residual / row_weight
            alpha = residual / row_weight
            x[indices] += alpha
            
            # Step 6: Track history if enabled
            if history is not None:
                history.record(residual_squared)
            
            # Step 7: Check stopping criterion
            conv_info = stopping.update(residual_squared)
            
            if conv_info.stopped:
                return conv_info
    
    def get_history(self) -> Optional[tuple]:
        """
        Get residual history if tracking was enabled.
        
        Returns:
            Tuple of (iterations, residuals) arrays, or None if not tracking
        """
        if self._history is not None:
            return self._history.to_arrays()
        return None


def solve_kaczmarz(
    A: SparseMatrix,
    b: np.ndarray,
    tolerance: float = 1e-6,
    max_iterations: int = 10_000_000,
    seed: Optional[int] = None,
    verbose: bool = False
) -> SolverResult:
    """
    Convenience function to solve Ax = b with randomized Kaczmarz.
    
    This is a simple wrapper around WeightedRandomizedKaczmarz for quick use.
    
    Args:
        A: Sparse matrix (implicit or explicit)
        b: Right-hand side vector
        tolerance: Convergence tolerance
        max_iterations: Maximum iterations
        seed: Random seed
        verbose: Print progress
        
    Returns:
        SolverResult with solution and convergence info
        
    Example:
        >>> from src.sparse_matrix import create_test_matrix
        >>> A = create_test_matrix(m=100, n=200, avg_nnz_per_row=10)
        >>> b = np.ones(100)
        >>> result = solve_kaczmarz(A, b, tolerance=1e-8)
        >>> print(result)
    """
    solver = WeightedRandomizedKaczmarz(
        tolerance=tolerance,
        max_iterations=max_iterations,
        seed=seed,
        verbose=verbose
    )
    return solver.solve(A, b)


def compute_full_residual(A: SparseMatrix, x: np.ndarray, b: np.ndarray) -> float:
    """
    Compute the full residual norm ||Ax - b||.
    
    Warning: This requires iterating over all rows, which is expensive
    for large systems. Use only for verification on small problems.
    
    Args:
        A: Sparse matrix
        x: Solution vector
        b: Right-hand side
        
    Returns:
        ||Ax - b||₂
    """
    m = A.shape[0]
    residual_sq_sum = 0.0
    
    for i in range(m):
        indices = A.get_row_indices(i)
        ax_i = np.sum(x[indices])  # Binary matrix dot product
        residual_sq_sum += (ax_i - b[i]) ** 2
    
    return np.sqrt(residual_sq_sum)


def compute_solution_norm(x: np.ndarray) -> float:
    """Compute ||x||₂."""
    return np.linalg.norm(x)
