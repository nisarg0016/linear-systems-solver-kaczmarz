"""
Adaptive Stopping Criteria for Iterative Solvers

This module implements convergence detection and inconsistency detection
for the randomized Kaczmarz algorithm using moving-window statistics.

Key features:
- Moving average residual tracking
- Adaptive tolerance checking
- Stagnation detection for inconsistent systems
- Graceful degradation to least-squares behavior

Author: Advanced Algorithms Student
Date: January 2026
"""

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Optional
import numpy as np


class StoppingReason(Enum):
    """Enum indicating why the solver stopped."""
    CONVERGED = "converged"           # Residual below tolerance
    MAX_ITERATIONS = "max_iterations" # Iteration limit reached
    STAGNATED = "stagnated"          # Residual stagnated (inconsistent system)
    DIVERGED = "diverged"            # Residual growing unboundedly


@dataclass
class ConvergenceInfo:
    """
    Detailed convergence information from the solver.
    
    Attributes:
        stopped: Whether stopping criterion was triggered
        reason: Why we stopped (if stopped)
        iterations: Number of iterations completed
        final_residual_mean: Moving average of squared residuals
        final_residual_std: Standard deviation of residual window
        coefficient_of_variation: Ratio of std to mean (stability indicator)
    """
    stopped: bool
    reason: Optional[StoppingReason]
    iterations: int
    final_residual_mean: float
    final_residual_std: float
    coefficient_of_variation: float
    
    def is_consistent(self) -> bool:
        """Check if the system appears consistent (has exact solution)."""
        return self.reason == StoppingReason.CONVERGED
    
    def __repr__(self) -> str:
        return (
            f"ConvergenceInfo(stopped={self.stopped}, reason={self.reason}, "
            f"iter={self.iterations}, residual_mean={self.final_residual_mean:.2e})"
        )


class AdaptiveStoppingCriterion:
    """
    Implements adaptive stopping based on moving-window residual statistics.
    
    The criterion monitors a sliding window of squared residuals and stops when:
    1. Mean squared residual falls below tolerance² (CONVERGED)
    2. Residual variance is low but mean is high (STAGNATED - inconsistent system)
    3. Residual is growing (DIVERGED)
    4. Maximum iterations reached (MAX_ITERATIONS)
    
    Stagnation Detection:
    For inconsistent systems (no exact solution), the residual will stabilize
    at a positive value. We detect this by checking if the coefficient of
    variation (std/mean) is small while the mean remains above threshold.
    
    Mathematical justification:
    - Consistent system: residual → 0, variance → 0
    - Inconsistent system: residual → r* > 0, variance → 0
    
    We distinguish these by checking if mean residual < tolerance.
    """
    
    def __init__(
        self,
        tolerance: float = 1e-6,
        window_size: int = 1000,
        max_iterations: int = 10_000_000,
        stagnation_cv_threshold: float = 0.1,
        divergence_factor: float = 100.0,
        verbose: bool = False
    ):
        """
        Initialize the stopping criterion.
        
        Args:
            tolerance: Stop when mean squared residual < tolerance²
            window_size: Number of recent residuals to track
            max_iterations: Hard limit on iterations
            stagnation_cv_threshold: Coefficient of variation below which
                                    residual is considered stable
            divergence_factor: Stop if residual exceeds initial by this factor
            verbose: Print progress information
        """
        self.tolerance = tolerance
        self.tolerance_sq = tolerance ** 2
        self.window_size = window_size
        self.max_iterations = max_iterations
        self.stagnation_cv_threshold = stagnation_cv_threshold
        self.divergence_factor = divergence_factor
        self.verbose = verbose
        
        # State
        self._residual_window: deque = deque(maxlen=window_size)
        self._iteration = 0
        self._initial_residual: Optional[float] = None
        self._window_sum = 0.0
        self._window_sum_sq = 0.0
    
    def reset(self) -> None:
        """Reset state for a new solve."""
        self._residual_window.clear()
        self._iteration = 0
        self._initial_residual = None
        self._window_sum = 0.0
        self._window_sum_sq = 0.0
    
    def update(self, residual_squared: float) -> ConvergenceInfo:
        """
        Update with new residual and check stopping criteria.
        
        Args:
            residual_squared: The squared residual |r_i|² for the current iteration
            
        Returns:
            ConvergenceInfo with current state and whether to stop
        """
        self._iteration += 1
        
        # Track initial residual for divergence detection
        if self._initial_residual is None:
            self._initial_residual = residual_squared
        
        # Efficient online statistics update
        if len(self._residual_window) == self.window_size:
            # Remove oldest from running sums
            oldest = self._residual_window[0]
            self._window_sum -= oldest
            self._window_sum_sq -= oldest * oldest
        
        # Add new residual
        self._residual_window.append(residual_squared)
        self._window_sum += residual_squared
        self._window_sum_sq += residual_squared * residual_squared
        
        # Check if window is full
        if len(self._residual_window) < self.window_size:
            # Not enough samples yet
            return ConvergenceInfo(
                stopped=False,
                reason=None,
                iterations=self._iteration,
                final_residual_mean=self._window_sum / len(self._residual_window),
                final_residual_std=0.0,
                coefficient_of_variation=float('inf')
            )
        
        # Compute statistics
        n = self.window_size
        mean = self._window_sum / n
        variance = (self._window_sum_sq / n) - (mean * mean)
        variance = max(0.0, variance)  # Handle numerical issues
        std = np.sqrt(variance)
        cv = std / mean if mean > 1e-15 else float('inf')
        
        # Check stopping conditions
        stopped = False
        reason = None
        
        # 1. Convergence check
        if mean < self.tolerance_sq:
            stopped = True
            reason = StoppingReason.CONVERGED
        
        # 2. Stagnation check (inconsistent system)
        elif cv < self.stagnation_cv_threshold and mean >= self.tolerance_sq:
            # Residual has stabilized but is still large
            stopped = True
            reason = StoppingReason.STAGNATED
            if self.verbose:
                print(f"Stagnation detected at iteration {self._iteration}: "
                      f"mean={mean:.2e}, cv={cv:.4f}")
        
        # 3. Divergence check
        elif mean > self.divergence_factor * self._initial_residual:
            stopped = True
            reason = StoppingReason.DIVERGED
            if self.verbose:
                print(f"Divergence detected at iteration {self._iteration}")
        
        # 4. Max iterations check
        elif self._iteration >= self.max_iterations:
            stopped = True
            reason = StoppingReason.MAX_ITERATIONS
        
        # Verbose progress reporting
        if self.verbose and self._iteration % (self.max_iterations // 100 + 1) == 0:
            print(f"Iteration {self._iteration}: mean_residual={np.sqrt(mean):.2e}, cv={cv:.4f}")
        
        return ConvergenceInfo(
            stopped=stopped,
            reason=reason,
            iterations=self._iteration,
            final_residual_mean=mean,
            final_residual_std=std,
            coefficient_of_variation=cv
        )
    
    @property
    def current_iteration(self) -> int:
        """Current iteration count."""
        return self._iteration


class ResidualHistory:
    """
    Tracks full residual history for analysis and plotting.
    
    Memory-intensive; only use for debugging or small problems.
    """
    
    def __init__(self, sample_interval: int = 1):
        """
        Args:
            sample_interval: Store every k-th residual to reduce memory
        """
        self.sample_interval = sample_interval
        self._residuals: list = []
        self._iterations: list = []
        self._count = 0
    
    def record(self, residual_squared: float) -> None:
        """Record a residual value."""
        self._count += 1
        if self._count % self.sample_interval == 0:
            self._residuals.append(residual_squared)
            self._iterations.append(self._count)
    
    def to_arrays(self) -> tuple:
        """Return iterations and residuals as numpy arrays."""
        return (
            np.array(self._iterations),
            np.array(self._residuals)
        )
    
    def reset(self) -> None:
        """Clear history."""
        self._residuals.clear()
        self._iterations.clear()
        self._count = 0
