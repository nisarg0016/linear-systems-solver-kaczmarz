"""
Source package for Weighted Randomized Kaczmarz Solver.

This package provides tools for solving extremely large sparse underdetermined
linear systems using the randomized Kaczmarz algorithm.

Main Components:
    - sparse_matrix: Implicit and explicit sparse matrix representations
    - sampling: Weighted random sampling with alias method
    - stopping: Adaptive convergence and stagnation detection
    - kaczmarz: Core algorithm implementation

Example:
    >>> from src import WeightedRandomizedKaczmarz, create_test_matrix
    >>> A = create_test_matrix(m=100, n=200, avg_nnz_per_row=10)
    >>> b = np.ones(100)
    >>> solver = WeightedRandomizedKaczmarz(tolerance=1e-8)
    >>> result = solver.solve(A, b)
"""

from .sparse_matrix import (
    ImplicitBitMatrix,
    ExplicitSparseRowMatrix,
    MatrixDimensions,
    create_test_matrix
)

from .sampling import (
    WeightedSampler,
    CumulativeSampler,
    verify_sampler
)

from .stopping import (
    AdaptiveStoppingCriterion,
    StoppingReason,
    ConvergenceInfo,
    ResidualHistory
)

from .kaczmarz import (
    WeightedRandomizedKaczmarz,
    SolverResult,
    solve_kaczmarz,
    compute_full_residual,
    compute_solution_norm
)

__version__ = "1.0.0"
__author__ = "Advanced Algorithms Student"

__all__ = [
    # Matrix classes
    "ImplicitBitMatrix",
    "ExplicitSparseRowMatrix", 
    "MatrixDimensions",
    "create_test_matrix",
    
    # Sampling
    "WeightedSampler",
    "CumulativeSampler",
    "verify_sampler",
    
    # Stopping criteria
    "AdaptiveStoppingCriterion",
    "StoppingReason",
    "ConvergenceInfo",
    "ResidualHistory",
    
    # Solver
    "WeightedRandomizedKaczmarz",
    "SolverResult",
    "solve_kaczmarz",
    "compute_full_residual",
    "compute_solution_norm",
]
