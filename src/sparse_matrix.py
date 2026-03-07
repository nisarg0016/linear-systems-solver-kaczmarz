"""
Sparse Matrix Infrastructure for Implicit Bit Matrices

This module provides an abstraction layer for extremely large sparse binary matrices
that cannot be explicitly materialized in memory. Access is provided only through
row-wise queries that return the indices of non-zero (1) entries.

Date: January 2026
"""

from typing import Callable, List, Optional, Sequence
from dataclasses import dataclass
import numpy as np


@dataclass
class MatrixDimensions:
    """Stores matrix dimensions without materializing the matrix."""
    m: int  # Number of rows
    n: int  # Number of columns
    
    def __post_init__(self):
        if self.m <= 0 or self.n <= 0:
            raise ValueError(f"Dimensions must be positive: got m={self.m}, n={self.n}")
    
    @property
    def is_underdetermined(self) -> bool:
        """Check if system is underdetermined (more unknowns than equations)."""
        return self.m < self.n


class ImplicitBitMatrix:
    """
    Represents an implicit binary matrix where explicit storage is infeasible.
    
    This class provides an interface to a large sparse binary matrix A ∈ {0,1}^{m×n}
    without ever materializing the full matrix. Access is provided through a 
    row-access function that returns indices of 1s for any given row.
    
    Memory Complexity: O(m) for caching row weights, not O(mn)
    
    Attributes:
        dims: MatrixDimensions object with m rows and n columns
        row_func: Function mapping row index to list of column indices with value 1
        _row_weights: Cached array of row sparsities (number of 1s per row)
        _total_weight: Sum of all row weights (equals nnz(A))
    """
    
    def __init__(
        self, 
        m: int, 
        n: int, 
        row_func: Callable[[int], Sequence[int]],
        precompute_weights: bool = True
    ):
        """
        Initialize an implicit bit matrix.
        
        Args:
            m: Number of rows
            n: Number of columns  
            row_func: Function that takes row index i and returns sequence of 
                     column indices j where A[i,j] = 1
            precompute_weights: If True, compute all row weights at initialization.
                               Set to False for lazy evaluation if memory is critical.
        
        Example:
            >>> def get_row(i):
            ...     return [2*i, 2*i+1]  # Each row has exactly 2 ones
            >>> A = ImplicitBitMatrix(m=100, n=200, row_func=get_row)
        """
        self.dims = MatrixDimensions(m, n)
        self._row_func = row_func
        self._row_weights: Optional[np.ndarray] = None
        self._total_weight: Optional[int] = None
        
        if precompute_weights:
            self._compute_all_weights()
    
    def _compute_all_weights(self) -> None:
        """
        Precompute weights (sparsities) for all rows.
        
        For binary matrices, ||a_i||^2 = number of 1s in row i.
        This is required for weighted sampling in the Kaczmarz algorithm.
        
        Time: O(m * average_sparsity)
        Space: O(m)
        """
        m = self.dims.m
        self._row_weights = np.zeros(m, dtype=np.float64)
        
        for i in range(m):
            indices = self._row_func(i)
            self._row_weights[i] = len(indices)
        
        self._total_weight = np.sum(self._row_weights)
    
    def get_row_indices(self, i: int) -> np.ndarray:
        """
        Get the column indices of non-zero entries in row i.
        
        Args:
            i: Row index (0 <= i < m)
            
        Returns:
            1D numpy array of column indices where A[i, j] = 1
            
        Raises:
            IndexError: If i is out of bounds
        """
        if not 0 <= i < self.dims.m:
            raise IndexError(f"Row index {i} out of bounds for matrix with {self.dims.m} rows")
        
        indices = self._row_func(i)
        return np.asarray(indices, dtype=np.int64)
    
    def get_row_weight(self, i: int) -> float:
        """
        Get the weight (squared norm) of row i.
        
        For binary matrices, this equals the number of 1s in the row.
        
        Args:
            i: Row index
            
        Returns:
            ||a_i||^2 = number of non-zeros in row i
        """
        if self._row_weights is not None:
            return self._row_weights[i]
        else:
            return float(len(self._row_func(i)))
    
    @property
    def row_weights(self) -> np.ndarray:
        """
        Get all row weights as a numpy array.
        
        Returns:
            Array of shape (m,) where weights[i] = ||a_i||^2
        """
        if self._row_weights is None:
            self._compute_all_weights()
        return self._row_weights
    
    @property
    def total_weight(self) -> float:
        """
        Get the total weight (squared Frobenius norm).
        
        For binary matrices: ||A||_F^2 = nnz(A)
        
        Returns:
            Sum of all row weights
        """
        if self._total_weight is None:
            self._compute_all_weights()
        return self._total_weight
    
    @property
    def m(self) -> int:
        """Number of rows."""
        return self.dims.m
    
    @property
    def n(self) -> int:
        """Number of columns."""
        return self.dims.n
    
    @property
    def shape(self) -> tuple:
        """Matrix shape as (m, n) tuple."""
        return (self.dims.m, self.dims.n)
    
    def sparse_dot(self, i: int, x: np.ndarray) -> float:
        """
        Compute the sparse dot product a_i^T * x.
        
        Since A is binary, this is simply the sum of x[j] for all j where A[i,j] = 1.
        
        Args:
            i: Row index
            x: Solution vector of length n
            
        Returns:
            Inner product <a_i, x>
            
        Time: O(sparsity of row i)
        """
        indices = self.get_row_indices(i)
        return np.sum(x[indices])
    
    def sparse_axpy(self, i: int, alpha: float, x: np.ndarray) -> None:
        """
        Perform the sparse AXPY update: x += alpha * a_i (in-place).
        
        Since A is binary, this adds alpha to x[j] for all j where A[i,j] = 1.
        
        Args:
            i: Row index
            alpha: Scalar multiplier
            x: Solution vector (modified in-place)
            
        Time: O(sparsity of row i)
        """
        indices = self.get_row_indices(i)
        x[indices] += alpha
    
    def __repr__(self) -> str:
        total = self._total_weight if self._total_weight else "unknown"
        avg_sparsity = self._total_weight / self.dims.m if self._total_weight else "unknown"
        return (
            f"ImplicitBitMatrix(m={self.dims.m:,}, n={self.dims.n:,}, "
            f"total_nnz={total}, avg_row_sparsity={avg_sparsity})"
        )


class ExplicitSparseRowMatrix:
    """
    A sparse matrix stored explicitly using Compressed Sparse Row (CSR) format.
    
    This is useful for small test cases where we can actually store the matrix.
    Provides the same interface as ImplicitBitMatrix for testing.
    """
    
    def __init__(self, row_indices_list: List[List[int]], n: int):
        """
        Create a sparse matrix from a list of row index lists.
        
        Args:
            row_indices_list: List of m lists, where row_indices_list[i] contains
                             the column indices of 1s in row i
            n: Number of columns
        """
        self.dims = MatrixDimensions(m=len(row_indices_list), n=n)
        self._rows = [np.asarray(row, dtype=np.int64) for row in row_indices_list]
        self._row_weights = np.array([len(row) for row in row_indices_list], dtype=np.float64)
        self._total_weight = np.sum(self._row_weights)
    
    def get_row_indices(self, i: int) -> np.ndarray:
        """Get column indices of non-zeros in row i."""
        return self._rows[i]
    
    def get_row_weight(self, i: int) -> float:
        """Get squared norm of row i."""
        return self._row_weights[i]
    
    @property
    def row_weights(self) -> np.ndarray:
        return self._row_weights
    
    @property
    def total_weight(self) -> float:
        return self._total_weight
    
    @property
    def m(self) -> int:
        return self.dims.m
    
    @property
    def n(self) -> int:
        return self.dims.n
    
    @property
    def shape(self) -> tuple:
        return (self.dims.m, self.dims.n)
    
    def sparse_dot(self, i: int, x: np.ndarray) -> float:
        """Compute a_i^T * x."""
        indices = self._rows[i]
        return np.sum(x[indices])
    
    def sparse_axpy(self, i: int, alpha: float, x: np.ndarray) -> None:
        """x += alpha * a_i."""
        indices = self._rows[i]
        x[indices] += alpha
    
    def to_dense(self) -> np.ndarray:
        """Convert to dense matrix (only for small test cases!)."""
        dense = np.zeros((self.dims.m, self.dims.n), dtype=np.float64)
        for i, indices in enumerate(self._rows):
            dense[i, indices] = 1.0
        return dense
    
    def __repr__(self) -> str:
        return (
            f"ExplicitSparseRowMatrix(m={self.dims.m}, n={self.dims.n}, "
            f"total_nnz={self._total_weight})"
        )


def create_test_matrix(m: int, n: int, avg_nnz_per_row: int, seed: int = 42) -> ExplicitSparseRowMatrix:
    """
    Create a random sparse binary matrix for testing.
    
    Args:
        m: Number of rows
        n: Number of columns
        avg_nnz_per_row: Target average number of 1s per row
        seed: Random seed for reproducibility
        
    Returns:
        ExplicitSparseRowMatrix with random sparsity pattern
    """
    rng = np.random.default_rng(seed)
    rows = []
    
    for _ in range(m):
        # Vary sparsity slightly around the average
        nnz = max(1, int(rng.poisson(avg_nnz_per_row)))
        nnz = min(nnz, n)  # Can't have more 1s than columns
        indices = rng.choice(n, size=nnz, replace=False)
        rows.append(sorted(indices.tolist()))
    
    return ExplicitSparseRowMatrix(rows, n)
