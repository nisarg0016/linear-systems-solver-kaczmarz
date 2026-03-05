"""
Demo: Running Kaczmarz Algorithm on Large Matrices

This script demonstrates how to use the Weighted Randomized Kaczmarz solver
on large sparse matrices - both explicit and implicit (memory-efficient).

Author: AA Project Demo
"""

import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sparse_matrix import ImplicitBitMatrix, ExplicitSparseRowMatrix, create_test_matrix
from src.kaczmarz import WeightedRandomizedKaczmarz, solve_kaczmarz, compute_full_residual


def demo_large_explicit_matrix():
    """
    Example 1: Large explicit sparse matrix
    
    Use this when you can store the matrix sparsity pattern in memory.
    Good for matrices up to ~100K rows with moderate sparsity.
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Large Explicit Sparse Matrix")
    print("="*70)
    
    # Matrix dimensions
    m = 10_000     # 10,000 rows (equations)
    n = 50_000     # 50,000 columns (unknowns) - underdetermined system
    avg_nnz = 20   # Average ~20 non-zeros per row
    
    print(f"\nCreating {m:,} × {n:,} sparse binary matrix...")
    print(f"Average {avg_nnz} non-zeros per row")
    
    # Create the test matrix
    A = create_test_matrix(m=m, n=n, avg_nnz_per_row=avg_nnz, seed=42)
    print(f"Matrix created: {A}")
    
    # Generate a consistent right-hand side: b = A @ x_true
    print("\nGenerating consistent system (b = Ax)...")
    x_true = np.random.RandomState(123).randn(n)
    b = np.zeros(m)
    for i in range(m):
        indices = A.get_row_indices(i)
        b[i] = np.sum(x_true[indices])
    
    # Solve using Kaczmarz
    print("\nSolving with Weighted Randomized Kaczmarz...")
    result = solve_kaczmarz(
        A, b,
        tolerance=1e-6,
        max_iterations=1_000_000,
        seed=42,
        verbose=True  # Print progress
    )
    
    print(f"\n{result}")
    print(f"Iterations per second: {result.iterations_per_second:,.0f}")
    
    return result


def demo_implicit_large_matrix():
    """
    Example 2: Implicit matrix for very large systems
    
    Use this when the matrix is too large to store explicitly.
    The matrix is defined by a function that computes row indices on-the-fly.
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Implicit Matrix (Memory Efficient)")
    print("="*70)
    
    # Very large matrix dimensions
    m = 100_000     # 100K rows
    n = 1_000_000   # 1 million columns!
    sparsity = 50   # Each row has exactly 50 non-zeros
    
    print(f"\nDefining {m:,} × {n:,} implicit sparse binary matrix")
    print(f"Each row has exactly {sparsity} non-zeros")
    print(f"This matrix would need ~400GB if stored densely!")
    
    # Define the row function - this generates row patterns on-the-fly
    # In practice, this would be your domain-specific row generation logic
    def row_function(i):
        """
        Generate the non-zero indices for row i.
        This example creates a deterministic pattern based on row index.
        """
        rng = np.random.RandomState(seed=i * 12345)  # Deterministic per row
        return sorted(rng.choice(n, size=sparsity, replace=False).tolist())
    
    # Create implicit matrix (only stores row weights, not actual indices)
    print("\nCreating implicit matrix representation...")
    A = ImplicitBitMatrix(m=m, n=n, row_func=row_function, precompute_weights=True)
    print(f"Matrix shape: {A.shape}")
    print(f"Total non-zeros: {A.total_weight:,.0f}")
    
    # Generate a consistent RHS
    print("\nGenerating consistent system...")
    x_true = np.zeros(n)
    # Make solution sparse for faster generation
    x_true[:1000] = np.random.RandomState(999).randn(1000)
    
    b = np.zeros(m)
    for i in range(m):
        indices = A.get_row_indices(i)
        b[i] = np.sum(x_true[indices])
    
    # Solve with higher iteration limit
    print("\nSolving with Weighted Randomized Kaczmarz...")
    solver = WeightedRandomizedKaczmarz(
        tolerance=1e-4,
        max_iterations=5_000_000,
        window_size=2000,
        seed=42,
        verbose=True
    )
    
    result = solver.solve(A, b)
    
    print(f"\n{result}")
    print(f"Iterations per second: {result.iterations_per_second:,.0f}")
    
    return result


def demo_custom_matrix():
    """
    Example 3: Define your own matrix structure
    
    This shows how to create a matrix with any custom sparsity pattern.
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Custom Matrix Structure")
    print("="*70)
    
    # Example: Banded matrix with random entries
    m = 5000
    n = 10000
    bandwidth = 100  # Each row has non-zeros in a band
    
    print(f"\nCreating {m:,} × {n:,} banded sparse matrix")
    print(f"Bandwidth: {bandwidth}")
    
    def banded_row(i):
        """Row i has non-zeros in columns around position 2*i"""
        center = min(2 * i, n - bandwidth // 2)
        center = max(center, bandwidth // 2)
        start = max(0, center - bandwidth // 2)
        end = min(n, center + bandwidth // 2)
        # Random subset within the band
        rng = np.random.RandomState(i)
        nnz = bandwidth // 4  # 25% fill within band
        cols = rng.choice(range(start, end), size=min(nnz, end-start), replace=False)
        return sorted(cols.tolist())
    
    A = ImplicitBitMatrix(m=m, n=n, row_func=banded_row)
    print(f"Matrix created with {A.total_weight:,.0f} non-zeros")
    
    # Create RHS
    x_true = np.random.RandomState(42).randn(n) * 0.1
    b = np.array([np.sum(x_true[A.get_row_indices(i)]) for i in range(m)])
    
    # Solve
    print("\nSolving...")
    result = solve_kaczmarz(A, b, tolerance=1e-6, max_iterations=500_000, verbose=True)
    
    print(f"\n{result}")
    
    return result


def quick_example():
    """
    Quick minimal example - copy this to get started fast!
    """
    print("\n" + "="*70)
    print("QUICK START EXAMPLE")
    print("="*70)
    
    # 1. Create a sparse matrix
    A = create_test_matrix(m=1000, n=5000, avg_nnz_per_row=10, seed=42)
    
    # 2. Create right-hand side vector
    b = np.random.randn(1000)
    
    # 3. Solve!
    result = solve_kaczmarz(A, b, tolerance=1e-6)
    
    # 4. Access the solution
    x = result.x
    print(f"Solution shape: {x.shape}")
    print(f"Converged: {result.converged}")
    print(f"Iterations: {result.convergence.iterations:,}")
    print(f"Time: {result.elapsed_time:.2f}s")
    
    return result


if __name__ == "__main__":
    print("="*70)
    print("WEIGHTED RANDOMIZED KACZMARZ - LARGE MATRIX DEMO")
    print("="*70)
    
    # Run the quick example first
    quick_example()
    
    # Then run the more detailed demos
    demo_large_explicit_matrix()
    
    # Uncomment to run the very large implicit matrix demo (takes longer)
    # demo_implicit_large_matrix()
    
    # demo_custom_matrix()
    
    print("\n" + "="*70)
    print("All demos completed!")
    print("="*70)
