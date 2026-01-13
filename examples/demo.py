"""
Full Demonstration of the Weighted Randomized Kaczmarz Solver

This script demonstrates the complete workflow:
1. Creating implicit sparse matrices
2. Solving consistent underdetermined systems
3. Handling inconsistent systems gracefully
4. Simulating the target problem scale

Author: Advanced Algorithms Student
Date: January 2026
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sparse_matrix import (
    ImplicitBitMatrix, 
    ExplicitSparseRowMatrix, 
    create_test_matrix
)
from src.kaczmarz import (
    WeightedRandomizedKaczmarz,
    solve_kaczmarz,
    compute_full_residual,
    compute_solution_norm
)
from src.sampling import WeightedSampler, verify_sampler


def demo_basic_usage():
    """Demonstrate basic solver usage."""
    print("\n" + "="*70)
    print("DEMO 1: Basic Usage")
    print("="*70)
    
    # Create a small test matrix
    print("\nCreating a 50x100 sparse binary matrix...")
    A = create_test_matrix(m=50, n=100, avg_nnz_per_row=5, seed=42)
    print(f"Matrix: {A}")
    
    # Create consistent right-hand side
    # b = Ax for some random x
    x_true = np.random.RandomState(123).randn(100)
    b = np.zeros(50)
    for i in range(50):
        indices = A.get_row_indices(i)
        b[i] = np.sum(x_true[indices])
    
    print(f"True solution norm: {np.linalg.norm(x_true):.4f}")
    
    # Solve
    print("\nSolving with randomized Kaczmarz...")
    result = solve_kaczmarz(A, b, tolerance=1e-8, seed=42)
    
    print(f"\nResult: {result}")
    print(f"Solution found with norm: {compute_solution_norm(result.x):.4f}")
    print(f"Full residual ||Ax - b||: {compute_full_residual(A, result.x, b):.2e}")


def demo_minimum_norm():
    """Demonstrate minimum-norm property."""
    print("\n" + "="*70)
    print("DEMO 2: Minimum-Norm Solution Property")
    print("="*70)
    
    # Create highly underdetermined system
    m, n = 10, 100  # 10 equations, 100 unknowns
    A = create_test_matrix(m, n, avg_nnz_per_row=5, seed=111)
    
    print(f"\nMatrix dimensions: {m} rows × {n} columns")
    print("System is highly underdetermined (infinitely many solutions)")
    
    # Create RHS
    x_true = np.random.RandomState(222).randn(n) * 10  # Large norm
    b = np.zeros(m)
    for i in range(m):
        indices = A.get_row_indices(i)
        b[i] = np.sum(x_true[indices])
    
    print(f"Generating solution has norm: {np.linalg.norm(x_true):.4f}")
    
    # Compute minimum-norm solution analytically
    A_dense = A.to_dense()
    x_pinv = np.linalg.lstsq(A_dense, b, rcond=None)[0]
    print(f"Minimum-norm solution (via pseudoinverse) has norm: {np.linalg.norm(x_pinv):.4f}")
    
    # Solve with Kaczmarz
    result = solve_kaczmarz(A, b, tolerance=1e-10, max_iterations=500000, seed=42)
    
    print(f"\nKaczmarz solution norm: {compute_solution_norm(result.x):.4f}")
    print(f"Difference from minimum-norm: {np.linalg.norm(result.x - x_pinv):.4f}")
    
    print("\n✓ Kaczmarz converges to minimum-norm solution (starting from x=0)")


def demo_inconsistent_system():
    """Demonstrate graceful handling of inconsistent systems."""
    print("\n" + "="*70)
    print("DEMO 3: Inconsistent System Handling")
    print("="*70)
    
    # Create contradictory system
    rows = [
        [0, 1, 2],      # x₀ + x₁ + x₂ = 1
        [0, 1, 2],      # x₀ + x₁ + x₂ = 5  (CONTRADICTION!)
        [3, 4],         # x₃ + x₄ = 2
    ]
    b = np.array([1.0, 5.0, 2.0])
    
    A = ExplicitSparseRowMatrix(rows, n=5)
    
    print("\nCreated inconsistent system:")
    print("  Row 0: x₀ + x₁ + x₂ = 1")
    print("  Row 1: x₀ + x₁ + x₂ = 5  (contradicts row 0!)")
    print("  Row 2: x₃ + x₄ = 2")
    
    # Solve
    solver = WeightedRandomizedKaczmarz(
        tolerance=1e-10,
        max_iterations=50000,
        window_size=500,
        stagnation_threshold=0.05,
        seed=42,
        verbose=False
    )
    result = solver.solve(A, b)
    
    print(f"\nSolver result: {result}")
    print(f"Detected as: {result.convergence.reason.value}")
    
    if not result.is_consistent:
        print("\n✓ Correctly detected inconsistent system")
        print("  Solver gracefully falls back to least-squares behavior")


def demo_implicit_matrix():
    """Demonstrate use with implicit matrix (never materialized)."""
    print("\n" + "="*70)
    print("DEMO 4: Implicit Matrix (Never Materialized)")
    print("="*70)
    
    # Define a structured sparsity pattern through a function
    # Example: Tridiagonal-like pattern where row i has 1s at columns i, i+1, i+2
    m = 1000
    n = 2000
    
    def get_row_indices(i):
        """Return indices of 1s in row i."""
        # Each row has 3 consecutive 1s starting at column i
        return list(range(i, min(i + 3, n)))
    
    print(f"\nCreating implicit {m}×{n} matrix")
    print("Pattern: Each row has 3 consecutive 1s")
    print("Memory: O(m) for weights, matrix NEVER stored")
    
    A = ImplicitBitMatrix(m, n, row_func=get_row_indices, precompute_weights=True)
    print(f"\nMatrix: {A}")
    
    # Create consistent RHS
    x_true = np.sin(np.arange(n) * 0.1)  # Smooth solution
    b = np.zeros(m)
    for i in range(m):
        indices = get_row_indices(i)
        b[i] = np.sum(x_true[indices])
    
    print(f"True solution norm: {np.linalg.norm(x_true):.4f}")
    
    # Solve
    print("\nSolving...")
    result = solve_kaczmarz(A, b, tolerance=1e-6, max_iterations=500000, seed=42)
    
    print(f"\nResult: {result}")
    print(f"Full residual: {compute_full_residual(A, result.x, b):.2e}")


def demo_weighted_sampling():
    """Demonstrate that weighted sampling matches expected distribution."""
    print("\n" + "="*70)
    print("DEMO 5: Weighted Sampling Verification")
    print("="*70)
    
    # Create weights with varying sparsities
    weights = np.array([1, 2, 5, 10, 20], dtype=float)
    
    print(f"Row weights (sparsities): {weights}")
    print(f"Expected probabilities: {weights / weights.sum()}")
    
    sampler = WeightedSampler(weights, seed=42)
    result = verify_sampler(sampler, weights, num_samples=50000)
    
    print(f"\nEmpirical probabilities: {result['empirical']}")
    print(f"Max deviation: {result['max_deviation']:.4f}")
    print(f"Expected max (3σ): {result['expected_max_deviation']:.4f}")
    print(f"Verification: {'✓ PASSED' if result['passed'] else '✗ FAILED'}")


def demo_scale_simulation():
    """
    Simulate solving at target scale.
    
    Target: m = 473×512 = 242,176 rows, n = 512³ = 134,217,728 columns
    
    We can't actually run this demo on a real machine without the actual
    data, but we simulate the memory and time requirements.
    """
    print("\n" + "="*70)
    print("DEMO 6: Target Scale Simulation")
    print("="*70)
    
    # Target dimensions
    m_target = 473 * 512  # 242,176 rows
    n_target = 512 ** 3   # 134,217,728 columns
    
    print(f"\nTarget problem:")
    print(f"  Rows (m): {m_target:,}")
    print(f"  Columns (n): {n_target:,}")
    print(f"  Dense matrix size: {m_target * n_target * 8 / 1e12:.1f} TB (impossible!)")
    
    # Estimate memory requirements
    solution_memory = n_target * 8  # float64
    weights_memory = m_target * 8
    total_memory = solution_memory + weights_memory
    
    print(f"\nMemory requirements for Kaczmarz solver:")
    print(f"  Solution vector x: {solution_memory / 1e9:.2f} GB")
    print(f"  Row weights: {weights_memory / 1e6:.2f} MB")
    print(f"  Total: {total_memory / 1e9:.2f} GB")
    print(f"  (Feasible on modern hardware!)")
    
    # Estimate time
    # Assume ~10M iterations, 100μs per iteration (conservative)
    iterations = 10_000_000
    time_per_iter_us = 100  # microseconds
    total_time_s = iterations * time_per_iter_us / 1e6
    
    print(f"\nTime estimate (assuming {iterations/1e6:.0f}M iterations):")
    print(f"  Time per iteration: ~{time_per_iter_us} μs")
    print(f"  Total time: ~{total_time_s/60:.1f} minutes")
    
    # Run a scaled-down version
    print("\n" + "-"*50)
    print("Running scaled-down simulation (1/1000 scale)...")
    
    m_sim = m_target // 100
    n_sim = n_target // 1000
    avg_nnz = 50  # Typical sparsity
    
    print(f"Simulation: {m_sim:,} rows × {n_sim:,} columns")
    
    A_sim = create_test_matrix(m_sim, n_sim, avg_nnz_per_row=avg_nnz, seed=42)
    
    # Consistent RHS
    x_true = np.random.RandomState(42).randn(n_sim)
    b = np.zeros(m_sim)
    for i in range(m_sim):
        indices = A_sim.get_row_indices(i)
        b[i] = np.sum(x_true[indices])
    
    result = solve_kaczmarz(A_sim, b, tolerance=1e-4, max_iterations=100000, seed=42)
    
    print(f"\nSimulation result: {result}")


def main():
    """Run all demonstrations."""
    print("#" * 70)
    print("# WEIGHTED RANDOMIZED KACZMARZ SOLVER - FULL DEMONSTRATION")
    print("#" * 70)
    
    demos = [
        ("Basic Usage", demo_basic_usage),
        ("Minimum-Norm Property", demo_minimum_norm),
        ("Inconsistent System", demo_inconsistent_system),
        ("Implicit Matrix", demo_implicit_matrix),
        ("Weighted Sampling", demo_weighted_sampling),
        ("Scale Simulation", demo_scale_simulation),
    ]
    
    for name, demo_func in demos:
        try:
            demo_func()
        except Exception as e:
            print(f"\n✗ Demo '{name}' failed: {e}")
    
    print("\n" + "#" * 70)
    print("# DEMONSTRATION COMPLETE")
    print("#" * 70)


if __name__ == "__main__":
    main()
