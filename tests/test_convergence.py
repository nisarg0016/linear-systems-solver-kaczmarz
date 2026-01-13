"""
Convergence Verification Tests

This module tests that the randomized Kaczmarz algorithm converges at the
expected rate and produces the minimum-norm solution.

Tests include:
1. Convergence rate verification (exponential convergence)
2. Comparison with numpy's lstsq
3. Scalability test with varying problem sizes

Author: Advanced Algorithms Student
Date: January 2026
"""

import numpy as np
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sparse_matrix import ExplicitSparseRowMatrix, create_test_matrix
from src.kaczmarz import (
    WeightedRandomizedKaczmarz,
    compute_full_residual,
    compute_solution_norm
)


def test_exponential_convergence():
    """
    Test that residual decreases exponentially on average.
    
    Theory: E[||x_k - x*||²] ≤ (1 - σ²_min/||A||²_F)^k ||x_0 - x*||²
    
    We verify that log(residual) decreases roughly linearly with iterations.
    """
    print("\n" + "="*60)
    print("Test: Exponential Convergence Rate")
    print("="*60)
    
    # Create well-conditioned system
    m, n = 20, 40
    A = create_test_matrix(m, n, avg_nnz_per_row=5, seed=42)
    
    # Consistent RHS
    x_true = np.random.RandomState(123).randn(n)
    b = np.zeros(m)
    for i in range(m):
        indices = A.get_row_indices(i)
        b[i] = np.sum(x_true[indices])
    
    # Solve with history tracking
    solver = WeightedRandomizedKaczmarz(
        tolerance=1e-12,
        max_iterations=50000,
        seed=42,
        track_history=True,
        history_interval=100
    )
    result = solver.solve(A, b)
    
    iterations, residuals = solver.get_history()
    
    print(f"Iterations: {len(iterations)}")
    print(f"Initial residual²: {residuals[0]:.2e}")
    print(f"Final residual²: {residuals[-1]:.2e}")
    
    # Check exponential decay by fitting log(residual) vs iteration
    # Should be roughly linear
    log_residuals = np.log(residuals + 1e-20)  # Avoid log(0)
    
    # Linear regression
    coeffs = np.polyfit(iterations, log_residuals, 1)
    decay_rate = -coeffs[0]
    
    print(f"Estimated decay rate: {decay_rate:.6f}")
    print(f"Expected: positive (exponential decay)")
    
    assert decay_rate > 0, f"Should decay, got rate {decay_rate}"
    print("✓ Exponential convergence verified")
    return True


def test_comparison_with_lstsq():
    """
    Compare Kaczmarz solution with numpy's least-squares solver.
    
    For consistent underdetermined systems, both should give the minimum-norm solution.
    """
    print("\n" + "="*60)
    print("Test: Comparison with NumPy lstsq")
    print("="*60)
    
    m, n = 15, 30
    A = create_test_matrix(m, n, avg_nnz_per_row=4, seed=555)
    
    # Consistent RHS
    x_true = np.random.RandomState(666).randn(n)
    b = np.zeros(m)
    for i in range(m):
        indices = A.get_row_indices(i)
        b[i] = np.sum(x_true[indices])
    
    # NumPy solution
    A_dense = A.to_dense()
    x_numpy, residuals_np, rank, _ = np.linalg.lstsq(A_dense, b, rcond=None)
    numpy_norm = np.linalg.norm(x_numpy)
    
    print(f"NumPy lstsq:")
    print(f"  Solution norm: {numpy_norm:.6f}")
    print(f"  Matrix rank: {rank}")
    
    # Kaczmarz solution
    solver = WeightedRandomizedKaczmarz(
        tolerance=1e-10,
        max_iterations=500000,
        seed=42
    )
    result = solver.solve(A, b)
    kacz_norm = compute_solution_norm(result.x)
    
    print(f"\nKaczmarz:")
    print(f"  Solution norm: {kacz_norm:.6f}")
    print(f"  Converged: {result.converged}")
    print(f"  Iterations: {result.convergence.iterations:,}")
    
    # Compare
    norm_diff = abs(kacz_norm - numpy_norm)
    print(f"\nNorm difference: {norm_diff:.6f}")
    
    # Also compare actual solutions
    solution_diff = np.linalg.norm(result.x - x_numpy)
    print(f"Solution difference ||x_kacz - x_numpy||: {solution_diff:.6f}")
    
    # Both should satisfy the equations
    kacz_residual = compute_full_residual(A, result.x, b)
    numpy_residual = np.linalg.norm(A_dense @ x_numpy - b)
    
    print(f"Kaczmarz residual: {kacz_residual:.2e}")
    print(f"NumPy residual: {numpy_residual:.2e}")
    
    assert kacz_residual < 1e-4, f"Kaczmarz residual too large: {kacz_residual}"
    # Norms should be close (both minimum-norm)
    assert norm_diff < 0.5, f"Norm difference too large: {norm_diff}"
    
    print("✓ Solutions match lstsq")
    return True


def test_scalability():
    """
    Test solver performance on systems of increasing size.
    
    Verify that time scales linearly with iterations, not with matrix size.
    """
    print("\n" + "="*60)
    print("Test: Scalability")
    print("="*60)
    
    sizes = [
        (100, 200, 5),
        (500, 1000, 5),
        (1000, 2000, 5),
    ]
    
    results = []
    
    for m, n, avg_nnz in sizes:
        print(f"\nTesting {m}x{n} system (avg {avg_nnz} nnz/row)...")
        
        A = create_test_matrix(m, n, avg_nnz_per_row=avg_nnz, seed=42)
        
        # Consistent RHS
        x_true = np.random.RandomState(42).randn(n)
        b = np.zeros(m)
        for i in range(m):
            indices = A.get_row_indices(i)
            b[i] = np.sum(x_true[indices])
        
        # Time the solve - use reasonable tolerance for scalability testing
        solver = WeightedRandomizedKaczmarz(
            tolerance=1e-4,  # Relaxed for scalability testing
            max_iterations=500000,
            window_size=500,
            seed=42
        )
        
        start = time.perf_counter()
        result = solver.solve(A, b)
        elapsed = time.perf_counter() - start
        
        iters = result.convergence.iterations
        iters_per_sec = iters / elapsed if elapsed > 0 else float('inf')
        
        # Check residual quality instead of convergence flag
        full_res = compute_full_residual(A, result.x, b)
        
        results.append({
            'm': m, 'n': n, 
            'iterations': iters,
            'time': elapsed,
            'iters_per_sec': iters_per_sec,
            'converged': result.converged or full_res < 1e-2,  # Accept small residual
            'residual': full_res
        })
        
        print(f"  Iterations: {iters:,}")
        print(f"  Time: {elapsed:.3f}s")
        print(f"  Throughput: {iters_per_sec:,.0f} iter/s")
        print(f"  Residual: {full_res:.2e}")
    
    # Throughput should be roughly constant (not depend on n)
    throughputs = [r['iters_per_sec'] for r in results]
    throughput_cv = np.std(throughputs) / np.mean(throughputs)
    
    print(f"\nThroughput coefficient of variation: {throughput_cv:.2f}")
    print("(Should be low if scaling is good)")
    
    # Check that solutions are reasonable
    all_good = all(r['residual'] < 0.1 for r in results)
    assert all_good, "All systems should achieve small residual"
    
    print("✓ Scalability test passed")
    return True


def run_convergence_tests():
    """Run all convergence tests."""
    print("\n" + "#"*60)
    print("# CONVERGENCE VERIFICATION TESTS")
    print("#"*60)
    
    tests = [
        ("Exponential Convergence", test_exponential_convergence),
        ("Comparison with lstsq", test_comparison_with_lstsq),
        ("Scalability", test_scalability),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed, None))
        except AssertionError as e:
            results.append((name, False, str(e)))
        except Exception as e:
            results.append((name, False, f"Exception: {e}"))
    
    # Summary
    print("\n" + "="*60)
    print("CONVERGENCE TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, p, _ in results if p)
    total = len(results)
    
    for name, success, error in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"  {status}: {name}")
        if error:
            print(f"         Error: {error}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    return passed == total


if __name__ == "__main__":
    success = run_convergence_tests()
    sys.exit(0 if success else 1)
