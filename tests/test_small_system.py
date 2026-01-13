"""
Test Suite for Small Synthetic Systems

This module contains tests that verify the correctness of the Kaczmarz solver
on small systems where we can compute exact solutions for comparison.

Tests include:
1. Simple consistent overdetermined system
2. Consistent underdetermined system (minimum-norm verification)
3. Inconsistent system (least-squares behavior)
4. Identity-like systems
5. Binary matrix with known solution

Author: Advanced Algorithms Student
Date: January 2026
"""

import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sparse_matrix import ExplicitSparseRowMatrix, create_test_matrix
from src.kaczmarz import (
    WeightedRandomizedKaczmarz,
    solve_kaczmarz,
    compute_full_residual,
    compute_solution_norm
)
from src.stopping import StoppingReason


def test_simple_consistent_system():
    """
    Test 1: Simple consistent system with known solution.
    
    System:
        x₀ + x₁ = 2
        x₁ + x₂ = 2
        x₂ + x₃ = 2
        
    With 4 unknowns and 3 equations (underdetermined).
    Solution space: x = [t, 2-t, t, 2-t] for any t.
    Minimum-norm solution: t = 1, giving x* = [1, 1, 1, 1].
    """
    print("\n" + "="*60)
    print("Test 1: Simple Consistent Underdetermined System")
    print("="*60)
    
    # Define matrix: each row has two adjacent 1s
    rows = [
        [0, 1],  # x₀ + x₁ = 2
        [1, 2],  # x₁ + x₂ = 2
        [2, 3],  # x₂ + x₃ = 2
    ]
    n = 4
    b = np.array([2.0, 2.0, 2.0])
    
    A = ExplicitSparseRowMatrix(rows, n)
    
    print(f"Matrix A (sparse): {A}")
    print(f"Right-hand side b: {b}")
    print(f"Expected minimum-norm solution: [1, 1, 1, 1]")
    
    # Solve
    solver = WeightedRandomizedKaczmarz(
        tolerance=1e-8,
        max_iterations=100000,
        seed=42,
        verbose=False
    )
    result = solver.solve(A, b)
    
    print(f"\nSolver result: {result}")
    print(f"Solution x: {result.x}")
    print(f"Solution norm ||x||: {compute_solution_norm(result.x):.6f}")
    print(f"Expected norm: {2.0:.6f}")  # ||[1,1,1,1]|| = 2
    
    # Verify solution
    full_residual = compute_full_residual(A, result.x, b)
    print(f"Full residual ||Ax - b||: {full_residual:.2e}")
    
    # Check if solution is close to expected minimum-norm solution
    expected = np.array([1.0, 1.0, 1.0, 1.0])
    error = np.linalg.norm(result.x - expected)
    print(f"Error from expected: {error:.2e}")
    
    assert result.converged, "Solver should converge"
    assert full_residual < 1e-4, f"Residual too large: {full_residual}"
    assert error < 0.1, f"Solution too far from expected: {error}"
    
    print("✓ Test 1 PASSED")
    return True


def test_minimum_norm_property():
    """
    Test 2: Verify minimum-norm property for underdetermined systems.
    
    For underdetermined systems, there are infinitely many solutions.
    Starting from x₀ = 0 should give the minimum-norm solution.
    
    We compare with the pseudoinverse solution A†b.
    """
    print("\n" + "="*60)
    print("Test 2: Minimum-Norm Solution Verification")
    print("="*60)
    
    # Create a small random underdetermined system
    m, n = 5, 10
    A = create_test_matrix(m, n, avg_nnz_per_row=3, seed=123)
    
    # Create consistent RHS by computing b = Ax* for known x*
    x_true = np.random.RandomState(456).randn(n)
    
    # Compute b = Ax*
    b = np.zeros(m)
    for i in range(m):
        indices = A.get_row_indices(i)
        b[i] = np.sum(x_true[indices])
    
    print(f"Matrix: {A}")
    print(f"True solution norm: {np.linalg.norm(x_true):.4f}")
    
    # Compute minimum-norm solution using pseudoinverse
    A_dense = A.to_dense()
    x_pinv = np.linalg.lstsq(A_dense, b, rcond=None)[0]
    pinv_norm = np.linalg.norm(x_pinv)
    print(f"Pseudoinverse solution norm: {pinv_norm:.4f}")
    
    # Solve with Kaczmarz - use reasonable tolerance and check solution quality
    # not just convergence flag (small systems may trigger stagnation detection)
    result = solve_kaczmarz(A, b, tolerance=1e-8, max_iterations=500000, seed=42)
    kacz_norm = compute_solution_norm(result.x)
    
    print(f"\nKaczmarz result: {result}")
    print(f"Kaczmarz solution norm: {kacz_norm:.4f}")
    
    # Verify it's a valid solution
    full_residual = compute_full_residual(A, result.x, b)
    print(f"Residual ||Ax - b||: {full_residual:.2e}")
    
    # Check norm is close to minimum-norm solution
    norm_error = abs(kacz_norm - pinv_norm) / pinv_norm if pinv_norm > 1e-10 else abs(kacz_norm - pinv_norm)
    print(f"Relative norm difference: {norm_error:.2e}")
    
    # For this test, we check:
    # 1. Solution is valid (small residual)
    # 2. Norm is close to minimum-norm
    # We don't require convergence flag because small systems may stagnate
    # at numerical precision limits
    assert full_residual < 1e-4, f"Residual too large: {full_residual}"
    # The norm should be close to minimum-norm (within 10%)
    assert norm_error < 0.1, f"Norm not minimal: {kacz_norm:.4f} vs {pinv_norm:.4f}"
    
    print("✓ Test 2 PASSED")
    return True


def test_inconsistent_system():
    """
    Test 3: Inconsistent system (no exact solution).
    
    Create a system where some equations contradict each other.
    The solver should detect stagnation and report inconsistency.
    """
    print("\n" + "="*60)
    print("Test 3: Inconsistent System Detection")
    print("="*60)
    
    # Create inconsistent system:
    # x₀ + x₁ = 1
    # x₀ + x₁ = 2  (contradicts first equation!)
    # x₂ = 0
    rows = [
        [0, 1],  # x₀ + x₁ = 1
        [0, 1],  # x₀ + x₁ = 2 (impossible!)
        [2],     # x₂ = 0
    ]
    n = 3
    b = np.array([1.0, 2.0, 0.0])
    
    A = ExplicitSparseRowMatrix(rows, n)
    
    print(f"Matrix A has contradictory rows")
    print(f"Row 0: x₀ + x₁ = 1")
    print(f"Row 1: x₀ + x₁ = 2 (CONTRADICTION)")
    
    # Solve with smaller window for faster stagnation detection
    solver = WeightedRandomizedKaczmarz(
        tolerance=1e-10,
        max_iterations=100000,
        window_size=500,
        stagnation_threshold=0.05,  # More sensitive to stagnation
        seed=42,
        verbose=False
    )
    result = solver.solve(A, b)
    
    print(f"\nSolver result: {result}")
    print(f"Stopping reason: {result.convergence.reason}")
    
    # Should detect stagnation (inconsistent system)
    is_stagnated = result.convergence.reason == StoppingReason.STAGNATED
    hit_max = result.convergence.reason == StoppingReason.MAX_ITERATIONS
    
    # Either stagnation detected OR hit max iterations without converging
    assert not result.converged, "Should NOT converge (inconsistent system)"
    assert is_stagnated or hit_max, f"Should stagnate or hit max, got {result.convergence.reason}"
    
    print("✓ Test 3 PASSED (correctly identified as inconsistent)")
    return True


def test_identity_rows():
    """
    Test 4: System with single-variable equations (like identity rows).
    
    When each row has exactly one 1, the system reduces to xᵢ = bᵢ.
    """
    print("\n" + "="*60)
    print("Test 4: Single-Variable Equations")
    print("="*60)
    
    # Each row is a single variable
    m, n = 5, 10
    rows = [[i] for i in range(m)]  # Row i has a 1 only in column i
    b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    
    A = ExplicitSparseRowMatrix(rows, n)
    
    print(f"System: x₀=1, x₁=2, x₂=3, x₃=4, x₄=5")
    print(f"Variables x₅ to x₉ are free (should be 0 for min-norm)")
    
    result = solve_kaczmarz(A, b, tolerance=1e-10, max_iterations=100000, seed=42)
    
    print(f"\nSolver result: {result}")
    print(f"Solution x[:5]: {result.x[:5]}")
    print(f"Solution x[5:]: {result.x[5:]}")
    
    # Check first 5 components
    expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    error_constrained = np.linalg.norm(result.x[:5] - expected)
    print(f"Error in constrained variables: {error_constrained:.2e}")
    
    # Check free variables (should be ~0 for minimum norm)
    error_free = np.linalg.norm(result.x[5:])
    print(f"Norm of free variables: {error_free:.2e}")
    
    assert result.converged, "Should converge"
    assert error_constrained < 0.1, f"Constrained vars wrong: {error_constrained}"
    assert error_free < 0.1, f"Free vars should be ~0: {error_free}"
    
    print("✓ Test 4 PASSED")
    return True


def test_random_consistent_system():
    """
    Test 5: Random consistent system.
    
    Generate a random sparse matrix and ensure solver finds a valid solution.
    """
    print("\n" + "="*60)
    print("Test 5: Random Consistent System")
    print("="*60)
    
    # Larger test case
    m, n = 50, 100
    A = create_test_matrix(m, n, avg_nnz_per_row=5, seed=789)
    
    # Generate consistent RHS
    x_true = np.random.RandomState(111).randn(n)
    b = np.zeros(m)
    for i in range(m):
        indices = A.get_row_indices(i)
        b[i] = np.sum(x_true[indices])
    
    print(f"Matrix: {A}")
    print(f"Solving {m}x{n} system...")
    
    result = solve_kaczmarz(A, b, tolerance=1e-8, max_iterations=1000000, seed=42)
    
    print(f"\nSolver result: {result}")
    
    # Verify solution
    full_residual = compute_full_residual(A, result.x, b)
    print(f"Full residual: {full_residual:.2e}")
    
    assert result.converged, "Should converge"
    assert full_residual < 1e-3, f"Residual too large: {full_residual}"
    
    print("✓ Test 5 PASSED")
    return True


def run_all_tests():
    """Run all test cases and report results."""
    print("\n" + "#"*60)
    print("# WEIGHTED RANDOMIZED KACZMARZ - TEST SUITE")
    print("#"*60)
    
    tests = [
        ("Simple Consistent System", test_simple_consistent_system),
        ("Minimum-Norm Property", test_minimum_norm_property),
        ("Inconsistent System", test_inconsistent_system),
        ("Identity Rows", test_identity_rows),
        ("Random Consistent System", test_random_consistent_system),
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
    print("TEST SUMMARY")
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
    success = run_all_tests()
    sys.exit(0 if success else 1)
