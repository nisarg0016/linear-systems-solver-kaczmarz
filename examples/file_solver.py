"""
File-Based Kaczmarz Solver

This script shows how to use the Kaczmarz solver like a traditional solver:
- Read matrix A and vector b from files
- Solve Ax = b
- Write solution x to output file

Supports multiple file formats: CSV, NPY (numpy), and custom sparse format.

Usage:
    python file_solver.py --matrix matrix.csv --rhs rhs.csv --output solution.csv
    python file_solver.py --matrix matrix.npz --rhs rhs.npy --output solution.npy
    
Author: AA Project
"""

import numpy as np
import argparse
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sparse_matrix import ExplicitSparseRowMatrix, ImplicitBitMatrix
from src.kaczmarz import solve_kaczmarz, compute_full_residual


# =============================================================================
# FILE I/O FUNCTIONS
# =============================================================================

def load_matrix_csv(filepath):
    """
    Load a sparse binary matrix from CSV.
    
    Expected format (each line is a row, values are column indices of 1s):
        0,5,12,45
        1,3,8,22,99
        2,7
        ...
    """
    rows = []
    max_col = 0
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            indices = [int(x) for x in line.split(',') if x.strip()]
            rows.append(sorted(indices))
            if indices:
                max_col = max(max_col, max(indices))
    
    n = max_col + 1  # Number of columns
    return ExplicitSparseRowMatrix(rows, n)


def load_matrix_dense_csv(filepath):
    """
    Load a dense matrix from CSV and convert to sparse binary.
    Values > 0.5 are treated as 1, others as 0.
    """
    data = np.loadtxt(filepath, delimiter=',')
    if data.ndim == 1:
        data = data.reshape(1, -1)
    
    m, n = data.shape
    rows = []
    for i in range(m):
        indices = np.where(data[i] > 0.5)[0].tolist()
        rows.append(indices)
    
    return ExplicitSparseRowMatrix(rows, n)


def load_matrix_npz(filepath):
    """
    Load sparse matrix from NumPy .npz file.
    
    Expected keys:
        - 'rows': list of arrays, each containing column indices for that row
        - 'n': number of columns
    Or:
        - 'data': dense matrix (converted to binary sparse)
    """
    loaded = np.load(filepath, allow_pickle=True)
    
    if 'rows' in loaded:
        rows = [list(r) for r in loaded['rows']]
        n = int(loaded['n'])
        return ExplicitSparseRowMatrix(rows, n)
    elif 'data' in loaded:
        data = loaded['data']
        m, n = data.shape
        rows = [np.where(data[i] > 0.5)[0].tolist() for i in range(m)]
        return ExplicitSparseRowMatrix(rows, n)
    else:
        raise ValueError("NPZ file must contain 'rows' and 'n', or 'data'")


def load_vector(filepath):
    """Load right-hand side vector from file."""
    ext = os.path.splitext(filepath)[1].lower()
    
    if ext == '.npy':
        return np.load(filepath)
    elif ext == '.csv' or ext == '.txt':
        return np.loadtxt(filepath, delimiter=',')
    else:
        # Try to auto-detect
        try:
            return np.load(filepath)
        except:
            return np.loadtxt(filepath, delimiter=',')


def save_vector(filepath, x):
    """Save solution vector to file."""
    ext = os.path.splitext(filepath)[1].lower()
    
    if ext == '.npy':
        np.save(filepath, x)
    elif ext == '.csv':
        np.savetxt(filepath, x, delimiter=',', fmt='%.10e')
    else:
        np.savetxt(filepath, x, delimiter=',', fmt='%.10e')
    
    print(f"Solution saved to: {filepath}")


def load_matrix(filepath):
    """Auto-detect matrix format and load."""
    ext = os.path.splitext(filepath)[1].lower()
    
    if ext == '.npz':
        return load_matrix_npz(filepath)
    elif ext == '.csv':
        # Try sparse format first, fall back to dense
        try:
            return load_matrix_csv(filepath)
        except:
            return load_matrix_dense_csv(filepath)
    else:
        raise ValueError(f"Unsupported matrix format: {ext}")


# =============================================================================
# MAIN SOLVER INTERFACE
# =============================================================================

def solve_from_files(matrix_file, rhs_file, output_file=None, 
                     tolerance=1e-6, max_iterations=1_000_000, verbose=True):
    """
    Complete file-based solving workflow.
    
    Args:
        matrix_file: Path to matrix file (CSV or NPZ)
        rhs_file: Path to right-hand side vector file
        output_file: Path to save solution (optional)
        tolerance: Convergence tolerance
        max_iterations: Maximum iterations
        verbose: Print progress
        
    Returns:
        Solution vector x
    """
    print("="*60)
    print("KACZMARZ SOLVER - File-Based Mode")
    print("="*60)
    
    # Load matrix
    print(f"\nLoading matrix from: {matrix_file}")
    A = load_matrix(matrix_file)
    print(f"  Matrix shape: {A.shape[0]} rows × {A.shape[1]} columns")
    print(f"  Total non-zeros: {A.total_weight:,.0f}")
    print(f"  Avg non-zeros/row: {A.total_weight / A.shape[0]:.1f}")
    
    # Load RHS
    print(f"\nLoading RHS from: {rhs_file}")
    b = load_vector(rhs_file)
    print(f"  Vector length: {len(b)}")
    
    # Validate dimensions
    if len(b) != A.shape[0]:
        raise ValueError(f"Dimension mismatch: matrix has {A.shape[0]} rows, "
                        f"but RHS has {len(b)} elements")
    
    # Solve
    print(f"\nSolving system...")
    print(f"  Tolerance: {tolerance}")
    print(f"  Max iterations: {max_iterations:,}")
    
    result = solve_kaczmarz(
        A, b,
        tolerance=tolerance,
        max_iterations=max_iterations,
        verbose=verbose
    )
    
    # Report results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"  Status: {'CONVERGED' if result.converged else 'NOT CONVERGED'}")
    print(f"  Reason: {result.convergence.reason.value}")
    print(f"  Iterations: {result.convergence.iterations:,}")
    print(f"  Time: {result.elapsed_time:.2f} seconds")
    print(f"  Throughput: {result.iterations_per_second:,.0f} iter/sec")
    print(f"  Final residual (RMS): {np.sqrt(result.convergence.final_residual_mean):.2e}")
    print(f"  Solution norm: {np.linalg.norm(result.x):.4f}")
    
    # Save output
    if output_file:
        save_vector(output_file, result.x)
    
    return result.x


# =============================================================================
# COMPARISON WITH NUMPY/SCIPY
# =============================================================================

def compare_with_numpy(A, b, kaczmarz_solution):
    """
    Compare Kaczmarz solution with NumPy's least-squares solver.
    Only works for small matrices that can be converted to dense!
    """
    print("\n" + "="*60)
    print("COMPARISON WITH NUMPY")
    print("="*60)
    
    if A.shape[0] > 5000 or A.shape[1] > 5000:
        print("  Matrix too large for dense comparison")
        return
    
    # Convert to dense
    A_dense = A.to_dense()
    
    # NumPy least-squares (minimum-norm for underdetermined)
    print("\nNumPy lstsq (pseudoinverse)...")
    start = time.time()
    x_numpy, residuals, rank, s = np.linalg.lstsq(A_dense, b, rcond=None)
    numpy_time = time.time() - start
    
    print(f"  Time: {numpy_time:.4f}s")
    print(f"  Solution norm: {np.linalg.norm(x_numpy):.4f}")
    print(f"  Residual ||Ax-b||: {np.linalg.norm(A_dense @ x_numpy - b):.2e}")
    
    # Compare solutions
    print(f"\nKaczmarz solution:")
    print(f"  Solution norm: {np.linalg.norm(kaczmarz_solution):.4f}")
    print(f"  Residual ||Ax-b||: {np.linalg.norm(A_dense @ kaczmarz_solution - b):.2e}")
    
    diff = np.linalg.norm(kaczmarz_solution - x_numpy)
    print(f"\nDifference between solutions: {diff:.2e}")


# =============================================================================
# CREATE SAMPLE INPUT FILES
# =============================================================================

def create_sample_files(output_dir="."):
    """Create sample input files for testing."""
    print("\nCreating sample input files...")
    
    # Sample sparse matrix (100 rows, 500 columns)
    m, n = 100, 500
    rng = np.random.RandomState(42)
    
    # Create sparse CSV format
    matrix_path = os.path.join(output_dir, "sample_matrix.csv")
    with open(matrix_path, 'w') as f:
        f.write("# Sparse binary matrix: each line is row indices of 1s\n")
        for i in range(m):
            nnz = rng.randint(5, 20)
            indices = sorted(rng.choice(n, size=nnz, replace=False))
            f.write(','.join(map(str, indices)) + '\n')
    print(f"  Created: {matrix_path}")
    
    # Create RHS vector
    rhs_path = os.path.join(output_dir, "sample_rhs.csv")
    b = rng.randn(m)
    np.savetxt(rhs_path, b, delimiter=',', fmt='%.10e')
    print(f"  Created: {rhs_path}")
    
    print(f"\nTo solve, run:")
    print(f"  python file_solver.py --matrix {matrix_path} --rhs {rhs_path} --output solution.csv")
    
    return matrix_path, rhs_path


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Solve Ax=b using Weighted Randomized Kaczmarz",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Solve from CSV files
  python file_solver.py --matrix A.csv --rhs b.csv --output x.csv
  
  # Solve from NumPy files with custom tolerance
  python file_solver.py --matrix A.npz --rhs b.npy --output x.npy --tol 1e-8
  
  # Create sample files for testing
  python file_solver.py --create-samples
  
  # Run built-in demo
  python file_solver.py --demo
        """
    )
    
    parser.add_argument('--matrix', '-A', help='Path to matrix file (CSV or NPZ)')
    parser.add_argument('--rhs', '-b', help='Path to right-hand side vector file')
    parser.add_argument('--output', '-o', help='Path to save solution')
    parser.add_argument('--tol', type=float, default=1e-6, help='Convergence tolerance')
    parser.add_argument('--maxiter', type=int, default=1_000_000, help='Max iterations')
    parser.add_argument('--quiet', action='store_true', help='Suppress progress output')
    parser.add_argument('--compare', action='store_true', help='Compare with NumPy solver')
    parser.add_argument('--create-samples', action='store_true', help='Create sample input files')
    parser.add_argument('--demo', action='store_true', help='Run built-in demo')
    
    args = parser.parse_args()
    
    if args.create_samples:
        create_sample_files()
        return
    
    if args.demo:
        # Run a self-contained demo
        from src.sparse_matrix import create_test_matrix
        
        print("Running built-in demo...\n")
        A = create_test_matrix(m=500, n=2000, avg_nnz_per_row=10, seed=42)
        b = np.random.RandomState(123).randn(500)
        
        result = solve_kaczmarz(A, b, tolerance=1e-6, verbose=True)
        print(f"\n{result}")
        
        if args.compare:
            compare_with_numpy(A, b, result.x)
        return
    
    if not args.matrix or not args.rhs:
        parser.print_help()
        print("\nError: --matrix and --rhs are required (or use --demo / --create-samples)")
        return 1
    
    # Solve from files
    x = solve_from_files(
        args.matrix, args.rhs, args.output,
        tolerance=args.tol,
        max_iterations=args.maxiter,
        verbose=not args.quiet
    )
    
    if args.compare:
        A = load_matrix(args.matrix)
        b = load_vector(args.rhs)
        compare_with_numpy(A, b, x)


if __name__ == "__main__":
    main()
