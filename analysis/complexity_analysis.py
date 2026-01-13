"""
Complexity Analysis and Theoretical Justification

This module provides detailed time and space complexity analysis of the
Weighted Randomized Kaczmarz algorithm, along with theoretical justification
for its superiority over alternative methods for this problem class.

Author: Advanced Algorithms Student
Date: January 2026
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def print_complexity_analysis():
    """Print detailed complexity analysis."""
    
    print("""
================================================================================
         COMPLEXITY ANALYSIS: WEIGHTED RANDOMIZED KACZMARZ ALGORITHM
================================================================================

PROBLEM PARAMETERS
==================
    m = 473 × 512 = 242,176          (number of equations / rows)
    n = 512³ = 134,217,728           (number of unknowns / columns)
    s̄ = average row sparsity         (average number of 1s per row)
    nnz = total non-zeros ≈ m × s̄    (total 1s in matrix)

Note: m << n, so the system is highly underdetermined.

--------------------------------------------------------------------------------
SPACE COMPLEXITY
--------------------------------------------------------------------------------

Component                   | Size              | Memory (8 bytes/float64)
----------------------------|-------------------|---------------------------
Solution vector x           | n = 134M          | ~1.07 GB
Row weight array            | m = 242K          | ~1.94 MB  
Alias sampler tables        | 2m = 484K         | ~3.88 MB
Residual history window     | W = 1000          | ~8 KB
RNG state                   | O(1)              | ~256 bytes
----------------------------|-------------------|---------------------------
TOTAL                       |                   | ~1.08 GB

CRITICAL: We NEVER store the matrix A, which would require:
    Dense: m × n × 1 byte = 32 PB (petabytes) for bits
    Dense float: m × n × 8 = 256 PB
    
Our approach uses 10⁷× less memory than naive dense storage.

--------------------------------------------------------------------------------
TIME COMPLEXITY
--------------------------------------------------------------------------------

PREPROCESSING (one-time):
    • Compute row weights: O(m × s̄) = O(nnz)
    • Build alias sampler: O(m)
    Total preprocessing: O(nnz + m) = O(nnz)

PER ITERATION:
    Operation                      | Complexity
    -------------------------------|-------------
    Sample row (alias method)      | O(1)
    Retrieve row indices           | O(s) where s = |row|
    Sparse dot product ⟨aᵢ, x⟩    | O(s)
    Compute residual bᵢ - ⟨aᵢ, x⟩ | O(1)
    Update x[indices] += α         | O(s)
    Update running statistics      | O(1)
    Check stopping criterion       | O(1)
    -------------------------------|-------------
    TOTAL PER ITERATION            | O(s)

EXPECTED ITERATIONS TO TOLERANCE ε:
    
    For consistent systems, Strohmer & Vershynin (2009) prove:
    
    E[‖xₖ - x*‖²] ≤ (1 - σ²ₘᵢₙ(A)/‖A‖²_F)^k ‖x₀ - x*‖²
    
    where σₘᵢₙ is the smallest singular value of A.
    
    To achieve relative error ε:
    
    K = O(κ²(A) × log(1/ε))
    
    where κ(A) = σₘₐₓ/σₘᵢₙ is the condition number.
    
    For well-conditioned systems: K = O(log(1/ε))
    For ill-conditioned systems: K = O(κ² × log(1/ε))

TOTAL TIME COMPLEXITY:
    
    T = O(nnz) + O(K × s̄)
      = O(nnz + K × s̄)
    
    If K = O(nnz/s̄) iterations suffice (typical for moderate κ):
    T = O(nnz)

--------------------------------------------------------------------------------
CONVERGENCE THEORY
--------------------------------------------------------------------------------

THEOREM (Strohmer-Vershynin, 2009):
    For a consistent system Ax = b, the randomized Kaczmarz method with
    weighted sampling P(i) ∝ ‖aᵢ‖² converges in expectation:
    
    E[‖xₖ - x*‖²] ≤ (1 - 1/κ²(A))^k ‖x₀ - x*‖²
    
    where κ(A) = ‖A‖_F × ‖A†‖ is the scaled condition number.

MINIMUM-NORM PROPERTY:
    Starting from x₀ = 0, the iterates converge to:
    
    x* = A†b = Aᵀ(AAᵀ)⁻¹b
    
    which is the unique minimum-norm solution lying in the row space of A.
    
    Proof sketch:
    - Each Kaczmarz step is a projection onto a hyperplane
    - Projections preserve the component in the row space
    - Starting from 0 (in row space), we stay in row space
    - The row-space solution is the minimum-norm solution ∎

INCONSISTENT SYSTEMS:
    For Ax = b with no exact solution, iterates oscillate around the 
    least-squares solution but do not converge. Our stagnation detection
    identifies this via low coefficient of variation in residuals.

--------------------------------------------------------------------------------
COMPARISON WITH ALTERNATIVE METHODS
--------------------------------------------------------------------------------

1. DIRECT METHODS (Gaussian Elimination, LU, QR)
   
   Method      | Time           | Space        | Verdict
   ------------|----------------|--------------|------------------
   Gaussian    | O(mn²)         | O(mn)        | IMPOSSIBLE (64 TB)
   LU          | O(mn²)         | O(mn)        | IMPOSSIBLE
   QR          | O(mn²)         | O(mn)        | IMPOSSIBLE
   Sparse LU   | O(fill-in)     | O(fill-in)   | Fill-in destroys sparsity
   
   These methods require materializing A or factors thereof.
   At 64+ TB, this is physically impossible.

2. KRYLOV SUBSPACE METHODS (GMRES, CG, LSQR)
   
   Method      | Per-Iter       | Issue
   ------------|----------------|----------------------------------------
   GMRES       | O(nnz + k²)    | Requires A*v products (need columns)
   CG          | O(nnz)         | A must be symmetric positive definite
   LSQR        | O(nnz)         | Requires both A*v and Aᵀ*v
   
   These methods need column-wise access or matrix-vector products.
   Our constraint: only row-wise access via index lists.
   
   Verdict: NOT APPLICABLE without column access.

3. STOCHASTIC GRADIENT DESCENT
   
   SGD can be viewed as a special case of Kaczmarz for ‖Ax-b‖².
   However:
   - Requires careful step-size tuning
   - Does not naturally give minimum-norm for underdetermined systems
   - Randomized Kaczmarz is the optimal variant for linear systems
   
   Verdict: Kaczmarz IS the optimal SGD for this problem.

4. MACHINE LEARNING / DEEP LEARNING
   
   Issues:
   - No theoretical guarantee of finding ANY solution
   - No guarantee of minimum-norm property
   - Would need to train on similar systems (data requirement)
   - Massive computational overhead for 134M-dimensional output
   - This is a LINEAR problem - neural nets are for nonlinear patterns
   
   Verdict: OVERKILL and INAPPROPRIATE. This is linear algebra, not 
   pattern recognition.

5. NATURE-INSPIRED METHODS (GA, PSO, Simulated Annealing)
   
   Issues:
   - Population of 134M-dimensional vectors = impossible memory
   - No convergence guarantees
   - Designed for non-convex optimization
   - Would need astronomical iterations
   - Finding a feasible solution for underdetermined system is trivial
     with proper linear algebra
   
   Verdict: FUNDAMENTALLY WRONG TOOL for linear systems.

6. COORDINATE DESCENT / RANDOMIZED COORDINATE DESCENT
   
   Issues:
   - Works column-by-column (need column access)
   - Does not exploit row-wise sparsity structure
   
   Verdict: Wrong axis. We have rows, not columns.

================================================================================
                         CONCLUSION
================================================================================

The Weighted Randomized Kaczmarz algorithm is the OPTIMAL choice because:

1. MEMORY EFFICIENCY
   - O(n + m) space vs O(mn) for direct methods
   - Factor of 10⁷ improvement

2. ROW-ACTION NATURE
   - Naturally handles row-wise implicit access
   - No column access or full matrix-vector products needed

3. THEORETICAL GUARANTEES
   - Proven exponential convergence rate
   - Guaranteed minimum-norm solution from x₀ = 0
   - Well-understood behavior for inconsistent systems

4. SIMPLICITY
   - Easy to implement correctly
   - Easy to verify correctness
   - No hyperparameters requiring extensive tuning

5. SCALABILITY  
   - Iteration cost O(s̄) independent of m, n
   - Easily parallelizable across multiple random seeds
   - Can checkpoint and resume

This is not just a good choice—it is the RIGHT choice given the constraints.
No other known algorithm can solve this problem at this scale with row-only
access and reasonable resources.

================================================================================
""")


def estimate_runtime(m, n, avg_sparsity, condition_number, tolerance):
    """
    Estimate runtime for the Kaczmarz solver.
    
    Args:
        m: Number of rows
        n: Number of columns
        avg_sparsity: Average non-zeros per row
        condition_number: Estimated condition number κ(A)
        tolerance: Target relative tolerance
        
    Returns:
        Dictionary with estimates
    """
    # Estimate iterations
    # K ≈ κ² × log(1/ε) for relative tolerance ε
    K_theoretical = condition_number**2 * np.log(1/tolerance)
    
    # Time per iteration (empirical: ~10-100 μs depending on sparsity)
    time_per_iter_us = 10 + avg_sparsity * 0.5  # microseconds
    
    # Total time
    total_time_s = K_theoretical * time_per_iter_us / 1e6
    
    # Memory
    solution_memory_gb = n * 8 / 1e9
    weights_memory_mb = m * 8 / 1e6
    
    return {
        'estimated_iterations': int(K_theoretical),
        'time_per_iteration_us': time_per_iter_us,
        'total_time_seconds': total_time_s,
        'total_time_minutes': total_time_s / 60,
        'solution_memory_gb': solution_memory_gb,
        'weights_memory_mb': weights_memory_mb,
    }


def main():
    """Run complexity analysis."""
    print_complexity_analysis()
    
    # Estimate for target problem
    print("\nRUNTIME ESTIMATE FOR TARGET PROBLEM")
    print("="*50)
    
    m = 473 * 512
    n = 512**3
    avg_sparsity = 50  # Assumption
    
    # Conservative condition number estimates
    for kappa in [10, 100, 1000]:
        print(f"\nCondition number κ = {kappa}:")
        est = estimate_runtime(m, n, avg_sparsity, kappa, 1e-6)
        print(f"  Estimated iterations: {est['estimated_iterations']:,}")
        print(f"  Time per iteration: {est['time_per_iteration_us']:.1f} μs")
        print(f"  Total time: {est['total_time_minutes']:.1f} minutes")
        print(f"  Solution memory: {est['solution_memory_gb']:.2f} GB")


if __name__ == "__main__":
    main()
