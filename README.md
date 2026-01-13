# Weighted Randomized Kaczmarz Solver for Large Sparse Underdetermined Systems

## Problem Statement

We seek to solve an extremely large, sparse, underdetermined system of linear equations:

$$Ax = b$$

where:
- $A \in \{0, 1\}^{m \times n}$ is a binary (bit) matrix
- $m < n$ (underdetermined system)
- $n = 512^3 = 134,217,728$ columns
- $m = 473 \times 512 = 242,176$ rows
- Explicit storage of $A$ is impossible (~64 TB for dense storage)
- Only row-wise access to indices of non-zero (1) entries is available

## Mathematical Background

### The Kaczmarz Algorithm

The classical Kaczmarz algorithm (also known as Algebraic Reconstruction Technique in CT imaging) is an iterative row-action method for solving linear systems. Given current iterate $x^{(k)}$, it projects onto the hyperplane defined by the $i$-th equation:

$$x^{(k+1)} = x^{(k)} + \frac{b_i - a_i^T x^{(k)}}{\|a_i\|^2} a_i$$

where $a_i$ is the $i$-th row of $A$.

### Randomized Kaczmarz (Strohmer & Vershynin, 2009)

Instead of cycling through rows, we select row $i$ with probability proportional to $\|a_i\|^2$:

$$P(i = k) = \frac{\|a_k\|^2}{\|A\|_F^2}$$

This achieves expected linear convergence:

$$\mathbb{E}[\|x^{(k)} - x^*\|^2] \leq \left(1 - \frac{\sigma_{\min}^2(A)}{\|A\|_F^2}\right)^k \|x^{(0)} - x^*\|^2$$

### Minimum-Norm Solution for Underdetermined Systems

For underdetermined systems ($m < n$), infinitely many solutions exist. By initializing $x^{(0)} = 0$, the Kaczmarz iterates converge to the **minimum-norm solution**:

$$x^* = A^T(AA^T)^{-1}b = A^\dagger b$$

This is the unique solution with smallest $\ell_2$ norm, which lies entirely in the row space of $A$.

### Weighted Sampling for Bit Matrices

For our binary matrix where all non-zeros are 1:
- $\|a_i\|^2 = \text{nnz}(a_i)$ (number of 1s in row $i$)
- The Frobenius norm squared equals total non-zeros: $\|A\|_F^2 = \text{nnz}(A)$

Sampling rows proportionally to their sparsity (number of 1s) gives denser rows higher selection probability, improving convergence for rows that constrain more variables.

## Algorithm Design

### Algorithm 1: Weighted Randomized Kaczmarz with Adaptive Stopping

```
Input: Row access function row(i), right-hand side b ∈ ℝᵐ, tolerance ε, window size W
Output: Approximate minimum-norm solution x ∈ ℝⁿ

1.  Initialize x = 0 ∈ ℝⁿ
2.  Compute row weights w[i] = |row(i)| for all i ∈ {0, ..., m-1}
3.  Total weight S = Σᵢ w[i]
4.  Construct sampling distribution P[i] = w[i] / S
5.  Initialize residual history H = empty deque of size W
6.  
7.  for k = 1, 2, ... do
8.      Sample row index i ~ P
9.      Retrieve non-zero indices J = row(i)
10.     Compute sparse dot product: ⟨aᵢ, x⟩ = Σⱼ∈J x[j]
11.     Compute residual: rᵢ = bᵢ - ⟨aᵢ, x⟩
12.     Compute step size: α = rᵢ / |J|        // Since ||aᵢ||² = |J| for binary rows
13.     Update: x[j] += α for all j ∈ J
14.     
15.     // Adaptive stopping criterion
16.     Append |rᵢ|² to H
17.     if |H| = W then
18.         μ = mean(H)
19.         if μ < ε² then return x                    // Converged
20.         if stagnation_detected(H, threshold) then
21.             report "System may be inconsistent"
22.             return x                               // Graceful fallback
23.         end if
24.     end if
25. end for
```

### Stagnation Detection

To detect inconsistent systems (no exact solution exists), we monitor the moving average of squared residuals. If the residual variance becomes very small but the mean remains above threshold, the system is likely inconsistent:

$$\text{stagnation} = \left(\frac{\sigma(H)}{\mu(H)} < \tau_{cv}\right) \land \left(\mu(H) > \epsilon^2\right)$$

where $\tau_{cv}$ is a coefficient of variation threshold (typically 0.1).

## Complexity Analysis

### Time Complexity

**Per iteration:**
- Row sampling: $O(1)$ with alias method, $O(\log m)$ with binary search
- Sparse dot product: $O(s)$ where $s = |J|$ is the row sparsity
- Update: $O(s)$

**Total:** $O(K \cdot \bar{s})$ where $K$ is iteration count and $\bar{s}$ is average row sparsity.

For convergence to tolerance $\epsilon$:
$$K = O\left(\frac{\|A\|_F^2}{\sigma_{\min}^2(A)} \log\frac{1}{\epsilon}\right)$$

### Space Complexity

- Solution vector $x$: $O(n)$
- Sampling weights: $O(m)$
- Per-row storage (implicit): $O(\text{nnz}(A) / m) = O(\bar{s})$ average
- Residual history: $O(W)$

**Total: $O(n + m)$** — linear in problem dimensions, independent of $mn$.

This is the critical advantage: we never store the $m \times n$ matrix.

## Why This Approach?

### vs. Direct Methods (Gaussian Elimination, LU, QR)
- **Memory:** Direct methods require $O(mn)$ storage — impossible at 64 TB
- **Sparsity:** Fill-in destroys sparsity during factorization
- **Underdetermined:** Not designed for minimum-norm solutions

### vs. Krylov Methods (GMRES, LSQR, CGLS)
- **Matrix-vector products:** Require computing $Ax$ or $A^Tx$ — implicitly requires materializing columns
- **Our constraint:** Only row-wise access available, not column-wise
- **Memory:** Still need $O(n)$ but with larger constants

### vs. Machine Learning / Deep Learning
- **Interpretability:** Black-box models provide no theoretical guarantees
- **Data requirements:** Would need training data from similar systems
- **Generalization:** No guarantee of finding minimum-norm solution
- **Overkill:** This is a well-understood linear algebra problem, not a pattern recognition task

### vs. Nature-Inspired Methods (Genetic Algorithms, Particle Swarm)
- **Convergence:** No theoretical convergence guarantees
- **Efficiency:** Population-based methods scale poorly with dimension $n \approx 10^8$
- **Precision:** Heuristics unlikely to achieve high numerical precision
- **Misapplication:** These are for non-convex optimization, not linear systems

### Why Randomized Kaczmarz Wins
1. **Memory efficiency:** $O(n + m)$ storage
2. **Row-action nature:** Naturally handles implicit row access
3. **Theoretical guarantees:** Proven exponential convergence
4. **Minimum-norm:** Initialization at zero guarantees minimum-norm solution
5. **Simplicity:** Easy to implement, debug, and analyze

## Project Structure

```
AA Project/
├── README.md                   # This file
├── src/
│   ├── __init__.py
│   ├── sparse_matrix.py       # Implicit sparse matrix interface
│   ├── kaczmarz.py            # Core algorithm implementation
│   ├── sampling.py            # Weighted sampling utilities
│   └── stopping.py            # Adaptive stopping criteria
├── tests/
│   ├── __init__.py
│   ├── test_small_system.py   # Small synthetic test
│   ├── test_convergence.py    # Convergence verification
│   └── test_inconsistent.py   # Inconsistency detection test
└── examples/
    └── demo.py                # Full demonstration
```

## Usage

```python
from src.kaczmarz import WeightedRandomizedKaczmarz
from src.sparse_matrix import ImplicitBitMatrix

# Define row access function
def get_row_indices(i):
    # Return indices where row i has value 1
    return [...]  # Your implementation

# Create implicit matrix
A = ImplicitBitMatrix(m=242176, n=134217728, row_func=get_row_indices)

# Define right-hand side
b = [...]  # Your values

# Solve
solver = WeightedRandomizedKaczmarz(tolerance=1e-6, window_size=1000)
x = solver.solve(A, b)
```

## References

1. Strohmer, T., & Vershynin, R. (2009). A randomized Kaczmarz algorithm with exponential convergence. *Journal of Fourier Analysis and Applications*, 15(2), 262-278.

2. Needell, D. (2010). Randomized Kaczmarz solver for noisy linear systems. *BIT Numerical Mathematics*, 50(2), 395-403.

3. Zouzias, A., & Freris, N. M. (2013). Randomized extended Kaczmarz for solving least squares. *SIAM Journal on Matrix Analysis and Applications*, 34(2), 773-793.

4. Kaczmarz, S. (1937). Angenäherte Auflösung von Systemen linearer Gleichungen. *Bulletin International de l'Académie Polonaise des Sciences et des Lettres*, 35, 355-357.

---
*Submitted as coursework for Advanced Algorithms*
*Author: [Student Name]*
*Date: January 2026*
