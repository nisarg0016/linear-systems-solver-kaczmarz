"""
Getting Input for Large Sparse Systems

For truly large systems (millions of rows/columns), the matrix is typically
NOT stored explicitly. Instead, it's defined by:
1. A mathematical rule or formula
2. A combinatorial structure (graphs, sets)
3. A database query
4. A streaming/generative process

This script shows practical examples of each approach.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sparse_matrix import ImplicitBitMatrix, ExplicitSparseRowMatrix
from src.kaczmarz import solve_kaczmarz


# =============================================================================
# EXAMPLE 1: Mathematical Formula (Combinatorics)
# =============================================================================

def example_combinatorial_constraints():
    """
    Example: Set covering / constraint satisfaction
    
    Imagine you have n=1,000,000 items and m=100,000 constraints.
    Each constraint says "at least one of these items must be selected".
    
    The matrix A is defined by: A[i,j] = 1 if item j appears in constraint i
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Combinatorial Constraints")
    print("="*70)
    
    n_items = 1_000_000       # 1 million items (columns)
    n_constraints = 100_000   # 100K constraints (rows)
    items_per_constraint = 50 # Each constraint involves ~50 items
    
    print(f"Problem: {n_constraints:,} constraints over {n_items:,} items")
    print(f"Matrix would need {n_constraints * n_items * 8 / 1e12:.1f} TB if stored dense!")
    
    def constraint_row(i):
        """
        Define which items appear in constraint i.
        In practice, this would come from your problem definition.
        """
        # Deterministic pseudo-random pattern based on constraint ID
        rng = np.random.RandomState(seed=i)
        # Each constraint involves a random subset of items
        return sorted(rng.choice(n_items, size=items_per_constraint, replace=False).tolist())
    
    # Create implicit matrix - only stores row weights, not the matrix itself!
    print("\nCreating implicit matrix (no storage of actual entries)...")
    A = ImplicitBitMatrix(
        m=n_constraints, 
        n=n_items, 
        row_func=constraint_row,
        precompute_weights=True  # Only stores m floats, not m*n
    )
    
    print(f"Memory for weights only: ~{n_constraints * 8 / 1e6:.1f} MB")
    print(f"Matrix shape: {A.shape}")
    
    # Generate a consistent RHS (this is just for demo)
    print("\nGenerating RHS vector...")
    b = np.random.RandomState(999).randn(n_constraints)
    
    # Solve (would take a while for this size - just showing the setup)
    print("\nMatrix is ready for solving!")
    print("Call: solve_kaczmarz(A, b, tolerance=1e-4)")
    
    return A, b


# =============================================================================
# EXAMPLE 2: Graph-Based Matrix
# =============================================================================

def example_graph_incidence():
    """
    Example: Graph incidence matrix
    
    For a graph with n nodes and m edges:
    - Each row represents an edge
    - A[edge, node] = 1 if node is endpoint of edge
    
    Common in network flow, graph algorithms, etc.
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Graph Incidence Matrix")
    print("="*70)
    
    n_nodes = 500_000   # 500K nodes
    n_edges = 2_000_000 # 2M edges
    
    print(f"Graph: {n_nodes:,} nodes, {n_edges:,} edges")
    
    # Simulate edge list (in practice, load from file or generate)
    print("Generating random graph edge list...")
    rng = np.random.RandomState(42)
    edges = []
    for _ in range(n_edges):
        u = rng.randint(0, n_nodes)
        v = rng.randint(0, n_nodes)
        while v == u:
            v = rng.randint(0, n_nodes)
        edges.append((min(u, v), max(u, v)))
    
    def edge_row(i):
        """Return the two endpoints of edge i."""
        return list(edges[i])
    
    A = ImplicitBitMatrix(m=n_edges, n=n_nodes, row_func=edge_row)
    print(f"Incidence matrix shape: {A.shape}")
    print(f"Each row has exactly 2 non-zeros (edge endpoints)")
    
    return A


# =============================================================================
# EXAMPLE 3: Database/File Streaming
# =============================================================================

def example_database_streaming():
    """
    Example: Matrix rows come from a database or large file
    
    Each query returns the non-zero column indices for one row.
    This is common when data is too large to load at once.
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Database/File Streaming")
    print("="*70)
    
    # Simulate a database with a simple file
    # In practice, this would be SQL queries or file reads
    
    class DatabaseMatrix:
        """Simulates reading matrix rows from a database."""
        
        def __init__(self, db_path, m, n):
            self.db_path = db_path
            self.m = m
            self.n = n
            self._cache = {}  # Optional caching
        
        def get_row(self, i):
            """
            Query row i from database.
            
            In practice:
                cursor.execute("SELECT col_idx FROM matrix WHERE row_id = ?", (i,))
                return [row[0] for row in cursor.fetchall()]
            """
            # Simulate with deterministic generation
            if i not in self._cache:
                rng = np.random.RandomState(seed=i * 31337)
                nnz = rng.randint(10, 50)
                self._cache[i] = sorted(rng.choice(self.n, size=nnz, replace=False).tolist())
            return self._cache[i]
    
    # Create the "database"
    m, n = 50_000, 200_000
    db = DatabaseMatrix("simulated_db", m, n)
    
    # Wrap in ImplicitBitMatrix
    A = ImplicitBitMatrix(m=m, n=n, row_func=db.get_row)
    
    print(f"Matrix from 'database': {A.shape}")
    print(f"Rows are fetched on-demand, not pre-loaded")
    
    return A


# =============================================================================
# EXAMPLE 4: Reading Sparse File Formats
# =============================================================================

def example_sparse_file_formats():
    """
    Example: Loading from standard sparse matrix formats
    
    For medium-large matrices that CAN fit in memory (up to ~100M non-zeros),
    you can use standard formats like:
    - Matrix Market (.mtx)
    - Coordinate format (COO)
    - Compressed Sparse Row (CSR)
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Sparse File Formats")
    print("="*70)
    
    print("""
Standard sparse formats for matrices that fit in memory:

1. MATRIX MARKET (.mtx) - Human readable
   %%MatrixMarket matrix coordinate real general
   1000 5000 15000
   1 5 1.0
   1 23 1.0
   2 7 1.0
   ...

2. COORDINATE (COO) - Simple CSV of (row, col, value)
   0,4,1
   0,22,1
   1,6,1
   ...

3. NUMPY SPARSE (.npz) - Efficient binary
   Save with: np.savez('matrix.npz', rows=row_lists, n=n_cols)
""")
    
    # Demo: Create and save a sparse matrix
    print("Creating sample sparse matrix file...")
    
    m, n = 1000, 5000
    rows = []
    rng = np.random.RandomState(42)
    for i in range(m):
        nnz = rng.randint(5, 20)
        rows.append(sorted(rng.choice(n, size=nnz, replace=False).tolist()))
    
    # Save in NPZ format
    output_path = "sample_sparse.npz"
    np.savez(output_path, rows=np.array(rows, dtype=object), n=n)
    print(f"Saved to: {output_path}")
    
    # Load it back
    loaded = np.load(output_path, allow_pickle=True)
    loaded_rows = [list(r) for r in loaded['rows']]
    loaded_n = int(loaded['n'])
    
    A = ExplicitSparseRowMatrix(loaded_rows, loaded_n)
    print(f"Loaded matrix: {A.shape}, {A.total_weight:.0f} non-zeros")
    
    return A


# =============================================================================
# EXAMPLE 5: Real-World Use Case - Feature Hashing
# =============================================================================

def example_feature_hashing():
    """
    Example: Feature hashing for machine learning
    
    In ML, you often have high-dimensional sparse features.
    The matrix A represents samples × features, where features
    are determined by hashing.
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: Feature Hashing (ML Use Case)")
    print("="*70)
    
    n_samples = 100_000
    n_features = 2**20  # 1 million feature buckets
    features_per_sample = 100
    
    print(f"Samples: {n_samples:,}")
    print(f"Feature space: {n_features:,} dimensions")
    
    def hash_features(sample_id):
        """
        Simulate feature hashing for a sample.
        
        In practice, this would hash actual feature names/values:
            features = []
            for word in document[sample_id].split():
                features.append(hash(word) % n_features)
            return features
        """
        rng = np.random.RandomState(seed=sample_id)
        return sorted(rng.choice(n_features, size=features_per_sample, replace=False).tolist())
    
    A = ImplicitBitMatrix(m=n_samples, n=n_features, row_func=hash_features)
    print(f"Feature matrix: {A.shape}")
    
    return A


# =============================================================================
# PRACTICAL TEMPLATE: Your Own Large Matrix
# =============================================================================

def your_matrix_template():
    """
    TEMPLATE: How to define your own large sparse matrix
    
    Copy and modify this for your specific problem!
    """
    print("\n" + "="*70)
    print("TEMPLATE: Define Your Own Large Matrix")
    print("="*70)
    
    print("""
# Step 1: Define your matrix dimensions
m = 1_000_000  # rows (equations/constraints)
n = 10_000_000 # columns (variables/unknowns)

# Step 2: Define a function that returns non-zero column indices for row i
def my_row_function(i):
    '''
    Return list of column indices where A[i, j] = 1
    
    This function encodes your problem structure!
    Examples:
    - Constraint i involves variables [j1, j2, j3, ...]
    - Edge i connects nodes [u, v]
    - Sample i has features [f1, f2, f3, ...]
    '''
    # Your logic here - this is problem-specific!
    # Example: row i has non-zeros at columns based on some formula
    cols = [i % n, (i * 2) % n, (i * 3 + 7) % n]
    return cols

# Step 3: Create the implicit matrix
from src.sparse_matrix import ImplicitBitMatrix
A = ImplicitBitMatrix(m=m, n=n, row_func=my_row_function)

# Step 4: Define your right-hand side vector b
import numpy as np
b = np.array([...])  # Your RHS values

# Step 5: Solve!
from src.kaczmarz import solve_kaczmarz
result = solve_kaczmarz(A, b, tolerance=1e-6, verbose=True)
x = result.x
""")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("LARGE SPARSE MATRIX INPUT METHODS")
    print("="*70)
    
    # Run examples (comment out the large ones if you want quick results)
    example_combinatorial_constraints()
    example_graph_incidence()
    example_database_streaming()
    example_sparse_file_formats()
    example_feature_hashing()
    your_matrix_template()
    
    print("\n" + "="*70)
    print("KEY TAKEAWAY")
    print("="*70)
    print("""
For large sparse systems, the matrix is usually IMPLICIT:

  ┌─────────────────────────────────────────────────────────────┐
  │  Don't store A[i,j] for all i,j                             │
  │  Instead, define: get_row(i) → [j1, j2, j3, ...]           │
  │  The Kaczmarz solver only needs ONE ROW at a time!          │
  └─────────────────────────────────────────────────────────────┘

Your row function encodes the problem structure:
  • Combinatorics → which items in constraint i
  • Graphs → which nodes edge i connects  
  • ML → which features sample i has
  • Physics → which variables equation i involves
""")
