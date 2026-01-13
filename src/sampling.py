"""
Weighted Sampling Utilities for Randomized Kaczmarz

This module implements efficient weighted random sampling for selecting rows
in the randomized Kaczmarz algorithm. Row selection probability is proportional
to row weight (squared norm).

For binary matrices: P(row i) = nnz(row_i) / nnz(A)

Author: Advanced Algorithms Student
Date: January 2026
"""

import numpy as np
from typing import Optional


class WeightedSampler:
    """
    Efficient weighted random sampler using Walker's Alias Method.
    
    The alias method allows O(1) sampling after O(n) preprocessing,
    compared to O(log n) for binary search on cumulative weights.
    
    For the Kaczmarz algorithm, we sample rows with probability proportional
    to their squared norms, which equals their sparsity for binary matrices.
    
    Reference:
        Walker, A. J. (1977). An efficient method for generating discrete 
        random variables with general distributions. ACM TOMS, 3(3), 253-256.
    """
    
    def __init__(self, weights: np.ndarray, seed: Optional[int] = None):
        """
        Initialize the sampler with given weights.
        
        Args:
            weights: Array of non-negative weights. Need not sum to 1.
            seed: Random seed for reproducibility
            
        Time: O(n) for preprocessing
        Space: O(n) for alias tables
        """
        self._n = len(weights)
        self._rng = np.random.default_rng(seed)
        
        # Normalize weights to probabilities
        total = np.sum(weights)
        if total <= 0:
            raise ValueError("Total weight must be positive")
        
        probs = weights.astype(np.float64) * self._n / total
        
        # Initialize alias tables
        self._prob = np.zeros(self._n, dtype=np.float64)
        self._alias = np.zeros(self._n, dtype=np.int64)
        
        # Partition indices into small (< 1) and large (>= 1) groups
        small = []
        large = []
        
        for i in range(self._n):
            if probs[i] < 1.0:
                small.append(i)
            else:
                large.append(i)
        
        # Build alias table using Robin Hood algorithm
        while small and large:
            s = small.pop()
            l = large.pop()
            
            self._prob[s] = probs[s]
            self._alias[s] = l
            
            probs[l] = probs[l] + probs[s] - 1.0
            
            if probs[l] < 1.0:
                small.append(l)
            else:
                large.append(l)
        
        # Handle remaining items (should all have prob ~= 1 due to numerical issues)
        while large:
            l = large.pop()
            self._prob[l] = 1.0
        
        while small:
            s = small.pop()
            self._prob[s] = 1.0
    
    def sample(self) -> int:
        """
        Draw one sample from the weighted distribution.
        
        Returns:
            Index sampled with probability proportional to its weight
            
        Time: O(1)
        """
        # Pick a random column
        i = self._rng.integers(0, self._n)
        
        # Flip a biased coin
        if self._rng.random() < self._prob[i]:
            return i
        else:
            return self._alias[i]
    
    def sample_batch(self, size: int) -> np.ndarray:
        """
        Draw multiple samples (with replacement).
        
        Args:
            size: Number of samples to draw
            
        Returns:
            Array of sampled indices
            
        Time: O(size)
        """
        indices = self._rng.integers(0, self._n, size=size)
        coins = self._rng.random(size=size)
        
        # Vectorized alias decision
        use_alias = coins >= self._prob[indices]
        result = np.where(use_alias, self._alias[indices], indices)
        
        return result


class CumulativeSampler:
    """
    Simpler weighted sampler using cumulative distribution and binary search.
    
    Less efficient than alias method (O(log n) vs O(1) per sample) but
    simpler to understand and verify. Useful for validation.
    """
    
    def __init__(self, weights: np.ndarray, seed: Optional[int] = None):
        """
        Initialize with cumulative weights.
        
        Args:
            weights: Non-negative weights array
            seed: Random seed
        """
        self._rng = np.random.default_rng(seed)
        self._cumsum = np.cumsum(weights.astype(np.float64))
        self._total = self._cumsum[-1]
        
        if self._total <= 0:
            raise ValueError("Total weight must be positive")
    
    def sample(self) -> int:
        """
        Draw one weighted sample using binary search.
        
        Time: O(log n)
        """
        u = self._rng.random() * self._total
        return np.searchsorted(self._cumsum, u, side='right')
    
    def sample_batch(self, size: int) -> np.ndarray:
        """
        Draw multiple samples.
        
        Time: O(size * log n)
        """
        u = self._rng.random(size=size) * self._total
        return np.searchsorted(self._cumsum, u, side='right')


def verify_sampler(sampler, weights: np.ndarray, num_samples: int = 100000) -> dict:
    """
    Verify that a sampler produces the correct distribution.
    
    Args:
        sampler: Sampler object with sample() method
        weights: Original weights
        num_samples: Number of samples to draw for verification
        
    Returns:
        Dictionary with empirical vs theoretical probabilities and max deviation
    """
    # Theoretical probabilities
    probs = weights / np.sum(weights)
    
    # Empirical counts
    counts = np.zeros(len(weights), dtype=np.int64)
    for _ in range(num_samples):
        i = sampler.sample()
        counts[i] += 1
    
    empirical = counts / num_samples
    max_deviation = np.max(np.abs(empirical - probs))
    
    # Expected deviation for multinomial sampling
    expected_std = np.sqrt(probs * (1 - probs) / num_samples)
    max_expected_dev = 3 * np.max(expected_std)  # 3-sigma bound
    
    return {
        'theoretical': probs,
        'empirical': empirical,
        'max_deviation': max_deviation,
        'expected_max_deviation': max_expected_dev,
        'passed': max_deviation < max_expected_dev * 2  # Allow some slack
    }
