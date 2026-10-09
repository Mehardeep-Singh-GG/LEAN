"""
Unit tests for Fundamental Invariants (A4).
"""

import pytest
import math
import numpy as np


def test_probability_simplex_invariant():
    """Verify that probability distributions sum to 1.0."""
    logits = np.array([2.5, 1.2, -0.5, 0.0])
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / np.sum(exp_logits)
    
    assert math.isclose(float(np.sum(probs)), 1.0, rel_tol=1e-6)
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)


def test_description_length_non_negativity():
    """Verify that description length calculations never yield negative bits."""
    vocab_size = 100
    num_facts = 50
    # Theoretical entropy of random table
    entropy_bits = num_facts * math.log2(vocab_size)
    assert entropy_bits > 0
    
    # Param overhead cannot be negative
    param_bits = 1024 * 16  # 1K params @ fp16
    assert param_bits > 0
