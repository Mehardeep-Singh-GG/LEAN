"""
Unit tests for Description Length Rulers A, B, C and Ground Truth Benchmarks.
"""

import pytest
import torch
import torch.nn as nn

from src.lean.toy.ground_truth_benchmarks import (
    BenchmarkLinear,
    BenchmarkMLP,
    BenchmarkComposite,
    generate_r1_random_table,
    generate_r2_pure_rule,
    generate_r3_rule_plus_table,
    generate_r4_noise_labels
)
from src.lean.measure.ruler_a import RulerA, quantize_tensor_rtn, svd_low_rank_compress, magnitude_prune
from src.lean.measure.ruler_b import calculate_prequential_code_length
from src.lean.measure.ruler_c import get_benchmark_program_complexity


def test_ground_truth_generation():
    # R1
    r1 = generate_r1_random_table(k_facts=64, vocab_size=32, seed=42)
    assert r1["x"].shape == (64, 64)
    assert r1["y"].shape == (64,)
    assert r1["theoretical_bits"] == 64 * 5.0
    
    # R2
    r2 = generate_r2_pure_rule(num_samples=100, prime_p=17, seed=42)
    assert r2["x"].shape == (100, 34)
    assert r2["y"].shape == (100,)
    assert r2["theoretical_rule_bits"] > 0
    
    # R3
    r3 = generate_r3_rule_plus_table(k_facts=64, vocab_size=32, num_samples=100, prime_p=32, seed=42)
    assert r3["x"].shape == (100, 64 + 32 + 32)
    assert r3["y"].shape == (100,)
    
    # R4
    r4 = generate_r4_noise_labels(num_samples=50, num_classes=10, seed=42)
    assert r4["x"].shape == (50, 64)
    assert r4["y"].shape == (50,)


def test_ruler_a_compression_primitives():
    w = torch.randn(32, 64)
    
    # RTN
    deq_rtn, dl_rtn = quantize_tensor_rtn(w, bits=4)
    assert deq_rtn.shape == w.shape
    assert dl_rtn > 0
    assert dl_rtn < w.numel() * 16
    
    # SVD
    deq_svd, dl_svd = svd_low_rank_compress(w, rank=4, factor_bits=4)
    assert deq_svd.shape == w.shape
    assert dl_svd > 0
    
    # Prune
    deq_prune, dl_prune = magnitude_prune(w, sparsity=0.5)
    assert deq_prune.shape == w.shape
    assert dl_prune > 0


def test_ruler_a_sweep():
    model = nn.Linear(32, 16)
    x = torch.randn(20, 32)
    y = torch.randint(0, 16, (20,))
    
    def eval_fn(m):
        with torch.no_grad():
            preds = torch.argmax(m(x), dim=1)
            return (preds == y).float().mean().item()
            
    ruler = RulerA(model, eval_fn=eval_fn, higher_is_better=True)
    res = ruler.find_dl_min(tau=0.90)
    assert "dl_min_bits" in res
    assert res["dl_min_bits"] > 0
    assert len(res["all_candidates"]) > 0


def test_ruler_b_prequential():
    x = torch.randn(40, 16)
    y = torch.randint(0, 4, (40,))
    
    model_fn = lambda: nn.Linear(16, 4)
    res = calculate_prequential_code_length(model_fn, x, y, num_blocks=2, epochs_per_block=5, lr=0.01, prior_param_bits=100)
    assert res["num_blocks"] == 2
    assert res["total_prequential_bits"] > 100


def test_ruler_c_program_complexity():
    res = get_benchmark_program_complexity("R2_Pure_Rule")
    assert res["program_bits"] > 0
    assert len(res["code"]) > 0
