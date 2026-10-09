"""
Unit and integration tests for Milestone M11: Theory -> Technology (KAC & MLX).
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path

from vault.guard import compute_sha256
from src.lean.compress.kac import KnapsackAwareCompressor
from src.lean.core.retrieval_store import BM25Index, build_canonical_knowledge_corpus, ThinkingCoreRetrievalPipeline
from src.lean.infer.mlx_engine import MLXBenchmarkEngine
from experiments.run_m11_technology import PREREG_M11_PATH, EXPECTED_M11_HASH


def test_m11_preregistration_hash():
    """Verify prereg/M11.md exists and matches locked cryptographic SHA-256 hash."""
    assert PREREG_M11_PATH.exists(), f"{PREREG_M11_PATH} missing!"
    actual_hash = compute_sha256(PREREG_M11_PATH)
    assert actual_hash == EXPECTED_M11_HASH, f"Hash mismatch: expected {EXPECTED_M11_HASH}, got {actual_hash}"


def test_kac_layer_classification_and_planning():
    """Verify KAC correctly classifies attention and MLP projections and calculates mixed-precision budget."""
    kac = KnapsackAwareCompressor(group_size=128)
    
    # Simple mock transformer block
    class MockBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(64, 64)
            self.gate_proj = nn.Linear(64, 128)
            self.ln = nn.LayerNorm(64)
            self.wte = nn.Embedding(100, 64)
            
    m = MockBlock()
    plan = kac.plan_bit_allocation(m, attention_bits=4, mlp_bits=2)
    
    assert plan["layer_plan"]["q_proj.weight"]["type"] == "attention"
    assert plan["layer_plan"]["q_proj.weight"]["bits"] == 4
    assert plan["layer_plan"]["gate_proj.weight"]["type"] == "mlp"
    assert plan["layer_plan"]["gate_proj.weight"]["bits"] == 2
    assert plan["layer_plan"]["wte.weight"]["quantize"] is False
    assert plan["layer_plan"]["ln.weight"]["quantize"] is False
    assert plan["compression_ratio"] > 1.0


def test_retrieval_store_bm25_search():
    """Verify BM25 index correctly indexes corpus and retrieves exact facts."""
    corpus = build_canonical_knowledge_corpus()
    assert len(corpus) >= 100
    
    idx = BM25Index()
    idx.fit(corpus)
    
    # Search for Paris
    results = idx.search("What is the capital of France?", top_k=1)
    assert len(results) == 1
    top_doc = results[0][0]
    assert "Paris" in top_doc["text"]
    assert "France" in top_doc["text"]


def test_mlx_benchmark_engine():
    """Verify native Apple Silicon MLX benchmark engine executes without errors."""
    engine = MLXBenchmarkEngine()
    res = engine.benchmark_matrix_decode_throughput(hidden_dim=256, intermediate_dim=512, n_tokens=16, n_layers=2)
    assert "mlx_tokens_per_sec" in res
    assert "pytorch_tokens_per_sec" in res
    assert res["speedup_factor"] > 0.0
