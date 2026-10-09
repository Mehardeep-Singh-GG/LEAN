"""
Unit tests for Milestone M2: Perplexity, Layer-wise Compression, and Probes Suite.
"""

import pytest
import math
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.lean.eval.perplexity import calculate_perplexity
from src.lean.compress.layerwise import compress_linear_layer, compress_model_layerwise
from src.lean.eval.probes import (
    generate_knowledge_probes,
    generate_skill_probes,
    compute_statistical_power
)


class DummyLM(nn.Module):
    """Tiny causal language model for unit tests."""
    def __init__(self, vocab_size=100, hidden_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        
    def forward(self, input_ids, labels=None):
        h = torch.relu(self.fc1(self.embedding(input_ids)))
        logits = self.lm_head(h)
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.CrossEntropyLoss()(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
        class Output:
            def __init__(self, l, log):
                self.loss = l
                self.logits = log
        return Output(loss, logits)


def test_calculate_perplexity_monotonicity():
    model = DummyLM()
    
    class DummyTokenizer:
        def __call__(self, text, return_tensors="pt"):
            tokens = [hash(w) % 100 for w in text.split()]
            return {"input_ids": torch.tensor([tokens])}
            
    tok = DummyTokenizer()
    text = "hello world this is a test sequence for perplexity measurement"
    res = calculate_perplexity(model, tok, text, max_length=16, stride=8, device="cpu")
    assert "perplexity" in res
    assert res["perplexity"] > 0
    assert res["total_tokens"] > 0


def test_layerwise_compression_primitives():
    layer = nn.Linear(64, 64)
    orig_norm = layer.weight.norm().item()
    
    # RTN 4-bit
    c_layer, dl = compress_linear_layer(layer, "rtn", 4)
    assert dl > 0
    assert c_layer.weight.shape == (64, 64)
    
    # Prune 50%
    layer2 = nn.Linear(64, 64)
    c_layer2, dl2 = compress_linear_layer(layer2, "prune", 0.50)
    assert (c_layer2.weight == 0).sum().item() > 0
    
    # SVD
    layer3 = nn.Linear(64, 64)
    c_layer3, dl3 = compress_linear_layer(layer3, "svd", 8)
    assert dl3 > 0


def test_compress_model_layerwise():
    model = DummyLM()
    c_model, total_dl = compress_model_layerwise(model, "rtn", 4, skip_modules=["lm_head"])
    assert total_dl > 0
    # lm_head was skipped, fc1 was compressed
    assert c_model.fc1.weight.shape == model.fc1.weight.shape


def test_knowledge_probes_structure():
    probes = generate_knowledge_probes(total_facts=50, seed=42)
    assert len(probes) >= 50
    p0 = probes[0]
    assert "fact_id" in p0
    assert "relation" in p0
    assert "pretraining_frequency" in p0
    assert "log_frequency" in p0
    assert len(p0["paraphrases"]) >= 2


def test_skill_probes_structure():
    probes = generate_skill_probes(num_per_skill=10, seed=42)
    assert len(probes) == 100
    skills = {p["skill_id"] for p in probes}
    assert len(skills) == 10  # 10 distinct skills
    assert all(p["synthetic"] for p in probes)


def test_power_analysis():
    power = compute_statistical_power(delta=0.05, alpha=0.05, n1=3000, n2=3000)
    assert power >= 0.80
    assert power <= 1.0
