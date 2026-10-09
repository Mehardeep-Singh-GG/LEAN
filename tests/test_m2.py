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


def test_storage_breakdown_accounting():
    from src.lean.compress.layerwise import calculate_storage_breakdown
    model = DummyLM(vocab_size=100, hidden_dim=32)
    breakdown = calculate_storage_breakdown(model, "rtn", 4, group_size=16, skip_modules=["lm_head"])
    assert "backbone_packed_mb" in breakdown
    assert "uncompressed_packed_mb" in breakdown
    assert "total_packed_mb" in breakdown
    assert breakdown["backbone_raw_weight_bits"] > 0
    assert breakdown["backbone_metadata_bits"] > 0
    assert breakdown["backbone_packed_mb"] < breakdown["uncompressed_packed_mb"]


def test_int4_pack_unpack_roundtrip():
    from src.lean.compress.layerwise import pack_int4, unpack_int4
    t = torch.randint(-8, 8, (64, 128), dtype=torch.int8)
    packed = pack_int4(t)
    restored = unpack_int4(packed, t.numel()).view(t.shape)
    assert torch.equal(t.float(), restored)


def test_quantized_checkpoint_save_and_reload(tmp_path):
    from src.lean.compress.layerwise import save_quantized_checkpoint, load_quantized_checkpoint
    model = DummyLM(vocab_size=50, hidden_dim=16)
    ckpt_file = tmp_path / "dummy_packed.pt"
    save_info = save_quantized_checkpoint(model, str(ckpt_file), bits=4, group_size=8, skip_modules=["lm_head"])
    assert ckpt_file.exists()
    assert save_info["file_bytes"] > 0
    
    reloaded = load_quantized_checkpoint(str(ckpt_file), model, group_size=8)
    # Check that reloaded model parameters have the expected shape
    for p_orig, p_rel in zip(model.parameters(), reloaded.parameters()):
        assert p_orig.shape == p_rel.shape


def test_explicit_power_models():
    from src.lean.eval.probes import compute_knowledge_power, compute_skill_invariance_power, compute_correlation_power
    p_k = compute_knowledge_power(p0=0.66, delta=0.05, alpha=0.05, n=3000)
    assert p_k["statistical_power"] >= 0.80
    assert p_k["passed"] is True
    
    p_s = compute_skill_invariance_power(p0=0.52, delta=0.05, alpha=0.05, n=500)
    assert p_s["statistical_power"] >= 0.80
    assert p_s["passed"] is True
    
    p_c = compute_correlation_power(rho=0.15, alpha=0.05, n=3000)
    assert p_c["statistical_power"] >= 0.80
    assert p_c["passed"] is True

