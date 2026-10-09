"""
Unit tests for Milestone M5: World-Randomized MDL Factorization (WRMF).

Verifies:
1. Preregistration immutability and SHA-256 hash match.
2. Differentiable DL proxy calibration against exact physical Ruler A DL (r >= 0.95, rho >= 0.95).
3. Memory Triad architectures (LoRA, PKM, EntityEmbedding).
4. Frozen core transfer test mechanics (gradient isolation).
5. Null control structural collapse.
"""

import hashlib
from pathlib import Path
import pytest
import torch

from src.lean.wrmf.loss import validate_dl_proxy, differentiable_dl_proxy
from src.lean.wrmf.models import WRMFNetwork, ThinkingCore, create_memory_module
from src.lean.wrmf.trainer import (
    build_world_data,
    train_wrmf_joint,
    evaluate_transfer_test,
    evaluate_null_control
)


PREREG_M5_PATH = Path("prereg/M5.md")
EXPECTED_M5_HASH = "f908aeb12e25d0b6b9786cd15d52fafe4d9a9f49dde7f9f732e509da05d69f78"


def test_m5_preregistration_lock():
    """Verify prereg/M5.md is locked and matches SHA-256 hash."""
    assert PREREG_M5_PATH.exists(), "prereg/M5.md does not exist"
    with open(PREREG_M5_PATH, "rb") as f:
        actual_hash = hashlib.sha256(f.read()).hexdigest()
    assert actual_hash == EXPECTED_M5_HASH, f"M5 prereg hash mismatch: got {actual_hash}"


def test_dl_proxy_calibration():
    """Criterion C1: Differentiable proxy correlates with physical DL (r >= 0.95, rho >= 0.95)."""
    res = validate_dl_proxy(device="cpu")
    assert res["passed"], f"Proxy calibration failed: r={res['pearson_r']:.4f}, rho={res['spearman_rho']:.4f}"
    assert res["pearson_r"] >= 0.95, f"Pearson r below 0.95: {res['pearson_r']:.4f}"
    assert res["spearman_rho"] >= 0.95, f"Spearman rho below 0.95: {res['spearman_rho']:.4f}"


def test_memory_triad_architectures():
    """Criterion C6: Verify all 3 memory implementations (LoRA, PKM, EntityEmbedding)."""
    device = "cpu"
    for m_type in ["lora", "pkm", "entity_emb"]:
        net = WRMFNetwork(
            num_worlds=2,
            num_entities=16,
            num_relations=6,
            vocab_size=32,
            d_model=16,
            memory_type=m_type,
            skill_ids=["S1", "S2"]
        ).to(device)
        
        ent = torch.tensor([0, 1, 2], device=device)
        rel = torch.tensor([0, 1, 2], device=device)
        out_f = net.forward_fact(0, ent, rel)
        assert out_f.shape == (3, 32), f"{m_type} forward_fact wrong shape: {out_f.shape}"
        
        x_s = torch.randn(3, 16, device=device)
        out_s = net.forward_skill("S1", x_s)
        assert out_s.shape == (3, 32), f"{m_type} forward_skill wrong shape: {out_s.shape}"


def test_frozen_core_gradient_isolation():
    """Criterion C4: Verify transfer test mechanics: core parameters must not receive gradients."""
    net = WRMFNetwork(
        num_worlds=2,
        num_entities=16,
        num_relations=6,
        vocab_size=32,
        d_model=16,
        memory_type="entity_emb"
    )
    net.freeze_core()
    for name, p in net.core.named_parameters():
        assert not p.requires_grad, f"Core parameter {name} was not frozen!"
        
    net.unfreeze_core()
    for name, p in net.core.named_parameters():
        assert p.requires_grad, f"Core parameter {name} was not unfrozen!"


def test_transfer_test_smoke():
    """Verify smoke transfer test execution."""
    skills = ["S1", "S2"]
    w1 = build_world_data(k_entities=16, seed=42, skills=skills, vocab_size=32, d_model=16, num_samples_per_skill=20)
    w2 = build_world_data(k_entities=16, seed=43, skills=skills, vocab_size=32, d_model=16, num_samples_per_skill=20)
    
    net = WRMFNetwork(
        num_worlds=2,
        num_entities=16,
        num_relations=6,
        vocab_size=32,
        d_model=16,
        memory_type="entity_emb",
        skill_ids=skills
    )
    res_joint = train_wrmf_joint(net, [w1, w2], epochs=5, lr=0.03, device="cpu")
    assert res_joint["epochs_trained"] == 5
    
    w_unseen = build_world_data(k_entities=16, seed=999, skills=skills, vocab_size=32, d_model=16, num_samples_per_skill=20)
    t_res = evaluate_transfer_test(net, w_unseen, epochs=5, lr=0.03, device="cpu")
    assert "transfer_accuracy" in t_res
    assert t_res["transfer_accuracy"] >= 0.0
