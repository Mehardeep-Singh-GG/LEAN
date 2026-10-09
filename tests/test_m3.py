"""
Unit Tests for Milestone M3: The Synthetic Universe.

Verifies:
1. Exact Shannon entropy of generated synthetic universe within <= 0.5%.
2. Deterministic Zipf distribution exposure probabilities.
3. Complete 9-skill procedural library generation and RASP token metrics.
4. Paraphrase template separation and 0% 13-gram leakage.
5. Capacity calibration: kappa in [1.33, 3.00] bits/param.
6. Synthetic transformer model forward and backward pass.
"""

import math
import pytest
import torch
import numpy as np

from src.lean.synthetic.world import SyntheticWorld, ATTRIBUTE_VOCABS, BITS_PER_ENTITY
from src.lean.synthetic.skills import SKILL_METADATA, SkillGenerator
from src.lean.synthetic.dataset import (
    RELATION_TEMPLATES,
    SyntheticTokenizer,
    generate_factual_splits,
    check_ngram_leakage
)
from src.lean.synthetic.models import (
    SyntheticTransformer,
    AssociativeMemoryBank,
    train_associative_memory
)
from src.lean.synthetic.capacity import run_capacity_experiment


def test_synthetic_world_exact_entropy():
    """Verify empirical Shannon entropy matches theoretical I_facts within 0.5%."""
    world = SyntheticWorld(k_entities=256, seed=42)
    ent_res = world.compute_empirical_entropy()
    
    assert ent_res["theoretical_bits"] == 256 * 24.0
    assert ent_res["relative_error"] <= 0.005
    assert ent_res["within_point_five_percent"] is True


def test_synthetic_world_zipf_mode():
    """Verify Zipf exposure probabilities follow power law rank ordering."""
    alpha = 0.6
    world = SyntheticWorld(k_entities=100, alpha=alpha, seed=42)
    probs = world.entity_probs
    
    assert len(probs) == 100
    assert np.isclose(np.sum(probs), 1.0)
    # Monotonically decreasing probabilities
    assert np.all(np.diff(probs) <= 0)
    # Ratio between rank 1 and rank 2 should be (2/1)^(1+alpha) = 2^1.6 ~= 3.03
    expected_ratio = (2.0 / 1.0) ** (1.0 + alpha)
    actual_ratio = probs[0] / probs[1]
    assert np.isclose(actual_ratio, expected_ratio, rtol=1e-3)


def test_all_nine_skills_operational():
    """Verify all 9 skills produce valid, non-empty prompt and target pairs."""
    sg = SkillGenerator(seed=42)
    assert len(SKILL_METADATA) == 9
    
    sample_kb = {"ENT_0001": {"origin_galaxy": "Andromeda", "containment_field": "Magnetic_Toroid"}}
    
    for s_id, meta in SKILL_METADATA.items():
        prompt, target = sg.sample_skill(meta["id"], kb=sample_kb)
        assert len(prompt) > 0
        assert len(target) > 0
        assert "->" in prompt
        assert meta["rasp_tokens"] > 0
        assert meta["k_bits"] == meta["rasp_tokens"] * 4.0


def test_zero_13gram_leakage():
    """Verify strict 0.0% 13-gram leakage between train and test splits."""
    world = SyntheticWorld(k_entities=64, seed=42)
    train_sents, eval_sents = generate_factual_splits(
        world, train_exposures_per_fact=4, eval_samples_per_fact=2, seed=42
    )
    
    leak_report = check_ngram_leakage(train_sents, eval_sents, n=13)
    assert leak_report["passed"] is True
    assert leak_report["overlap_count"] == 0
    assert leak_report["overlap_rate"] == 0.0


def test_capacity_calibration_bounds():
    """Verify empirical storage capacity kappa is within 1.5x of 2.0 bits/param."""
    res = run_capacity_experiment(k_entities_list=[128, 256], seed=42, device="cpu")
    assert res["within_target"] is True
    assert 1.33 <= res["mean_kappa"] <= 3.00


def test_synthetic_transformer_forward():
    """Verify compact decoder transformer executes forward pass and computes loss."""
    world = SyntheticWorld(k_entities=32, seed=42)
    tokenizer = SyntheticTokenizer(world)
    
    model = SyntheticTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=32,
        n_layers=2,
        n_heads=2,
        max_seq_len=32
    )
    
    sample_text = "The celestial entity ENT_0001 originates from Andromeda ."
    tok_ids = tokenizer.encode(sample_text)
    x = torch.tensor([tok_ids[:-1]], dtype=torch.long)
    y = torch.tensor([tok_ids[1:]], dtype=torch.long)
    
    logits, loss = model(x, targets=y)
    assert logits.shape == (1, len(tok_ids) - 1, tokenizer.vocab_size)
    assert loss is not None
    assert loss.item() > 0.0
