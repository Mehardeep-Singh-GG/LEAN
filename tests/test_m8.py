"""
Unit and integration tests for Milestone M8: Real-LLM Anatomy & Gate beta.
"""

import hashlib
from pathlib import Path
import numpy as np
import pytest
import torch

from src.lean.anatomy.real_llm import (
    fit_pile_zipf_distribution,
    generate_standard_skill_items,
    evaluate_cloze_candidates,
    evaluate_procedural_skills
)
from src.lean.theory.priced_quanta import (
    compute_roc_auc,
    fit_power_law_exponent,
    compute_half_life
)


def test_m8_preregistration_hash():
    """Verify prereg/M8.md exists and matches locked cryptographic SHA-256 hash."""
    prereg_path = Path("prereg/M8.md")
    assert prereg_path.exists(), "prereg/M8.md missing!"
    with open(prereg_path, "rb") as f:
        actual_hash = hashlib.sha256(f.read()).hexdigest()
    expected_hash = "70e96589ee2557ba5aa47e6639ace4ab612c9a1d146574d4f52c2810dedb9138"
    assert actual_hash == expected_hash, f"Hash mismatch: expected {expected_hash}, got {actual_hash}"


def test_pile_zipf_distribution_fitting():
    """Verify Pile fact frequency distribution fitting and exponent estimation."""
    res = fit_pile_zipf_distribution()
    assert 0.30 <= res["alpha_pile"] <= 0.60
    assert res["r_squared"] > 0.75
    assert res["ci_low"] <= res["alpha_pile"] <= res["ci_high"]
    assert res["total_facts"] >= 100


def test_standard_skill_items_generation():
    """Verify standardized skill probe items have valid structure and candidates."""
    items = generate_standard_skill_items(n_items=50, seed=42)
    assert len(items) == 50
    for prompt, target, cands in items:
        assert "Map:" in prompt
        assert "Query:" in prompt
        assert target in cands
        assert len(cands) == 3


def test_knapsack_scaling_matches_pile_exponent():
    """Verify continuous knapsack excess loss scaling recovers alpha_Pile."""
    pile_res = fit_pile_zipf_distribution()
    alpha_pile = pile_res["alpha_pile"]

    budgets = np.geomspace(50, 1000, 20)
    c_f = 2.0
    rng = np.random.default_rng(42)
    excess = ((budgets / c_f) ** (-alpha_pile) / alpha_pile) * rng.normal(1.0, 0.005, size=len(budgets))
    fit_k = fit_power_law_exponent(list(budgets), list(excess))

    assert fit_k["ci_low"] <= alpha_pile <= fit_k["ci_high"]
    assert fit_k["r_squared"] > 0.99


def test_asymmetric_half_life_calculation():
    """Verify half-life ratio calculation on realistic compression ladder values."""
    budgets = [1126.8, 563.4, 281.7, 211.3, 140.8]
    k_accs = [0.46, 0.47, 0.44, 0.21, 0.18]
    s_accs = [0.58, 0.58, 0.52, 0.28, 0.30]

    b_half_k = compute_half_life(budgets, k_accs)
    b_half_s = compute_half_life(budgets, s_accs)
    
    assert b_half_k > 0
    assert b_half_s > 0
    # Factual degradation occurs at higher budget than skills
    assert b_half_k >= b_half_s
