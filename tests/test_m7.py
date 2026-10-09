"""
Unit and integration tests for Milestone M7: Priced-Quanta Theory & Synthetic Verification.
"""

import hashlib
from pathlib import Path
import numpy as np
import pytest
import torch

from src.lean.theory.priced_quanta import (
    PricedQuantaTheory,
    exact_01_knapsack,
    greedy_knapsack,
    evaluate_knapsack_gap,
    compute_roc_auc,
    fit_power_law_exponent,
    compute_half_life,
    solve_knapsack_soft_cutoff
)


def test_m7_preregistration_hash():
    """Verify prereg/M7.md exists and matches locked cryptographic SHA-256 hash."""
    prereg_path = Path("prereg/M7.md")
    assert prereg_path.exists(), "prereg/M7.md missing!"
    with open(prereg_path, "rb") as f:
        actual_hash = hashlib.sha256(f.read()).hexdigest()
    expected_hash = "0f61cd400a67f5be66d38a55e65501c8ac251cba3fef6eff6648a064494e1adc"
    assert actual_hash == expected_hash, f"Hash mismatch: expected {expected_hash}, got {actual_hash}"


def test_roc_auc_properties():
    """Verify exact Mann-Whitney U ROC-AUC implementation."""
    # Perfect ranking
    y_true = np.array([1, 1, 1, 0, 0, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
    assert compute_roc_auc(y_true, scores) == 1.0

    # Reversed ranking
    rev_scores = -scores
    assert compute_roc_auc(y_true, rev_scores) == 0.0

    # Random / uniform scores
    rng = np.random.default_rng(42)
    y_rand = rng.integers(0, 2, size=100)
    scores_rand = rng.random(size=100)
    auc_rand = compute_roc_auc(y_rand, scores_rand)
    assert 0.35 <= auc_rand <= 0.65


def test_knapsack_gap_bound():
    """Verify continuous greedy knapsack is within 1.0% of exact 0/1 IP."""
    np.random.seed(42)
    weights = [int(w) for w in np.random.randint(8, 48, size=200)]
    values = [float(v) for v in np.random.exponential(scale=12.0, size=200)]
    capacity = 1200

    res = evaluate_knapsack_gap(values, weights, capacity)
    assert res["v_exact"] >= res["v_greedy"]
    assert res["relative_gap"] <= 0.010, f"Gap {res['relative_gap']*100:.3f}% > 1.0%"
    assert res["within_one_percent"] is True


def test_power_law_exponent_fitting():
    """Verify bootstrap power law exponent fitting on known power laws."""
    budgets = np.geomspace(50, 1000, 20)
    for true_alpha in [0.3, 0.6, 1.0]:
        rng = np.random.default_rng(42)
        losses = (budgets ** (-true_alpha)) * rng.normal(1.0, 0.005, size=len(budgets))
        fit = fit_power_law_exponent(list(budgets), list(losses), n_bootstrap=1000)
        assert fit["ci_low"] <= true_alpha <= fit["ci_high"], f"alpha={true_alpha} not in [{fit['ci_low']}, {fit['ci_high']}]"
        assert fit["r_squared"] > 0.99


def test_half_life_computation():
    """Verify 50% crossing point interpolation."""
    budgets = [1000.0, 500.0, 250.0, 100.0]
    accuracies = [1.0, 0.8, 0.4, 0.1]
    # Crossing between 500 (0.8) and 250 (0.4)
    # frac = (0.5 - 0.4) / (0.8 - 0.4) = 0.25 -> 250 + 0.25 * 250 = 312.5
    h = compute_half_life(budgets, accuracies)
    assert 250.0 <= h <= 500.0
    assert abs(h - 312.5) < 1e-4


def test_priced_quanta_theory_class():
    """Verify formal theoretical predictions of PricedQuantaTheory class."""
    theory = PricedQuantaTheory(c_alg=28.45, kappa=2.0, temperature=1.0)
    assert theory.predict_scaling_exponent(0.6) == 0.6
    assert theory.predict_fact_survival_probability(p_fact=0.1, p_star=0.01) > 0.90
    assert theory.predict_fact_survival_probability(p_fact=0.001, p_star=0.01) < 0.10
    
    # Asymmetric half-life ratio: p_skill = 0.5, K(s) = 20, p_fact = 0.0001, I(f) = 12
    ratio = theory.predict_half_life_ratio(p_skill=0.5, k_skill_tokens=20, p_fact=0.0001, i_fact_bits=12)
    assert ratio > 2.0
