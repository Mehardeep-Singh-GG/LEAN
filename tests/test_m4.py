"""
Unit Tests for Milestone M4: Composition Law and Gate alpha.

Verifies:
1. Preregistration document hash integrity (Audit A8).
2. Parameter recovery and bootstrap CIs for LEAN Composition Law.
3. Competing model selection: LOOCV R^2, AIC, BIC computations.
4. Additivity gap invariance (Delta_comp <= 5%).
5. Bounded thinking core saturation exponent (gamma <= 0.30).
"""

import math
import hashlib
from pathlib import Path
import pytest
import numpy as np

from src.lean.theory.composition import (
    fit_composition_law,
    compare_competing_models,
    evaluate_additivity_gap,
    evaluate_skill_saturation
)
from experiments.run_m4_composition import verify_preregistration, EXPECTED_PREREG_HASH


def test_m4_preregistration_hash():
    """Verify prereg/M4.md exists and matches locked SHA-256 hash."""
    passed, actual_hash = verify_preregistration()
    assert passed is True
    assert actual_hash == EXPECTED_PREREG_HASH


def test_additivity_gap_evaluator():
    """Verify additivity gap calculation and threshold enforcement."""
    # Scenario 1: Nearly exact additivity
    res_clean = evaluate_additivity_gap(
        dl_composite=10000.0,
        dl_facts_only=6000.0,
        dl_skills_only=5000.0,
        delta_baseline=1000.0  # Expected sum = 6000 + 5000 - 1000 = 10000
    )
    assert res_clean["passed"] is True
    assert res_clean["gap_percent"] == 0.0

    # Scenario 2: Within 5% margin (3% gap)
    res_margin = evaluate_additivity_gap(
        dl_composite=10000.0,
        dl_facts_only=6100.0,
        dl_skills_only=5200.0,
        delta_baseline=1000.0  # Expected = 10300 => gap = 300 / 10000 = 3%
    )
    assert res_margin["passed"] is True
    assert np.isclose(res_margin["gap_percent"], 3.0)

    # Scenario 3: Fails > 5% margin (10% gap)
    res_fail = evaluate_additivity_gap(
        dl_composite=10000.0,
        dl_facts_only=7000.0,
        dl_skills_only=5000.0,
        delta_baseline=1000.0  # Expected = 11000 => gap = 1000 / 10000 = 10%
    )
    assert res_fail["passed"] is False
    assert np.isclose(res_fail["gap_percent"], 10.0)


def test_fit_composition_law_recovery():
    """Verify composition solver accurately recovers known ground-truth constants."""
    c_alg_true = 1.50
    inv_kappa_true = 0.50
    delta_true = 2000.0
    
    synthetic_data = []
    rng = np.random.RandomState(42)
    for K in [0, 200, 500, 1000]:
        for I in [0, 2000, 5000, 10000]:
            noise = rng.normal(0, 1.0)
            dl = c_alg_true * K + inv_kappa_true * I + delta_true + noise
            synthetic_data.append({
                "k_skill_bits": float(K),
                "i_fact_bits": float(I),
                "dl_bits": float(dl)
            })
            
    fit_res = fit_composition_law(synthetic_data, n_resamples=100)
    assert np.isclose(fit_res["c_alg"], c_alg_true, atol=0.05)
    assert np.isclose(fit_res["kappa"], 1.0 / inv_kappa_true, atol=0.05)
    assert np.isclose(fit_res["delta"], delta_true, atol=20.0)
    assert fit_res["r_squared"] >= 0.999
    assert fit_res["c_alg_relative_ci_width"] <= 0.30


def test_compare_competing_models():
    """Verify model comparison correctly ranks LEAN over naive, multiplicative, and no-skill models."""
    synthetic_data = []
    for K in [100, 500, 1000]:
        for I in [1000, 5000, 10000]:
            dl = 1.4 * K + 0.5 * I + 1500.0
            synthetic_data.append({
                "k_skill_bits": float(K),
                "i_fact_bits": float(I),
                "n_params": int(1500 + K + I),
                "dl_bits": float(dl)
            })
            
    comp_res = compare_competing_models(synthetic_data)
    assert comp_res["lean_beats_all"] is True
    assert comp_res["model_1_lean"]["r2_loocv"] >= 0.99
    assert comp_res["model_2_naive"]["delta_aic"] > 0
    assert comp_res["model_3_multiplicative"]["delta_aic"] > 0
    assert comp_res["model_4_pure_facts"]["delta_aic"] > 0


def test_skill_saturation_bounded_core():
    """Verify saturation solver calculates gamma and enforces gamma <= 0.30."""
    # Sub-linear saturation scale points: DL_skill grows like log(N)
    scale_points = [
        (2500, 5000.0),
        (5000, 5020.0),
        (10000, 5045.0),
        (25000, 5080.0),
        (50000, 5110.0)
    ]
    sat_res = evaluate_skill_saturation(scale_points, n_resamples=100)
    assert sat_res["saturates"] is True
    assert sat_res["gamma_95_ci"][1] <= 0.30
    assert sat_res["gamma_point"] < 0.10
