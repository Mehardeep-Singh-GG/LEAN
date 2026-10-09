"""
Unit and integration tests for Milestone M9: Assemble and LOCK the Equation v1.
"""

import json
import hashlib
from pathlib import Path
import pytest
import numpy as np

from vault.guard import compute_sha256, verify_unlock_token
from experiments.run_m9_lock import (
    PREREG_LOCK_PATH,
    EXPECTED_LOCK_HASH,
    evaluate_dev_model_comparison,
    compile_blind_predictions_t1,
    run_milestone_m9
)


def test_m9_lock_preregistration_hash():
    """Verify prereg/LOCK_v1.md exists and matches locked cryptographic SHA-256 hash."""
    assert PREREG_LOCK_PATH.exists(), f"{PREREG_LOCK_PATH} missing!"
    actual_hash = compute_sha256(PREREG_LOCK_PATH)
    assert actual_hash == EXPECTED_LOCK_HASH, f"Hash mismatch: expected {EXPECTED_LOCK_HASH}, got {actual_hash}"
    assert verify_unlock_token(actual_hash), "Vault guard does not accept LOCK_v1 hash!"


def test_m9_frozen_constants_validity():
    """Verify core physical constants are strictly frozen within empirical CIs."""
    results_path = Path("results/m9_lock_results.json")
    assert results_path.exists(), "results/m9_lock_results.json missing!"
    with open(results_path) as f:
        data = json.load(f)
    constants = data["frozen_constants"]
    
    # c_alg: circuit cost in bits/tok
    assert 24.8 <= constants["c_alg"]["value"] <= 32.1
    # kappa: factual capacity in bits/param
    assert 1.42 <= constants["kappa"]["value"] <= 1.90
    # T: transition temperature
    assert 0.85 <= constants["T"]["value"] <= 1.25
    # alpha_pile: pretraining Zipf exponent
    assert 0.36 <= constants["alpha_pile"]["value"] <= 0.46


def test_m9_model_comparison_framework():
    """Verify Model A (LEAN) beats competing models B, C, D on out-of-sample dev benchmark."""
    res = evaluate_dev_model_comparison()
    assert res["lean_wins"] is True
    assert res["loocv_r2"] >= 0.95
    models = res["models"]
    
    # LEAN has lower MAPE than Chinchilla
    assert models["Model_A_LEAN"]["mape"] < models["Model_B_Chinchilla"]["mape"]
    # Delta AIC >= 10.0 against all alternatives
    assert models["Model_B_Chinchilla"]["delta_aic"] >= 10.0
    assert models["Model_C_Precision"]["delta_aic"] >= 10.0
    assert models["Model_D_FrequencyOnly"]["delta_aic"] >= 10.0


def test_m9_blind_predictions_structure():
    """Verify all 6 blind predictions have complete 95% PIs and falsification conditions."""
    preds = compile_blind_predictions_t1()
    assert len(preds) == 6
    pred_ids = {p["id"] for p in preds}
    assert pred_ids == {"P_LOCK_1", "P_LOCK_2", "P_LOCK_3", "P_LOCK_4", "P_LOCK_5", "P_LOCK_6"}
    
    for p in preds:
        assert p["pi_low"] < p["point_pred"] < p["pi_high"]
        assert len(p["falsification_condition"]) > 0
        assert len(p["unit"]) > 0


def test_m9_red_team_checklist():
    """Verify red-team audit checklist contains at least 25 questions with empirical answers."""
    results_path = Path("results/m9_lock_results.json")
    with open(results_path) as f:
        data = json.load(f)
    rt = data["red_team_audit"]
    assert rt["questions_evaluated"] >= 25
    assert rt["passed"] is True


def test_m9_execution_verdict():
    """Verify end-to-end execution of run_milestone_m9 passes all criteria."""
    res = run_milestone_m9()
    assert res["verdict"] == "PASS"
    assert res["all_criteria_passed"] is True
    assert (Path(res["provenance_path"]) / "hardware.json").exists()
    assert (Path(res["provenance_path"]) / "config.yaml").exists()
    assert (Path(res["provenance_path"]) / "predictions.parquet").exists()
