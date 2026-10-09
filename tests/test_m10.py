"""
Unit and integration tests for Milestone M10: Held-Out Validation Tranche T1.
"""

import os
import json
import hashlib
from pathlib import Path
import pytest
import numpy as np

from vault.guard import compute_sha256, open_vault_file, VaultLockViolation
from experiments.run_m10_validation import (
    PREREG_M10_PATH,
    EXPECTED_M10_HASH,
    LOCK_V1_HASH,
    run_milestone_m10
)


def test_m10_preregistration_hash():
    """Verify prereg/M10.md exists and matches locked cryptographic SHA-256 hash."""
    assert PREREG_M10_PATH.exists(), f"{PREREG_M10_PATH} missing!"
    actual_hash = compute_sha256(PREREG_M10_PATH)
    assert actual_hash == EXPECTED_M10_HASH, f"Hash mismatch: expected {EXPECTED_M10_HASH}, got {actual_hash}"


def test_m10_vault_unseal_with_token():
    """Verify vault unsealing requires valid LOCK_v1 token and denies access without it."""
    # Without token
    saved_token = os.environ.get("LEAN_UNLOCK")
    if "LEAN_UNLOCK" in os.environ:
        del os.environ["LEAN_UNLOCK"]
    with pytest.raises(VaultLockViolation):
        open_vault_file("locked/t1_test.txt")
        
    # With valid token
    os.environ["LEAN_UNLOCK"] = LOCK_V1_HASH
    with open_vault_file("locked/t1_test.txt") as f:
        header = f.readline()
        assert len(header) > 0
    if saved_token:
        os.environ["LEAN_UNLOCK"] = saved_token


def test_m10_prediction_scoring():
    """Verify results/m10_validation_results.json exists and scores all 6 blind predictions."""
    results_path = Path("results/m10_validation_results.json")
    assert results_path.exists(), "results/m10_validation_results.json missing!"
    with open(results_path) as f:
        data = json.load(f)
    scores = data["prediction_scores"]
    assert len(scores) == 6
    for pid in ["P_LOCK_1", "P_LOCK_2", "P_LOCK_3", "P_LOCK_4", "P_LOCK_5", "P_LOCK_6"]:
        assert pid in scores
        assert "actual" in scores[pid]
        assert "within_pi" in scores[pid]


def test_m10_pi_coverage():
    """Verify PI coverage is >= 83.3% (at least 5 of 6 predictions within 95% PI)."""
    results_path = Path("results/m10_validation_results.json")
    with open(results_path) as f:
        data = json.load(f)
    summary = data["summary"]
    assert summary["inside_pi_count"] >= 5
    assert summary["pi_coverage"] >= 5.0 / 6.0
    assert summary["coverage_passed"] is True


def test_m10_half_life_asymmetry():
    """Verify Asymmetric Half-Life Law holds on held-out tranche: Ratio >= 2.0x."""
    results_path = Path("results/m10_validation_results.json")
    with open(results_path) as f:
        data = json.load(f)
    ratio = data["prediction_scores"]["P_LOCK_3"]["actual"]
    assert ratio >= 2.0
    assert 2.5 <= ratio <= 5.7


def test_m10_fact_survival_roc_auc():
    """Verify factual survival monotonicity ROC-AUC >= 0.680 on held-out tranche."""
    results_path = Path("results/m10_validation_results.json")
    with open(results_path) as f:
        data = json.load(f)
    auc = data["prediction_scores"]["P_LOCK_2"]["actual"]
    assert 0.680 <= auc <= 0.760


def test_m10_provenance_files():
    """Verify all run provenance artifacts are present on disk."""
    results_path = Path("results/m10_validation_results.json")
    with open(results_path) as f:
        data = json.load(f)
    prov_dir = Path(data["provenance_path"])
    assert (prov_dir / "hardware.json").exists()
    assert (prov_dir / "config.yaml").exists()
    assert (prov_dir / "metrics.jsonl").exists()
    assert (prov_dir / "predictions.parquet").exists()
