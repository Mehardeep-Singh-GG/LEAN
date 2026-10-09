"""
Unit and integration tests for Milestone M10B: Multi-Model Feasible-Scale Validation.
"""

import os
import json
import pytest
from pathlib import Path

from vault.guard import compute_sha256
from experiments.run_m10b_feasible import (
    PREREG_M10B_PATH,
    EXPECTED_M10B_HASH,
    PARENT_LOCK_HASH,
    FEASIBLE_MODELS,
    verify_disk_safeguard,
    evaluate_model_quanta_ladder,
    run_milestone_m10b
)


def test_m10b_preregistration_hash():
    """Verify prereg/M10B.md exists and matches locked cryptographic SHA-256 hash."""
    assert PREREG_M10B_PATH.exists(), f"{PREREG_M10B_PATH} missing!"
    actual_hash = compute_sha256(PREREG_M10B_PATH)
    assert actual_hash == EXPECTED_M10B_HASH, f"Hash mismatch: expected {EXPECTED_M10B_HASH}, got {actual_hash}"


def test_m10b_feasible_models_scope():
    """Verify all 5 feasible models satisfy parameter limit <= 500M and diverse families."""
    assert len(FEASIBLE_MODELS) >= 5
    families = {m["family"] for m in FEASIBLE_MODELS}
    assert len(families) >= 4, "Must cover at least 4 distinct architectural families"
    for m in FEASIBLE_MODELS:
        assert m["params"] <= 500_000_000, f"Model {m['model_id']} exceeds 500M parameter limit!"


def test_m10b_disk_safeguard():
    """Verify disk safeguard detects available space and enforces safety floor."""
    free_gb = verify_disk_safeguard(min_free_gb=1.0)
    assert free_gb >= 1.0, f"Free disk space {free_gb} GB is below test threshold"


def test_m10b_quanta_ladder_invariants():
    """Verify single model ladder satisfies physical bits monotonicity and half-life asymmetry."""
    test_meta = FEASIBLE_MODELS[0]
    res = evaluate_model_quanta_ladder(test_meta, seed=42)
    ladder = res["ladder"]
    assert len(ladder) == 5
    
    # Check bit budgets
    bits = [r["bits"] for r in ladder]
    assert bits == [16, 8, 4, 3, 2]
    
    # Check physical DL monotonicity
    dl_bits = [r["physical_dl_bits"] for r in ladder]
    assert dl_bits == sorted(dl_bits, reverse=True)
    
    # Check asymmetric half-life ratio >= 1.0x on uncorrected single seed fallback
    assert res["half_life_ratio"] >= 1.0
