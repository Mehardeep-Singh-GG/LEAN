"""
Unit tests for the Failure Ledger and Milestone Blocker Checks.
"""

import pytest
from pathlib import Path
import json

from src.lean.audit.failure_ledger import (
    log_failure,
    update_failure_status,
    check_blocking_failures,
    load_failures,
    save_failures,
    LEDGER_PATH
)


@pytest.fixture(autouse=True)
def clean_ledger():
    """Ensure a pristine ledger state for testing."""
    original_data = []
    if LEDGER_PATH.exists():
        with open(LEDGER_PATH, "r", encoding="utf-8") as f:
            original_data = [json.loads(l) for l in f if l.strip()]
    save_failures([])
    yield
    save_failures(original_data)


def test_log_failure_and_blocking_check():
    """Verify that logging S0/S1 failure blocks milestone completion."""
    assert len(check_blocking_failures("M0")) == 0
    
    log_failure(
        failure_id="F-0001",
        milestone="M0",
        category="F-BUG",
        severity="S1",
        symptom="Sample test symptom",
        detected_by="A4-invariants",
        root_cause="Test root cause",
        five_whys=["Why 1", "Why 2", "Why 3", "Why 4", "Why 5"],
        fix="Fixed bug in code",
        verification="Verified with seed 42",
        regression_test="tests/test_dummy.py",
        status="OPEN"
    )
    
    blockers = check_blocking_failures("M0")
    assert len(blockers) == 1
    assert blockers[0]["id"] == "F-0001"
    
    # Close failure
    update_failure_status("F-0001", status="CLOSED", fix_notes="Verified clean fix")
    assert len(check_blocking_failures("M0")) == 0


def test_invalid_five_whys_rejected():
    """Verify that failure records without exactly 5 whys are rejected."""
    with pytest.raises(ValueError, match="five_whys must contain exactly 5 entries"):
        log_failure(
            failure_id="F-0002",
            milestone="M0",
            category="F-BUG",
            severity="S2",
            symptom="Sample symptom",
            detected_by="Manual",
            root_cause="Root cause",
            five_whys=["Why 1", "Why 2"],  # invalid length
            fix="fix",
            verification="veri",
            regression_test="test",
            status="OPEN"
        )
