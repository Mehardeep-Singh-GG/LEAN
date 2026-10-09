"""
Unit tests for the Cryptographic Vault Guard (A3 / Security Invariant).
"""

import os
import pytest
from pathlib import Path
import tempfile
import json

from vault.guard import (
    check_split_permission,
    VaultLockViolation,
    VaultIntegrityError,
    register_vault_file,
    deregister_vault_file,
    open_vault_file,
    compute_sha256,
    VAULT_DIR
)


def test_open_splits_accessible_without_token():
    """Verify that calibration and dev splits do not require tokens."""
    for split in ["calibration", "train", "dev", "val"]:
        # Should not raise exception
        check_split_permission(split)


def test_locked_splits_blocked_without_token(monkeypatch):
    """Verify that held-out test splits raise VaultLockViolation without token."""
    monkeypatch.delenv("LEAN_UNLOCK", raising=False)
    for split in ["test", "t1", "t2", "t3", "locked"]:
        with pytest.raises(VaultLockViolation):
            check_split_permission(split)


def test_locked_splits_blocked_with_invalid_token(monkeypatch):
    """Verify that arbitrary or invalid tokens are rejected."""
    monkeypatch.setenv("LEAN_UNLOCK", "invalid_token_12345")
    with pytest.raises(VaultLockViolation):
        check_split_permission("t1")


def test_vault_tamper_detection(tmp_path, monkeypatch):
    """Verify that modifying a vault file triggers VaultIntegrityError."""
    # Create test file inside vault raw directory
    test_file = VAULT_DIR / "raw" / "test_sample.txt"
    test_file.parent.mkdir(parents=True, exist_ok=True)
    rel_path = "raw/test_sample.txt"
    
    try:
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("ORIGINAL_CONTENT_AAA")
            
        register_vault_file(rel_path, split="dev")
        
        # Reading intact file should succeed
        with open_vault_file(rel_path) as f:
            assert f.read() == "ORIGINAL_CONTENT_AAA"
            
        # Tamper with file
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("TAMPERED_CONTENT_BBB")
            
        with pytest.raises(VaultIntegrityError):
            open_vault_file(rel_path)
    finally:
        deregister_vault_file(rel_path)
        test_file.unlink(missing_ok=True)
