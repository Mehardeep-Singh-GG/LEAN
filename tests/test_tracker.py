"""
Unit tests for the Standardized Run Tracker.
"""

import json
import pytest
from pathlib import Path

from src.lean.measure.tracker import (
    RunContext,
    get_git_sha,
    get_hardware_info,
    set_deterministic_seed
)


def test_hardware_profiling():
    """Verify hardware profile contains essential runtime fields."""
    info = get_hardware_info()
    assert "platform" in info
    assert "python_version" in info
    assert "mps_available" in info
    assert "torch_version" in info


def test_run_tracker_lifecycle(tmp_path):
    """Verify RunContext generates complete provenance artifacts."""
    config = {"lr": 1e-4, "batch_size": 16, "model": "test_transformer"}
    seed = 42
    
    with RunContext(experiment_name="unit_test_run", config=config, seed=seed) as run:
        run.log("Unit test message")
        run.log_metrics(step=1, metrics={"loss": 2.5, "accuracy": 0.8})
        run.save_per_item_results([{"item_id": 1, "correct": True}])
        run_dir = run.run_dir
        
    assert run_dir.exists()
    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "hardware.json").exists()
    assert (run_dir / "log.txt").exists()
    assert (run_dir / "metrics.jsonl").exists()
    assert (run_dir / "per_item.jsonl").exists()
    
    with open(run_dir / "metrics.jsonl", "r", encoding="utf-8") as f:
        metrics = json.loads(f.readline())
        assert metrics["step"] == 1
        assert metrics["loss"] == 2.5
        assert metrics["accuracy"] == 0.8
        
    # Clean up test run directory
    import shutil
    shutil.rmtree(run_dir, ignore_errors=True)
