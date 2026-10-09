"""
Standardized Experiment Run Tracker for Project LEAN.

Captures for every execution:
- config.yaml (frozen execution parameters)
- git_sha (current repository commit hash)
- hardware.json (system specifications and available accelerators)
- seed (reproducibility RNG seed)
- log.txt (execution logs)
- metrics.jsonl (scalar metrics logged during and after run)
- per_item (fine-grained sample-level outputs for paired bootstrap tests)

Run ID is deterministic: hash(config + git_sha + seed).
"""

import os
import sys
import json
import yaml
import hashlib
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional

import torch

RUNS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "results" / "runs"


def get_git_sha() -> str:
    """Return the current git commit SHA."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        return sha
    except Exception:
        return "UNKNOWN_DIRTY"


def get_hardware_info() -> Dict[str, Any]:
    """Capture hardware profiling metadata."""
    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "mps_available": torch.backends.mps.is_available(),
        "mps_built": torch.backends.mps.is_built(),
        "cuda_available": torch.cuda.is_available(),
    }
    
    # Apple Silicon specific sysctl query if on Darwin
    if platform.system() == "Darwin":
        try:
            chip = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], stderr=subprocess.DEVNULL
            ).decode("utf-8").strip()
            mem = subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"], stderr=subprocess.DEVNULL
            ).decode("utf-8").strip()
            info["apple_silicon_chip"] = chip
            info["ram_bytes"] = int(mem)
            info["ram_gb"] = round(int(mem) / (1024**3), 2)
        except Exception:
            pass
            
    return info


def set_deterministic_seed(seed: int) -> None:
    """Set seeds across Python, NumPy, and PyTorch (CPU and MPS)."""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class RunContext:
    """Context manager and tracker for an experimental run."""
    
    def __init__(self, experiment_name: str, config: Dict[str, Any], seed: int = 42):
        self.experiment_name = experiment_name
        self.config = config
        self.seed = seed
        self.git_sha = get_git_sha()
        
        # Calculate deterministic run ID
        config_str = json.dumps(config, sort_keys=True)
        raw_sig = f"{config_str}_{self.git_sha}_{seed}".encode("utf-8")
        self.run_hash = hashlib.sha256(raw_sig).hexdigest()[:12]
        self.run_id = f"{experiment_name}_{self.run_hash}_s{seed}"
        
        self.run_dir = RUNS_DIR / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics_file = self.run_dir / "metrics.jsonl"
        self.log_file = self.run_dir / "log.txt"
        self.start_time = None
        self.end_time = None
        
    def __enter__(self):
        self.start_time = datetime.now(timezone.utc)
        set_deterministic_seed(self.seed)
        
        # Write config.yaml
        with open(self.run_dir / "config.yaml", "w", encoding="utf-8") as f:
            yaml.dump(self.config, f, sort_keys=True)
            
        # Write hardware.json
        hardware = get_hardware_info()
        hardware["git_sha"] = self.git_sha
        hardware["seed"] = self.seed
        hardware["run_id"] = self.run_id
        hardware["start_time"] = self.start_time.isoformat()
        with open(self.run_dir / "hardware.json", "w", encoding="utf-8") as f:
            json.dump(hardware, f, indent=2)
            
        self.log(f"--- START RUN: {self.run_id} ---")
        self.log(f"Git SHA: {self.git_sha}")
        self.log(f"Seed: {self.seed}")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = datetime.now(timezone.utc)
        elapsed = (self.end_time - self.start_time).total_seconds()
        
        status = "COMPLETED" if exc_type is None else f"FAILED: {exc_val}"
        self.log(f"--- END RUN: {self.run_id} | Status: {status} | Elapsed: {elapsed:.2f}s ---")
        
        # Update hardware.json with finish timestamp and status
        hardware_path = self.run_dir / "hardware.json"
        if hardware_path.exists():
            with open(hardware_path, "r", encoding="utf-8") as f:
                hw = json.load(f)
            hw["end_time"] = self.end_time.isoformat()
            hw["elapsed_seconds"] = elapsed
            hw["status"] = status
            with open(hardware_path, "w", encoding="utf-8") as f:
                json.dump(hw, f, indent=2)
                
    def log(self, message: str) -> None:
        """Write timestamped message to log.txt and stdout."""
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        line = f"[{ts}] {message}"
        print(line)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            
    def log_metrics(self, step: int, metrics: Dict[str, Any]) -> None:
        """Append a record to metrics.jsonl."""
        record = {
            "step": step,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **metrics
        }
        with open(self.metrics_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
            
    def save_per_item_results(self, records: list) -> None:
        """Save fine-grained per-item results for paired statistical tests."""
        out_path = self.run_dir / "per_item.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
