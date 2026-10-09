"""
Hardware and MPS Determinism / Noise-Floor Evaluation for Project LEAN.

Runs identical forward-pass and loss evaluations 3x on CPU and 3x on MPS
to establish the empirical hardware noise floor sigma_0.
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import torch
import torch.nn as nn

from src.lean.measure.tracker import set_deterministic_seed, get_hardware_info


class BenchmarkTransformer(nn.Module):
    """Minimal deterministic transformer for hardware noise-floor calibration."""
    def __init__(self, vocab_size: int = 1000, d_model: int = 128, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=256, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        h = self.embedding(x)
        out = self.transformer(h)
        logits = self.head(out)
        return logits


def evaluate_device_noise_floor(device_str: str, num_repeats: int = 3, seed: int = 1337) -> Dict[str, Any]:
    """Execute identical computation num_repeats times and calculate variance."""
    device = torch.device(device_str)
    
    # Static input batch
    batch_size = 4
    seq_len = 64
    vocab_size = 1000
    
    losses: List[float] = []
    latencies: List[float] = []
    all_logits: List[torch.Tensor] = []
    
    for i in range(num_repeats):
        set_deterministic_seed(seed)
        model = BenchmarkTransformer(vocab_size=vocab_size).to(device)
        model.eval()
        
        # Fixed deterministic input
        x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        target = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        criterion = nn.CrossEntropyLoss()
        
        t0 = time.perf_counter()
        with torch.no_grad():
            logits = model(x)
            loss = criterion(logits.view(-1, vocab_size), target.view(-1))
            if device.type == "mps":
                torch.mps.synchronize()
        t1 = time.perf_counter()
        
        latencies.append((t1 - t0) * 1000.0)
        losses.append(loss.item())
        all_logits.append(logits.detach().cpu())
        
    # Compute maximum pairwise logit difference across runs
    max_logit_diff = 0.0
    for i in range(num_repeats):
        for j in range(i + 1, num_repeats):
            diff = torch.max(torch.abs(all_logits[i] - all_logits[j])).item()
            if diff > max_logit_diff:
                max_logit_diff = diff
                
    loss_arr = np.array(losses)
    sigma_0 = float(np.std(loss_arr))
    
    return {
        "device": device_str,
        "repeats": num_repeats,
        "seed": seed,
        "losses": losses,
        "loss_mean": float(np.mean(loss_arr)),
        "loss_std_sigma0": sigma_0,
        "max_pairwise_logit_diff": max_logit_diff,
        "latency_ms_mean": float(np.mean(latencies)),
        "latency_ms_std": float(np.std(latencies)),
        "deterministic": max_logit_diff < 1e-5
    }


def run_full_noise_floor_study() -> Dict[str, Any]:
    """Evaluate CPU and MPS noise floors."""
    results = {
        "hardware": get_hardware_info(),
        "cpu": evaluate_device_noise_floor("cpu"),
    }
    
    if torch.backends.mps.is_available():
        results["mps"] = evaluate_device_noise_floor("mps")
    else:
        results["mps"] = {"available": False}
        
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "noise_floor.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    return results


if __name__ == "__main__":
    res = run_full_noise_floor_study()
    print("\n--- HARDWARE NOISE FLOOR REPORT ---")
    print(f"CPU Loss sigma_0: {res['cpu']['loss_std_sigma0']:.8f} (Max logit diff: {res['cpu']['max_pairwise_logit_diff']:.2e})")
    if res.get("mps", {}).get("available") is not False:
        print(f"MPS Loss sigma_0: {res['mps']['loss_std_sigma0']:.8f} (Max logit diff: {res['mps']['max_pairwise_logit_diff']:.2e})")
    print("Results saved to results/noise_floor.json\n")
