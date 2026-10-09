"""
Multi-Model Matched-Budget Real-Artifact Benchmark.

Executes controlled, storage-matched benchmarks across multiple model families
and random seeds, logging comprehensive statistical comparisons:
- KAC (Attention-prioritized) vs Matched-Uniform Baseline (0.000% storage delta)
- Full FP16 reference and Uniform 3-bit lower bound
- Retrieval augmentation accounting with physical BM25 serialized index footprint
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any

from experiments.run_matched_budget_benchmark import run_matched_budget_experiment

REPO_ROOT = Path("/Users/mac/llm")

MODELS = [
    "HuggingFaceTB/SmolLM2-360M",
    "EleutherAI/pythia-160m",
]

SEEDS = [1042, 2042]


def run_multimodel_benchmark():
    all_results = []
    
    print("=" * 80)
    print("STARTING PROJECT LEAN MULTI-MODEL MATCHED-BUDGET BENCHMARK")
    print(f"Models: {MODELS}")
    print(f"Seeds: {SEEDS}")
    print("=" * 80)
    
    for model_id in MODELS:
        for seed in SEEDS:
            print(f"\n>>> Running Benchmark: Model={model_id}, Seed={seed} <<<")
            res = run_matched_budget_experiment(model_id=model_id, seed=seed, device="cpu")
            all_results.append(res)
            
    summary_path = REPO_ROOT / "results" / "matched_budget_multimodel_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
        
    print(f"\nAll benchmark runs completed! Summary serialized to: {summary_path}")
    return all_results


if __name__ == "__main__":
    run_multimodel_benchmark()
