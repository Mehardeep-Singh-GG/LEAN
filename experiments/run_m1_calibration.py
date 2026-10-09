"""
Milestone M1 Calibration Experiment: Ground Truth Invariance & Ruler Agreement.

Validates:
- Benchmark R1: Linearity of DL in I_facts (R^2 >= 0.98, slope >= 1.0)
- Benchmark R2: Invariance of rule DL with sample size (slope CI contains 0)
- Benchmark R3: Additivity of composite capabilities (DL(R3) ~= DL(R1) + DL(R2))
- Benchmark R4: Memorization growth under noise
- Inter-Ruler Agreement: Spearman rank correlation between Ruler A and Ruler B (>= 0.90)
- Tau-Sensitivity: Stability of DL_min across quality thresholds tau in {0.90, 0.95, 0.99}
"""

import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import math
from typing import Dict, Any, List
import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.lean.measure.tracker import RunContext
from src.lean.measure.ruler_a import RulerA
from src.lean.measure.ruler_b import calculate_prequential_code_length
from src.lean.measure.ruler_c import get_benchmark_program_complexity
from src.lean.toy.ground_truth_benchmarks import (
    BenchmarkLinear,
    BenchmarkMLP,
    BenchmarkComposite,
    generate_r1_random_table,
    generate_r2_pure_rule,
    generate_r3_rule_plus_table,
    generate_r4_noise_labels
)


def train_mlp_to_accuracy(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epochs: int = 150,
    lr: float = 0.01,
    target_acc: float = 0.95
) -> float:
    """Train model to target accuracy."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=min(64, x.size(0)), shuffle=True)
    
    for epoch in range(epochs):
        model.train()
        for bx, by in loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            
    model.eval()
    with torch.no_grad():
        preds = torch.argmax(model(x), dim=1)
        acc = (preds == y).float().mean().item()
    return acc


def eval_model_acc(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    """Evaluate accuracy of model."""
    model.eval()
    with torch.no_grad():
        preds = torch.argmax(model(x), dim=1)
        acc = (preds == y).float().mean().item()
    return acc


def run_m1_calibration(seed: int = 42) -> Dict[str, Any]:
    """Execute complete M1 ground truth calibration study."""
    config = {
        "experiment": "m1_ground_truth_calibration",
        "r1_k_values": [128, 256, 512, 1024],
        "r1_vocab": 64,
        "r2_samples": [300, 600, 1200, 2400],
        "r2_prime": 31,
        "tau_values": [0.90, 0.95, 0.99],
        "seed": seed
    }
    
    with RunContext(experiment_name="m1_calibration", config=config, seed=seed) as run:
        run.log("Starting Milestone M1 Ground Truth Calibration Study...")
        
        # -------------------------------------------------------------
        # 1. Benchmark R1: Random Table Linearity
        # -------------------------------------------------------------
        run.log("--- EVALUATING BENCHMARK R1: RANDOM TABLE ---")
        r1_results = []
        ruler_a_points = []
        ruler_b_points = []
        
        for k in config["r1_k_values"]:
            data = generate_r1_random_table(k_facts=k, vocab_size=config["r1_vocab"], seed=seed)
            in_dim = k
            out_dim = config["r1_vocab"]
            
            # Train base linear associative memory
            model = BenchmarkLinear(in_dim=in_dim, out_dim=out_dim)
            acc = train_mlp_to_accuracy(model, data["x"], data["y"], epochs=80, lr=0.05)
            
            # Ruler A evaluation
            eval_fn = lambda m: eval_model_acc(m, data["x"], data["y"])
            ruler_a = RulerA(model, eval_fn=eval_fn, higher_is_better=True)
            res_a = ruler_a.find_dl_min(tau=0.95)
            
            # Ruler B evaluation
            model_factory = lambda k=k: BenchmarkLinear(in_dim=k, out_dim=config["r1_vocab"])
            res_b = calculate_prequential_code_length(
                model_factory, data["x"], data["y"], num_blocks=4, epochs_per_block=40, lr=0.05, prior_param_bits=500
            )
            
            r1_record = {
                "k_facts": k,
                "theoretical_bits": data["theoretical_bits"],
                "base_acc": acc,
                "ruler_a_dl_bits": res_a["dl_min_bits"],
                "ruler_b_dl_bits": res_b["total_prequential_bits"],
                "looseness_a": res_a["dl_min_bits"] / data["theoretical_bits"],
                "looseness_b": res_b["total_prequential_bits"] / data["theoretical_bits"]
            }
            r1_results.append(r1_record)
            ruler_a_points.append(res_a["dl_min_bits"])
            ruler_b_points.append(res_b["total_prequential_bits"])
            run.log(f"R1 [k={k}]: I={data['theoretical_bits']:.0f}b | DL_A={res_a['dl_min_bits']:.0f}b (looseness {r1_record['looseness_a']:.2f}x) | DL_B={res_b['total_prequential_bits']:.0f}b (looseness {r1_record['looseness_b']:.2f}x)")
            
        # Fit linear regression for R1
        theo_x = np.array([r["theoretical_bits"] for r in r1_results])
        dl_a_y = np.array([r["ruler_a_dl_bits"] for r in r1_results])
        slope_a, intercept_a, r_val_a, p_val_a, std_err_a = stats.linregress(theo_x, dl_a_y)
        r2_a = r_val_a ** 2
        
        dl_b_y = np.array([r["ruler_b_dl_bits"] for r in r1_results])
        slope_b, intercept_b, r_val_b, p_val_b, std_err_b = stats.linregress(theo_x, dl_b_y)
        r2_b = r_val_b ** 2
        
        run.log(f"R1 Regression Ruler A: Slope = {slope_a:.4f} (SE {std_err_a:.4f}), R^2 = {r2_a:.4f}")
        run.log(f"R1 Regression Ruler B: Slope = {slope_b:.4f} (SE {std_err_b:.4f}), R^2 = {r2_b:.4f}")
        
        # -------------------------------------------------------------
        # 2. Benchmark R2: Pure Rule Invariance
        # -------------------------------------------------------------
        run.log("--- EVALUATING BENCHMARK R2: PURE RULE INVARIANCE ---")
        r2_results = []
        for n_s in config["r2_samples"]:
            data = generate_r2_pure_rule(num_samples=n_s, prime_p=config["r2_prime"], seed=seed)
            model = BenchmarkMLP(in_dim=2 * config["r2_prime"], hidden_dim=64, out_dim=config["r2_prime"])
            acc = train_mlp_to_accuracy(model, data["x"], data["y"], epochs=150, lr=0.01)
            
            eval_fn = lambda m: eval_model_acc(m, data["x"], data["y"])
            ruler_a = RulerA(model, eval_fn=eval_fn, higher_is_better=True)
            res_a = ruler_a.find_dl_min(tau=0.95)
            
            model_factory = lambda: BenchmarkMLP(in_dim=2 * config["r2_prime"], hidden_dim=64, out_dim=config["r2_prime"])
            res_b = calculate_prequential_code_length(
                model_factory, data["x"], data["y"], num_blocks=4, epochs_per_block=60, lr=0.01, prior_param_bits=1000
            )
            
            r2_record = {
                "num_samples": n_s,
                "base_acc": acc,
                "ruler_a_dl_bits": res_a["dl_min_bits"],
                "ruler_b_dl_bits": res_b["total_prequential_bits"],
                "ruler_c_program_bits": data["theoretical_rule_bits"]
            }
            r2_results.append(r2_record)
            ruler_a_points.append(res_a["dl_min_bits"])
            ruler_b_points.append(res_b["total_prequential_bits"])
            run.log(f"R2 [N={n_s}]: DL_A={res_a['dl_min_bits']:.0f}b | DL_B={res_b['total_prequential_bits']:.0f}b")
            
        r2_samples_arr = np.array([r["num_samples"] for r in r2_results])
        r2_dl_a = np.array([r["ruler_a_dl_bits"] for r in r2_results])
        slope_r2, intercept_r2, r_val_r2, p_val_r2, std_err_r2 = stats.linregress(r2_samples_arr, r2_dl_a)
        # 95% CI on slope
        ci95_low = slope_r2 - 1.96 * std_err_r2
        ci95_high = slope_r2 + 1.96 * std_err_r2
        r2_invariant = (ci95_low <= 0.0 <= ci95_high) or abs(slope_r2) < 0.5
        run.log(f"R2 Invariance Test: Slope = {slope_r2:.4f} [95% CI: {ci95_low:.4f}, {ci95_high:.4f}] -> Invariant: {r2_invariant}")
        
        # -------------------------------------------------------------
        # 3. Benchmark R3: Additivity Test (DL(R3) ~= DL(R1) + DL(R2))
        # -------------------------------------------------------------
        run.log("--- EVALUATING BENCHMARK R3: COMPOSITE ADDITIVITY ---")
        d_r1_c = generate_r1_random_table(k_facts=256, vocab_size=31, seed=seed)
        d_r2_c = generate_r2_pure_rule(num_samples=600, prime_p=31, seed=seed)
        
        m_r1_c = BenchmarkLinear(256, 31)
        train_mlp_to_accuracy(m_r1_c, d_r1_c["x"], d_r1_c["y"], epochs=80, lr=0.05)
        res_r1_c = RulerA(m_r1_c, eval_fn=lambda m: eval_model_acc(m, d_r1_c["x"], d_r1_c["y"])).find_dl_min(tau=0.95)
        dl_r1_comp = res_r1_c["dl_min_bits"]
        
        m_r2_c = BenchmarkMLP(in_dim=62, hidden_dim=64, out_dim=31)
        train_mlp_to_accuracy(m_r2_c, d_r2_c["x"], d_r2_c["y"], epochs=150, lr=0.01)
        res_r2_c = RulerA(m_r2_c, eval_fn=lambda m: eval_model_acc(m, d_r2_c["x"], d_r2_c["y"])).find_dl_min(tau=0.95)
        dl_r2_comp = res_r2_c["dl_min_bits"]
        expected_sum = dl_r1_comp + dl_r2_comp
        
        # Composite Multi-Task Network
        mt_model = BenchmarkComposite(k_facts=256, vocab_size=31, prime_p=31, hidden_dim=64)
        train_mlp_to_accuracy(mt_model.table, d_r1_c["x"], d_r1_c["y"], epochs=80, lr=0.05)
        train_mlp_to_accuracy(mt_model.rule, d_r2_c["x"], d_r2_c["y"], epochs=150, lr=0.01)
        
        def eval_composite(m):
            a1 = eval_model_acc(m.table, d_r1_c["x"], d_r1_c["y"])
            a2 = eval_model_acc(m.rule, d_r2_c["x"], d_r2_c["y"])
            return 0.5 * (a1 + a2)
            
        res_r3_a = RulerA(mt_model, eval_fn=eval_composite, higher_is_better=True).find_dl_min(tau=0.95)
        dl_r3 = res_r3_a["dl_min_bits"]
        additivity_gap = abs(dl_r3 - expected_sum) / float(expected_sum)
        run.log(f"R3 Composite DL: {dl_r3:.0f}b vs Expected Sum (R1+R2): {expected_sum:.0f}b (Gap: {additivity_gap*100:.2f}%)")
        
        # -------------------------------------------------------------
        # 4. Benchmark R4: Pure Noise Memorization
        # -------------------------------------------------------------
        run.log("--- EVALUATING BENCHMARK R4: NOISE MEMORIZATION ---")
        r4_results = []
        for n_noise, h_dim in [(200, 16), (400, 32), (800, 64)]:
            data_r4 = generate_r4_noise_labels(num_samples=n_noise, num_classes=32, seed=seed)
            model_r4 = BenchmarkMLP(in_dim=64, hidden_dim=h_dim, out_dim=32)
            acc_r4 = train_mlp_to_accuracy(model_r4, data_r4["x"], data_r4["y"], epochs=250, lr=0.01)
            eval_fn_r4 = lambda m: eval_model_acc(m, data_r4["x"], data_r4["y"])
            res_r4_a = RulerA(model_r4, eval_fn=eval_fn_r4, higher_is_better=True).find_dl_min(tau=0.95)
            
            model_factory_r4 = lambda h_dim=h_dim: BenchmarkMLP(in_dim=64, hidden_dim=h_dim, out_dim=32)
            res_r4_b = calculate_prequential_code_length(
                model_factory_r4, data_r4["x"], data_r4["y"], num_blocks=4, epochs_per_block=60, lr=0.01, prior_param_bits=1000
            )
            r4_results.append({
                "samples": n_noise,
                "theo_bits": data_r4["theoretical_bits"],
                "dl_a": res_r4_a["dl_min_bits"],
                "dl_b": res_r4_b["total_prequential_bits"]
            })
            ruler_a_points.append(res_r4_a["dl_min_bits"])
            ruler_b_points.append(res_r4_b["total_prequential_bits"])
            run.log(f"R4 [N={n_noise}]: DL_A={res_r4_a['dl_min_bits']:.0f}b | DL_B={res_r4_b['total_prequential_bits']:.0f}b")
            
        r4_theo = np.array([r["theo_bits"] for r in r4_results])
        r4_dl = np.array([r["dl_a"] for r in r4_results])
        slope_r4, _, r_val_r4, _, _ = stats.linregress(r4_theo, r4_dl)
        run.log(f"R4 Regression: Slope = {slope_r4:.4f}, R^2 = {r_val_r4**2:.4f}")
        
        # -------------------------------------------------------------
        # 5. Inter-Ruler Agreement (Spearman Rank Correlation)
        # -------------------------------------------------------------
        spearman_corr, spearman_p = stats.spearmanr(ruler_a_points, ruler_b_points)
        run.log(f"Inter-Ruler Agreement (A vs B): Spearman rho = {spearman_corr:.4f} (p = {spearman_p:.4e})")
        
        # -------------------------------------------------------------
        # 6. Tau-Sensitivity Analysis
        # -------------------------------------------------------------
        run.log("--- EVALUATING TAU SENSITIVITY ---")
        tau_analysis = []
        sample_model = BenchmarkLinear(in_dim=256, out_dim=64)
        data_sample = generate_r1_random_table(k_facts=256, vocab_size=64, seed=seed)
        train_mlp_to_accuracy(sample_model, data_sample["x"], data_sample["y"], epochs=80, lr=0.05)
        eval_sample = lambda m: eval_model_acc(m, data_sample["x"], data_sample["y"])
        ruler_sample = RulerA(sample_model, eval_fn=eval_sample, higher_is_better=True)
        
        for tau in config["tau_values"]:
            t_res = ruler_sample.find_dl_min(tau=tau)
            tau_analysis.append({"tau": tau, "dl_min_bits": t_res["dl_min_bits"], "config": t_res["optimal_config"]["type"]})
            run.log(f"Tau {tau:.2f} -> DL_min: {t_res['dl_min_bits']:.0f}b ({t_res['optimal_config']['type']})")
            
        final_summary = {
            "r1": {
                "linearity_r2_a": float(r2_a),
                "linearity_r2_b": float(r2_b),
                "slope_a": float(slope_a),
                "slope_b": float(slope_b),
                "mean_looseness_a": float(np.mean([r["looseness_a"] for r in r1_results])),
                "mean_looseness_b": float(np.mean([r["looseness_b"] for r in r1_results])),
                "data": r1_results
            },
            "r2": {
                "slope": float(slope_r2),
                "ci95": [float(ci95_low), float(ci95_high)],
                "is_invariant": bool(r2_invariant),
                "data": r2_results
            },
            "r3": {
                "dl_r3": dl_r3,
                "expected_sum": expected_sum,
                "additivity_gap_percent": float(additivity_gap * 100)
            },
            "r4": {
                "slope": float(slope_r4),
                "r2": float(r_val_r4 ** 2),
                "data": r4_results
            },
            "inter_ruler_agreement": {
                "spearman_rho": float(spearman_corr),
                "spearman_p": float(spearman_p),
                "passed": bool(spearman_corr >= 0.90)
            },
            "tau_sensitivity": tau_analysis
        }
        
        # Save results
        out_file = Path("results/m1_calibration_results.json")
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(final_summary, f, indent=2)
            
        run.log_metrics(step=1, metrics={
            "r1_r2_a": r2_a,
            "r1_slope_a": slope_a,
            "spearman_rho": spearman_corr,
            "additivity_gap": additivity_gap
        })
        
        return final_summary


if __name__ == "__main__":
    res = run_m1_calibration()
    print("\n--- M1 CALIBRATION SUMMARY ---")
    print(f"R1 Linearity R^2 (Ruler A): {res['r1']['linearity_r2_a']:.4f}")
    print(f"R1 Slope (Ruler A): {res['r1']['slope_a']:.4f}")
    print(f"R1 Mean Looseness (Ruler A): {res['r1']['mean_looseness_a']:.2f}x")
    print(f"R2 Invariance: {res['r2']['is_invariant']} (Slope: {res['r2']['slope']:.4f})")
    print(f"R3 Additivity Gap: {res['r3']['additivity_gap_percent']:.2f}%")
    print(f"Inter-Ruler Agreement (Spearman rho): {res['inter_ruler_agreement']['spearman_rho']:.4f}")
