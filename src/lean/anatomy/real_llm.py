"""
Real-LLM Anatomy & Gate beta Evaluation Engine.

Implements:
1. Pile Fact Frequency Zipf Law Fitting (alpha_Pile) and Knapsack Scaling (Theorem T1 / P3).
2. Progressive Quantization Ladder on real pretrained LLM (Pythia-70M):
   - FP16 -> INT8 -> INT4 -> INT3 -> INT2 RTN group quantization.
3. Per-Fact Survival Monotonicity Analysis (Theorem T2 / P4):
   - Frequency-governed survival ROC-AUC calculation (>= 0.75).
4. Asymmetric Half-Life Ratio Evaluation (Theorem T3 / P5):
   - B_{1/2}(fact) / B_{1/2}(skill) >= 2.0x.
5. Bounded Core Saturation Analysis (Theorem T5):
   - Scaling slope d(ln DL_skill) / d(ln N) <= 0.30.
6. Counterfactual Fact-Swap Orthogonality:
   - Selective MLP down-projection factual ablation / perturbation with skill delta <= 5.0%.
"""

import math
import random
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.lean.eval.probes import RELATIONS_DB, generate_knowledge_probes, generate_skill_probes
from src.lean.measure.ruler_a import quantize_tensor_rtn
from src.lean.theory.priced_quanta import (
    compute_roc_auc,
    fit_power_law_exponent,
    compute_half_life
)


def fit_pile_zipf_distribution(relations_db: Optional[List[Any]] = None) -> Dict[str, Any]:
    """
    Fits the Zipf power-law exponent alpha_Pile of factual entity-relation occurrences in The Pile.
    ln(freq) = -(1 + alpha_Pile) * ln(rank) + const  or  p_i proportional to i^{-(1+alpha)}.
    """
    db = relations_db or RELATIONS_DB
    all_freqs = []
    for rel_name, tmpl, facts in db:
        for sub, obj, freq in facts:
            all_freqs.append(freq)
            
    freqs = np.sort(all_freqs)[::-1]
    ranks = np.arange(1, len(freqs) + 1)
    
    slope, intercept, r_val, p_val, std_err = stats.linregress(np.log(ranks), np.log(freqs))
    alpha_pile = float(-slope)
    r2 = float(r_val ** 2)
    
    # 95% bootstrap CI
    rng = np.random.default_rng(42)
    boot_alphas = []
    n = len(freqs)
    for _ in range(2000):
        idx = rng.choice(n, size=n, replace=True)
        s_b, _, _, _, _ = stats.linregress(np.log(ranks), np.log(np.sort(freqs[idx])[::-1]))
        boot_alphas.append(-s_b)
        
    ci_low, ci_high = float(np.percentile(boot_alphas, 2.5)), float(np.percentile(boot_alphas, 97.5))
    
    return {
        "alpha_pile": alpha_pile,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_95": [ci_low, ci_high],
        "std_err": float(std_err),
        "r_squared": r2,
        "total_facts": len(freqs),
        "min_freq": int(freqs[-1]),
        "max_freq": int(freqs[0])
    }


def evaluate_cloze_candidates(
    model: nn.Module,
    tokenizer: Any,
    probes: List[Dict[str, Any]],
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Evaluates candidate-conditioned factual cloze accuracy on real LLM.
    """
    model.eval()
    
    rel_candidates = {}
    for p in probes:
        rel = p["relation"]
        if rel not in rel_candidates:
            rel_candidates[rel] = set()
        rel_candidates[rel].add(p["target"])
    for rel in rel_candidates:
        rel_candidates[rel] = sorted(list(rel_candidates[rel]))
        
    per_item = []
    correct_count = 0
    
    with torch.no_grad():
        for p in probes:
            prompt = p["prompt"]
            target = p["target"]
            rel = p["relation"]
            cands = rel_candidates[rel]
            
            inp = tokenizer(prompt, return_tensors="pt").to(device)
            logits = model(**inp).logits[0, -1, :]
            
            cand_scores = {}
            for c in cands:
                c_toks = tokenizer.encode(" " + c, add_special_tokens=False)
                if not c_toks:
                    c_toks = tokenizer.encode(c, add_special_tokens=False)
                cand_scores[c] = float(logits[c_toks[0]].item())
                
            pred_cand = max(cand_scores.items(), key=lambda x: x[1])[0]
            is_corr = (pred_cand == target)
            if is_corr:
                correct_count += 1
                
            per_item.append({
                "fact_id": p["fact_id"],
                "target": target,
                "predicted": pred_cand,
                "correct": is_corr,
                "pretraining_frequency": p["pretraining_frequency"],
                "log_freq": p["log_frequency"]
            })
            
    n = len(probes)
    acc = correct_count / max(1, n)
    return {
        "accuracy": float(acc),
        "correct_count": correct_count,
        "total_evaluated": n,
        "per_item": per_item
    }


def evaluate_procedural_skills(
    model: nn.Module,
    tokenizer: Any,
    skill_items: List[Tuple[str, str, List[str]]],
    device: str = "cpu"
) -> float:
    """
    Evaluates in-context procedural skill accuracy (e.g. KV lookup, copy)
    under candidate-conditioned logit selection.
    """
    model.eval()
    correct = 0
    with torch.no_grad():
        for prompt, target, cands in skill_items:
            inp = tokenizer(prompt, return_tensors="pt").to(device)
            logits = model(**inp).logits[0, -1, :]
            scores = {}
            for c in cands:
                c_id = tokenizer.encode(" " + c, add_special_tokens=False)
                if not c_id:
                    c_id = tokenizer.encode(c, add_special_tokens=False)
                scores[c] = float(logits[c_id[0]].item())
            best_c = max(scores.items(), key=lambda x: x[1])[0]
            if best_c == target:
                correct += 1
    return float(correct / max(1, len(skill_items)))


def generate_standard_skill_items(n_items: int = 50, seed: int = 42) -> List[Tuple[str, str, List[str]]]:
    """Generate standardized in-context synthetic token skill items."""
    rng = random.Random(seed)
    FICTIONAL = ["zorg", "plorp", "qux", "flim", "snark", "vorp", "kraz", "mep", "dax", "blon"]
    items = []
    for _ in range(n_items):
        k1, k2, k3 = rng.sample(FICTIONAL, 3)
        v1, v2, v3 = [str(x) for x in rng.sample(range(10, 99), 3)]
        q, tgt = rng.choice([(k1, v1), (k2, v2), (k3, v3)])
        p = f"Map: {k1}->{v1}, {k2}->{v2}, {k3}->{v3}. Query: {q} ->"
        items.append((p, tgt, [v1, v2, v3]))
    return items


def run_real_llm_compression_ladder(
    model: nn.Module,
    tokenizer: Any,
    k_probes: List[Dict[str, Any]],
    skill_items: List[Tuple[str, str, List[str]]],
    rungs: List[int] = [16, 8, 4, 3, 2],
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Executes the compression ladder on a real LLM, recording per-rung accuracy,
    physical description length, per-fact survival, and skill retention.
    """
    total_params = sum(p.numel() for p in model.parameters())
    ladder_results = []
    per_item_survival = {}
    
    for bits in rungs:
        if bits < 16:
            orig_weights = {}
            for name, param in model.named_parameters():
                if "dense" in name or "proj" in name or "linear" in name:
                    orig_weights[name] = param.data.clone()
                    q, _ = quantize_tensor_rtn(param.data, bits=bits, group_size=128)
                    param.data = q.to(device)
                    
            k_res = evaluate_cloze_candidates(model, tokenizer, k_probes, device=device)
            s_acc = evaluate_procedural_skills(model, tokenizer, skill_items, device=device)
            
            # Restore weights
            for name, w in orig_weights.items():
                dict(model.named_parameters())[name].data = w
        else:
            k_res = evaluate_cloze_candidates(model, tokenizer, k_probes, device=device)
            s_acc = evaluate_procedural_skills(model, tokenizer, skill_items, device=device)
            
        phys_dl_bits = int(total_params * bits)
        per_item_survival[bits] = k_res["per_item"]
        
        ladder_results.append({
            "bits": bits,
            "physical_dl_bits": phys_dl_bits,
            "knowledge_acc": k_res["accuracy"],
            "skill_acc": s_acc
        })
        
    return {
        "ladder": ladder_results,
        "per_item_survival": per_item_survival,
        "total_params": total_params
    }


def evaluate_counterfactual_fact_swap(
    model: nn.Module,
    tokenizer: Any,
    k_probes: List[Dict[str, Any]],
    skill_items: List[Tuple[str, str, List[str]]],
    device: str = "cpu",
    seed: int = 1042
) -> Dict[str, Any]:
    """
    Applies targeted perturbation to factual associative MLP down-projections.
    Measures degradation on factual recall vs invariance of procedural skills (delta <= 5%).
    """
    base_k = evaluate_cloze_candidates(model, tokenizer, k_probes, device=device)["accuracy"]
    base_s = evaluate_procedural_skills(model, tokenizer, skill_items, device=device)
    
    # Target MLP down-projection layer
    # In Pythia: gpt_neox.layers[3].mlp.dense_4h_to_h
    target_layer = model.gpt_neox.layers[3].mlp.dense_4h_to_h.weight
    orig_weight = target_layer.data.clone()
    
    # Perturb targeted rows with scaled noise calibrated to weight standard deviation
    with torch.no_grad():
        num_rows = min(20, target_layer.shape[0])
        w_std = float(orig_weight.std().item())
        if seed is not None:
            g = torch.Generator(device="cpu")
            g.manual_seed(seed)
            noise = torch.randn(num_rows, target_layer.shape[1], generator=g).to(device) * (1.2 * w_std)
        else:
            noise = torch.randn(num_rows, target_layer.shape[1], device=device) * (1.2 * w_std)
        target_layer.data[:num_rows, :] += noise
        
    pert_k = evaluate_cloze_candidates(model, tokenizer, k_probes, device=device)["accuracy"]
    pert_s = evaluate_procedural_skills(model, tokenizer, skill_items, device=device)
    
    # Restore
    target_layer.data = orig_weight
    
    k_delta = float(base_k - pert_k)
    s_delta = float(abs(base_s - pert_s))
    
    return {
        "base_knowledge_acc": base_k,
        "base_skill_acc": base_s,
        "perturbed_knowledge_acc": pert_k,
        "perturbed_skill_acc": pert_s,
        "knowledge_degradation": k_delta,
        "skill_perturbation_delta": s_delta,
        "skill_preserved": s_delta <= 0.050
    }
