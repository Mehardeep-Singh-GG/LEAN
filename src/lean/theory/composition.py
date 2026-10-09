"""
Composition Law Engine and Model Comparison Suite for Milestone M4 (Gate alpha).

Implements:
1. LEAN Composition Law Solver:
       DL*(M) = c_alg * sum_{s in S} K(s) + I_facts / kappa + delta(N) + eta * I_interaction
2. Model Comparison: LEAN vs. Naive N vs. Multiplicative vs. Pure Memorization (AIC, BIC, LOOCV R^2).
3. Additivity Gap Evaluator: Delta_comp = |DL(F U S) - (DL(F) + DL(S) - delta)| / DL(F U S).
4. Bounded Thinking Core Analysis: gamma = d ln(DL_skill) / d ln(N) with 95% bootstrap CI.
"""

import math
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from scipy import stats
from scipy.optimize import minimize


def fit_composition_law(
    data: List[Dict[str, Any]],
    n_resamples: int = 1000,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Fit the LEAN Composition Law to empirical data:
        DL = c_alg * K_total + (1 / kappa) * I_facts + delta + eta * I_interaction
    Using robust Huber loss in parameter space with non-parametric bootstrap CIs.
    """
    K_total = np.array([d["k_skill_bits"] for d in data], dtype=np.float64)
    I_facts = np.array([d["i_fact_bits"] for d in data], dtype=np.float64)
    I_inter = np.array([d.get("i_interaction_bits", 0.0) for d in data], dtype=np.float64)
    DL_obs = np.array([d["dl_bits"] for d in data], dtype=np.float64)
    N_samples = len(data)
    
    # Feature matrix X: [K_total, I_facts, ones, I_inter]
    X = np.column_stack([K_total, I_facts, np.ones(N_samples), I_inter])
    
    # Robust fit via Huber-like loss or OLS with non-negative constraints
    def loss_fn(params):
        c_alg, inv_kappa, delta, eta = params
        pred = c_alg * K_total + inv_kappa * I_facts + delta + eta * I_inter
        res = DL_obs - pred
        # Smooth Huber loss
        delta_h = 200.0
        huber = np.where(np.abs(res) <= delta_h, 0.5 * res**2, delta_h * (np.abs(res) - 0.5 * delta_h))
        return np.sum(huber)
        
    init_params = [1.2, 0.5, 1000.0, 0.1]
    bounds = [(0.1, 10.0), (0.1, 2.0), (0.0, 50000.0), (0.0, 5.0)]
    res_opt = minimize(loss_fn, init_params, bounds=bounds, method="L-BFGS-B")
    
    c_alg_fit, inv_kappa_fit, delta_fit, eta_fit = res_opt.x
    kappa_fit = 1.0 / inv_kappa_fit
    
    # Predictions and residuals
    preds = c_alg_fit * K_total + inv_kappa_fit * I_facts + delta_fit + eta_fit * I_inter
    ss_res = np.sum((DL_obs - preds)**2)
    ss_tot = np.sum((DL_obs - np.mean(DL_obs))**2)
    r2 = 1.0 - (ss_res / max(ss_tot, 1e-8))
    
    # Bootstrap confidence intervals
    boot_c_alg = []
    boot_kappa = []
    boot_delta = []
    boot_r2 = []
    
    rng = np.random.RandomState(42)
    for _ in range(n_resamples):
        indices = rng.choice(N_samples, size=N_samples, replace=True)
        K_b = K_total[indices]
        I_b = I_facts[indices]
        Inter_b = I_inter[indices]
        DL_b = DL_obs[indices]
        
        def b_loss(p):
            ca, ik, d, e = p
            pr = ca * K_b + ik * I_b + d + e * Inter_b
            r = DL_b - pr
            return np.sum(np.where(np.abs(r) <= 200.0, 0.5 * r**2, 200.0 * (np.abs(r) - 100.0)))
            
        b_res = minimize(b_loss, [c_alg_fit, inv_kappa_fit, delta_fit, eta_fit], bounds=bounds, method="L-BFGS-B")
        ca_b, ik_b, d_b, _ = b_res.x
        boot_c_alg.append(ca_b)
        boot_kappa.append(1.0 / max(ik_b, 1e-4))
        boot_delta.append(d_b)
        
    c_alg_ci = [float(np.percentile(boot_c_alg, 2.5)), float(np.percentile(boot_c_alg, 97.5))]
    kappa_ci = [float(np.percentile(boot_kappa, 2.5)), float(np.percentile(boot_kappa, 97.5))]
    delta_ci = [float(np.percentile(boot_delta, 2.5)), float(np.percentile(boot_delta, 97.5))]
    
    c_alg_width = c_alg_ci[1] - c_alg_ci[0]
    relative_c_alg_ci_width = c_alg_width / c_alg_fit
    
    return {
        "c_alg": float(c_alg_fit),
        "c_alg_95_ci": c_alg_ci,
        "c_alg_relative_ci_width": float(relative_c_alg_ci_width),
        "kappa": float(kappa_fit),
        "kappa_95_ci": kappa_ci,
        "delta": float(delta_fit),
        "delta_95_ci": delta_ci,
        "eta": float(eta_fit),
        "r_squared": float(r2),
        "ss_res": float(ss_res),
        "ss_tot": float(ss_tot)
    }


def compare_competing_models(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compare LEAN Composition Law against 3 competing models:
    1. Model 1 (LEAN): DL = c_alg * K + inv_kappa * I_facts + delta
    2. Model 2 (Naive N): DL = beta_0 + beta_1 * N
    3. Model 3 (Multiplicative): DL = beta_0 + beta_1 * (I_facts * K)
    4. Model 4 (Pure Facts): DL = beta_0 + beta_1 * I_facts
    
    Computes AIC, BIC, and Leave-One-Out Cross-Validation (LOOCV) R^2.
    """
    N_samples = len(data)
    y = np.array([d["dl_bits"] for d in data], dtype=np.float64)
    ss_tot = np.sum((y - np.mean(y))**2)
    
    models = {}
    
    # 1. LEAN Composition Model
    K_total = np.array([d["k_skill_bits"] for d in data], dtype=np.float64)
    I_facts = np.array([d["i_fact_bits"] for d in data], dtype=np.float64)
    X_lean = np.column_stack([K_total, I_facts, np.ones(N_samples)])
    p_lean, _, _, _ = np.linalg.lstsq(X_lean, y, rcond=None)
    y_pred_lean = X_lean @ p_lean
    rss_lean = np.sum((y - y_pred_lean)**2)
    k_lean = 3
    
    # 2. Naive Parameter Count Model
    N_params = np.array([d.get("n_params", 1000) for d in data], dtype=np.float64)
    X_naive = np.column_stack([N_params, np.ones(N_samples)])
    p_naive, _, _, _ = np.linalg.lstsq(X_naive, y, rcond=None)
    y_pred_naive = X_naive @ p_naive
    rss_naive = np.sum((y - y_pred_naive)**2)
    k_naive = 2
    
    # 3. Multiplicative Interaction Model
    K_norm = K_total / 1000.0
    I_norm = I_facts / 1000.0
    Mult = K_norm * I_norm
    X_mult = np.column_stack([Mult, np.ones(N_samples)])
    p_mult, _, _, _ = np.linalg.lstsq(X_mult, y, rcond=None)
    y_pred_mult = X_mult @ p_mult
    rss_mult = np.sum((y - y_pred_mult)**2)
    k_mult = 2
    
    # 4. Pure Facts (No Skill Term)
    X_facts = np.column_stack([I_facts, np.ones(N_samples)])
    p_facts, _, _, _ = np.linalg.lstsq(X_facts, y, rcond=None)
    y_pred_facts = X_facts @ p_facts
    rss_facts = np.sum((y - y_pred_facts)**2)
    k_facts = 2
    
    # Information criteria helper
    def calc_ic(rss, k, n):
        aic = n * np.log(max(rss / n, 1e-12)) + 2 * k
        bic = n * np.log(max(rss / n, 1e-12)) + k * np.log(n)
        return float(aic), float(bic)
        
    # Leave-One-Out Cross-Validation helper
    def calc_loocv(X_mat, y_vec):
        loocv_errors = []
        n = len(y_vec)
        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            p_cv, _, _, _ = np.linalg.lstsq(X_mat[mask], y_vec[mask], rcond=None)
            pred_i = X_mat[i] @ p_cv
            loocv_errors.append((y_vec[i] - pred_i)**2)
        mse_cv = np.mean(loocv_errors)
        r2_cv = 1.0 - (np.sum(loocv_errors) / max(ss_tot, 1e-8))
        return float(r2_cv), float(mse_cv)
        
    loocv_lean, _ = calc_loocv(X_lean, y)
    loocv_naive, _ = calc_loocv(X_naive, y)
    loocv_mult, _ = calc_loocv(X_mult, y)
    loocv_facts, _ = calc_loocv(X_facts, y)
    
    aic_lean, bic_lean = calc_ic(rss_lean, k_lean, N_samples)
    aic_naive, bic_naive = calc_ic(rss_naive, k_naive, N_samples)
    aic_mult, bic_mult = calc_ic(rss_mult, k_mult, N_samples)
    aic_facts, bic_facts = calc_ic(rss_facts, k_facts, N_samples)
    
    return {
        "model_1_lean": {
            "name": "LEAN Composition Law (c_alg * K + I_facts / kappa + delta)",
            "rss": float(rss_lean),
            "r2_in_sample": float(1.0 - rss_lean / ss_tot),
            "r2_loocv": float(loocv_lean),
            "aic": aic_lean,
            "bic": bic_lean
        },
        "model_2_naive": {
            "name": "Naive Scaling (beta_0 + beta_1 * N)",
            "rss": float(rss_naive),
            "r2_in_sample": float(1.0 - rss_naive / ss_tot),
            "r2_loocv": float(loocv_naive),
            "aic": aic_naive,
            "bic": bic_naive,
            "delta_aic": aic_naive - aic_lean,
            "delta_bic": bic_naive - bic_lean
        },
        "model_3_multiplicative": {
            "name": "Multiplicative Entanglement (beta_0 + beta_1 * (I * K))",
            "rss": float(rss_mult),
            "r2_in_sample": float(1.0 - rss_mult / ss_tot),
            "r2_loocv": float(loocv_mult),
            "aic": aic_mult,
            "bic": bic_mult,
            "delta_aic": aic_mult - aic_lean,
            "delta_bic": bic_mult - bic_lean
        },
        "model_4_pure_facts": {
            "name": "Pure Facts / No Skill Term (beta_0 + beta_1 * I_facts)",
            "rss": float(rss_facts),
            "r2_in_sample": float(1.0 - rss_facts / ss_tot),
            "r2_loocv": float(loocv_facts),
            "aic": aic_facts,
            "bic": bic_facts,
            "delta_aic": aic_facts - aic_lean,
            "delta_bic": bic_facts - bic_lean
        },
        "lean_beats_all": (
            loocv_lean >= 0.90 and
            aic_lean < min(aic_naive, aic_mult, aic_facts) and
            bic_lean < min(bic_naive, bic_mult, bic_facts)
        )
    }


def evaluate_additivity_gap(
    dl_composite: float,
    dl_facts_only: float,
    dl_skills_only: float,
    delta_baseline: float
) -> Dict[str, Any]:
    """
    Measure additivity gap between composite capability and independent parts:
        Delta_comp = |DL(F U S) - (DL(F) + DL(S) - delta)| / DL(F U S)
    Must be <= 0.05 (5%) to pass Gate alpha.
    """
    expected_sum = dl_facts_only + dl_skills_only - delta_baseline
    gap_bits = abs(dl_composite - expected_sum)
    gap_fraction = gap_bits / max(dl_composite, 1.0)
    passed = (gap_fraction <= 0.05)
    
    return {
        "dl_composite": dl_composite,
        "dl_facts_only": dl_facts_only,
        "dl_skills_only": dl_skills_only,
        "delta_baseline": delta_baseline,
        "expected_sum": expected_sum,
        "gap_bits": gap_bits,
        "gap_fraction": gap_fraction,
        "gap_percent": gap_fraction * 100.0,
        "passed": passed
    }


def evaluate_skill_saturation(
    scale_points: List[Tuple[int, float]],
    n_resamples: int = 1000
) -> Dict[str, Any]:
    """
    Test Bounded Thinking Core: fit log(DL_skill) = beta_0 + gamma * log(N).
    Verify gamma upper CI <= 0.30 (sub-linear saturation).
    """
    N_vals = np.array([p[0] for p in scale_points], dtype=np.float64)
    DL_vals = np.array([p[1] for p in scale_points], dtype=np.float64)
    
    log_N = np.log(N_vals)
    log_DL = np.log(DL_vals)
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_N, log_DL)
    
    # Bootstrap CI on slope gamma
    boot_slopes = []
    rng = np.random.RandomState(42)
    n = len(scale_points)
    for _ in range(n_resamples):
        idx = rng.choice(n, size=n, replace=True)
        if len(np.unique(log_N[idx])) > 1:
            s_b, _, _, _, _ = stats.linregress(log_N[idx], log_DL[idx])
            boot_slopes.append(s_b)
            
    if boot_slopes:
        gamma_low = float(np.percentile(boot_slopes, 2.5))
        gamma_high = float(np.percentile(boot_slopes, 97.5))
    else:
        gamma_low = float(slope - 1.96 * std_err)
        gamma_high = float(slope + 1.96 * std_err)
        
    saturates = (gamma_high <= 0.30)
    return {
        "gamma_point": float(slope),
        "gamma_95_ci": [gamma_low, gamma_high],
        "std_err": float(std_err),
        "r_squared": float(r_value**2),
        "saturates": saturates
    }
