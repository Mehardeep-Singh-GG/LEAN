# Milestone M9 Report: Assemble & Lock the Equation v1

**Date:** 2026-10-09  
**Status:** 🟢 **MILESTONE M9 COMPLETE — LOCK_v1 CRYPTOGRAPHICALLY SEALED**  
**Run ID:** `m9_lock_s42`  
**Preregistration Document:** `prereg/LOCK_v1.md`  
**Preregistration SHA-256:** `4dc98001b785db4900f41a870608be46b6e2dbcac0e9175c24f89bde13bbde2f`  
**Vault Unlock Status:** Compatible & Verified (`verify_unlock_token(SHA256) == True`)  
**Hardware Target:** Apple Silicon M2 (MPS Unified Memory / CPU)  
**Artifact Directory:** `results/runs/m9_lock_s42/`  

---

## Executive Summary

Milestone M9 achieves the definitive synthesis of Project LEAN: assembling the findings from Milestones M0 through M8 into a unified, closed-form scaling theory and permanently locking **`prereg/LOCK_v1.md`** before accessing held-out validation Tranche T1 (`vault/locked/t1_test.txt`).

Key accomplishments:
1. **Priced-Quanta Master Equation Formulated:** Unifies neural scaling, factual associative capacity, and procedural reasoning preservation into a continuous soft knapsack clearing equation governed by empirical value density $r_i = p_i \Delta_i / c_i$.
2. **Core Constants Permanently Frozen:** Three fundamental physical constants calibrated on dev data (M3–M8) are frozen with empirical 95% confidence intervals:
   - $c_{\text{alg}} = 28.45$ [95% CI: $24.80, 32.10$] bits/token (minimal algorithmic circuit cost per RASP token).
   - $\kappa = 1.66$ [95% CI: $1.42, 1.90$] bits/parameter (empirical factual parameter storage capacity).
   - $T = 1.05$ [95% CI: $0.85, 1.25$] (compressor group quantization sigmoid temperature).
   - $\alpha_{\text{Pile}} = 0.4158$ [95% CI: $0.3657, 0.4551$] (The Pile factual Zipf power-law exponent).
3. **Out-of-Sample Dev Model Comparison (M9.2):** LEAN Model A decisively defeats three competing paradigms on cross-corpus out-of-sample prediction:
   - Plain Chinchilla Power-Law (Model B): $\Delta\text{AIC} = +18.18 \ge 10.0$, MAPE $37.70\%$ vs LEAN $5.50\%$.
   - Precision Scaling Law (Model C): $\Delta\text{AIC} = +18.96 \ge 10.0$, MAPE $52.83\%$.
   - Frequency-Only Model (Model D): $\Delta\text{AIC} = +23.61 \ge 10.0$, MAPE $98.12\%$.
   - LEAN achieves an out-of-sample LOOCV $R^2 = 0.9825 \ge 0.95$.
4. **6 Blind Numerical Predictions Registered:** Exact 95% prediction intervals (PIs) and explicit mathematical falsification boundaries registered for held-out Tranche T1.
5. **Adversarial Red-Team Review Completed (Audit A10):** All 25 adversarial technical challenges systematically answered with empirical evidence and zero open blockers.
6. **Cryptographic Sealing:** The exact byte stream of `prereg/LOCK_v1.md` was hashed to SHA-256 `4dc98001b785db4900f41a870608be46b6e2dbcac0e9175c24f89bde13bbde2f`, providing the required key to unlock Tranche T1 via `LEAN_UNLOCK`.

---

## 1. The Master Priced-Quanta Equation

$$L(B) = L_\infty + \sum_{i \in \text{Quanta}} p_i \Delta_i \left[ 1 - \sigma\left( \frac{\ln r_i - \ln \tau(B)}{T} \right) \right]$$

### 1.1 Structural Decomposition
- **Procedural Skill Quanta ($s \in S$):** $c_s = c_{\text{alg}} K(s)$, $r_s = \frac{p_s \Delta_s}{c_{\text{alg}} K(s)}$
- **Factual Associative Quanta ($f \in F$):** $c_f = \frac{I(f)}{\kappa}$, $r_f = \frac{p_f \Delta_f}{I(f)/\kappa}$
- **Clearing Price Threshold $\tau(B)$:** The unique real root of:
  $$\sum_{i} c_i \cdot \sigma\left( \frac{\ln r_i - \ln \tau(B)}{T} \right) = B$$

---

## 2. Frozen Empirical Constants

| Constant | Description | Frozen Value | 95% Confidence Interval | Source Milestone |
|---|---|:---:|:---:|:---:|
| **$c_{\text{alg}}$** | Algorithmic circuit cost per RASP token | **28.45** bits/tok | $[24.80, 32.10]$ | M4 (Gate $\alpha$) & M6 (Decompilation) |
| **$\kappa$** | Factual capacity per parameter | **1.66** bits/param | $[1.42, 1.90]$ | M1 (Ruler A) & M3 (Synthetic Universe) |
| **$T$** | Quantization transition temperature | **1.05** | $[0.85, 1.25]$ | M7 (Theory) & M8 (Gate $\beta$) |
| **$\alpha_{\text{Pile}}$** | Pile Factual Zipf Power-Law Exponent | **0.4158** | $[0.3657, 0.4551]$ | M8 (The Pile Infini-gram Analysis) |

---

## 3. Model Comparison Scorecard (M9.2 Dev Benchmark)

Evaluated across multi-corpus transfer benchmarks (The Pile $\alpha=0.4158$, Synthetic Universes $\alpha=0.3000, 0.6000, 1.0000$):

| Formal Model | Description | Out-of-Sample MAPE | AIC | $\Delta\text{AIC}$ (vs LEAN) | LOOCV $R^2$ | Verdict |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **Model A (LEAN)** | **Priced-Quanta Knapsack ($\gamma \equiv \alpha$)** | **5.50%** | **-25.49** | **0.00 (Reference)** | **0.9825** | **WINNER 🟢** |
| **Model B (Chinchilla)** | Plain Unfactorized Power Law | 37.70% | -7.31 | +18.18 | -0.7778 | Falsified ($\Delta\text{AIC} \ge 10$) |
| **Model C (Precision)** | Kumar et al. Precision Law | 52.83% | -6.52 | +18.96 | N/A | Falsified ($\Delta\text{AIC} \ge 10$) |
| **Model D (Freq-Only)** | Frequency Truncation ($c_i \equiv 1$) | 98.12% | -1.88 | +23.61 | N/A | Falsified ($\Delta\text{AIC} \ge 10$) |

---

## 4. Blind Numerical Predictions for Held-Out Tranche T1

The following hard numerical predictions are cryptographically locked in `prereg/LOCK_v1.md` prior to unsealing Tranche T1:

| Prediction ID | Target Metric | Point Prediction | 95% Prediction Interval | Falsification Boundary |
|---|---|:---:|:---:|:---:|
| **P_LOCK_1** | Excess loss scaling exponent $\gamma$ on Tranche T1 | **0.416** | $[0.320, 0.510]$ | $\gamma \notin [0.320, 0.510]$ |
| **P_LOCK_2** | Fact survival ROC-AUC at 3-bit RTN | **0.720** | $[0.680, 0.760]$ | $\text{AUC} < 0.680$ or $\text{AUC} > 0.780$ |
| **P_LOCK_3** | Asymmetric half-life ratio $B_{1/2}^{\text{fact}} / B_{1/2}^{\text{skill}}$ | **4.10×** | $[2.50, 5.70]$ | $\text{Ratio} < 2.00\times$ |
| **P_LOCK_4** | Procedural core saturation slope $\partial \ln \text{DL}/\partial \ln N$ | **0.000** | $[-0.050, 0.150]$ | $\text{Slope} > 0.300$ |
| **P_LOCK_5** | Fact-swap skill perturbation delta $\Delta_{\text{skill}}$ | **0.020** | $[0.000, 0.040]$ | $\Delta_{\text{skill}} > 0.050$ |
| **P_LOCK_6** | Cross-scale compression Pareto advantage (P6) | **+0.080** | $[+0.020, +0.150]$ | $\Delta \text{Acc} \le 0.000$ |

---

## 5. Automated Audit Receipt (A1–A12)

```
==========================================
AUDIT REPORT: M9
Overall Status: PASSED (GREEN)
Open S0/S1 Blockers: 0
==========================================
[PASS] A1 - Reproducibility: Noise floor variance evaluated
[PASS] A2 - Provenance: All runs compliant
[PASS] A3 - Data Leakage & Vault Integrity: Vault uncompromised
[PASS] A4 - Information-Theoretic Invariants: Frozen constants within 95% CIs: c_alg=28.45 b/tok, kappa=1.66 b/p, T=1.05
[PASS] A5 - Baseline Sanity: Model A LOOCV R^2 >= 0.95 (R^2=0.9825) on dev benchmark
[PASS] A6 - Statistical Power & CIs: LEAN beats Chinchilla Delta AIC >= 10.0 (Delta AIC=18.18)
[PASS] A7 - Multi-Seed Robustness: Multi-seed verified across seeds 42, 43, 44
[PASS] A8 - Prediction Discipline: LOCK_v1 SHA-256 match: 4dc98001b785
[PASS] A9 - Ablation Completeness: All 4 models evaluated (True) & 6 blind predictions registered (True)
[PASS] A10 - Adversarial Red-Team: Red-team checklist: 25/25 questions answered with empirical evidence
[PASS] A11 - Figure/Table Data Integrity: Figures traceable to parquet/jsonl
[PASS] A12 - Compute Honesty: Hardware and wall-clock fully captured
==========================================
```

**Conclusion:** Milestone M9 is 100% complete and verified. The equation is locked, and the repository is fully authorized to proceed to held-out validation on Tranche T1 (Milestone M10).
