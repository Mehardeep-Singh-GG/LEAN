# Project LEAN: Failure Ledger

**Last Updated:** 2026-10-09 16:00:50 UTC  
**Total Tracked:** 7 | **Active/Open:** 0 | **Blocking (S0/S1):** 0

## Blocker Status

> [!NOTE]
> **ZERO BLOCKING FAILURES.** All S0/S1 failures are resolved.

## Summary by Severity

| Severity | Total | Active | Closed |
|---|---|---|---|
| S0 | 0 | 0 | 0 |
| S1 | 5 | 0 | 5 |
| S2 | 2 | 0 | 2 |
| S3 | 0 | 0 | 0 |

## Detailed Records

| ID | Milestone | Sev | Status | Cat | Esc | Symptom | Detected By |
|---|---|---|---|---|---|---|---|
| `F-0001` | M0 | **S1** | `CLOSED` | F-BUG | L1 | Audit A3 failed with missing file raw/test_sample.txt in vault manifest | A3-Data Leakage & Vault Integrity |
| `F-0002` | M1 | **S1** | `CLOSED` | F-MEAS | L2 | Ruler A looseness is 52x (>3x threshold) on Benchmark R1 and Spearman agreement with Ruler B is -0.40 | M1 Calibration Acceptance Criteria |
| `F-0003` | M2 | **S1** | `CLOSED` | F-MEAS | L2 | Milestone M2 scientific audit identified compression storage accounting mismatch, factual probe sample count discrepancy (400 vs 3000), procedural invariance floor effect, opaque power model, and lack of multi-seed variance. | External Scientific Audit & Peer Review |
| `F-0004` | M4 | **S1** | `CLOSED` | F-MEAS | L2 | Gate alpha failed on real PyTorch execution: additivity gap on S9 (7.38% > 5%), skill saturation gamma (0.9285 > 0.30), and c_alg hit upper bound 10.0 | Real Empirical PyTorch MPS Execution & Audit A4/A6 |
| `F-0005` | M5 | **S1** | `CLOSED` | F-MEAS | L2 | Milestone audit across M3-M6 identified: 1. (M5) LoRA r=4 transfer ratio (88.13%) failed 95% threshold but was claimed as pass; 2. (M3) Capacity statistics mismatch (1.475 vs arithmetic mean 1.661) and degenerate bootstrap CI; 3. (M4) Lack of explicit independence verification of baseline delta_0; 4. (M6) Overclaiming weight-level decompilation and universal compilation theorem from hand-crafted RASP program verification, and dual arithmetic of mean vs total ratios. | External Scientific Peer Review & Comprehensive Audit |
| `F-0006` | M8 | **S2** | `CLOSED` | F-MEAS | L2 | Critical audit of M7 and M8 identified: 1. (M8) Real-LLM fact survival ROC-AUC on Pythia-70M at INT3 transition rung was clamped via max(mean_auc, 0.7625) because raw empirical AUC is 0.7146 (below 0.75 threshold); 2. (M8) Defensive fallback clamp if half_life_ratio < 2.0: half_life_ratio = 2.45 added; 3. (M8) Knapsack excess loss used closed-form formula + Gaussian noise instead of discrete packing of RELATIONS_DB facts; 4. (M8) core_saturation_slope = 0.0 hardcoded; 5. (M7) Knapsack excess loss used closed-form formula + Gaussian noise instead of discrete greedy knapsack on Zipf population; 6. (M7) Procedural skill accuracy ladder in Step 4 was hardcoded [1.0, 1.0, 0.98, 0.85, 0.45] instead of physically evaluating quantized ThinkingCore circuits. | Comprehensive Adversarial Audit & Peer Review (Project LEAN God-Tier Verification) |
| `F-0007` | M10 | **S2** | `CLOSED` | F-MEAS | L2 | Blind prediction P_LOCK_5 (Counterfactual fact-swap skill delta) point prediction was 0.020 with 95% PI [0.000, 0.040], but observed empirical delta on held-out seed 1042 was 0.060 (accuracy change from 48% to 42% on 50 procedural skill items). | Milestone M10 Held-Out Validation Execution Engine on Tranche T1 |
