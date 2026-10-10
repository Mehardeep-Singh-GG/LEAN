# Project LEAN: Failure Ledger

**Last Updated:** 2026-10-10 11:48:56 UTC  
**Total Tracked:** 10 | **Active/Open:** 0 | **Blocking (S0/S1):** 0

## Blocker Status

> [!NOTE]
> **ZERO BLOCKING FAILURES.** All S0/S1 failures are resolved.

## Summary by Severity

| Severity | Total | Active | Closed |
|---|---|---|---|
| S0 | 0 | 0 | 0 |
| S1 | 6 | 0 | 6 |
| S2 | 4 | 0 | 4 |
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
| `F-0007` | M10 | **S2** | `CLOSED` | F-MEAS | L2 | Blind prediction P_LOCK_5 (Counterfactual fact-swap skill delta) was FALSIFIED: point prediction was 0.020 with 95% PI [0.000, 0.040] and falsification boundary Delta > 0.050; observed empirical delta on held-out seed 1042 was 0.060. | Milestone M10 Held-Out Validation Execution Engine on Tranche T1 |
| `F-0008` | M10 | **S1** | `CLOSED` | F-MEAS | L2 | M10 report claimed factual half-life of 1126.83 Mb and skill half-life of 273.88 Mb yielding 4.11x ratio. However, empirical cloze knowledge baseline was 46% (half-life target 23%), which crossed at 217.40 Mb, while procedural skill baseline was 48% (half-life target 24%), which remained at 34% at 2-bit (140.85 Mb) and 40% at 1-bit (70.42 Mb). Skills never crossed the 24% target within the evaluated ladder. The 4.11x point estimate was an artifact of compute_half_life() checking absolute 0.50 accuracy. | Adversarial Scientific Audit of Milestone M10 Report |
| `F-0009` | M11 | **S2** | `CLOSED` | F-MEAS | L2 | Preregistered prediction P_M11_1 (KAC skill advantage over uniform RTN at equal bits) was FALSIFIED: point prediction was +0.080 with 95% PI [+0.030, +0.150] and falsification boundary Delta <= 0.000; observed empirical delta on SmolLM2-360M was 0.0000 (KAC skill = 40.0%, RTN2 skill = 40.0%). | Milestone M11 Technology Execution Engine (SmolLM2-360M) |
| `F-0010` | M11 | **S2** | `CLOSED` | F-MEAS | L2 | Preregistered prediction P_M11_6 (Native Apple Silicon MLX decode speedup factor) was FALSIFIED: point prediction was 3.80x with 95% PI [3.00x, 5.50x] and falsification boundary < 2.50x; observed empirical speedup on Apple Silicon M2 was 2.30x (MLX 140.7 tok/s vs PyTorch CPU 61.1 tok/s). | Milestone M11 MLX Acceleration Benchmark on Apple M2 |
