# Project LEAN: Failure Ledger

**Last Updated:** 2026-10-09 13:51:32 UTC  
**Total Tracked:** 4 | **Active/Open:** 0 | **Blocking (S0/S1):** 0

## Blocker Status

> [!NOTE]
> **ZERO BLOCKING FAILURES.** All S0/S1 failures are resolved.

## Summary by Severity

| Severity | Total | Active | Closed |
|---|---|---|---|
| S0 | 0 | 0 | 0 |
| S1 | 4 | 0 | 4 |
| S2 | 0 | 0 | 0 |
| S3 | 0 | 0 | 0 |

## Detailed Records

| ID | Milestone | Sev | Status | Cat | Esc | Symptom | Detected By |
|---|---|---|---|---|---|---|---|
| `F-0001` | M0 | **S1** | `CLOSED` | F-BUG | L1 | Audit A3 failed with missing file raw/test_sample.txt in vault manifest | A3-Data Leakage & Vault Integrity |
| `F-0002` | M1 | **S1** | `CLOSED` | F-MEAS | L2 | Ruler A looseness is 52x (>3x threshold) on Benchmark R1 and Spearman agreement with Ruler B is -0.40 | M1 Calibration Acceptance Criteria |
| `F-0003` | M2 | **S1** | `CLOSED` | F-MEAS | L2 | Milestone M2 scientific audit identified compression storage accounting mismatch, factual probe sample count discrepancy (400 vs 3000), procedural invariance floor effect, opaque power model, and lack of multi-seed variance. | External Scientific Audit & Peer Review |
| `F-0004` | M4 | **S1** | `CLOSED` | F-MEAS | L2 | Gate alpha failed on real PyTorch execution: additivity gap on S9 (7.38% > 5%), skill saturation gamma (0.9285 > 0.30), and c_alg hit upper bound 10.0 | Real Empirical PyTorch MPS Execution & Audit A4/A6 |
