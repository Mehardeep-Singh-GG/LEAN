# M0 Report: Foundations and Audit Infrastructure

## 0. Verdict
**Status:** **PASS (COMPLETED)**  
**Open S0/S1 Blockers:** 0  
**Audit Suite A1–A12:** 100% PASS (GREEN)  
**Unit Tests:** 15/15 PASSED (100%)  

---

## 1. Objective
Establish a reproducible, tamper-evident research environment before executing scientific experiments:
- Pinned virtual environment with Apple Silicon MPS acceleration.
- Determinism and hardware noise-floor calibration ($\sigma_0$).
- Cryptographic vault guard protecting held-out test tranches (T1, T2, T3).
- Failure Ledger tracking with automated blocker enforcement.
- Automated audit suite (A1–A12).

---

## 2. Steps Checklist
- [x] **0.1 Directory Skeleton:** `src/lean/`, `audit/`, `vault/`, `prereg/`, `reports/`, `results/runs/`, `tests/`, `configs/` created.
- [x] **0.2 Environment Setup:** Python 3.11.17 isolated virtual environment initialized via `uv` with `env.lock`.
- [x] **0.3 Core Dependencies:** `torch==2.14.1` (MPS verified), `transformers==5.19.0`, `mlx==0.32.3`, `scipy==1.17.1`, `pytest==9.1.1`, `ruff==0.16.10` installed.
- [x] **0.4 Run Tracker:** `src/lean/measure/tracker.py` implemented with deterministic run ID `hash(config + SHA + seed)`.
- [x] **0.5 Hardware Noise Floor:** `src/lean/measure/noise_floor.py` evaluated 3x on CPU and 3x on MPS.
- [x] **0.6 Vault Guard & Ingestion:** `vault/guard.py` and `src/lean/eval/data_prep.py` implemented with SHA-256 manifest.
- [x] **0.7 Failure Ledger:** `audit/FAILURES.jsonl`, schema, and `src/lean/audit/failure_ledger.py` implemented and tested with live failure cycle.
- [x] **0.8 Testing Suite:** 15 unit tests covering vault security, DL accounting, invariants, and tracking passing in 1.11s.

---

## 3. Results

### Hardware Profiling & Determinism Calibration (`results/noise_floor.json`)
- **Host:** macOS-27.0-arm64 (Apple M2, 8.0 GB RAM)
- **PyTorch:** 2.14.1 (MPS Native: `True`)
- **MLX Core:** 0.32.3

| Device | Metric | Mean | Std ($\sigma_0$) | Max Pairwise Diff | Deterministic? |
|---|---|---|---|---|---|
| **CPU** | Cross-Entropy Loss | 7.08416176 | **0.00000000** | **0.00e+00** | **YES** |
| **MPS** | Cross-Entropy Loss | 7.11298466 | **0.00000000** | **0.00e+00** | **YES** |

*Finding:* Both CPU and Apple Silicon MPS exhibit exact determinism ($\sigma_0 = 0.0$) under identical random seeds across repeat executions.

---

## 4. Acceptance Criteria
- [x] CPU rerun gives identical metrics ($|\Delta| < 1\times 10^{-6}$). Achieved: $|\Delta| = 0.00$.
- [x] MPS rerun stays within recorded noise floor $\sigma_0$. Achieved: $|\Delta| = 0.00$.
- [x] Vault guard test passes: reading test splits without valid `LEAN_UNLOCK` raises `VaultLockViolation`.
- [x] 100% of infrastructure unit tests pass (15/15 passed).
- [x] Audits A1–A12 pass with 0 open S0/S1 blockers.

---

## 5. Audit Suite Status (A1 - A12)
```
[PASS] A1 - Reproducibility: Noise floor variance evaluated (sigma_0 = 0.0)
[PASS] A2 - Provenance: All runs compliant
[PASS] A3 - Data Leakage & Vault Integrity: Vault uncompromised (5 files verified)
[PASS] A4 - Information-Theoretic Invariants: Invariants verified
[PASS] A5 - Baseline Sanity: Baselines calibrated
[PASS] A6 - Statistical Power & CIs: Statistics standard applied
[PASS] A7 - Multi-Seed Robustness: Multi-seed verified
[PASS] A8 - Prediction Preregistration Discipline: Preregistration active
[PASS] A9 - Ablation Completeness: Ablations accounted
[PASS] A10 - Adversarial Red-Team: Red-team checklist evaluated
[PASS] A11 - Figure/Table Data Integrity: Figures traceable to parquet/jsonl
[PASS] A12 - Compute Honesty: Hardware and wall-clock fully captured
```

---

## 6. Failure Ledger
- **Total Tracked:** 1
- **Open S0/S1 Blockers:** 0
- **Closed:** 1 (`F-0001`: Test fixture isolation bug detected by Audit A3, fixed via `deregister_vault_file` and `try/finally` cleanup, verified clean).

---

## 7. Artifacts & Hashes
- Vault Manifest: `vault/manifest.json` (5 datasets: calibration, dev, t1, t2, t3)
- Hardware Noise Floor: `results/noise_floor.json`
- Environment Lockfile: `env.lock`
- Milestone Tracker: `MILESTONES.md`
