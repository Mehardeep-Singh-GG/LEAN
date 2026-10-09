# Comprehensive Scientific Re-Audit: Milestones M9 and M10

**Audit Date:** 2026-10-09  
**Auditor:** Adversarial Scientific Review Board & Project LEAN Lead Investigator  
**Scope:** Milestone M9 (Assemble & Lock Equation v1) and Milestone M10 (Held-Out Validation Tranche T1)  
**Governing Standard:** Project LEAN Scientific Constitution (Audits A1–A12, Failure Protocol L1–L4)  
**Status:** 🟡 **CRITICAL AUDIT COMPLETE — FINDINGS ACTIONED & LEDGER CLOSED**  

---

## 1. Executive Summary & Audit Mandate

A rigorous scientific critique challenged three fundamental claims presented in the Milestone M9 and M10 reports:
1. **The $4.11\times$ Half-Life Ratio:** Whether the reported empirical accuracy ladder supports the claimed factual-vs-skill half-life ratio.
2. **The M9 Model Comparison:** Whether the claimed superiority over Chinchilla and Kumar et al. models was evaluated on real neural compression or knapsack simulations.
3. **The Blind-Test Chronology:** Whether accessing `vault/locked/t1_test.txt` prior to locking `prereg/LOCK_v1.md` compromised the held-out status of Tranche T1.
4. **Prediction Interval Coverage Calibration:** Whether 5 out of 6 successes on nominal 95% intervals is statistically decisive, and whether P_LOCK_5 can be dismissed as finite-sample noise.

This re-audit was conducted under **zero tolerance for scientific self-deception**. Every calculation was independently re-derived from raw measurements, git commit histories were verified, and the failure ledger was updated.

---

## 2. Issue 1: The Asymmetric Half-Life Ratio Defect (Failure `F-0008`)

### 2.1 The Mathematical Problem
The M10 report claimed:
- Factual memory half-life: $B_{1/2}^{\text{fact}} = 1,126.83\text{ Mbits}$
- Procedural skill half-life: $B_{1/2}^{\text{skill}} = 273.88\text{ Mbits}$
- Ratio: $\frac{1,126.83}{273.88} = \mathbf{4.11\times}$

However, the raw accuracy measurements on the 5-rung ladder for `EleutherAI/pythia-70m` were:
| Quantization Rung | Total Physical Bits | Knowledge Accuracy ($A_0 = 46.0\%$) | Skill Accuracy ($A_0 = 48.0\%$) |
|---|:---:|:---:|:---:|
| **FP16** | 1,126.83 Mbits | 46.00% | 48.00% |
| **8-bit RTN** | 563.41 Mbits | 47.00% | 48.00% |
| **4-bit RTN** | 281.71 Mbits | 44.00% | 52.00% |
| **3-bit RTN** | 211.28 Mbits | 21.00% | 34.00% |
| **2-bit RTN** | 140.85 Mbits | 18.00% | 34.00% |

Under the standard definition of capacity half-life:
$$A(B_{1/2}) = 0.50 \cdot A_0$$
- Knowledge 50% target: $0.50 \times 46.0\% = \mathbf{23.00\%}$.
- Skill 50% target: $0.50 \times 48.0\% = \mathbf{24.00\%}$.

From the raw table:
- Knowledge drops below $23\%$ between 4-bit ($44\%$) and 3-bit ($21\%$).
- Skills **never drop below $24\%$ anywhere on the ladder** (reaching $34\%$ at 2-bit).

### 2.2 Root Cause Analysis (Five Whys)
1. **Why was $4.11\times$ reported?** Because $B_{1/2}^{\text{fact}}$ evaluated to $1,126.83\text{ Mb}$ and $B_{1/2}^{\text{skill}}$ evaluated to $273.88\text{ Mb}$.
2. **Why did $B_{1/2}^{\text{fact}}$ evaluate to $1,126.83\text{ Mb}$?** Because `compute_half_life()` evaluated absolute $0.50$ accuracy threshold, not relative $50\%$ baseline retention ($0.50 \cdot A_0$). Since knowledge accuracy never exceeded $47\%$, `np.all(acc < 0.50)` defaulted to the highest budget ($1,126.83\text{ Mb}$, FP16).
3. **Why did $B_{1/2}^{\text{skill}}$ evaluate to $273.88\text{ Mb}$?** Because skill accuracy temporarily reached $52\%$ at 4-bit ($281.71\text{ Mb}$) and dropped to $48\%$ at 8-bit, crossing absolute $0.50$ at $273.88\text{ Mb}$.
4. **Why did the code use absolute $0.50$?** It was written for synthetic universes (Milestone M3/M7) where uncompressed baseline accuracy was exactly $1.00$ ($100\%$), making absolute $0.50$ identical to relative $50\%$.
5. **Why was this not caught earlier?** In Milestone M8, the same function produced $1,126.83 / 275.84 = 4.09\times$, which was accepted without verifying the relative crossing points, and then frozen into `prereg/LOCK_v1.md` as prediction P_LOCK_3 ($4.10\times$).

### 2.3 The Task Chance-Floor Barrier
Why did procedural skills never cross $24.0\%$?
The procedural skill probe is an in-context synthetic mapping task:
`Map: k1->v1, k2->v2, k3->v3. Query: q ->`
The candidate set evaluated by logit scoring consists of $[v1, v2, v3]$—exactly 3 choices.
The random guessing expectation on 3 choices is:
$$C_{\text{skill}} = \frac{1}{3} \approx 33.33\%$$
Because the uncompressed accuracy was $48.0\%$, the $50\%$ relative target is:
$$0.50 \times 48.0\% = 24.0\% < 33.33\%$$
**A model degrading purely into noise can never fall below the chance floor of $33.33\%$.** Thus, $24.0\%$ was mathematically unobservable under random noise!

### 2.4 Mathematical Resolution & Re-Derivations

1. **Direct Empirical Crossing (Factual Memory):**
   $$B_{1/2}^{\text{fact}} = 211.28 + \frac{0.230 - 0.210}{0.440 - 0.210} \times (281.71 - 211.28) = \mathbf{217.40\text{ Mbits}}$$

2. **Direct Empirical Ladder Bound (Skills):**
   Because skill accuracy remains at $34.0\% > 24.0\%$ at 2-bit ($140.85\text{ Mbits}$):
   $$B_{1/2}^{\text{skill}} < 140.85\text{ Mbits} \implies \text{Ratio} = \frac{B_{1/2}^{\text{fact}}}{B_{1/2}^{\text{skill}}} > \frac{217.40}{140.85} = \mathbf{1.54\times}$$

3. **Linear Extrapolation past Chance Floor:**
   Extrapolating the observed decay from 4-bit ($52\%$) to 2-bit ($34\%$) down to $24\%$:
   $$\text{Slope} = \frac{0.34 - 0.52}{140.85 - 281.71} = 0.001278\text{ acc/Mbit}$$
   $$B_{1/2}^{\text{skill, extrap}} = 140.85 - \frac{0.34 - 0.24}{0.001278} = 140.85 - 78.25 = \mathbf{62.60\text{ Mbits}}$$
   $$\text{Ratio}_{\text{extrap}} = \frac{217.40}{62.60} = \mathbf{3.47\times}$$

4. **1-Bit Extended Quantization Ladder:**
   Evaluating 1-bit RTN ($70.42\text{ Mbits}$), skill accuracy remains at $40.0\% > 24.0\%$:
   $$B_{1/2}^{\text{skill}} < 70.42\text{ Mbits} \implies \text{Ratio} > \frac{217.40}{70.42} = \mathbf{3.09\times \ge 2.0\times}$$

5. **Chance-Corrected Excess Retention Metric:**
   Evaluating retention over task chance floors:
   $$R(B) = \frac{A(B) - C}{A_0 - C} = 0.50 \implies A(B) = C + 0.50(A_0 - C)$$
   - Skills ($C = 0.3333$, $A_0 = 0.48$): Target $= 0.4067 \implies B_{1/2} = \mathbf{237.37\text{ Mbits}}$
   - Factuals ($C = 0.1000$, $A_0 = 0.46$): Target $= 0.2800 \implies B_{1/2} = \mathbf{232.71\text{ Mbits}}$
   - Ratio: $\frac{232.71}{237.37} = \mathbf{0.98\times}$ (No asymmetry under chance-corrected metric on this probe battery!).

**Conclusion on Issue 1:** The claimed $4.11\times$ ratio was a reporting and calculation error (logged as `F-0008`). While skills qualitatively exhibit superior compression resilience (surviving down to 2-bit and 1-bit above 34%), the quantitative ratio depends heavily on extrapolation and task chance floors. The half-life law is **not quantitatively validated** as a constant $4.11\times$ law.

---

## 3. Issue 2: Audit of Milestone M9 Model Comparison

### 3.1 Critique
M9 claimed that LEAN beat Chinchilla ($\Delta\text{AIC} = +18.18$) and Kumar et al. ($\Delta\text{AIC} = +18.96$) with 5.50% MAPE. However, the execution log shows that excess-loss curves were constructed from factual token frequencies and `greedy_knapsack` simulations.

### 3.2 Audit Findings
1. **Evaluation Target:** The M9 model comparison evaluated **scaling exponent transfer across data distributions** ($\gamma \equiv \alpha$), comparing:
   - Model A (LEAN): Exponent equals corpus Zipf exponent ($\gamma = \alpha$).
   - Model B (Chinchilla-style): Single constant power law exponent across all distributions.
   - Model C (Kumar et al.): Fixed precision scaling exponent.
   - Model D (Frequency-only): Constant cutoff ($\gamma = 1.0$).
2. **Data Scope:** Evaluated across The Pile ($\alpha = 0.4158$) and three synthetic universes ($\alpha = 0.30, 0.60, 1.00$).
3. **Clarification:** This comparison demonstrates that within discrete knapsack capacity allocation, the scaling exponent is governed by the underlying data Zipf distribution. It does **not** prove that LEAN predicts continuous cross-entropy loss trajectories on trained neural networks better than empirical scaling laws.
4. **Action Taken:** `reports/M9_report.md` Section 3.1 was updated with prominent disclosure of this scope.

---

## 4. Issue 3: Blind-Test Chronology and Provenance

### 4.1 Chronology Audit
- `vault/locked/t1_test.txt` was created on **2026-10-09 17:41:11 UTC** (Commit `cbd3590`, Milestone M0).
- `prereg/LOCK_v1.md` was created on **2026-10-09 21:31:24 UTC** (Commit `26853e2`, Milestone M9).
- File contents of `vault/locked/t1_test.txt`:
  ```
  Held-out test tranche 1 evaluation split: Qwen2.5 and SmolLM2 benchmark texts.
  This text is cryptographically protected by the Project LEAN vault guard.
  Premature evaluation before formal preregistration is strictly prohibited.
  ```
- **Zero Label Leakage:** The file contains exactly 228 bytes of lock sentinel text. No test questions, facts, answers, or model evaluations were present.
- **Provenance Qualification:** Actual evaluation probes were generated at runtime via `generate_knowledge_probes(..., seed=1042)`. While seed 1042 was strictly disjoint from development seeds (seed 42), T1 is a **synthetic pseudo-random held-out split**, not an independently held benchmark.

---

## 5. Issue 4: Prediction Calibration and P_LOCK_5 Falsification

### 5.1 Falsification of P_LOCK_5
- Registered Point Prediction: $0.020$ [95% PI: $0.000, 0.040$]
- Registered Falsification Boundary: $\Delta_{\text{skill}} > 0.050$
- Observed Outcome: $\Delta_{\text{skill}} = \mathbf{0.060}$
- **Verdict:** **CONFIRMED EMPIRICAL FALSIFICATION** (Logged as `F-0007`).
- **Scientific Accounting:** The initial report attempted to excuse this as finite-sample discretization ($N=50$, 1 item = $0.02$). This re-audit rejects that excuse. When a preregistered prediction lands outside its 95% PI and crosses its explicit falsification boundary, it must be recorded as an empirical failure. Factual associative weights in MLP down-projections are not completely orthogonal to procedural skills at this probe scale.

### 5.2 Statistical Significance of 5/6 Successes
With $N=6$ predictions:
- Observing $K=5$ successes on nominal 95% intervals:
  $$P(K \le 5 \mid p=0.95) = 1 - 0.95^6 = 26.5\%$$
- Exact Clopper-Pearson 95% confidence interval for $5/6 = 83.33\%$:
  $$\mathbf{[35.9\%, 99.6\%]}$$
- An empirical success rate of 5/6 on 6 trials does **not** provide statistically decisive evidence that the 95% prediction intervals are calibrated.

---

## 6. Scientific Failure Ledger Entries

| Failure ID | Milestone | Severity | Category | Status | Summary & Resolution |
|---|:---:|:---:|:---:|:---:|---|
| `F-0007` | M10 | **S2** | `CLOSED` | F-THEORY | P_LOCK_5 observed $\Delta = 0.060$ vs PI $[0.000, 0.040]$ and boundary $> 0.050$. Retained as formal prediction failure. Probe size recommendation $N \ge 200$ for future validation. |
| `F-0008` | M10 | **S1** | `CLOSED` | F-MEAS | `compute_half_life` evaluated absolute $0.50$ accuracy, producing spurious $4.11\times$ ratio. Refactored to relative retention and chance-corrected metrics. Point estimate retracted. |

---

## 7. Action Plan & Milestone Governance

1. **Milestone M11 (Paper Draft) Is Formally Placed On Hold.**
   No claims of a "universal law of neural compression" will be submitted to peer review until the half-life ratio is tested on tasks with high candidate cardinality (where $0.50 \cdot A_0 > C_{\text{chance}}$).
2. **Preregistration of Milestone M10B (Multi-Model Feasible Battery):**
   Evaluate cross-family scaling on `pythia-160m`, `pythia-410m`, `SmolLM2-360M`, `Qwen2.5-0.5B`, and `gpt2` with:
   - Expanded probe batteries ($N \ge 200$)
   - Chance-corrected retention metrics
   - 1-bit and fractional-bit extended ladders
3. **All Audit Gates Re-Verified:**
   - 79/79 automated pytest tests passing.
   - Audits A1–A12 green under revised criteria.
