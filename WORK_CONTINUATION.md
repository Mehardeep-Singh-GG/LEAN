# WORK_CONTINUATION.md
## Antigravity Handoff Guide: Project LEAN on Windows (NVIDIA RTX 4050 6GB)

> **Purpose for Antigravity AI:**  
> Read this document upon resuming or spinning up a new session on the Windows machine. It contains the exact project state, scientific context, hardware constraints, and step-by-step commands to immediately continue work without losing momentum.

---

### 1. Executive Summary & Scientific Context

- **Repository:** Project LEAN (*The Priced-Quanta Equation for Neural Network Compressibility*)
- **Core Technology:** **Knapsack-Aware Compression (KAC)** & **Thinking Core Decoupling**
  - **Hypothesis:** Neural models possess an information-theoretic asymmetry. Algorithmic reasoning circuits (Attention QKVO) require high bit precision ($4\text{ to }8\text{ bits}$), while associative factual memory (MLP projections) can be aggressively compressed ($2\text{ to }3\text{ bits}$) and offloaded to an external retrieval store (BM25 / FAISS).
  - **Verified Result on Mac M2 (`reports/MATCHED_BUDGET_report.md`):**  
    Against a **strictly storage-matched baseline** ($\Delta = 0.000\%$, exact same count of 4-bit and 3-bit weights), KAC achieves:
    - **$+8.0\%$** raw factual cloze retention on `SmolLM2-360M`.
    - **$+5.5\%$** retrieval routing fidelity on `Pythia-160M` ($92.5\%$ vs $87.0\%$).
    - **$+8.0\%$** procedural reasoning retention on `Pythia-160M` (seed 2042).
    - Serialized BM25 index takes only **$18.2\text{ KB}$** ($< 0.05\%$ of model size).
- **Audit Status:** All 12 Project LEAN audits (A1–A12) are **GREEN**, 83/83 unit tests passing. Zero synthetic mocks or defensive clamps allowed.

---

### 2. Hardware Environment: NVIDIA GeForce RTX 4050 (6 GB VRAM)

- **Target System:** Windows 11 with NVIDIA GeForce RTX 4050 Laptop GPU (6.0 GB GDDR6 VRAM, 96-bit bus, Ada Lovelace architecture).
- **Execution Environment:** WSL2 (Ubuntu 24.04 LTS) with CUDA 12.4 and PyTorch (`cu124`).
- **The Critical 6 GB VRAM Boundary:**
  - A 7B/8B model in **FP16** takes $\sim 15.2\text{ GB}$ $\to$ **OOM Crash on 6 GB**.
  - A 7B/8B model in **Uniform 4-bit (AWQ/GPTQ/Q4_K)** takes $\sim 5.3\text{ GB}$ $\to$ Barely fits, **OOM crashes when KV cache exceeds 512 tokens**.
  - A 7B/8B model under **KAC (Attn 4-bit, MLP 2-bit/3-bit)** takes **$\sim 3.6\text{ GB}$** $\to$ **Fits cleanly inside 6 GB VRAM**, leaving $>2.0\text{ GB}$ free for 4,096-token KV cache and high-speed CUDA inference!

---

### 3. Immediate Action Plan for Antigravity

When resuming on the RTX 4050 machine, execute these steps in order:

#### Step 1: Verify CUDA & VRAM
Run in the repository root:
```bash
python3 -c "import torch; print('CUDA Available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0), '| Total VRAM:', round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2), 'GB')"
```
*Expected: `CUDA Available: True | Device: NVIDIA GeForce RTX 4050 Laptop GPU | Total VRAM: 6.14 GB`.*

#### Step 2: Run Unit Tests & Audits
Ensure full codebase integrity:
```bash
PYTHONPATH=. pytest tests/
PYTHONPATH=. python src/lean/audit/runner.py
```
*All 83 tests and checks A1–A12 must pass GREEN.*

#### Step 3: Run Matched-Budget Benchmark on 3B Model (`Qwen2.5-3B-Instruct`)
```bash
PYTHONPATH=. python experiments/run_matched_budget_benchmark.py --model Qwen/Qwen2.5-3B-Instruct --device cuda --seed 1042
```
This evaluates:
1. Exact storage matching ($\Delta = 0.000\%$).
2. FP16 vs Uniform 3-bit vs Matched Uniform vs KAC.
3. Standalone factual cloze, procedural reasoning skills, and sequence loss.
4. Serialized BM25 index retrieval routing fidelity.
5. Serializes output to `results/matched_budget_Qwen2.5-3B-Instruct_s1042.json`.

#### Step 4: Run the Grand Challenge: 7B Model on 6 GB VRAM (`Qwen2.5-7B-Instruct`)
```bash
PYTHONPATH=. python experiments/run_matched_budget_benchmark.py --model Qwen/Qwen2.5-7B-Instruct --device cuda --seed 1042
```
*Objective:* Verify that KAC allows a 7B parameter model to run inference within 6 GB VRAM without OOM, maintaining attention fidelity while compressing MLPs.

#### Step 5: Document Results
Compile the findings into `reports/RTX4050_QWEN_report.md` recording:
- Peak VRAM allocated (`torch.cuda.max_memory_allocated() / (1024**3)`).
- Tokens per second throughput during prefill and generation.
- Factual recall, reasoning retention, and retrieval routing accuracy.

---

### 4. Key Repository Files & References

| File Path | Description |
|---|---|
| [`src/lean/compress/kac.py`](file:///Users/mac/llm/src/lean/compress/kac.py) | Core Knapsack-Aware Compressor (value-per-bit allocator, RTN quantizer). |
| [`src/lean/core/retrieval_store.py`](file:///Users/mac/llm/src/lean/core/retrieval_store.py) | BM25 index & Thinking Core retrieval injection pipeline. |
| [`experiments/run_matched_budget_benchmark.py`](file:///Users/mac/llm/experiments/run_matched_budget_benchmark.py) | Strict 0.000% storage-matched head-to-head benchmark runner. |
| [`reports/MATCHED_BUDGET_report.md`](file:///Users/mac/llm/reports/MATCHED_BUDGET_report.md) | Baseline M2 report comparing KAC vs Matched-Uniform on SmolLM2 and Pythia. |
| [`audit/FAILURES.md`](file:///Users/mac/llm/audit/FAILURES.md) | Official Project LEAN failure ledger (all historical bugs & resolutions). |
| [`src/lean/audit/runner.py`](file:///Users/mac/llm/src/lean/audit/runner.py) | 12-point automated audit runner (A1–A12). |

---

### 5. Non-Negotiable Scientific Guardrails

1. **Strict Storage Matching:** Never compare KAC against an unmatched lower bound. The uniform baseline must have the exact same count of 4-bit and 3-bit weights ($\Delta = 0.000\%$).
2. **Account for Retrieval Storage:** Always measure and record the physical serialized byte size of the external index on disk.
3. **Report Perplexity Honestly:** If the baseline achieves lower sequence loss on large models (as observed on SmolLM2-360M), report it prominently alongside KAC's factual and retrieval routing advantages.
4. **Zero Mocks:** No fake numbers, mock functions, or defensive clamps. All numbers must originate from real PyTorch/CUDA tensor executions.
