# WORK_CONTINUATION.md
## Antigravity Handoff Guide: Universal Half-Life Asymmetry Testing on NVIDIA RTX

> **ATTENTION ANTIGRAVITY AI ON RTX MACHINE:**  
> Read this document immediately upon spinning up or resuming on the RTX GPU machine. It contains the exact context, mathematical scrutiny, hardware runbook, and non-negotiable scientific guardrails to validate the **Universal Half-Life Asymmetry Theorem** across models from 70M to 14B.

---

### 1. Context & The Pre-Crash Event
- **Repository:** `Project LEAN` (*The Priced-Quanta Equation for Neural Network Compressibility*)
- **Core Subject:** Testing whether the **Universal Half-Life Asymmetry Theorem** ($\Delta h_{1/2} > 0$) strictly holds across pre-trained model weights.
- **The Crash:** The evaluation script on Mac M2 attempted an unbatched float32 allocation of `(2000, 2048, 512)` ($\sim 8.4\text{ GB}$), instantly exhausting the 8GB RAM.
- **The Fix:** [`experiments/test_universal_half_life_claims.py`](file:///Users/mac/llm/experiments/test_universal_half_life_claims.py) is now fully patched with memory-safe mini-batching (`--batch-size 50`), CUDA acceleration (`--device cuda`), and multi-tier model support.

---

### 2. The Core Theoretical Audit: The Minus Sign & Scaling Jump
In [`paper/MANUSCRIPT_EXTENDED.md`](file:///Users/mac/llm/paper/MANUSCRIPT_EXTENDED.md#L265-L320), Theorem 1 derives:
$$h_{1/2}(\text{proc}) = -\frac{1}{2}\log_2(d_k) + \text{offset}_1$$
$$h_{1/2}(\text{fact}) = +\frac{1}{2}\log_2(d_{\text{mlp}}) + \text{offset}_2$$
Subtracting them gives:
$$\Delta h_{1/2} = h_{1/2}(\text{fact}) - h_{1/2}(\text{proc}) = \frac{1}{2}\log_2(d_{\text{mlp}}) - \left( -\frac{1}{2}\log_2(d_k) \right) + \Delta \text{offset} = \mathbf{\frac{1}{2}\log_2(d_{\text{mlp}} \cdot d_k) + \Delta \text{offset}}$$
Notice that the manuscript currently writes $\mathbf{\frac{1}{2}\log_2(d_{\text{mlp}} / d_k)}$ (a ratio rather than a product).
**Your Theoretical Mission on RTX:** Determine empirically and analytically whether the half-life gap scales with the product $d_{\text{mlp}} \cdot d_k$ (which would make attention even more dramatically robust as $d_k$ grows) or the ratio $d_{\text{mlp}} / d_k$, or if $K_{\text{attn}}$ has an additional $1/d_k$ normalization in real transformer attention layers.

---

### 3. Model Testing Matrix on RTX

Execute evaluations across the three parameter tiers:

- **Tier 1 (Small / Edge):** `pythia70m`, `pythia160m`, `gpt2`, `smollm2_360m`, `pythia410m`
- **Tier 2 (Mid-Scale):** `pythia1.4b`, `qwen2.5_1.5b`, `llama3.2_1b`, `llama3.2_3b`, `qwen2.5_3b`, `pythia2.8b`
- **Tier 3 (Grand Challenge):** `qwen2.5_7b`, `llama3_8b`, `mistral7b`, `qwen2.5_14b`

---

### 4. Step-by-Step Command Runbook for RTX

#### Step 1: Verify Hardware & CUDA Environment
```bash
python3 -c "
import torch
print('CUDA Available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('Device:', torch.cuda.get_device_name(0), '| Total VRAM:', round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2), 'GB')
"
```

#### Step 2: Run Unit Tests & Audit Suite
```bash
pytest tests/
python src/lean/audit/runner.py
```
*All 83 unit tests and audits A1–A12 must be GREEN.*

#### Step 3: Run Circuit-Level Analysis (Prong A)

##### Tier 1 (Lightweight Checkpoints):
```bash
python experiments/test_universal_half_life_claims.py \
    --tier tier1 \
    --device cuda \
    --dtype float32 \
    --n-trials 5000 \
    --batch-size 50 \
    --allow-download \
    --output results/universal_half_life_tier1.json
```

##### Tier 2 (Mid-Scale 1B–3B Checkpoints):
```bash
python experiments/test_universal_half_life_claims.py \
    --tier tier2 \
    --device cuda \
    --dtype float16 \
    --n-trials 5000 \
    --batch-size 50 \
    --allow-download \
    --output results/universal_half_life_tier2.json
```

##### Tier 3 (Grand Challenge 7B–14B Checkpoints):
*Use `--dtype float16 --batch-size 25` if on 6GB–8GB VRAM; use `--batch-size 50` on $\ge 16\text{GB}$.*
```bash
python experiments/test_universal_half_life_claims.py \
    --models qwen2.5_7b llama3_8b mistral7b \
    --device cuda \
    --dtype float16 \
    --n-trials 2000 \
    --batch-size 25 \
    --allow-download \
    --output results/universal_half_life_tier3.json
```

#### Step 4: Run Real Task Benchmarks (Prong B)
Evaluate full forward passes under progressive quantization on real tasks (factual cloze vs procedural reasoning):
```bash
python experiments/run_matched_budget_benchmark.py --model Qwen/Qwen2.5-3B-Instruct --device cuda --seed 1042
python experiments/run_matched_budget_benchmark.py --model Qwen/Qwen2.5-7B-Instruct --device cuda --seed 1042
```

#### Step 5: Synthesize Results & Update Archival Documents
1. Check whether $\Delta h_{1/2} > 0$ holds across all evaluated models.
2. If any model violates the condition, record it immediately in `audit/FAILURES.md` as an S1 failure.
3. Update Table 2B in [`paper/MANUSCRIPT_EXTENDED.md`](file:///Users/mac/llm/paper/MANUSCRIPT_EXTENDED.md#L339) with the real empirical numbers.

---

### 5. Non-Negotiable Scientific Guardrails
1. **Zero Defensive Clamps:** Never use `max()`, `min()`, or hardcoded fallbacks to force $\Delta h_{1/2} > 0$. If an empirical number is negative or violates the theorem, document it honestly.
2. **Strict Storage Parity:** In Prong B, all comparisons must be strictly storage-matched ($\Delta = 0.000\%$).
3. **Retrieval Accounting:** Always include the physical on-disk size of the BM25 external retrieval index ($18.2\text{ KB}$).
4. **VRAM Hygiene:** Wrap large model forward passes in `torch.no_grad()` and invoke `torch.cuda.empty_cache()` between iterations.
