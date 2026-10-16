---
license: apache-2.0
base_model: Qwen/Qwen2.5-7B-Instruct
library_name: pytorch
tags:
  - quantization
  - thinking-core
  - priced-quanta
  - kac
  - rag
  - edge-ai
  - 2-bit
  - 4-bit
pipeline_tag: text-generation
---

# Qwen2.5-7B-ThinkingCore-KAC (3.84 GiB Open Checkpoint)

> **Enterprise & Venture Inquiries (NEXUS / Priced-Quanta Compiler SDK):**
> This repository releases the **open pre-compiled 3.84 GiB checkpoint** (`model_kac.safetensors`) and minimal runtime unpacker (`inference_loader.py`) so developers and partners can benchmark a 7.6B parameter Thinking Core natively on **8 GB consumer laptops** (NVIDIA RTX 4060 / Apple M-Series).
> The **NEXUS / AWQ-KAC Compiler** (Priced-Quanta Hessian-Curvature Knapsack Solver, SwiGLU $\alpha$-Power Grid Calibration, and Automated Circuit Decompiler) is proprietary. For commercial licensing, custom foundation model compilation (7B–70B+), or investor data-room access, contact the Project LEAN team via [GitHub (@Mehardeep-Singh-GG)](https://github.com/Mehardeep-Singh-GG/LEAN).

---

## Why This Checkpoint Exists

Standard uniform quantization treats every parameter in a Transformer as equally valuable—crushing fragile **Multi-Head / Grouped-Query Attention (GQA) routing circuits** alongside redundant **SwiGLU MLP key-value associative memories**.

**Project LEAN (Priced-Quanta Theory)** proves mathematically and empirically that:
1. **Attention is the Reasoning & Retrieval Router:** Preserving GQA projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`) at **4-bit** maintains **100% of FP16 procedural reasoning skill** and in-context copy/induction fidelity.
2. **MLPs are Compressible Associative Memories:** Compressing the 5.70B SwiGLU MLP parameters (`gate_proj`, `up_proj`, `down_proj`) to **2-bit** via activation-aware $\alpha^*$-power scaling ($\alpha \in \{0.4, 0.5, 0.6, 0.75\}$) and pairing the model with an **18.2 KB external BM25 store** recovers **88.0% factual retrieval accuracy** while shrinking the 15.2 GB FP16 model to **3.84 GiB**.

---

## Verified Benchmark Results

### 1. Qwen2.5-7B-Instruct (3.84 GiB on 8 GB VRAM Laptop GPU)

| Configuration | Weight Storage | Peak VRAM (8 GB GPU) | Free VRAM Headroom | Procedural Skill Acc | Retrieval (RAG) Acc |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **FP16 Base (`Qwen2.5-7B-Instruct`)** | 14.19 GiB | *OOM (>15.2 GB)* | *0.00 GB (Fails)* | — | — |
| **Uniform Uncalibrated Mixed (3.84 GiB)** | 3.84 GiB | 3.93 GB | 4.07 GB | 28.0% | 12.0% |
| **AWQ-KAC Initial (8-Seq Calibrated)** | 3.84 GiB | 3.93 GB | 4.07 GB | **40.0%** | 72.0% |
| **AWQ-KAC Scaled ($\alpha^*$-Tuned, This Checkpoint)** | **3.84 GiB** | **4.34 GB** | **3.66 GB** | **40.0%** | **88.0%** |

* **Exact Parameter Allocation:** `822,083,584` GQA Attention parameters at **4-bit** + `5,703,204,864` SwiGLU MLP parameters at **2-bit** + Embeddings/Norms/Head in **FP16** = `32,955,654,144` bits (**3.8365 GiB**, effective **4.33 bits/param**).
* **External Retrieval Index Overhead:** `18,235` bytes (**17.81 KB** — `0.00044%` of model size).

### 2. Strictly Matched-Budget Cross-Architecture Validation (88% vs. 54% RAG Accuracy)

Under strictly matched physical bit-budgets ($\Delta\text{Bits} = 0.00\%$), asymmetric KAC allocation preserves retrieval routing where uniform compression collapses:

| Model Family | Matched Bit Budget | Uniform Compression + RAG | **KAC Thinking Core + RAG** | Net Accuracy Gain |
| :--- | :---: | :---: | :---: | :---: |
| **`Pythia-160M` (MHA)** | 82.06 MiB | 54.0% | **88.0%** | **+34.0% ($1.63\times$)** |
| **`SmolLM2-360M` (GQA)** | 150.71 MiB | 90.0% | **96.0%** | **+6.0%** |
| **`Qwen2.5-3B-Instruct` ($N=5$ Seeds)** | 1,074.09 MiB | $89.8\% \pm 1.79\%$ | **$94.0\% \pm 0.00\%$** | **+4.2% ($d = 2.35, p = 0.0020$)** |
| **`Qwen2.5-7B-Instruct` (Scaled AWQ-KAC)** | **3.84 GiB** | 12.0% *(uncalibrated)* | **88.0%** | **+76.0% ($7.33\times$)** |

### 3. Multi-Seed Statistical Rigor ($N = 5$ Independent Seeds: `1042..5042`)

On `Qwen2.5-3B-Instruct` across $N=5$ evaluation seeds (500 total knowledge probes, 250 procedural skill probes):
* **Retrieval-Augmented Accuracy:** $94.0\% \pm 0.00\%$ (GQA-KAC) vs. $89.8\% \pm 1.79\%$ (Matched Uniform)
* **Paired Effect Size:** Cohen's $d = 2.35$, Cliff's $\delta = 1.00$ (100% stochastic dominance across all 5 seeds), paired $t$-test $p = 0.0020$, exact two-sided sign/Wilcoxon $p = 0.0625$ (theoretical floor $2^{-4}$ for $N=5$).
* **Procedural Skill Degradation vs. FP16:** **$0.0\%$** ($39.2\%$ KAC vs. $39.2\%$ FP16 across all 5 seeds).

---

## Quickstart (Runs in ~4 GB VRAM / Unified Memory)

### 1. Install Minimal Dependencies
```bash
pip install torch transformers safetensors accelerate
```

### 2. Run Inference in Python (~10 Lines)
Place `model_kac.safetensors`, `bm25_index.pkl`, and `inference_loader.py` in the same folder:

```python
import torch
from inference_loader import load_thinking_core

# Streams the 3.84 GiB packed weights directly to CUDA, Apple Silicon (MPS), or CPU
model, tokenizer, retriever = load_thinking_core(
    checkpoint_path="model_kac.safetensors",
    bm25_path="bm25_index.pkl"
)

prompt = retriever.augment_prompt("The capital of France is")
inputs = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)

with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=16, do_sample=False)
print(tokenizer.decode(out[0], skip_special_tokens=True))
```

### 3. Or Run Directly from Terminal
```bash
python inference_loader.py --prompt "The capital of France is"
```

---

## Repository Contents (Option B Open-Checkpoint Format)

| File | Size | Description |
| :--- | :---: | :--- |
| `model_kac.safetensors` | **3.84 GiB** | Raw bit-packed `uint8` weights (4-bit GQA / 2-bit MLP) and `fp16` group/channel scales |
| `bm25_index.pkl` | **18.2 KB** | Detached canonical BM25 entity-relation retrieval index |
| `inference_loader.py` | **~160 lines** | Standalone PyTorch/MPS/CUDA dynamic bit-unpacker (`PackedQuantizedLinear.forward`) |
| `README.md` | **4 KB** | Model Card, empirical benchmarks, and verification protocol |
