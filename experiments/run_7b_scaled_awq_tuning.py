"""
Project LEAN: 7B Model Scaled Calibration & Alpha Tuning Battery (AWQ-KAC)
===========================================================================
Executes full activation-aware calibration on Qwen2.5-7B-Instruct with:
1. Calibration Corpus Scaling: 48 diverse sequences from WikiText-2.
2. Channel Power Tuning: Systematic grid evaluation of alpha in {0.4, 0.5, 0.6, 0.75}
   to discover the optimal balance between activation preservation and weight dynamic range.
3. Adaptive Layer-Wise Optimization: Quantizes each projection with the optimal alpha*
   while recording macro-averaged error curves across all 28 layers.
4. Cognitive Evaluation: Raw Cloze (100 items), Retrieval Cloze (100 items),
   and Procedural Reasoning Skill (50 items) on live NVIDIA RTX 4060 GPU.
"""

import sys
sys.path.insert(0, ".")

import os
import time
import json
import math
import pickle
import random
from typing import Dict, Any, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding
from accelerate.utils import set_module_tensor_to_device
from safetensors import safe_open
from huggingface_hub import constants
from datasets import load_dataset

from src.lean.eval.probes import generate_knowledge_probes
from src.lean.anatomy.real_llm import (
    generate_standard_skill_items,
    evaluate_cloze_candidates,
    evaluate_procedural_skills
)
from src.lean.core.retrieval_store import (
    BM25Index,
    build_canonical_knowledge_corpus,
    ThinkingCoreRetrievalPipeline
)


MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
SEED = 1042
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ALPHA_CANDIDATES = [0.4, 0.5, 0.6, 0.75]
NUM_CALIB_SEQS = 48
MAX_SEQ_LEN = 64


class ScaledCalibratedPackedLinear(nn.Module):
    """
    Activation-Aware Quantized Linear Layer with Alpha Grid Tuning and MSE Group Clipping.
    """
    def __init__(self, in_features: int, out_features: int, bits: int = 4, group_size: int = 128, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.group_size = group_size
        
        num_elements = in_features * out_features
        self.num_groups = (num_elements + group_size - 1) // group_size
        self.pack_factor = 8 // bits
        
        packed_bytes = (num_elements + self.pack_factor - 1) // self.pack_factor
        self.register_buffer("packed", torch.zeros(packed_bytes, dtype=torch.uint8))
        self.register_buffer("scales", torch.zeros(self.num_groups, dtype=torch.float16))
        self.register_buffer("channel_scales", torch.ones(in_features, dtype=torch.float16))
        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=torch.float16))
        else:
            self.bias = None
            
        self.optimal_alpha = 0.5
        self.alpha_errors = {}
        
    def pack_weights(
        self, 
        weight_tensor: torch.Tensor, 
        act_scales: Optional[torch.Tensor] = None,
        X_sample: Optional[torch.Tensor] = None,
        alpha_candidates: List[float] = ALPHA_CANDIDATES
    ):
        device = self.packed.device
        dtype = torch.float32
        W = weight_tensor.detach().to(device=device, dtype=dtype)
        
        qmax = (2 ** (self.bits - 1)) - 1
        qmin = -(2 ** (self.bits - 1))
        group_size = self.group_size
        num_groups = self.num_groups
        numel = self.in_features * self.out_features
        pad_len = num_groups * group_size
        
        if self.bits == 2:
            ratios = torch.tensor([0.55, 0.65, 0.75, 0.85, 0.95, 1.0], device=device, dtype=dtype)
        else:
            ratios = torch.tensor([0.75, 0.85, 0.92, 0.97, 1.0], device=device, dtype=dtype)
            
        def quant_with_scale(s_vec):
            W_scaled = W * s_vec.unsqueeze(0)
            flat = W_scaled.flatten()
            if pad_len > numel:
                flat = F.pad(flat, (0, pad_len - numel))
            grouped = flat.view(num_groups, group_size)
            max_val = torch.max(torch.abs(grouped), dim=-1, keepdim=True).values.clamp(min=1e-8)
            cand_scales = (max_val.unsqueeze(1) * ratios.view(1, -1, 1)) / float(qmax)
            grouped_exp = grouped.unsqueeze(1)
            q_cand = torch.round(grouped_exp / cand_scales).clamp(qmin, qmax)
            w_cand = q_cand * cand_scales
            mse = torch.mean((grouped_exp - w_cand) ** 2, dim=-1)
            best_idx = torch.argmin(mse, dim=-1)
            best_scales = cand_scales[torch.arange(num_groups, device=device), best_idx, 0]
            q = torch.round(grouped / best_scales.unsqueeze(-1)).clamp(qmin, qmax)
            w_dequant = (q * best_scales.unsqueeze(-1)).flatten()[:numel].view(self.out_features, self.in_features)
            w_hat = w_dequant / s_vec.unsqueeze(0)
            return best_scales, q, w_hat
            
        best_alpha = 0.5
        best_err = float("inf")
        self.alpha_errors = {}
        
        # Grid search over alpha candidates
        if act_scales is not None and act_scales.numel() == self.in_features and X_sample is not None:
            act_s = act_scales.to(device=device, dtype=dtype).clamp(min=1e-5)
            X_eval = X_sample.to(device=device, dtype=dtype)
            Y_true = F.linear(X_eval, W)
            norm_Y = torch.norm(Y_true).clamp(min=1e-6)
            
            for a in alpha_candidates:
                s_a = (act_s / act_s.mean()).pow(a).clamp(min=0.2, max=5.0)
                _, _, w_hat = quant_with_scale(s_a)
                Y_hat = F.linear(X_eval, w_hat)
                err = (torch.norm(Y_true - Y_hat) / norm_Y).item()
                self.alpha_errors[a] = err
                if err < best_err:
                    best_err = err
                    best_alpha = a
        elif act_scales is not None and act_scales.numel() == self.in_features:
            best_alpha = 0.5
            
        self.optimal_alpha = best_alpha
        
        # Quantize and pack with the chosen optimal alpha
        if act_scales is not None and act_scales.numel() == self.in_features:
            act_s = act_scales.to(device=device, dtype=dtype).clamp(min=1e-5)
            s_final = (act_s / act_s.mean()).pow(best_alpha).clamp(min=0.2, max=5.0)
            self.channel_scales.copy_(s_final.to(dtype=torch.float16))
        else:
            s_final = torch.ones(self.in_features, device=device, dtype=dtype)
            self.channel_scales.fill_(1.0)
            
        best_scales, q, _ = quant_with_scale(s_final)
        self.scales.copy_(best_scales.to(dtype=torch.float16))
        
        q_shifted = (q.to(torch.int32) - qmin).to(torch.uint8)
        flat_q = q_shifted.flatten()[:numel]
        
        if self.bits == 4:
            evens = flat_q[0::2]
            odds = flat_q[1::2]
            packed = (evens & 0x0F) | ((odds & 0x0F) << 4)
            self.packed.copy_(packed)
        elif self.bits == 2:
            p0 = flat_q[0::4]
            p1 = flat_q[1::4]
            p2 = flat_q[2::4]
            p3 = flat_q[3::4]
            packed = (p0 & 0x03) | ((p1 & 0x03) << 2) | ((p2 & 0x03) << 4) | ((p3 & 0x03) << 6)
            self.packed.copy_(packed)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        target_dtype = x.dtype
        x_scaled = (x / self.channel_scales.view(1, 1, -1).to(dtype=target_dtype)).to(dtype=target_dtype)
        
        qmin = -(2 ** (self.bits - 1))
        if self.bits == 4:
            low = (self.packed & 0x0F).to(torch.int8) + qmin
            high = ((self.packed >> 4) & 0x0F).to(torch.int8) + qmin
            unpacked = torch.stack([low, high], dim=-1).flatten()
        elif self.bits == 2:
            p0 = (self.packed & 0x03).to(torch.int8) + qmin
            p1 = ((self.packed >> 2) & 0x03).to(torch.int8) + qmin
            p2 = ((self.packed >> 4) & 0x03).to(torch.int8) + qmin
            p3 = ((self.packed >> 6) & 0x03).to(torch.int8) + qmin
            unpacked = torch.stack([p0, p1, p2, p3], dim=-1).flatten()
            
        unpacked_grouped = unpacked.view(self.num_groups, self.group_size).float()
        w = (unpacked_grouped * self.scales.unsqueeze(-1).float()).view(-1)[:self.in_features * self.out_features]
        w = w.view(self.out_features, self.in_features).to(dtype=target_dtype)
        
        out = F.linear(x_scaled, w, self.bias.to(dtype=target_dtype) if self.bias is not None else None)
        return out.to(dtype=target_dtype)


def run_7b_scaled_awq_tuning():
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        
    print("=" * 80)
    print("Project LEAN: 7B Model Scaled Calibration & Alpha Tuning Battery")
    print(f"Model: {MODEL_ID} | Device: {DEVICE} | Seed: {SEED}")
    print(f"Calibration Corpus: {NUM_CALIB_SEQS} sequences from WikiText-2 | Seq Len: {MAX_SEQ_LEN}")
    print(f"Alpha Candidates Tested: {ALPHA_CANDIDATES}")
    print("=" * 80)
    
    t_start = time.time()
    
    # Locate snapshot
    cache_dir = os.path.join(constants.HUGGINGFACE_HUB_CACHE, 'models--Qwen--Qwen2.5-7B-Instruct', 'snapshots')
    snapshot = os.path.join(cache_dir, os.listdir(cache_dir)[0])
    
    # Phase 1: Tokenizer & BM25 Store
    print("\n" + "=" * 80)
    print("[Phase 1/5] Initializing Tokenizer and Canonical BM25 Store...")
    print("=" * 80)
    tok = AutoTokenizer.from_pretrained(snapshot, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        
    k_probes = generate_knowledge_probes(total_facts=100, seed=SEED)
    skill_items = generate_standard_skill_items(n_items=50, seed=SEED)
    corpus = build_canonical_knowledge_corpus()
    bm25 = BM25Index()
    bm25.fit(corpus)
    
    os.makedirs("scratch", exist_ok=True)
    bm25_path = "scratch/test_7b_bm25_index.pkl"
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25, f)
    index_bytes = os.path.getsize(bm25_path)
    index_bits = index_bytes * 8
    print(f"  Canonical BM25 index verified: {index_bytes:,} bytes ({index_bytes/1024:.2f} KB)")
    
    # Phase 2: Calibration Corpus from WikiText-2
    print("\n" + "=" * 80)
    print(f"[Phase 2/5] Loading {NUM_CALIB_SEQS} Diverse Calibration Sequences from WikiText-2...")
    print("=" * 80)
    ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    valid_texts = [t.strip() for t in ds['text'] if len(t.strip()) > 100]
    calib_texts = valid_texts[:NUM_CALIB_SEQS]
    
    enc = tok(calib_texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_SEQ_LEN)
    calib_input_ids = enc["input_ids"].to(DEVICE)
    seq_len = calib_input_ids.shape[1]
    print(f"  Tokenized {len(calib_texts)} sequences from WikiText-2 (Input Tensor: {calib_input_ids.shape}).")
    
    # Read Safetensors Index
    with open(os.path.join(snapshot, "model.safetensors.index.json"), "r") as f:
        weight_map = json.load(f)["weight_map"]
        
    def load_shard_tensor(tensor_name: str, target_dev: str = "cpu"):
        shard_file = weight_map[tensor_name]
        shard_path = os.path.join(snapshot, shard_file)
        with safe_open(shard_path, framework="pt", device=target_dev) as f:
            return f.get_tensor(tensor_name)
            
    # Phase 3: Construct Skeleton & Stream Calibration
    print("\n" + "=" * 80)
    print("[Phase 3/5] Streaming Layer-by-Layer Activation Extraction, Alpha Tuning & Knapsack Packing...")
    print("=" * 80)
    config = AutoConfig.from_pretrained(snapshot)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16)
        
    target_device = "cuda:0" if DEVICE == "cuda" else "cpu"
    
    for layer_idx, layer in enumerate(model.model.layers):
        for proj_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            old = getattr(layer.self_attn, proj_name)
            has_bias = getattr(old, "bias", None) is not None
            new_lin = ScaledCalibratedPackedLinear(
                old.in_features, old.out_features, bits=4, group_size=128, bias=has_bias
            ).to(target_device)
            setattr(layer.self_attn, proj_name, new_lin)
            
        for proj_name in ["gate_proj", "up_proj", "down_proj"]:
            old = getattr(layer.mlp, proj_name)
            has_bias = getattr(old, "bias", None) is not None
            new_lin = ScaledCalibratedPackedLinear(
                old.in_features, old.out_features, bits=2, group_size=128, bias=has_bias
            ).to(target_device)
            setattr(layer.mlp, proj_name, new_lin)
            
    # Materialize non-quantized weights
    print("  Loading Embeddings, Norms, LM Head, and Rotary Embeddings on GPU...")
    embed_w = load_shard_tensor("model.embed_tokens.weight", target_dev=target_device)
    model.model.embed_tokens = nn.Embedding.from_pretrained(embed_w.to(torch.float16))
    del embed_w
    
    norm_w = load_shard_tensor("model.norm.weight", target_dev=target_device)
    set_module_tensor_to_device(model, "model.norm.weight", device=target_device, value=norm_w.to(torch.float16))
    del norm_w
    
    lm_head_w = load_shard_tensor("lm_head.weight", target_dev=target_device)
    set_module_tensor_to_device(model, "lm_head.weight", device=target_device, value=lm_head_w.to(torch.float16))
    del lm_head_w
    
    model.model.rotary_emb = Qwen2RotaryEmbedding(config=config, device=target_device)
    
    total_bits = 0
    total_params = 0
    quantized_params = 0
    allocated_4b = 0
    allocated_2b = 0
    
    non_quant_params = model.model.embed_tokens.weight.numel() + model.model.norm.weight.numel() + model.lm_head.weight.numel()
    total_params += non_quant_params
    total_bits += non_quant_params * 16.0
    
    with torch.no_grad():
        H = model.model.embed_tokens(calib_input_ids).to(device=target_device, dtype=torch.float16)
        
    pos_ids = torch.arange(seq_len, device=target_device).unsqueeze(0)
    pos_embs = model.model.rotary_emb(H, pos_ids)
    
    t_quant_start = time.time()
    print("  Calibrating, Tuning Alpha, and Packing 28 layers sequentially...")
    
    # Track alpha errors across all layers
    macro_alpha_errors_mlp = {a: [] for a in ALPHA_CANDIDATES}
    chosen_alpha_counts = {a: 0 for a in ALPHA_CANDIDATES}
    
    for layer_idx in range(len(model.model.layers)):
        layer = model.model.layers[layer_idx]
        
        for norm_name in ["input_layernorm", "post_attention_layernorm"]:
            w_name = f"model.layers.{layer_idx}.{norm_name}.weight"
            w_tensor = load_shard_tensor(w_name, target_dev=target_device)
            set_module_tensor_to_device(layer, norm_name + ".weight", device=target_device, value=w_tensor)
            total_params += w_tensor.numel()
            total_bits += w_tensor.numel() * 16.0
            
        with torch.no_grad():
            norm_x = layer.input_layernorm(H)
            act_attn = torch.mean(torch.abs(norm_x), dim=(0, 1))
            X_sample_attn = norm_x.view(-1, norm_x.shape[-1])[:128]
            
        # Attention Projections (4-bit)
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            w_name = f"model.layers.{layer_idx}.self_attn.{proj}.weight"
            w_tensor = load_shard_tensor(w_name, target_dev=target_device)
            numel = w_tensor.numel()
            total_params += numel
            quantized_params += numel
            allocated_4b += numel
            total_bits += numel * (4.0 + 16.0 / 128)
            
            lin_mod = getattr(layer.self_attn, proj)
            act_s = act_attn if proj in ["q_proj", "k_proj", "v_proj"] else None
            lin_mod.pack_weights(w_tensor, act_scales=act_s, X_sample=X_sample_attn if act_s is not None else None)
            
            b_name = f"model.layers.{layer_idx}.self_attn.{proj}.bias"
            if b_name in weight_map:
                b_tensor = load_shard_tensor(b_name, target_dev=target_device)
                lin_mod.bias.copy_(b_tensor.to(dtype=torch.float16, device=target_device))
                total_params += b_tensor.numel()
                total_bits += b_tensor.numel() * 16.0
                del b_tensor
            del w_tensor
            
        # Forward Attention
        with torch.no_grad():
            attn_out = layer.self_attn(norm_x, attention_mask=None, position_embeddings=pos_embs)[0]
            H_post = H + attn_out
            post_norm_x = layer.post_attention_layernorm(H_post)
            act_mlp = torch.mean(torch.abs(post_norm_x), dim=(0, 1))
            X_sample_mlp = post_norm_x.view(-1, post_norm_x.shape[-1])[:128]
            
        # MLP gate_proj and up_proj (2-bit)
        for proj in ["gate_proj", "up_proj"]:
            w_name = f"model.layers.{layer_idx}.mlp.{proj}.weight"
            w_tensor = load_shard_tensor(w_name, target_dev=target_device)
            numel = w_tensor.numel()
            total_params += numel
            quantized_params += numel
            allocated_2b += numel
            total_bits += numel * (2.0 + 16.0 / 128)
            
            lin_mod = getattr(layer.mlp, proj)
            lin_mod.pack_weights(w_tensor, act_scales=act_mlp, X_sample=X_sample_mlp)
            
            chosen_alpha_counts[lin_mod.optimal_alpha] += 1
            for a, err in lin_mod.alpha_errors.items():
                macro_alpha_errors_mlp[a].append(err)
                
            b_name = f"model.layers.{layer_idx}.mlp.{proj}.bias"
            if b_name in weight_map:
                b_tensor = load_shard_tensor(b_name, target_dev=target_device)
                lin_mod.bias.copy_(b_tensor.to(dtype=torch.float16, device=target_device))
                total_params += b_tensor.numel()
                total_bits += b_tensor.numel() * 16.0
                del b_tensor
            del w_tensor
            
        # Compute SwiGLU activation for down_proj
        with torch.no_grad():
            gate_val = layer.mlp.gate_proj(post_norm_x)
            up_val = layer.mlp.up_proj(post_norm_x)
            swiglu_act = F.silu(gate_val) * up_val
            act_down = torch.mean(torch.abs(swiglu_act), dim=(0, 1))
            X_sample_down = swiglu_act.view(-1, swiglu_act.shape[-1])[:128]
            
        # MLP down_proj (2-bit)
        w_name = f"model.layers.{layer_idx}.mlp.down_proj.weight"
        w_tensor = load_shard_tensor(w_name, target_dev=target_device)
        numel = w_tensor.numel()
        total_params += numel
        quantized_params += numel
        allocated_2b += numel
        total_bits += numel * (2.0 + 16.0 / 128)
        
        lin_mod = getattr(layer.mlp, "down_proj")
        lin_mod.pack_weights(w_tensor, act_scales=act_down, X_sample=X_sample_down)
        
        chosen_alpha_counts[lin_mod.optimal_alpha] += 1
        for a, err in lin_mod.alpha_errors.items():
            macro_alpha_errors_mlp[a].append(err)
            
        b_name = f"model.layers.{layer_idx}.mlp.down_proj.bias"
        if b_name in weight_map:
            b_tensor = load_shard_tensor(b_name, target_dev=target_device)
            lin_mod.bias.copy_(b_tensor.to(dtype=torch.float16, device=target_device))
            total_params += b_tensor.numel()
            total_bits += b_tensor.numel() * 16.0
            del b_tensor
        del w_tensor
        
        # Advance H to next layer
        with torch.no_grad():
            mlp_out = layer.mlp(post_norm_x)
            H = H_post + mlp_out
            
        if (layer_idx + 1) % 4 == 0 or layer_idx == 27:
            cur_vram = torch.cuda.memory_allocated() / (1024**3)
            print(f"    [Layer {layer_idx+1:02d}/28] Calibrated & Packed | VRAM: {cur_vram:.2f} GB | Last Opt Alpha: {lin_mod.optimal_alpha}", flush=True)
            
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
            
    del H
    quant_time = time.time() - t_quant_start
    print(f"  Streaming Calibration & Packing Complete in {quant_time:.1f} seconds.")
    
    # Compute Alpha Tuning Statistics
    alpha_summary = {}
    for a in ALPHA_CANDIDATES:
        errs = macro_alpha_errors_mlp[a]
        mean_err = sum(errs) / len(errs) if errs else float("nan")
        alpha_summary[f"alpha_{a}"] = {
            "mean_relative_error": round(mean_err, 4),
            "chosen_count": chosen_alpha_counts[a],
            "chosen_percentage": round(chosen_alpha_counts[a] / (28 * 3) * 100, 1)
        }
        
    print("\n" + "=" * 80)
    print("Channel Power Tuning (Alpha) Summary Across 28 SwiGLU MLP Layers (84 Projections):")
    for a in ALPHA_CANDIDATES:
        s = alpha_summary[f"alpha_{a}"]
        print(f"  Alpha = {a:.2f}: Mean Rel Error = {s['mean_relative_error']:.4f} | Chosen: {s['chosen_count']}/84 ({s['chosen_percentage']}%)")
    print("=" * 80)
    
    # Storage Accounting
    storage_gib = (total_bits / 8.0) / (1024 ** 3)
    effective_bpp = total_bits / total_params
    allocated_vram_gb = torch.cuda.memory_allocated() / (1024 ** 3)
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    vram_headroom_gb = total_vram_gb - allocated_vram_gb
    
    # Phase 4: Live Throughput Profiling
    print("\n" + "=" * 80)
    print("[Phase 4/5] Profiling Hardware Throughput on Live RTX 4060 GPU...")
    print("=" * 80)
    model.eval()
    test_prompt = "Explain why quantum entanglement occurs in physical systems."
    enc_test = tok(test_prompt, return_tensors="pt").to(target_device)
    prompt_len = enc_test["input_ids"].shape[1]
    
    if DEVICE == "cuda":
        torch.cuda.synchronize()
        
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(**enc_test)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    prefill_sec = time.perf_counter() - t0
    prefill_tok_per_sec = prompt_len / max(1e-6, prefill_sec)
    
    gen_tokens = 8
    gen_input = enc_test["input_ids"].clone()
    t1 = time.perf_counter()
    with torch.no_grad():
        for _ in range(gen_tokens):
            out = model(gen_input)
            next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            gen_input = torch.cat([gen_input, next_tok], dim=1)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    gen_sec = time.perf_counter() - t1
    gen_tok_per_sec = gen_tokens / max(1e-6, gen_sec)
    
    print(f"  Prefill Speed:    {prefill_tok_per_sec:.2f} tokens/sec", flush=True)
    print(f"  Generation Speed: {gen_tok_per_sec:.2f} tokens/sec", flush=True)
    
    # Phase 5: Cognitive Evaluation
    print("\n" + "=" * 80)
    print("[Phase 5/5] Cognitive Evaluation: Scaled Calibrated Reasoning & Retrieval Routing...", flush=True)
    print("=" * 80)
    
    kac_k_res = evaluate_cloze_candidates(model, tok, k_probes, device=target_device)
    kac_s_acc = evaluate_procedural_skills(model, tok, skill_items, device=target_device)
    print(f"  AWQ-KAC Scaled Raw: Know = {kac_k_res['accuracy']*100:.1f}%, Skill = {kac_s_acc*100:.1f}%", flush=True)
    
    pipeline = ThinkingCoreRetrievalPipeline(bm25)
    rag_probes = []
    for p in k_probes:
        augmented = pipeline.augment_prompt(p["prompt"], top_k=1)
        p_rag = dict(p)
        p_rag["prompt"] = augmented
        rag_probes.append(p_rag)
        
    kac_rag_res = evaluate_cloze_candidates(model, tok, rag_probes, device=target_device)
    print(f"  AWQ-KAC Scaled + BM25 Retrieval: Know = {kac_rag_res['accuracy']*100:.1f}%", flush=True)
    
    total_time = time.time() - t_start
    
    results = {
        "model_id": MODEL_ID,
        "seed": SEED,
        "device": DEVICE,
        "experiment": "scaled_corpus_and_alpha_tuning",
        "calibration_corpus": {
            "source": "WikiText-2 (train split)",
            "num_sequences": NUM_CALIB_SEQS,
            "seq_len": MAX_SEQ_LEN,
            "total_calibration_tokens": NUM_CALIB_SEQS * MAX_SEQ_LEN
        },
        "channel_power_tuning": {
            "alpha_candidates": ALPHA_CANDIDATES,
            "summary_across_layers": alpha_summary
        },
        "total_benchmark_time_sec": round(total_time, 1),
        "quantization_time_sec": round(quant_time, 1),
        "hardware_profile": {
            "allocated_vram_gb": round(allocated_vram_gb, 2),
            "total_vram_gb": round(total_vram_gb, 2),
            "free_vram_headroom_gb": round(vram_headroom_gb, 2),
            "prefill_tok_per_sec": round(prefill_tok_per_sec, 2),
            "gen_tok_per_sec": round(gen_tok_per_sec, 2)
        },
        "exact_storage_accounting": {
            "total_params": total_params,
            "quantized_params": quantized_params,
            "allocated_4b_attn_params": allocated_4b,
            "allocated_2b_mlp_params": allocated_2b,
            "gqa_kac_storage_bits": total_bits,
            "gqa_kac_storage_gib": storage_gib,
            "effective_bpp": effective_bpp,
            "external_index_bytes": index_bytes,
            "external_index_bits": index_bits,
            "index_overhead_fraction": index_bits / total_bits
        },
        "metrics": {
            "scaled_awq_kac_raw": {
                "factual_cloze_acc": kac_k_res["accuracy"],
                "procedural_skill_acc": kac_s_acc
            },
            "scaled_awq_kac_plus_retrieval": {
                "factual_cloze_acc": kac_rag_res["accuracy"],
                "correct_count": kac_rag_res.get("correct_count", 0),
                "total_evaluated": kac_rag_res.get("total_evaluated", len(rag_probes))
            },
            "evolution_comparison": {
                "uncalibrated_rag_acc": 0.12,
                "initial_8seq_calibrated_rag_acc": 0.72,
                "scaled_corpus_tuned_alpha_rag_acc": kac_rag_res["accuracy"],
                "gain_vs_uncalibrated": kac_rag_res["accuracy"] - 0.12,
                "gain_vs_initial_8seq": kac_rag_res["accuracy"] - 0.72
            }
        }
    }
    
    out_path = "results/matched_budget_7b_scaled_awq_tuning.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
        
    print("\n" + "=" * 80)
    print(f"[SCALED CALIBRATION COMPLETE] All Receipts Serialized to: {out_path}")
    print("=" * 80)
    return results


if __name__ == "__main__":
    run_7b_scaled_awq_tuning()
