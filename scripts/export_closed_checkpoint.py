"""
scripts/export_closed_checkpoint.py
===================================
Project LEAN: Closed-Compiler Checkpoint Exporter for Qwen2.5-7B-Instruct.

Bakes the activation-calibrated Knapsack-Aware Compression (KAC) weights into
two GitHub-Release-compliant safetensor shards (~1.92 GiB each, total 3.84 GiB),
while keeping the calibration search and compiler logic private:

Outputs:
  - release/model_kac-00001-of-00002.safetensors  (Layers 00..13 + Embeddings)
  - release/model_kac-00002-of-00002.safetensors  (Layers 14..27 + Norm + LM Head)
  - release/model_kac.safetensors.index.json      (Safetensors Shard Map)
  - release/bm25_index.pkl                       (18 KB Canonical Knowledge Index)
  - release/config.json                          (Model Architecture Config)
"""

import os
import sys
import json
import time
import shutil
import pickle
from pathlib import Path
from typing import Dict, Any, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding
from accelerate.utils import set_module_tensor_to_device
from huggingface_hub import constants

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ALPHA_OPTIMAL = 0.40  # Statistically discovered optimal channel power in LEAN grand challenge
GROUP_SIZE = 128


class CalibratedPackedQuantizedLinear(nn.Module):
    """
    Activation-Aware Memory-Efficient Quantized Linear Layer.
    Applies per-channel input scaling (AWQ) and optimal MSE group clipping.
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
            
    def pack_weights(self, weight_tensor: torch.Tensor, act_scales: Optional[torch.Tensor] = None, alpha: float = ALPHA_OPTIMAL):
        device = self.packed.device
        dtype = torch.float32
        W = weight_tensor.detach().to(device=device, dtype=dtype)
        
        # 1. Activation-Aware Channel Scaling (AWQ)
        if act_scales is not None and act_scales.numel() == self.in_features:
            act_s = act_scales.to(device=device, dtype=dtype).clamp(min=1e-5)
            s = (act_s / act_s.mean()).pow(alpha).clamp(min=0.2, max=5.0)
            W_scaled = W * s.unsqueeze(0)
            self.channel_scales.copy_(s.to(dtype=torch.float16))
        else:
            W_scaled = W
            self.channel_scales.fill_(1.0)
            
        flat = W_scaled.flatten()
        numel = flat.numel()
        pad_len = self.num_groups * self.group_size
        if pad_len > numel:
            flat = F.pad(flat, (0, pad_len - numel))
        grouped = flat.view(self.num_groups, self.group_size)
        
        qmax = (2 ** (self.bits - 1)) - 1
        qmin = -(2 ** (self.bits - 1))
        max_val = torch.max(torch.abs(grouped), dim=-1, keepdim=True).values.clamp(min=1e-8)
        
        # 2. Optimal Clipping Scale Search (MSE Minimization)
        if self.bits == 2:
            ratios = torch.tensor([0.55, 0.65, 0.75, 0.85, 0.95, 1.0], device=device, dtype=dtype)
        else:
            ratios = torch.tensor([0.75, 0.85, 0.92, 0.97, 1.0], device=device, dtype=dtype)
            
        cand_scales = (max_val.unsqueeze(1) * ratios.view(1, -1, 1)) / float(qmax)  # [G, R, 1]
        grouped_exp = grouped.unsqueeze(1)  # [G, 1, S]
        
        q_cand = torch.round(grouped_exp / cand_scales).clamp(qmin, qmax)
        w_cand = q_cand * cand_scales
        mse = torch.mean((grouped_exp - w_cand) ** 2, dim=-1)  # [G, R]
        best_idx = torch.argmin(mse, dim=-1)  # [G]
        
        best_scales = cand_scales[torch.arange(self.num_groups, device=device), best_idx, 0]  # [G]
        self.scales.copy_(best_scales.to(dtype=torch.float16))
        
        # Quantize with best scale
        q = torch.round(grouped / best_scales.unsqueeze(-1)).clamp(qmin, qmax).to(torch.int32)
        q_shifted = (q - qmin).to(torch.uint8)
        flat_q = q_shifted.flatten()[:numel]
        
        # 3. Bitpacking into uint8 buffers
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


def main():
    print("=" * 80)
    print("Project LEAN: Closed-Compiler Checkpoint Exporter (Qwen2.5-7B-Instruct)")
    print(f"Device: {DEVICE.upper()} | Group Size: {GROUP_SIZE} | Optimal Alpha: {ALPHA_OPTIMAL}")
    print("Target Checkpoints: release/model_kac-00001-of-00002.safetensors")
    print("                    release/model_kac-00002-of-00002.safetensors")
    print("=" * 80)

    # 1. Locate Hugging Face Snapshot
    cache_dir = os.path.join(constants.HUGGINGFACE_HUB_CACHE, "models--Qwen--Qwen2.5-7B-Instruct", "snapshots")
    if not os.path.exists(cache_dir):
        raise FileNotFoundError(f"Cached snapshot not found at {cache_dir}")
    snapshot = os.path.join(cache_dir, os.listdir(cache_dir)[0])
    print(f"Source Snapshot: {snapshot}")

    release_dir = Path(REPO_ROOT) / "release"
    release_dir.mkdir(parents=True, exist_ok=True)

    # 2. Tokenizer & Calibration Sequences
    tok = AutoTokenizer.from_pretrained(snapshot, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    calib_texts = [
        "The capital of France is Paris. French is the official language spoken in France.",
        "Python was developed by Guido van Rossum and is widely used for data science and AI.",
        "Albert Einstein proposed the theory of special relativity in 1905, altering physics forever.",
        "In astronomy, the Moon moves in orbit around the Earth as its primary satellite.",
        "A surgeon specializes in medicine and performs complex operative procedures.",
        "The headquarters of Google is located in Mountain View, California.",
        "Water chemically predominantly consists of hydrogen and oxygen molecules.",
        "Hamlet is a celebrated tragedy written by William Shakespeare."
    ]
    enc = tok(calib_texts, return_tensors="pt", padding=True, truncation=True, max_length=64)
    calib_input_ids = enc["input_ids"].to(DEVICE)
    seq_len = calib_input_ids.shape[1]

    # 3. Read Safetensors Shard Map
    with open(os.path.join(snapshot, "model.safetensors.index.json"), "r") as f:
        weight_map = json.load(f)["weight_map"]

    def load_shard_tensor(tensor_name: str, target_dev: str = "cpu"):
        shard_file = weight_map[tensor_name]
        shard_path = os.path.join(snapshot, shard_file)
        with safe_open(shard_path, framework="pt", device=target_dev) as f:
            return f.get_tensor(tensor_name)

    # 4. Construct Skeleton on meta & patch with CalibratedPackedQuantizedLinear
    print("\n[Phase 1/3] Building Model Skeleton & Loading Non-Quantized Modules...")
    config = AutoConfig.from_pretrained(snapshot)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16)

    target_device = "cuda:0" if DEVICE == "cuda" else "cpu"

    for layer_idx, layer in enumerate(model.model.layers):
        for proj_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            old = getattr(layer.self_attn, proj_name)
            has_bias = getattr(old, "bias", None) is not None or proj_name in ["q_proj", "k_proj", "v_proj"]
            new_lin = CalibratedPackedQuantizedLinear(
                old.in_features, old.out_features, bits=4, group_size=GROUP_SIZE, bias=has_bias
            ).to(target_device)
            setattr(layer.self_attn, proj_name, new_lin)
            
        for proj_name in ["gate_proj", "up_proj", "down_proj"]:
            old = getattr(layer.mlp, proj_name)
            has_bias = getattr(old, "bias", None) is not None
            new_lin = CalibratedPackedQuantizedLinear(
                old.in_features, old.out_features, bits=2, group_size=GROUP_SIZE, bias=has_bias
            ).to(target_device)
            setattr(layer.mlp, proj_name, new_lin)

    # Materialize non-quantized global weights in FP16
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

    # Initial hidden state
    with torch.no_grad():
        H = model.model.embed_tokens(calib_input_ids).to(device=target_device, dtype=torch.float16)
        
    pos_ids = torch.arange(seq_len, device=target_device).unsqueeze(0)
    pos_embs = model.model.rotary_emb(H, pos_ids)

    # 5. Stream Calibration & Packing Layer-by-Layer
    print("\n[Phase 2/3] Streaming Layer-by-Layer Calibration & Packing (28 Layers)...")
    t_start = time.time()

    for layer_idx in range(len(model.model.layers)):
        layer = model.model.layers[layer_idx]
        
        # Load layernorms
        for norm_name in ["input_layernorm", "post_attention_layernorm"]:
            w_name = f"model.layers.{layer_idx}.{norm_name}.weight"
            w_tensor = load_shard_tensor(w_name, target_dev=target_device)
            set_module_tensor_to_device(layer, norm_name + ".weight", device=target_device, value=w_tensor.to(torch.float16))
            del w_tensor
            
        # Attention activations
        with torch.no_grad():
            norm_x = layer.input_layernorm(H)
            act_attn = torch.mean(torch.abs(norm_x), dim=(0, 1))
            
        # Attention Projections (4-bit)
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            w_name = f"model.layers.{layer_idx}.self_attn.{proj}.weight"
            w_tensor = load_shard_tensor(w_name, target_dev=target_device)
            lin_mod = getattr(layer.self_attn, proj)
            act_s = act_attn if proj in ["q_proj", "k_proj", "v_proj"] else None
            lin_mod.pack_weights(w_tensor, act_scales=act_s)
            
            b_name = f"model.layers.{layer_idx}.self_attn.{proj}.bias"
            if b_name in weight_map and lin_mod.bias is not None:
                b_tensor = load_shard_tensor(b_name, target_dev=target_device)
                lin_mod.bias.copy_(b_tensor.to(dtype=torch.float16, device=target_device))
                del b_tensor
            del w_tensor
            
        # Forward Attention
        with torch.no_grad():
            attn_out = layer.self_attn(norm_x, attention_mask=None, position_embeddings=pos_embs)[0]
            H_post = H + attn_out
            post_norm_x = layer.post_attention_layernorm(H_post)
            act_mlp = torch.mean(torch.abs(post_norm_x), dim=(0, 1))
            
        # MLP gate_proj and up_proj (2-bit)
        for proj in ["gate_proj", "up_proj"]:
            w_name = f"model.layers.{layer_idx}.mlp.{proj}.weight"
            w_tensor = load_shard_tensor(w_name, target_dev=target_device)
            lin_mod = getattr(layer.mlp, proj)
            lin_mod.pack_weights(w_tensor, act_scales=act_mlp, alpha=ALPHA_OPTIMAL)
            
            b_name = f"model.layers.{layer_idx}.mlp.{proj}.bias"
            if b_name in weight_map and lin_mod.bias is not None:
                b_tensor = load_shard_tensor(b_name, target_dev=target_device)
                lin_mod.bias.copy_(b_tensor.to(dtype=torch.float16, device=target_device))
                del b_tensor
            del w_tensor
            
        # SwiGLU activation for down_proj
        with torch.no_grad():
            gate_val = layer.mlp.gate_proj(post_norm_x)
            up_val = layer.mlp.up_proj(post_norm_x)
            swiglu_act = F.silu(gate_val) * up_val
            act_down = torch.mean(torch.abs(swiglu_act), dim=(0, 1))
            
        # MLP down_proj (2-bit)
        w_down = load_shard_tensor(f"model.layers.{layer_idx}.mlp.down_proj.weight", target_dev=target_device)
        lin_mod = getattr(layer.mlp, "down_proj")
        lin_mod.pack_weights(w_down, act_scales=act_down, alpha=ALPHA_OPTIMAL)
        
        b_name = f"model.layers.{layer_idx}.mlp.down_proj.bias"
        if b_name in weight_map and lin_mod.bias is not None:
            b_tensor = load_shard_tensor(b_name, target_dev=target_device)
            lin_mod.bias.copy_(b_tensor.to(dtype=torch.float16, device=target_device))
            del b_tensor
        del w_down
        
        # Advance H
        with torch.no_grad():
            mlp_out = layer.mlp(post_norm_x)
            H = H_post + mlp_out
            
        if (layer_idx + 1) % 4 == 0 or layer_idx == 27:
            cur_vram = torch.cuda.memory_allocated() / (1024**3) if DEVICE == "cuda" else 0.0
            print(f"  -> [Layer {layer_idx+1:02d}/28] Calibrated & Packed | VRAM: {cur_vram:.2f} GB", flush=True)

        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    del H, pos_embs, pos_ids
    calib_time = time.time() - t_start
    print(f"Calibration and packing complete in {calib_time:.1f} seconds.")

    # 6. Partition into 2 Release Shards (< 2.0 GiB each)
    print("\n[Phase 3/3] Partitioning into Release Shards and Serializing...")
    shard1_name = "model_kac-00001-of-00002.safetensors"
    shard2_name = "model_kac-00002-of-00002.safetensors"
    
    shard1_tensors: Dict[str, torch.Tensor] = {}
    shard2_tensors: Dict[str, torch.Tensor] = {}
    weight_map_out: Dict[str, str] = {}

    # Shard 1: Embeddings
    shard1_tensors["model.embed_tokens.weight"] = model.model.embed_tokens.weight.data.contiguous().to("cpu")
    weight_map_out["model.embed_tokens.weight"] = shard1_name

    # Layers 0..13 -> Shard 1, Layers 14..27 -> Shard 2
    for layer_idx, layer in enumerate(model.model.layers):
        target_shard = shard1_tensors if layer_idx < 14 else shard2_tensors
        target_name = shard1_name if layer_idx < 14 else shard2_name
        
        prefix = f"model.layers.{layer_idx}"
        for name, param in layer.named_parameters():
            k = f"{prefix}.{name}"
            target_shard[k] = param.data.contiguous().to("cpu")
            weight_map_out[k] = target_name
            
        for name, buf in layer.named_buffers():
            k = f"{prefix}.{name}"
            target_shard[k] = buf.data.contiguous().to("cpu")
            weight_map_out[k] = target_name

    # Shard 2: Norm + LM Head
    shard2_tensors["model.norm.weight"] = model.model.norm.weight.data.contiguous().to("cpu")
    weight_map_out["model.norm.weight"] = shard2_name

    shard2_tensors["lm_head.weight"] = model.lm_head.weight.data.contiguous().to("cpu")
    weight_map_out["lm_head.weight"] = shard2_name

    # Save Shards
    p1 = release_dir / shard1_name
    p2 = release_dir / shard2_name

    print(f"  Saving Shard 1: {p1}...")
    save_file(shard1_tensors, str(p1))
    s1_size = p1.stat().st_size
    print(f"  [OK] Shard 1 Saved: {s1_size:,} bytes ({s1_size / (1024**3):.2f} GiB)")

    print(f"  Saving Shard 2: {p2}...")
    save_file(shard2_tensors, str(p2))
    s2_size = p2.stat().st_size
    print(f"  [OK] Shard 2 Saved: {s2_size:,} bytes ({s2_size / (1024**3):.2f} GiB)")

    total_size = s1_size + s2_size
    print(f"  [OK] Total Compressed Checkpoint Size: {total_size:,} bytes ({total_size / (1024**3):.2f} GiB)")

    # Save Index File
    index_file = release_dir / "model_kac.safetensors.index.json"
    with open(index_file, "w") as f:
        json.dump({
            "metadata": {"total_size": total_size, "format": "kac_packed_v1", "model_id": MODEL_ID},
            "weight_map": weight_map_out
        }, f, indent=2)
    print(f"  [OK] Saved Shard Index: {index_file}")

    # Copy Standalone BM25 Index
    bm25_src = Path(REPO_ROOT) / "scratch" / "test_7b_bm25_index.pkl"
    bm25_dst = release_dir / "bm25_index.pkl"
    if bm25_src.exists():
        with open(bm25_src, "rb") as f:
            bm25_obj = pickle.load(f)
        bm25_clean = {
            "k1": getattr(bm25_obj, "k1", 1.5),
            "b": getattr(bm25_obj, "b", 0.75),
            "corpus": getattr(bm25_obj, "corpus", []),
            "doc_freqs": dict(getattr(bm25_obj, "doc_freqs", {})),
            "doc_lens": getattr(bm25_obj, "doc_lens", []),
            "avg_doc_len": getattr(bm25_obj, "avg_doc_len", 0.0),
        }
        with open(bm25_dst, "wb") as f:
            pickle.dump(bm25_clean, f)
        print(f"  [OK] Saved Standalone BM25 Index: {bm25_dst} ({bm25_dst.stat().st_size:,} bytes)")

    # Copy Model Config to release
    for cfg_file in ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"]:
        src_f = os.path.join(snapshot, cfg_file)
        if os.path.exists(src_f):
            shutil.copy(src_f, release_dir / cfg_file)
            print(f"  [OK] Copied {cfg_file} to release/")

    print("\n" + "=" * 80)
    print("SUCCESS: Closed Checkpoint Baked and Ready for Release!")
    print(f"Output Directory: {release_dir.resolve()}")
    print("=" * 80)


if __name__ == "__main__":
    main()
