"""
Project LEAN — Private Compiler to Open-Checkpoint Exporter (Option B)
======================================================================
Runs the proprietary AWQ-KAC streaming compiler on Qwen/Qwen2.5-7B-Instruct
and serializes ONLY the final packed buffers (`packed`, `scales`, `channel_scales`)
to `release/model_kac.safetensors` (3.84 GiB) + `release/bm25_index.pkl` (18 KB).

Usage:
  # Full export (on RTX 4060 or Mac):
  python scripts/export_closed_checkpoint.py --output-dir release

  # Fast export using pre-discovered optimal alpha=0.4 (3x faster compile):
  python scripts/export_closed_checkpoint.py --output-dir release --fast-alpha
"""

import sys
sys.path.insert(0, ".")

import os
import time
import json
import pickle
import random
import shutil
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding
from safetensors import safe_open
from safetensors.torch import save_file
from huggingface_hub import snapshot_download
from datasets import load_dataset

from src.lean.core.retrieval_store import BM25Index, build_canonical_knowledge_corpus
from experiments.run_7b_scaled_awq_tuning import (
    MODEL_ID,
    SEED,
    ALPHA_CANDIDATES,
    NUM_CALIB_SEQS,
    MAX_SEQ_LEN,
    ScaledCalibratedPackedLinear,
)


def set_module_tensor_to_device(module: nn.Module, tensor_name: str, device: str, value: torch.Tensor):
    parts = tensor_name.split(".")
    for p in parts[:-1]:
        module = getattr(module, p)
    attr_name = parts[-1]
    old = getattr(module, attr_name, None)
    if isinstance(old, nn.Parameter):
        setattr(module, attr_name, nn.Parameter(value.to(device), requires_grad=False))
    else:
        module.register_buffer(attr_name, value.to(device))


def export_standalone_bm25(output_dir: str) -> str:
    """Exports the 18 KB BM25 index as a pure Python dict with zero src.lean class dependencies."""
    os.makedirs(output_dir, exist_ok=True)
    corpus = build_canonical_knowledge_corpus()
    bm25 = BM25Index()
    bm25.fit(corpus)

    pure_payload = {
        "k1": bm25.k1,
        "b": bm25.b,
        "corpus": bm25.corpus,
        "doc_freqs": dict(bm25.doc_freqs),
        "doc_lens": list(bm25.doc_lens),
        "avg_doc_len": float(bm25.avg_doc_len),
    }
    bm25_path = os.path.join(output_dir, "bm25_index.pkl")
    with open(bm25_path, "wb") as f:
        pickle.dump(pure_payload, f, protocol=4)
    size_kb = os.path.getsize(bm25_path) / 1024.0
    print(f"[Exporter] Saved standalone BM25 index to {bm25_path} ({size_kb:.2f} KB)")
    return bm25_path


def detect_compute_device() -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def export_closed_checkpoint(
    output_dir: str = "release",
    fast_alpha: bool = False,
    cleanup_hf_cache: bool = False,
):
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    os.makedirs(output_dir, exist_ok=True)
    export_standalone_bm25(output_dir)

    target_device = detect_compute_device()
    alpha_list = [0.4] if fast_alpha else ALPHA_CANDIDATES

    print("=" * 80)
    print("Project LEAN — Option B Closed-Compiler Checkpoint Exporter")
    print(f"Base Model: {MODEL_ID} | Target Device: {target_device} | Alpha Grid: {alpha_list}")
    print("=" * 80)

    t0 = time.time()
    print("[1/4] Locating or downloading Qwen/Qwen2.5-7B-Instruct snapshot...")
    snapshot = snapshot_download(repo_id=MODEL_ID)
    print(f"  Snapshot located at: {snapshot}")

    tok = AutoTokenizer.from_pretrained(snapshot, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    config = AutoConfig.from_pretrained(snapshot, trust_remote_code=True)

    # Save config & tokenizer into release/ so the checkpoint is self-contained
    tok.save_pretrained(output_dir)
    config.save_pretrained(output_dir)

    print(f"[2/4] Loading {NUM_CALIB_SEQS} WikiText-2 calibration sequences...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    valid_texts = [t.strip() for t in ds["text"] if len(t.strip()) > 100]
    calib_texts = valid_texts[:NUM_CALIB_SEQS]
    enc = tok(calib_texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_SEQ_LEN)
    calib_input_ids = enc["input_ids"].to(target_device)
    seq_len = calib_input_ids.shape[1]

    with open(os.path.join(snapshot, "model.safetensors.index.json"), "r") as f:
        weight_map = json.load(f)["weight_map"]

    def load_shard_tensor(tensor_name: str, target_dev: str = "cpu"):
        shard_file = weight_map[tensor_name]
        shard_path = os.path.join(snapshot, shard_file)
        open_dev = target_dev if target_dev.startswith("cuda") else "cpu"
        with safe_open(shard_path, framework="pt", device=open_dev) as f:
            t = f.get_tensor(tensor_name)
        return t.to(target_dev) if target_dev != open_dev else t

    print("[3/4] Streaming 28 Layers: Calibrating & Packing into 4-bit GQA / 2-bit SwiGLU MLP...")
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16)

    for layer in model.model.layers:
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

    with torch.no_grad():
        H = model.model.embed_tokens(calib_input_ids).to(device=target_device, dtype=torch.float16)
    pos_ids = torch.arange(seq_len, device=target_device).unsqueeze(0)
    pos_embs = model.model.rotary_emb(H, pos_ids)

    for layer_idx, layer in enumerate(model.model.layers):
        t_layer = time.time()
        for norm_name in ["input_layernorm", "post_attention_layernorm"]:
            w_name = f"model.layers.{layer_idx}.{norm_name}.weight"
            w_tensor = load_shard_tensor(w_name, target_dev=target_device)
            set_module_tensor_to_device(layer, norm_name + ".weight", device=target_device, value=w_tensor.to(torch.float16))
            del w_tensor

        with torch.no_grad():
            norm_x = layer.input_layernorm(H)
            act_attn = torch.mean(torch.abs(norm_x), dim=(0, 1))
            X_sample_attn = norm_x.reshape(-1, norm_x.shape[-1])[:128]

        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            w_name = f"model.layers.{layer_idx}.self_attn.{proj}.weight"
            w_tensor = load_shard_tensor(w_name, target_dev=target_device)
            lin_mod = getattr(layer.self_attn, proj)
            act_s = act_attn if proj in ["q_proj", "k_proj", "v_proj"] else None
            lin_mod.pack_weights(
                w_tensor,
                act_scales=act_s,
                X_sample=X_sample_attn if act_s is not None else None,
                alpha_candidates=alpha_list,
            )
            b_name = f"model.layers.{layer_idx}.self_attn.{proj}.bias"
            if b_name in weight_map:
                b_tensor = load_shard_tensor(b_name, target_dev=target_device)
                lin_mod.bias.copy_(b_tensor.to(dtype=torch.float16, device=target_device))
                del b_tensor
            del w_tensor

        with torch.no_grad():
            attn_out = layer.self_attn(norm_x, attention_mask=None, position_embeddings=pos_embs)[0]
            H_post = H + attn_out
            post_norm_x = layer.post_attention_layernorm(H_post)
            act_mlp = torch.mean(torch.abs(post_norm_x), dim=(0, 1))
            X_sample_mlp = post_norm_x.reshape(-1, post_norm_x.shape[-1])[:128]

        for proj in ["gate_proj", "up_proj"]:
            w_name = f"model.layers.{layer_idx}.mlp.{proj}.weight"
            w_tensor = load_shard_tensor(w_name, target_dev=target_device)
            lin_mod = getattr(layer.mlp, proj)
            lin_mod.pack_weights(w_tensor, act_scales=act_mlp, X_sample=X_sample_mlp, alpha_candidates=alpha_list)
            del w_tensor

        with torch.no_grad():
            gate_val = layer.mlp.gate_proj(post_norm_x)
            up_val = layer.mlp.up_proj(post_norm_x)
            swiglu_act = F.silu(gate_val) * up_val
            act_down = torch.mean(torch.abs(swiglu_act), dim=(0, 1))
            X_sample_down = swiglu_act.reshape(-1, swiglu_act.shape[-1])[:128]

        w_name = f"model.layers.{layer_idx}.mlp.down_proj.weight"
        w_tensor = load_shard_tensor(w_name, target_dev=target_device)
        lin_mod = getattr(layer.mlp, "down_proj")
        lin_mod.pack_weights(w_tensor, act_scales=act_down, X_sample=X_sample_down, alpha_candidates=alpha_list)
        del w_tensor

        with torch.no_grad():
            mlp_out = layer.mlp(post_norm_x)
            H = H_post + mlp_out

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()

        print(
            f"  [Layer {layer_idx + 1:02d}/28] Packed in {time.time() - t_layer:.1f}s | "
            f"Optimal Alpha: {lin_mod.optimal_alpha}",
            flush=True,
        )

    del H
    safetensors_path = os.path.join(output_dir, "model_kac.safetensors")
    print(f"[4/4] Serializing closed-compiler checkpoint to {safetensors_path}...")
    state_dict = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    save_file(state_dict, safetensors_path)

    file_gib = os.path.getsize(safetensors_path) / (1024 ** 3)
    print("=" * 80)
    print(f"[SUCCESS] Exported {safetensors_path} ({file_gib:.3f} GiB) in {time.time() - t0:.1f}s")
    print("=" * 80)

    if cleanup_hf_cache and os.path.exists(snapshot):
        cache_repo_dir = os.path.dirname(os.path.dirname(snapshot))
        print(f"[Cleanup] Removing raw FP16 cache at {cache_repo_dir} to reclaim 15.2 GB...")
        shutil.rmtree(cache_repo_dir, ignore_errors=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Qwen2.5-7B-ThinkingCore-KAC Open Checkpoint")
    parser.add_argument("--output-dir", type=str, default="release")
    parser.add_argument("--bm25-only", action="store_true", help="Only export the 18 KB BM25 index")
    parser.add_argument("--fast-alpha", action="store_true", help="Use optimal alpha=0.4 directly for 3x faster export")
    parser.add_argument("--cleanup-hf-cache", action="store_true", help="Delete 15.2 GB raw HF cache after exporting")
    args = parser.parse_args()

    if args.bm25_only:
        export_standalone_bm25(args.output_dir)
    else:
        export_closed_checkpoint(
            output_dir=args.output_dir,
            fast_alpha=args.fast_alpha,
            cleanup_hf_cache=args.cleanup_hf_cache,
        )
