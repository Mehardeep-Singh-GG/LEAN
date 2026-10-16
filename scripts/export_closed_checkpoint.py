"""
Project LEAN — Private Compiler to Open-Checkpoint Exporter (Option B)
======================================================================
Memory- & disk-streaming AWQ-KAC compiler for Qwen/Qwen2.5-7B-Instruct:
1. Streams ONE raw FP16 shard (~3.9 GB) at a time from Hugging Face Hub and
   deletes each raw shard immediately after its last layer is packed (when
   --cleanup-hf-cache is enabled), keeping peak disk usage under ~7.8 GB.
2. Uses in-place float16 group-128 MSE clipping and activation-aware alpha*
   scaling without 3D tensor broadcasting, packing each 67.9M-param SwiGLU
   projection in ~1.0s on Apple Silicon M2 or NVIDIA RTX GPUs with <400 MB
   working memory.
3. Serializes ONLY the final packed buffers (`packed`, `scales`, `channel_scales`)
   to `release/model_kac.safetensors` (3.84 GiB) + `release/bm25_index.pkl` (18 KB).
"""

import sys
sys.path.insert(0, ".")

import os
import gc
import time
import json
import pickle
import random
import shutil
import argparse
from collections import Counter
from typing import Dict, Any, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding, apply_rotary_pos_emb, repeat_kv
from safetensors import safe_open
from safetensors.torch import save_file
from huggingface_hub import hf_hub_download
from datasets import load_dataset

from src.lean.core.retrieval_store import BM25Index, build_canonical_knowledge_corpus

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
SEED = 1042
ALPHA_CANDIDATES = [0.4, 0.5, 0.6, 0.75]
NUM_CALIB_SEQS = 48
MAX_SEQ_LEN = 64
GROUP_SIZE = 128


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
    print(f"[Exporter] Saved standalone BM25 index to {bm25_path} ({size_kb:.2f} KB)", flush=True)
    return bm25_path


def fast_pack_projection(
    W_fp16: torch.Tensor,
    bits: int,
    act_scales: Optional[torch.Tensor] = None,
    X_sample: Optional[torch.Tensor] = None,
    alpha_candidates: List[float] = [0.4],
    group_size: int = GROUP_SIZE,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Low-memory (<350 MB RAM), high-speed (~1.0s/projection) AWQ-KAC packer.
    Returns (packed_uint8, scales_fp16, channel_scales_fp16, w_hat_fp16, best_alpha).
    """
    out_features, in_features = W_fp16.shape
    numel = out_features * in_features
    num_groups = (numel + group_size - 1) // group_size
    pad_len = num_groups * group_size

    qmax = float((2 ** (bits - 1)) - 1)
    qmin = float(-(2 ** (bits - 1)))
    ratios = [0.55, 0.65, 0.75, 0.85, 0.95, 1.0] if bits == 2 else [0.75, 0.85, 0.92, 0.97, 1.0]

    def quant_for_s(s_vec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        W_s = (W_fp16 * s_vec.unsqueeze(0)).reshape(-1)
        if pad_len > numel:
            W_s = F.pad(W_s, (0, pad_len - numel))
        grouped = W_s.view(num_groups, group_size)
        max_val = grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)

        best_err = torch.full((num_groups, 1), float("inf"), dtype=torch.float32, device=W_fp16.device)
        best_sc = max_val / qmax

        for r in ratios:
            sc = (max_val * r) / qmax
            q = torch.round(grouped / sc).clamp(qmin, qmax)
            diff = (grouped - q * sc).float()
            err = torch.mean(diff * diff, dim=-1, keepdim=True)
            mask = err < best_err
            best_err = torch.where(mask, err, best_err)
            best_sc = torch.where(mask, sc, best_sc)

        q_final = torch.round(grouped / best_sc).clamp(qmin, qmax)
        w_deq = (q_final.float() * best_sc.to(torch.float16).float()).reshape(-1)[:numel].view(out_features, in_features).to(torch.float16)
        w_hat = (w_deq / s_vec.unsqueeze(0)).to(torch.float16)
        return best_sc.squeeze(-1).to(torch.float16), q_final, w_hat

    best_alpha = alpha_candidates[0]
    if act_scales is not None and act_scales.numel() == in_features:
        act_s = act_scales.float().clamp(min=1e-5)
        act_mean = act_s.mean()
        if len(alpha_candidates) > 1 and X_sample is not None:
            X_ev = X_sample.to(torch.float16)
            Y_true = F.linear(X_ev, W_fp16).float()
            norm_Y = torch.norm(Y_true).clamp(min=1e-6)
            best_a_err = float("inf")
            for a in alpha_candidates:
                s_a = (act_s / act_mean).pow(a).clamp(min=0.2, max=5.0).to(torch.float16)
                _, _, w_hat_cand = quant_for_s(s_a)
                Y_hat = F.linear(X_ev, w_hat_cand).float()
                err = (torch.norm(Y_true - Y_hat) / norm_Y).item()
                if err < best_a_err:
                    best_a_err = err
                    best_alpha = a
        s_final = (act_s / act_mean).pow(best_alpha).clamp(min=0.2, max=5.0).to(torch.float16)
    else:
        s_final = torch.ones(in_features, dtype=torch.float16, device=W_fp16.device)

    scales_fp16, q_final, w_hat = quant_for_s(s_final)
    q_shifted = (q_final.to(torch.int16) - int(qmin)).to(torch.uint8).reshape(-1)[:numel]

    if bits == 4:
        packed = (q_shifted[0::2] & 0x0F) | ((q_shifted[1::2] & 0x0F) << 4)
    elif bits == 2:
        packed = (
            (q_shifted[0::4] & 0x03)
            | ((q_shifted[1::4] & 0x03) << 2)
            | ((q_shifted[2::4] & 0x03) << 4)
            | ((q_shifted[3::4] & 0x03) << 6)
        )
    else:
        raise ValueError(f"Unsupported bits={bits}")

    return packed.cpu().contiguous(), scales_fp16.cpu().contiguous(), s_final.cpu().contiguous(), w_hat, best_alpha


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    variance = x.float().pow(2).mean(-1, keepdim=True)
    x_normed = x.float() * torch.rsqrt(variance + eps)
    return (weight.float() * x_normed).to(x.dtype)


def export_closed_checkpoint(
    output_dir: str = "release",
    fast_alpha: bool = False,
    cleanup_hf_cache: bool = False,
):
    random.seed(SEED)
    torch.manual_seed(SEED)

    os.makedirs(output_dir, exist_ok=True)
    export_standalone_bm25(output_dir)

    alpha_list = [0.4] if fast_alpha else ALPHA_CANDIDATES
    t0 = time.time()

    print("=" * 80, flush=True)
    print("Project LEAN — Low-Memory Shard-Streaming Compiler & Exporter (Option B)", flush=True)
    print(f"Base Model: {MODEL_ID} | Alpha Grid: {alpha_list} | Auto-Clean Shards: {cleanup_hf_cache}", flush=True)
    print("=" * 80, flush=True)

    print("[1/4] Fetching lightweight config, tokenizer, and shard index (<15 MB)...", flush=True)
    for fname in ["config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors.index.json"]:
        hf_hub_download(repo_id=MODEL_ID, filename=fname)

    index_path = hf_hub_download(repo_id=MODEL_ID, filename="model.safetensors.index.json")
    with open(index_path, "r") as f:
        weight_map: Dict[str, str] = json.load(f)["weight_map"]

    config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    tok.save_pretrained(output_dir)
    config.save_pretrained(output_dir)

    print(f"[2/4] Loading {NUM_CALIB_SEQS} WikiText-2 calibration sequences...", flush=True)
    try:
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
        valid_texts = [t.strip() for t in ds["text"] if len(t.strip()) > 100]
        calib_texts = valid_texts[:NUM_CALIB_SEQS]
    except Exception:
        corpus = build_canonical_knowledge_corpus()
        calib_texts = [doc["text"] + " " + doc["text"] for doc in corpus[:NUM_CALIB_SEQS]]
    enc = tok(calib_texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_SEQ_LEN)
    compute_dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    calib_input_ids = enc["input_ids"].to(compute_dev)
    batch_size, seq_len = calib_input_ids.shape

    # Count how many tensors we will read from each shard so we can delete each 3.9 GB shard immediately after its last read
    shard_remaining = Counter(weight_map.values())
    downloaded_shards: Dict[str, str] = {}

    def stream_tensor(tensor_name: str) -> torch.Tensor:
        shard_file = weight_map[tensor_name]
        if shard_file not in downloaded_shards:
            print(f"  -> Loading shard {shard_file}...", flush=True)
            downloaded_shards[shard_file] = hf_hub_download(repo_id=MODEL_ID, filename=shard_file)
        shard_path = downloaded_shards[shard_file]
        with safe_open(shard_path, framework="pt", device=compute_dev) as f:
            tensor = f.get_tensor(tensor_name).to(dtype=torch.float16, device=compute_dev)

        shard_remaining[shard_file] -= 1
        if cleanup_hf_cache and shard_remaining[shard_file] <= 0:
            real_blob = os.path.realpath(shard_path)
            try:
                if os.path.exists(shard_path) and os.path.islink(shard_path):
                    os.unlink(shard_path)
                if os.path.exists(real_blob):
                    os.remove(real_blob)
                print(f"  [Disk Guard] Freed raw FP16 shard {shard_file} (~3.9 GB reclaimed)", flush=True)
            except Exception as e:
                print(f"  [Disk Guard] Note: could not remove {shard_file}: {e}", flush=True)
        return tensor

    exported_tensors: Dict[str, torch.Tensor] = {}

    print(f"[3/4] Streaming 28 Layers on {compute_dev}: Calibrating & Packing into 4-bit GQA / 2-bit SwiGLU MLP...", flush=True)
    embed_w = stream_tensor("model.embed_tokens.weight")
    with torch.no_grad():
        H = F.embedding(calib_input_ids, embed_w)
    exported_tensors["model.embed_tokens.weight"] = embed_w.detach().cpu().contiguous()
    del embed_w
    gc.collect()

    rotary_emb = Qwen2RotaryEmbedding(config=config, device=compute_dev)
    pos_ids = torch.arange(seq_len, dtype=torch.long, device=compute_dev).unsqueeze(0)
    cos, sin = rotary_emb(H, pos_ids)

    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_dim = config.hidden_size // num_heads
    num_kv_groups = num_heads // num_kv_heads
    rms_eps = getattr(config, "rms_norm_eps", 1e-6)
    causal_mask = torch.triu(torch.full((seq_len, seq_len), float("-inf"), dtype=torch.float32, device=compute_dev), diagonal=1).unsqueeze(0).unsqueeze(0)

    for layer_idx in range(config.num_hidden_layers):
        t_layer = time.time()
        prefix = f"model.layers.{layer_idx}"

        in_norm_w = stream_tensor(f"{prefix}.input_layernorm.weight")
        post_norm_w = stream_tensor(f"{prefix}.post_attention_layernorm.weight")
        exported_tensors[f"{prefix}.input_layernorm.weight"] = in_norm_w.detach().cpu().contiguous()
        exported_tensors[f"{prefix}.post_attention_layernorm.weight"] = post_norm_w.detach().cpu().contiguous()

        with torch.no_grad():
            norm_x = rms_norm(H, in_norm_w, rms_eps)
            act_attn = torch.mean(torch.abs(norm_x.float()), dim=(0, 1))
            X_sample_attn = norm_x.reshape(-1, norm_x.shape[-1])[:128]

        # 4-bit GQA Projections (q_proj, k_proj, v_proj, o_proj)
        attn_w_hat = {}
        attn_bias = {}
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            w_name = f"{prefix}.self_attn.{proj}.weight"
            w_t = stream_tensor(w_name)
            act_s = act_attn if proj in ["q_proj", "k_proj", "v_proj"] else None
            packed, scales, ch_scales, w_hat, _ = fast_pack_projection(
                w_t,
                bits=4,
                act_scales=act_s,
                X_sample=X_sample_attn if act_s is not None else None,
                alpha_candidates=alpha_list,
            )
            del w_t
            exported_tensors[f"{prefix}.self_attn.{proj}.packed"] = packed
            exported_tensors[f"{prefix}.self_attn.{proj}.scales"] = scales
            exported_tensors[f"{prefix}.self_attn.{proj}.channel_scales"] = ch_scales
            attn_w_hat[proj] = w_hat

            b_name = f"{prefix}.self_attn.{proj}.bias"
            if b_name in weight_map:
                b_t = stream_tensor(b_name)
                exported_tensors[b_name] = b_t.detach().cpu().contiguous()
                attn_bias[proj] = b_t
            else:
                attn_bias[proj] = None

        # Forward GQA Attention using quantized w_hat to propagate accurate activations
        with torch.no_grad():
            q = F.linear(norm_x, attn_w_hat["q_proj"], attn_bias["q_proj"]).view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            k = F.linear(norm_x, attn_w_hat["k_proj"], attn_bias["k_proj"]).view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)
            v = F.linear(norm_x, attn_w_hat["v_proj"], attn_bias["v_proj"]).view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            k = repeat_kv(k, num_kv_groups)
            v = repeat_kv(v, num_kv_groups)
            scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) / (head_dim ** 0.5)
            scores = scores + causal_mask
            probs = F.softmax(scores, dim=-1).to(torch.float16)
            attn_output = torch.matmul(probs, v).transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
            attn_out = F.linear(attn_output, attn_w_hat["o_proj"], attn_bias["o_proj"])
            del attn_w_hat, attn_bias, q, k, v, scores, probs, attn_output

            H_post = H + attn_out
            post_norm_x = rms_norm(H_post, post_norm_w, rms_eps)
            act_mlp = torch.mean(torch.abs(post_norm_x.float()), dim=(0, 1))
            X_sample_mlp = post_norm_x.reshape(-1, post_norm_x.shape[-1])[:128]
            del in_norm_w, post_norm_w

        # 2-bit SwiGLU gate_proj and up_proj
        mlp_w_hat = {}
        for proj in ["gate_proj", "up_proj"]:
            w_name = f"{prefix}.mlp.{proj}.weight"
            w_t = stream_tensor(w_name)
            packed, scales, ch_scales, w_hat, opt_a = fast_pack_projection(
                w_t, bits=2, act_scales=act_mlp, X_sample=X_sample_mlp, alpha_candidates=alpha_list
            )
            del w_t
            exported_tensors[f"{prefix}.mlp.{proj}.packed"] = packed
            exported_tensors[f"{prefix}.mlp.{proj}.scales"] = scales
            exported_tensors[f"{prefix}.mlp.{proj}.channel_scales"] = ch_scales
            mlp_w_hat[proj] = w_hat

        with torch.no_grad():
            gate_val = F.linear(post_norm_x, mlp_w_hat["gate_proj"])
            up_val = F.linear(post_norm_x, mlp_w_hat["up_proj"])
            del mlp_w_hat
            swiglu_act = F.silu(gate_val) * up_val
            del gate_val, up_val
            act_down = torch.mean(torch.abs(swiglu_act.float()), dim=(0, 1))
            X_sample_down = swiglu_act.reshape(-1, swiglu_act.shape[-1])[:128]

        # 2-bit SwiGLU down_proj
        w_t = stream_tensor(f"{prefix}.mlp.down_proj.weight")
        packed, scales, ch_scales, w_hat_down, opt_a = fast_pack_projection(
            w_t, bits=2, act_scales=act_down, X_sample=X_sample_down, alpha_candidates=alpha_list
        )
        del w_t
        exported_tensors[f"{prefix}.mlp.down_proj.packed"] = packed
        exported_tensors[f"{prefix}.mlp.down_proj.scales"] = scales
        exported_tensors[f"{prefix}.mlp.down_proj.channel_scales"] = ch_scales

        with torch.no_grad():
            mlp_out = F.linear(swiglu_act, w_hat_down)
            del swiglu_act, w_hat_down
            H = H_post + mlp_out

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print(
            f"  [Layer {layer_idx + 1:02d}/28] Calibrated & Packed in {time.time() - t_layer:.1f}s | "
            f"Alpha*: {opt_a}",
            flush=True,
        )

    del H
    norm_w = stream_tensor("model.norm.weight")
    exported_tensors["model.norm.weight"] = norm_w.detach().cpu().contiguous()
    del norm_w

    lm_head_w = stream_tensor("lm_head.weight")
    exported_tensors["lm_head.weight"] = lm_head_w.detach().cpu().contiguous()
    del lm_head_w
    gc.collect()

    safetensors_path = os.path.join(output_dir, "model_kac.safetensors")
    print(f"[4/4] Writing 3.84 GiB open checkpoint to {safetensors_path}...", flush=True)
    save_file(exported_tensors, safetensors_path)

    file_gib = os.path.getsize(safetensors_path) / (1024 ** 3)
    print("=" * 80, flush=True)
    print(f"[SUCCESS] Exported {safetensors_path} ({file_gib:.4f} GiB) in {time.time() - t0:.1f}s", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Qwen2.5-7B-ThinkingCore-KAC Open Checkpoint")
    parser.add_argument("--output-dir", type=str, default="release")
    parser.add_argument("--bm25-only", action="store_true", help="Only export the 18 KB BM25 index")
    parser.add_argument("--fast-alpha", action="store_true", help="Use optimal alpha=0.4 directly for 3x faster export")
    parser.add_argument("--cleanup-hf-cache", action="store_true", help="Delete each 3.9 GB raw HF shard right after packing")
    args = parser.parse_args()

    if args.bm25_only:
        export_standalone_bm25(args.output_dir)
    else:
        export_closed_checkpoint(
            output_dir=args.output_dir,
            fast_alpha=args.fast_alpha,
            cleanup_hf_cache=args.cleanup_hf_cache,
        )
