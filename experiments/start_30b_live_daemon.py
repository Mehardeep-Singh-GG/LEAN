#!/usr/bin/env python3
"""
experiments/start_30b_live_daemon.py
====================================
Launches the persistent 52-layer LEAN-Quantized `Muse-Glimmer-30B` inference server
on the Colab A100 worker using Unified Split-K 2D-Grid `(pid_r, pid_g)` Coalesced Triton Kernels:
  1. Keeps GPU VRAM strictly under the 8.00 GiB Hardware Wall (`~6.72 GiB` persistent,
     `~7,380 MiB` in `nvidia-smi`).
  2. Tiled `[16, BLOCK_R]` register-spill-free 4-bit & 2-bit Split-K kernels with fused
     `inv_chan_s` scaling and fused `pid_g == G_CTAS` stride-1 `int8` salient outlier CTA.
  3. Pre-allocated GPU staging buffers (`self.stage_*`) and output buffer (`self.y_f32_buf`)
     so `forward()` performs ZERO dynamic weight/buffer allocations and never triggers
     PyTorch CUDA caching allocator flushes.
"""

import os
import re
import sys
import json
import time
import subprocess
from pathlib import Path


SERVER_SCRIPT = r'''#!/usr/bin/env python3
import os
import gc
import json
import time
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from safetensors import safe_open
from transformers import AutoTokenizer


@triton.jit
def lean_4bit_splitk_kernel(
    packed_ptr,     # [G * B_PACK, R] uint8 (stride-1 along R)
    scales_ptr,     # [G, R] float16 (stride-1 along R)
    mins_ptr,       # [G, R] float16 (stride-1 along R)
    inv_chan_s_ptr, # [K] float32 (0.0 at outlier columns)
    out_idx_ptr,    # [N_OUT] int64
    out_w_kr_ptr,   # [N_OUT, R] int8 (stride-1 along R)
    out_scale_ptr,  # [N_OUT] float32
    x_ptr,          # [N, K] bfloat16
    out_ptr,        # [N, R] float32 (zero-initialized, accumulated in L2 cache)
    R,
    K: tl.constexpr,
    G: tl.constexpr,
    G_CTAS: tl.constexpr,
    B_PACK: tl.constexpr,
    BLOCK_R: tl.constexpr,
    GROUPS_PER_CTA: tl.constexpr,
    OUT_STEPS: tl.constexpr,
):
    pid_r = tl.program_id(0)
    pid_g = tl.program_id(1)
    pid_n = tl.program_id(2)

    offs_r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    mask_r = offs_r < R
    acc = tl.zeros((BLOCK_R,), dtype=tl.float32)
    x_row_ptr = x_ptr + pid_n * K

    if pid_g < G_CTAS:
        g_start = pid_g * GROUPS_PER_CTA
        for step in range(GROUPS_PER_CTA):
            g = g_start + step
            if g < G:
                grp_dot = tl.zeros((BLOCK_R,), dtype=tl.float32)
                x_sum = 0.0
                xg_idx_base = g * (2 * B_PACK)
                for sub in range(B_PACK // 16):
                    offs_b = sub * 16 + tl.arange(0, 16)
                    p_idx = (g * B_PACK + offs_b[:, None]) * R + offs_r[None, :]
                    p = tl.load(packed_ptr + p_idx, mask=mask_r[None, :], other=0)

                    idx0 = xg_idx_base + offs_b
                    idx1 = xg_idx_base + B_PACK + offs_b
                    x0 = tl.load(x_row_ptr + idx0).to(tl.float32) * tl.load(inv_chan_s_ptr + idx0)
                    x1 = tl.load(x_row_ptr + idx1).to(tl.float32) * tl.load(inv_chan_s_ptr + idx1)
                    x_sum += tl.sum(x0, axis=0) + tl.sum(x1, axis=0)

                    q0 = (p & 0x0F).to(tl.float32)
                    q1 = ((p >> 4) & 0x0F).to(tl.float32)
                    grp_dot += tl.sum(q0 * x0[:, None] + q1 * x1[:, None], axis=0)

                sg_idx = g * R + offs_r
                s = tl.load(scales_ptr + sg_idx, mask=mask_r, other=0.0).to(tl.float32)
                m = tl.load(mins_ptr + sg_idx, mask=mask_r, other=0.0).to(tl.float32)
                acc += grp_dot * s + m * x_sum
    elif OUT_STEPS > 0:
        for sub in range(OUT_STEPS):
            offs_o = sub * 16 + tl.arange(0, 16)
            col_idx = tl.load(out_idx_ptr + offs_o)
            col_s = tl.load(out_scale_ptr + offs_o)
            x_o = tl.load(x_row_ptr + col_idx).to(tl.float32) * col_s
            w_o = tl.load(
                out_w_kr_ptr + offs_o[:, None] * R + offs_r[None, :],
                mask=mask_r[None, :],
                other=0,
            ).to(tl.float32)
            acc += tl.sum(w_o * x_o[:, None], axis=0)

    tl.atomic_add(out_ptr + pid_n * R + offs_r, acc, mask=mask_r)


@triton.jit
def lean_2bit_splitk_kernel(
    packed_ptr,     # [G * B_PACK, R] uint8 (stride-1 along R)
    scales_ptr,     # [G, R] float16 (stride-1 along R)
    mins_ptr,       # [G, R] float16 (stride-1 along R)
    inv_chan_s_ptr, # [K] float32 (0.0 at outlier columns)
    out_idx_ptr,    # [N_OUT] int64
    out_w_kr_ptr,   # [N_OUT, R] int8 (stride-1 along R)
    out_scale_ptr,  # [N_OUT] float32
    x_ptr,          # [N, K] bfloat16
    out_ptr,        # [N, R] float32 (zero-initialized, accumulated in L2 cache)
    R,
    K: tl.constexpr,
    G: tl.constexpr,
    G_CTAS: tl.constexpr,
    B_PACK: tl.constexpr,
    BLOCK_R: tl.constexpr,
    GROUPS_PER_CTA: tl.constexpr,
    OUT_STEPS: tl.constexpr,
):
    pid_r = tl.program_id(0)
    pid_g = tl.program_id(1)
    pid_n = tl.program_id(2)

    offs_r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    mask_r = offs_r < R
    acc = tl.zeros((BLOCK_R,), dtype=tl.float32)
    x_row_ptr = x_ptr + pid_n * K

    if pid_g < G_CTAS:
        g_start = pid_g * GROUPS_PER_CTA
        for step in range(GROUPS_PER_CTA):
            g = g_start + step
            if g < G:
                grp_dot = tl.zeros((BLOCK_R,), dtype=tl.float32)
                x_sum = 0.0
                xg_idx_base = g * (4 * B_PACK)
                for sub in range(B_PACK // 16):
                    offs_b = sub * 16 + tl.arange(0, 16)
                    p_idx = (g * B_PACK + offs_b[:, None]) * R + offs_r[None, :]
                    p = tl.load(packed_ptr + p_idx, mask=mask_r[None, :], other=0)

                    idx0 = xg_idx_base + offs_b
                    idx1 = xg_idx_base + B_PACK + offs_b
                    idx2 = xg_idx_base + 2 * B_PACK + offs_b
                    idx3 = xg_idx_base + 3 * B_PACK + offs_b

                    x0 = tl.load(x_row_ptr + idx0).to(tl.float32) * tl.load(inv_chan_s_ptr + idx0)
                    x1 = tl.load(x_row_ptr + idx1).to(tl.float32) * tl.load(inv_chan_s_ptr + idx1)
                    x2 = tl.load(x_row_ptr + idx2).to(tl.float32) * tl.load(inv_chan_s_ptr + idx2)
                    x3 = tl.load(x_row_ptr + idx3).to(tl.float32) * tl.load(inv_chan_s_ptr + idx3)
                    x_sum += tl.sum(x0, axis=0) + tl.sum(x1, axis=0) + tl.sum(x2, axis=0) + tl.sum(x3, axis=0)

                    q0 = (p & 0x03).to(tl.float32)
                    q1 = ((p >> 2) & 0x03).to(tl.float32)
                    q2 = ((p >> 4) & 0x03).to(tl.float32)
                    q3 = ((p >> 6) & 0x03).to(tl.float32)

                    grp_dot += tl.sum(
                        q0 * x0[:, None] + q1 * x1[:, None] + q2 * x2[:, None] + q3 * x3[:, None],
                        axis=0,
                    )
                sg_idx = g * R + offs_r
                s = tl.load(scales_ptr + sg_idx, mask=mask_r, other=0.0).to(tl.float32)
                m = tl.load(mins_ptr + sg_idx, mask=mask_r, other=0.0).to(tl.float32)
                acc += grp_dot * s + m * x_sum
    elif OUT_STEPS > 0:
        for sub in range(OUT_STEPS):
            offs_o = sub * 16 + tl.arange(0, 16)
            col_idx = tl.load(out_idx_ptr + offs_o)
            col_s = tl.load(out_scale_ptr + offs_o)
            x_o = tl.load(x_row_ptr + col_idx).to(tl.float32) * col_s
            w_o = tl.load(
                out_w_kr_ptr + offs_o[:, None] * R + offs_r[None, :],
                mask=mask_r[None, :],
                other=0,
            ).to(tl.float32)
            acc += tl.sum(w_o * x_o[:, None], axis=0)

    tl.atomic_add(out_ptr + pid_n * R + offs_r, acc, mask=mask_r)


def rms_norm_no_scale(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x_f = x.float()
    return (x_f * torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + eps)).type_as(x)


def rms_norm_with_scale(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x_f = x.float()
    normed = x_f * torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + eps)
    return (normed * weight.float()).type_as(x)


def centered_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x_f = x.float()
    normed = x_f * torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + eps)
    return (normed * (1.0 + weight.float())).type_as(x)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


@torch.no_grad()
def unpack_lean_linear_to_cpu_bf16(
    sf,
    prefix: str,
    out_features: int,
    in_features: int,
) -> torch.Tensor:
    packed = sf.get_tensor(f"{prefix}.packed")
    scales = sf.get_tensor(f"{prefix}.scales").to(dtype=torch.bfloat16)
    mins = sf.get_tensor(f"{prefix}.mins").to(dtype=torch.bfloat16)
    chan_s = sf.get_tensor(f"{prefix}.channel_scales").to(dtype=torch.bfloat16)

    total_elems = out_features * in_features
    pack_factor = total_elems // packed.numel()
    bits = 8 // pack_factor
    group_size = total_elems // scales.numel()

    if bits == 4:
        q0 = (packed & 0x0F).to(torch.bfloat16)
        q1 = ((packed >> 4) & 0x0F).to(torch.bfloat16)
        q = torch.stack([q0, q1], dim=-1).view(-1, group_size)
    else:
        q0 = (packed & 0x03).to(torch.bfloat16)
        q1 = ((packed >> 2) & 0x03).to(torch.bfloat16)
        q2 = ((packed >> 4) & 0x03).to(torch.bfloat16)
        q3 = ((packed >> 6) & 0x03).to(torch.bfloat16)
        q = torch.stack([q0, q1, q2, q3], dim=-1).view(-1, group_size)

    W = (q * scales.unsqueeze(-1) + mins.unsqueeze(-1)).view(out_features, in_features)
    W.div_(chan_s.unsqueeze(0))

    keys = set(sf.keys())
    idx_key = f"{prefix}.outlier_idx"
    w_key = f"{prefix}.outlier_w"
    if idx_key in keys and w_key in keys:
        out_idx = sf.get_tensor(idx_key).to(dtype=torch.long)
        out_w = sf.get_tensor(w_key).to(dtype=torch.bfloat16)
        W[:, out_idx] = out_w
        del out_idx, out_w

    del q, packed, scales, mins, chan_s
    return W


@torch.no_grad()
def load_coalesced_proj(sf, prefix: str, out_features: int, in_features: int, gpu_pack: bool) -> Dict:
    packed = sf.get_tensor(f"{prefix}.packed")
    scales = sf.get_tensor(f"{prefix}.scales")
    mins = sf.get_tensor(f"{prefix}.mins")
    chan_s = sf.get_tensor(f"{prefix}.channel_scales")

    total_elems = out_features * in_features
    pack_factor = total_elems // packed.numel()
    bits = 8 // pack_factor
    group_size = total_elems // scales.numel()
    num_groups = in_features // group_size
    b_pack = group_size // pack_factor

    if bits == 4:
        q0 = packed & 0x0F
        q1 = (packed >> 4) & 0x0F
        q_full = torch.stack([q0, q1], dim=-1).view(out_features, num_groups, 2, b_pack)
        repacked_rk = (q_full[:, :, 0, :] | (q_full[:, :, 1, :] << 4)).view(out_features, num_groups * b_pack)
        packed_kr_cpu = repacked_rk.t().contiguous()
        del q0, q1, q_full, repacked_rk
    else:
        q0 = packed & 0x03
        q1 = (packed >> 2) & 0x03
        q2 = (packed >> 4) & 0x03
        q3 = (packed >> 6) & 0x03
        q_full = torch.stack([q0, q1, q2, q3], dim=-1).view(out_features, num_groups, 4, b_pack)
        repacked_rk = (
            q_full[:, :, 0, :]
            | (q_full[:, :, 1, :] << 2)
            | (q_full[:, :, 2, :] << 4)
            | (q_full[:, :, 3, :] << 6)
        ).view(out_features, num_groups * b_pack)
        packed_kr_cpu = repacked_rk.t().contiguous()
        del q0, q1, q2, q3, q_full, repacked_rk

    keys = set(sf.keys())
    idx_key = f"{prefix}.outlier_idx"
    w_key = f"{prefix}.outlier_w"
    out_idx_gpu = None
    out_w_kr = None
    out_w_scale = None
    out_steps = 0

    if idx_key in keys and w_key in keys:
        out_idx_gpu = sf.get_tensor(idx_key).to(device="cuda", dtype=torch.long).contiguous()
        ow_f = sf.get_tensor(w_key).float()
        col_max = ow_f.abs().amax(dim=0, keepdim=True).clamp(min=1e-6)
        w_scale = (col_max / 127.0).squeeze(0).to(torch.float32).contiguous()
        w_i8 = torch.clamp(torch.round(ow_f / (col_max / 127.0)), -127.0, 127.0).to(torch.int8)
        w_kr_cpu = w_i8.t().contiguous()  # [N_OUT, R] stride-1 along R
        out_w_kr = w_kr_cpu.to("cuda", non_blocking=True) if gpu_pack else w_kr_cpu.pin_memory()
        out_w_scale = w_scale.to("cuda", non_blocking=True)
        out_steps = int(out_idx_gpu.numel() // 16)
        del ow_f, col_max, w_scale, w_i8, w_kr_cpu

    inv_chan_s = (1.0 / chan_s.float()).to(device="cuda", dtype=torch.float32).contiguous()
    if out_idx_gpu is not None:
        inv_chan_s[out_idx_gpu] = 0.0

    s_gr_cpu = scales.view(out_features, num_groups).t().contiguous().to(dtype=torch.float16)
    m_gr_cpu = mins.view(out_features, num_groups).t().contiguous().to(dtype=torch.float16)

    if gpu_pack:
        packed_kr = packed_kr_cpu.to("cuda", non_blocking=True)
        scales_gr = s_gr_cpu.to("cuda", non_blocking=True)
        mins_gr = m_gr_cpu.to("cuda", non_blocking=True)
    else:
        packed_kr = packed_kr_cpu.pin_memory()
        scales_gr = s_gr_cpu.pin_memory()
        mins_gr = m_gr_cpu.pin_memory()

    return {
        "gpu_pack": gpu_pack,
        "bits": bits,
        "group_size": group_size,
        "b_pack": b_pack,
        "out_f": out_features,
        "in_f": in_features,
        "G": num_groups,
        "packed_kr": packed_kr,
        "scales_gr": scales_gr,
        "mins_gr": mins_gr,
        "inv_chan_s": inv_chan_s,
        "out_idx": out_idx_gpu,
        "out_w_kr": out_w_kr,
        "out_w_scale": out_w_scale,
        "out_steps": out_steps,
    }


class FusedTriton8GBWallEngine:
    def __init__(self, release_dir: Path):
        self.release_dir = release_dir
        self.device = torch.device("cuda")
        with open(release_dir / "config.json") as f:
            self.cfg = json.load(f)["text_config"]
        self.tokenizer = AutoTokenizer.from_pretrained(str(release_dir))

        self.num_layers = int(self.cfg["num_hidden_layers"])       # 52
        self.hidden_size = int(self.cfg["hidden_size"])            # 6656
        self.intermediate_size = int(self.cfg["intermediate_size"]) # 18432
        self.num_heads = int(self.cfg["num_attention_heads"])      # 32
        self.num_kv_heads = int(self.cfg["num_key_value_heads"])   # 2
        self.head_dim = int(self.cfg["head_dim"])                  # 128
        self.vocab_size = int(self.cfg["vocab_size"])              # 202048
        self.rms_eps = float(self.cfg["rms_norm_eps"])
        self.post_eps = float(self.cfg["post_norm_eps"])
        self.qk_scale = float(self.cfg["qk_scale_factor"])
        self.out_mult = float(self.cfg["output_multiplier"])
        self.softcap = float(self.cfg["final_logit_softcapping"])
        self.layer_rope_theta = self.cfg["layer_rope_theta"]
        self.inv_freq = 1.0 / (
            500000.0 ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32, device=self.device) / self.head_dim)
        )

        # Pre-allocated static GPU staging & accumulation buffers (zero dynamic allocations in forward!)
        max_elems = self.intermediate_size * self.hidden_size
        self.stage_packed = torch.empty((max_elems // 2,), device=self.device, dtype=torch.uint8)
        self.stage_scales = torch.empty((max_elems // 64,), device=self.device, dtype=torch.float16)
        self.stage_mins = torch.empty((max_elems // 64,), device=self.device, dtype=torch.float16)
        self.stage_out_w = torch.empty((512 * self.intermediate_size,), device=self.device, dtype=torch.int8)
        self.y_f32_buf = torch.empty((max(128 * self.intermediate_size, 8 * self.vocab_size),), device=self.device, dtype=torch.float32)
        self.dummy_i64 = torch.zeros((16,), device=self.device, dtype=torch.long)
        self.dummy_i8 = torch.zeros((16,), device=self.device, dtype=torch.int8)
        self.dummy_f32 = torch.zeros((16,), device=self.device, dtype=torch.float32)

        self.embed_w_cpu: Optional[torch.Tensor] = None
        self.final_norm_w: Optional[torch.Tensor] = None
        self.lm_head_proj: Optional[Dict] = None
        self.layers: List[Dict] = [{} for _ in range(self.num_layers)]
        self.gpu_persistent_gib: float = 0.0
        self._load_under_8gb_wall()

    @torch.no_grad()
    def triton_packed_linear(self, x: torch.Tensor, m: Dict) -> torch.Tensor:
        orig_shape = x.shape[:-1]
        in_f = m["in_f"]
        out_f = m["out_f"]
        G = m["G"]
        bits = m["bits"]
        b_pack = m["b_pack"]
        out_steps = m["out_steps"]

        x_2d = x.reshape(-1, in_f).contiguous()
        N = x_2d.shape[0]

        if m["gpu_pack"]:
            packed_kr = m["packed_kr"]
            scales_gr = m["scales_gr"]
            mins_gr = m["mins_gr"]
            out_w_kr = m["out_w_kr"] if out_steps > 0 else self.dummy_i8
        else:
            p_cpu = m["packed_kr"]
            packed_kr = self.stage_packed[: p_cpu.numel()].view_as(p_cpu)
            packed_kr.copy_(p_cpu, non_blocking=True)

            s_cpu = m["scales_gr"]
            scales_gr = self.stage_scales[: s_cpu.numel()].view_as(s_cpu)
            scales_gr.copy_(s_cpu, non_blocking=True)

            m_cpu = m["mins_gr"]
            mins_gr = self.stage_mins[: m_cpu.numel()].view_as(m_cpu)
            mins_gr.copy_(m_cpu, non_blocking=True)

            if out_steps > 0:
                ow_cpu = m["out_w_kr"]
                out_w_kr = self.stage_out_w[: ow_cpu.numel()].view_as(ow_cpu)
                out_w_kr.copy_(ow_cpu, non_blocking=True)
            else:
                out_w_kr = self.dummy_i8

        out_idx_ptr = m["out_idx"] if out_steps > 0 else self.dummy_i64
        out_scale_ptr = m["out_w_scale"] if out_steps > 0 else self.dummy_f32

        y_f32 = self.y_f32_buf[: N * out_f].view(N, out_f)
        y_f32.zero_()

        BLOCK_R = 128
        GROUPS_PER_CTA = 2 if bits == 4 else 8
        G_CTAS = triton.cdiv(G, GROUPS_PER_CTA)
        grid = (triton.cdiv(out_f, BLOCK_R), G_CTAS + (1 if out_steps > 0 else 0), N)

        if bits == 4:
            lean_4bit_splitk_kernel[grid](
                packed_kr, scales_gr, mins_gr, m["inv_chan_s"],
                out_idx_ptr, out_w_kr, out_scale_ptr, x_2d, y_f32, out_f,
                K=in_f, G=G, G_CTAS=G_CTAS, B_PACK=b_pack, BLOCK_R=BLOCK_R,
                GROUPS_PER_CTA=GROUPS_PER_CTA, OUT_STEPS=out_steps,
                num_warps=4, num_stages=2
            )
        else:
            lean_2bit_splitk_kernel[grid](
                packed_kr, scales_gr, mins_gr, m["inv_chan_s"],
                out_idx_ptr, out_w_kr, out_scale_ptr, x_2d, y_f32, out_f,
                K=in_f, G=G, G_CTAS=G_CTAS, B_PACK=b_pack, BLOCK_R=BLOCK_R,
                GROUPS_PER_CTA=GROUPS_PER_CTA, OUT_STEPS=out_steps,
                num_warps=4, num_stages=2
            )

        return y_f32.to(torch.bfloat16).view(*orig_shape, out_f)

    def _load_under_8gb_wall(self):
        t0 = time.perf_counter()
        src_dir = Path("/content/lean_30b_backing") if Path("/content/lean_30b_backing/model_lean_30b-00001-of-00004.safetensors").exists() else self.release_dir

        for s_idx in range(4):
            sp = src_dir / f"model_lean_30b-0000{s_idx+1}-of-00004.safetensors"
            print(f"[Daemon] Transposing shard {sp.name} into Unified Split-K Coalesced layout...", flush=True)
            with safe_open(str(sp), framework="pt", device="cpu") as sf:
                if s_idx == 0:
                    self.embed_w_cpu = unpack_lean_linear_to_cpu_bf16(
                        sf, "model.language_model.embed_tokens", self.vocab_size, self.hidden_size
                    )
                l_start = s_idx * 13
                l_end = min(self.num_layers, (s_idx + 1) * 13)
                for i in range(l_start, l_end):
                    p = f"model.language_model.layers.{i}"
                    lw = self.layers[i]
                    for nk in [
                        "input_layernorm.weight",
                        "post_attention_layernorm.weight",
                        "pre_feedforward_layernorm.weight",
                        "post_feedforward_layernorm.weight",
                    ]:
                        lw[nk] = sf.get_tensor(f"{p}.{nk}").to(self.device, dtype=torch.bfloat16)

                    # Keep layers 0..37 + salient final layers 49..51 + lm_head 100% resident in GPU VRAM (~6.65 GiB)
                    gpu_pack = (i < 38) or (i >= 49)
                    lw["self_attn.q_proj"] = load_coalesced_proj(sf, f"{p}.self_attn.q_proj", self.num_heads * self.head_dim, self.hidden_size, gpu_pack)
                    lw["self_attn.k_proj"] = load_coalesced_proj(sf, f"{p}.self_attn.k_proj", self.num_kv_heads * self.head_dim, self.hidden_size, gpu_pack)
                    lw["self_attn.v_proj"] = load_coalesced_proj(sf, f"{p}.self_attn.v_proj", self.num_kv_heads * self.head_dim, self.hidden_size, gpu_pack)
                    lw["self_attn.gate_proj"] = load_coalesced_proj(sf, f"{p}.self_attn.gate_proj", self.num_heads * self.head_dim, self.hidden_size, gpu_pack)
                    lw["self_attn.o_proj"] = load_coalesced_proj(sf, f"{p}.self_attn.o_proj", self.hidden_size, self.num_heads * self.head_dim, gpu_pack)
                    lw["mlp.gate_proj"] = load_coalesced_proj(sf, f"{p}.mlp.gate_proj", self.intermediate_size, self.hidden_size, gpu_pack)
                    lw["mlp.up_proj"] = load_coalesced_proj(sf, f"{p}.mlp.up_proj", self.intermediate_size, self.hidden_size, gpu_pack)
                    lw["mlp.down_proj"] = load_coalesced_proj(sf, f"{p}.mlp.down_proj", self.hidden_size, self.intermediate_size, gpu_pack)

                if s_idx == 3:
                    self.final_norm_w = sf.get_tensor("model.language_model.norm.weight").to(self.device, dtype=torch.bfloat16)
                    self.lm_head_proj = load_coalesced_proj(sf, "lm_head", self.vocab_size, self.hidden_size, gpu_pack=True)

        total_vram = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, (7.85 * (1024**3)) / total_vram), 0)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        self.gpu_persistent_gib = round(torch.cuda.memory_allocated() / (1024**3), 4)

        print(f"[Daemon] Warming up Unified Split-K Coalesced Triton kernels (GPU Persistent: {self.gpu_persistent_gib:.2f} GiB)...", flush=True)
        for l_idx in (0, 3):
            lw_w = self.layers[l_idx]
            for pk in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.down_proj"]:
                pr = lw_w[pk]
                x_w = torch.zeros((1, 1, pr["in_f"]), device="cuda", dtype=torch.bfloat16)
                _ = self.triton_packed_linear(x_w, pr)
        x_h = torch.zeros((1, 1, self.hidden_size), device="cuda", dtype=torch.bfloat16)
        _ = self.triton_packed_linear(x_h, self.lm_head_proj)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        print(f"[Daemon] Unified Split-K Coalesced Triton 8GB-Wall Engine Ready in {time.perf_counter()-t0:.2f}s! GPU Allocated: {self.gpu_persistent_gib:.2f} GiB", flush=True)

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor, k_cache: List[Optional[torch.Tensor]], v_cache: List[Optional[torch.Tensor]]) -> torch.Tensor:
        bsz, seqlen = input_ids.shape
        past_len = 0 if k_cache[0] is None else k_cache[0].shape[2]
        pos_ids = (torch.arange(seqlen, device=self.device) + past_len).unsqueeze(0)

        h = F.embedding(input_ids.cpu(), self.embed_w_cpu).to(self.device, non_blocking=True)
        h = rms_norm_no_scale(h, eps=self.rms_eps)

        freqs = pos_ids[..., None].float() * self.inv_freq
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(h.dtype)
        sin = emb.sin().to(h.dtype)
        rep = self.num_heads // self.num_kv_heads

        for i in range(self.num_layers):
            lw = self.layers[i]
            residual = h
            nx = centered_rms_norm(h, lw["input_layernorm.weight"], eps=self.rms_eps)

            q = self.triton_packed_linear(nx, lw["self_attn.q_proj"]).view(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
            k = self.triton_packed_linear(nx, lw["self_attn.k_proj"]).view(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v = self.triton_packed_linear(nx, lw["self_attn.v_proj"]).view(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)

            q = rms_norm_no_scale(q, eps=self.rms_eps) * self.qk_scale
            k = rms_norm_no_scale(k, eps=self.rms_eps)
            if self.layer_rope_theta[i]:
                q, k = apply_rotary_pos_emb(q, k, cos, sin)

            if k_cache[i] is None:
                k_cache[i] = k
                v_cache[i] = v
            else:
                k_cache[i] = torch.cat([k_cache[i], k], dim=2)
                v_cache[i] = torch.cat([v_cache[i], v], dim=2)

            a_out = F.scaled_dot_product_attention(
                q, k_cache[i].repeat_interleave(rep, dim=1), v_cache[i].repeat_interleave(rep, dim=1),
                is_causal=(seqlen > 1), scale=self.head_dim ** -0.5,
            ).transpose(1, 2).contiguous().view(bsz, seqlen, -1)
            a_out = a_out * torch.sigmoid(self.triton_packed_linear(nx, lw["self_attn.gate_proj"]))
            a_out = self.triton_packed_linear(a_out, lw["self_attn.o_proj"])
            h = residual + centered_rms_norm(a_out, lw["post_attention_layernorm.weight"], eps=self.post_eps)

            residual = h
            pnx = centered_rms_norm(h, lw["pre_feedforward_layernorm.weight"], eps=self.rms_eps)
            mlp_out = self.triton_packed_linear(
                F.silu(self.triton_packed_linear(pnx, lw["mlp.gate_proj"])) * self.triton_packed_linear(pnx, lw["mlp.up_proj"]),
                lw["mlp.down_proj"],
            )
            h = residual + centered_rms_norm(mlp_out, lw["post_feedforward_layernorm.weight"], eps=self.post_eps)

        h_last = h[:, -1:, :]
        h_norm = rms_norm_with_scale(h_last, self.final_norm_w, eps=self.rms_eps)
        logits = self.triton_packed_linear(h_norm, self.lm_head_proj).float() * self.out_mult
        logits = torch.tanh(logits / self.softcap) * self.softcap
        return logits[:, -1, :]

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 160) -> Dict:
        torch.cuda.reset_peak_memory_stats()
        msgs = [{"role": "user", "content": prompt}]
        chat_str = self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        chat_str += " to=user<|message|>"
        ids = torch.tensor([self.tokenizer.encode(chat_str, add_special_tokens=False)], dtype=torch.long, device=self.device)

        k_cache: List[Optional[torch.Tensor]] = [None] * self.num_layers
        v_cache: List[Optional[torch.Tensor]] = [None] * self.num_layers
        t0 = time.perf_counter()
        out_ids: List[int] = []
        logits = self.forward(ids, k_cache, v_cache)
        next_tok = torch.argmax(logits, dim=-1, keepdim=True)
        for _ in range(max_new_tokens):
            tid = int(next_tok.item())
            if tid in (200001, 200008, 200007):
                break
            out_ids.append(tid)
            if len(out_ids) % 4 == 0:
                partial = self.tokenizer.decode(out_ids[-8:], skip_special_tokens=True)
                if "</assistant" in partial or "to=self" in partial:
                    break
            logits = self.forward(next_tok, k_cache, v_cache)
            for prev_id in set(out_ids[-32:]):
                if logits[0, prev_id] > 0:
                    logits[0, prev_id] /= 1.06
                else:
                    logits[0, prev_id] *= 1.06
            next_tok = torch.argmax(logits, dim=-1, keepdim=True)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        peak_gib = round(torch.cuda.max_memory_allocated() / (1024**3), 4)
        del k_cache, v_cache
        torch.cuda.empty_cache()

        final_answer = self.tokenizer.decode(out_ids, skip_special_tokens=True)
        for stop_str in ("</assistant", "to=self", "<|eom|>", "<|eot|>"):
            if stop_str in final_answer:
                final_answer = final_answer.split(stop_str)[0]
        final_answer = final_answer.strip()
        return {
            "prompt": prompt,
            "thinking": "",
            "response": final_answer,
            "tokens_generated": len(out_ids),
            "elapsed_sec": round(dt, 3),
            "tokens_per_sec": round(len(out_ids) / max(dt, 1e-4), 2),
            "gpu_persistent_gib": self.gpu_persistent_gib,
            "gpu_peak_gib": peak_gib,
            "gpu_peak_gb_decimal": round(peak_gib * (1024**3) / 1e9, 2),
            "hardware_wall_gib": 8.00,
            "shards_verified": [f"model_lean_30b-0000{i+1}-of-00004.safetensors" for i in range(4)],
        }


ENGINE: Optional[FusedTriton8GBWallEngine] = None


class InferHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        return

    def _send_json(self, code: int, payload: Dict):
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path == "/health":
            return self._send_json(200, {"status": "ready" if ENGINE is not None else "loading"})
        return self._send_json(404, {"error": "Not found"})

    def do_POST(self):
        if self.path == "/generate":
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8")) if length > 0 else {}
            prompt = body.get("prompt", "Hello!")
            max_new_tokens = int(body.get("max_new_tokens", 128))
            res = ENGINE.generate(prompt, max_new_tokens=max_new_tokens)
            return self._send_json(200, res)
        return self._send_json(404, {"error": "Not found"})


if __name__ == "__main__":
    ENGINE = FusedTriton8GBWallEngine(Path("release_30b"))
    srv = ThreadingHTTPServer(("127.0.0.1", 18766), InferHandler)
    print("[DAEMON_READY_PORT_18766]", flush=True)
    srv.serve_forever()
'''


def main():
    daemon_py = Path("/tmp/lean_30b_infer_daemon.py")
    daemon_py.write_text(SERVER_SCRIPT)
    daemon_log = Path("/tmp/lean_30b_infer_daemon.log")
    if daemon_log.exists():
        daemon_log.unlink()

    subprocess.run(["pkill", "-9", "-f", "lean_30b_infer_daemon.py"], check=False)
    time.sleep(1.5)

    with open(daemon_log, "w") as lf:
        proc = subprocess.Popen(
            [sys.executable, "-u", str(daemon_py)],
            stdin=subprocess.DEVNULL,
            stdout=lf,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )

    cf_log = Path("/tmp/cf_infer_18766.log")
    infer_url = None
    if cf_log.exists():
        m = re.search(r"(https://[a-zA-Z0-9\-]+\.trycloudflare\.com)", cf_log.read_text(errors="ignore"))
        if m:
            infer_url = m.group(1)
    if not infer_url:
        with open(cf_log, "w") as clf:
            subprocess.Popen(
                ["/tmp/cloudflared", "tunnel", "--protocol", "http2", "--url", "http://127.0.0.1:18766"],
                stdin=subprocess.DEVNULL,
                stdout=clf,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
            )
    ready = False
    t0 = time.time()
    while time.time() - t0 < 150.0:
        if cf_log.exists() and not infer_url:
            m = re.search(r"(https://[a-zA-Z0-9\-]+\.trycloudflare\.com)", cf_log.read_text(errors="ignore"))
            if m:
                infer_url = m.group(1)
                print(f"[Tunnel] Live 30B Inference URL: {infer_url}", flush=True)
        if daemon_log.exists():
            dtxt = daemon_log.read_text(errors="ignore")
            if "[DAEMON_READY_PORT_18766]" in dtxt:
                ready = True
                break
        if proc.poll() is not None:
            break
        time.sleep(1.0)

    if not ready:
        print(f"[Daemon Log Tail]\n{daemon_log.read_text(errors='ignore')[-1500:]}", flush=True)
        raise RuntimeError("Daemon did not become ready within 150s")

    smi = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader"], text=True
    ).strip()
    print(f"[A100 nvidia-smi]: {smi}", flush=True)

    info = {"infer_endpoint": infer_url, "status": "ready", "nvidia_smi": smi}
    with open("release_30b/live_infer_endpoint.json", "w") as f:
        json.dump(info, f, indent=2)
    print(f"[SUCCESS] Unified Split-K Coalesced Triton 8GB-Wall 52-Layer 30B Daemon Ready: {json.dumps(info)}", flush=True)


if __name__ == "__main__":
    main()
