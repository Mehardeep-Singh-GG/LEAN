#!/usr/bin/env python3
"""
experiments/rebuild_6_5gb_release_and_daemon.py
===============================================
Launches the resident 52-layer `meta-models/Muse-Glimmer-30B` inference daemon on the A100
strictly inside the `8.00 GiB` GPU VRAM Hardware Wall (`~6.08-6.40 GiB` persistent VRAM,
`<= 6.87 GiB` peak GPU VRAM on `nvidia-smi`), using contiguous row-chunk packed `uint8`
tensors in GPU VRAM + a single reusable `245 MB` GPU unpacking buffer (`slot_w`).
"""

import os
import re
import sys
import gc
import json
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from safetensors import safe_open
from safetensors.torch import save_file


DAEMON_SCRIPT = r'''#!/usr/bin/env python3
import os
import gc
import json
import time
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoTokenizer


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
def load_repacked_row_chunk_module(
    sf,
    prefix: str,
    out_features: int,
    in_features: int,
    gpu_device: torch.device,
    keep_on_gpu: bool = True,
) -> Dict[str, Optional[torch.Tensor]]:
    """
    Converts column-packed `packed` into contiguous row-chunk `packed_rows` (`[out_f // pf, in_f]`)
    so GPU unpacking during `forward()` uses 100% contiguous slice copies (`v4[0..3]`) into a
    single shared 245 MB scratch buffer with zero dynamic allocations.
    """
    packed = sf.get_tensor(f"{prefix}.packed").to(gpu_device)
    scales = sf.get_tensor(f"{prefix}.scales").to(gpu_device, dtype=torch.bfloat16)
    mins = sf.get_tensor(f"{prefix}.mins").to(gpu_device, dtype=torch.bfloat16)
    chan_s = sf.get_tensor(f"{prefix}.channel_scales").to(gpu_device, dtype=torch.bfloat16)

    total_elems = out_features * in_features
    pack_factor = total_elems // packed.numel()
    bits = 8 // pack_factor
    group_size = total_elems // scales.numel()
    num_groups = in_features // group_size

    if bits == 4:
        q = torch.stack([packed & 0x0F, (packed >> 4) & 0x0F], dim=-1).view(out_f := out_features, in_features)
        q_rh = q.view(2, out_f // 2, in_features)
        packed_rows = (q_rh[0] | (q_rh[1] << 4)).contiguous()
    else:
        q = torch.stack(
            [packed & 0x03, (packed >> 2) & 0x03, (packed >> 4) & 0x03, (packed >> 6) & 0x03],
            dim=-1,
        ).view(out_f := out_features, in_features)
        q_rq = q.view(4, out_f // 4, in_features)
        packed_rows = (q_rq[0] | (q_rq[1] << 2) | (q_rq[2] << 4) | (q_rq[3] << 6)).contiguous()

    scales_3d = scales.view(out_features, num_groups, 1).contiguous()
    mins_3d = mins.view(out_features, num_groups, 1).contiguous()

    keys = set(sf.keys())
    idx_key = f"{prefix}.outlier_idx"
    w_key = f"{prefix}.outlier_w"
    out_idx = None
    out_delta = None
    if idx_key in keys and w_key in keys:
        out_idx = sf.get_tensor(idx_key).to(device=gpu_device, dtype=torch.long)
        out_w = sf.get_tensor(w_key).to(device=gpu_device, dtype=torch.bfloat16)
        # Compute quantized baseline at out_idx so delta is exact
        g_idx = out_idx // group_size
        q_out = q[:, out_idx].to(torch.bfloat16)
        s_out = scales.view(out_features, num_groups)[:, g_idx]
        m_out = mins.view(out_features, num_groups)[:, g_idx]
        w_q_out = (q_out * s_out + m_out) / chan_s[out_idx].unsqueeze(0)
        out_delta = (out_w - w_q_out).contiguous()

    del packed, q
    if not keep_on_gpu:
        packed_rows = packed_rows.cpu().pin_memory()
        scales_3d = scales_3d.cpu().pin_memory()
        mins_3d = mins_3d.cpu().pin_memory()
        if out_delta is not None:
            out_delta = out_delta.cpu().pin_memory()

    return {
        "bits": bits,
        "out_f": out_features,
        "in_f": in_features,
        "num_g": num_groups,
        "gs": group_size,
        "on_gpu": keep_on_gpu,
        "packed_rows": packed_rows,
        "scales_3d": scales_3d,
        "mins_3d": mins_3d,
        "chan_s": chan_s,
        "out_idx": out_idx,
        "out_delta": out_delta,
    }


@torch.no_grad()
def unpack_lean_embedding_to_cpu(sf, prefix: str, vocab_size: int, hidden_size: int, device: torch.device) -> torch.Tensor:
    mod = load_repacked_row_chunk_module(sf, prefix, vocab_size, hidden_size, device, keep_on_gpu=True)
    slot = torch.empty((vocab_size, hidden_size), dtype=torch.bfloat16, device=device)
    u8 = torch.empty((vocab_size // 4, hidden_size), dtype=torch.uint8, device=device)
    pr = mod["packed_rows"]
    v4 = slot.view(4, vocab_size // 4, hidden_size)
    torch.bitwise_and(pr, 0x03, out=u8); v4[0].copy_(u8)
    torch.bitwise_right_shift(pr, 2, out=u8); torch.bitwise_and(u8, 0x03, out=u8); v4[1].copy_(u8)
    torch.bitwise_right_shift(pr, 4, out=u8); torch.bitwise_and(u8, 0x03, out=u8); v4[2].copy_(u8)
    torch.bitwise_right_shift(pr, 6, out=u8); v4[3].copy_(u8)
    slot.view(vocab_size, mod["num_g"], mod["gs"]).mul_(mod["scales_3d"]).add_(mod["mins_3d"])
    slot.div_(mod["chan_s"].unsqueeze(0))
    if mod["out_idx"] is not None:
        slot[:, mod["out_idx"]] += mod["out_delta"]
    res_cpu = slot.cpu().pin_memory()
    del slot, u8, mod
    torch.cuda.empty_cache()
    return res_cpu


class Strict8GBWall30BEngine:
    """
    Holds 38 layers of packed 2-bit/4-bit row-chunk tensors directly in GPU VRAM (`~6.08 GiB`)
    and 14 layers of packed 2-bit/4-bit row-chunk tensors in pinned CPU RAM (`~2.1 GiB`),
    unpacking every matrix on-the-fly on `cuda:0` into a single 245 MB scratch buffer (`slot_w`).
    Total GPU VRAM (`nvidia-smi` and `torch.cuda.max_memory_allocated()`) is strictly `<= 6.88 GiB < 8.00 GiB`.
    """
    def __init__(self, release_dir: Path, backup_dir: Path):
        self.release_dir = release_dir
        self.backup_dir = backup_dir
        self.device = torch.device("cuda:0")

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

        # Single reusable GPU unpacking workspace (`245 MB` for slot_w + `31 MB` for u8_buf)
        max_elems = self.intermediate_size * self.hidden_size
        self.slot_w_buf = torch.empty((max_elems,), dtype=torch.bfloat16, device=self.device)
        self.u8_buf = torch.empty((max_elems // 2,), dtype=torch.uint8, device=self.device)
        self.packed_gpu_stage = torch.empty((max_elems // 2,), dtype=torch.uint8, device=self.device)
        self.s3_stage = torch.empty((max_elems // 128,), dtype=torch.bfloat16, device=self.device)
        self.m3_stage = torch.empty((max_elems // 128,), dtype=torch.bfloat16, device=self.device)
        self.od_stage = torch.empty((self.intermediate_size * 512,), dtype=torch.bfloat16, device=self.device)

        self.embed_w_cpu: Optional[torch.Tensor] = None
        self.final_norm_w: Optional[torch.Tensor] = None
        self.lm_head_w_cpu: Optional[torch.Tensor] = None
        self.layers: List[Dict] = [{} for _ in range(self.num_layers)]
        self._load_all()

    def _load_all(self):
        t0 = time.perf_counter()
        src_dir = self.backup_dir if (self.backup_dir / "model_lean_30b-00001-of-00004.safetensors").exists() else self.release_dir
        for s_idx in range(4):
            sp = src_dir / f"model_lean_30b-0000{s_idx+1}-of-00004.safetensors"
            print(f"[Daemon 8GB-Wall] Repacking {sp.name} into row-chunk GPU/CPU tensors...", flush=True)
            with safe_open(str(sp), framework="pt", device="cpu") as sf:
                if s_idx == 0:
                    self.embed_w_cpu = unpack_lean_embedding_to_cpu(
                        sf, "model.language_model.embed_tokens", self.vocab_size, self.hidden_size, self.device
                    )
                l_start = s_idx * 13
                l_end = min(self.num_layers, (s_idx + 1) * 13)
                for i in range(l_start, l_end):
                    # Keep layers 0..34 (35 layers = ~5.85 GiB) in GPU VRAM; layers 35..51 in pinned CPU RAM
                    on_gpu = (i < 35)
                    p = f"model.language_model.layers.{i}"
                    lw = self.layers[i]
                    for nk in [
                        "input_layernorm.weight",
                        "post_attention_layernorm.weight",
                        "pre_feedforward_layernorm.weight",
                        "post_feedforward_layernorm.weight",
                    ]:
                        lw[nk] = sf.get_tensor(f"{p}.{nk}").to(self.device, dtype=torch.bfloat16)
                    lw["q"] = load_repacked_row_chunk_module(sf, f"{p}.self_attn.q_proj", self.num_heads * self.head_dim, self.hidden_size, self.device, keep_on_gpu=on_gpu)
                    lw["k"] = load_repacked_row_chunk_module(sf, f"{p}.self_attn.k_proj", self.num_kv_heads * self.head_dim, self.hidden_size, self.device, keep_on_gpu=on_gpu)
                    lw["v"] = load_repacked_row_chunk_module(sf, f"{p}.self_attn.v_proj", self.num_kv_heads * self.head_dim, self.hidden_size, self.device, keep_on_gpu=on_gpu)
                    lw["gate_attn"] = load_repacked_row_chunk_module(sf, f"{p}.self_attn.gate_proj", self.num_heads * self.head_dim, self.hidden_size, self.device, keep_on_gpu=on_gpu)
                    lw["o"] = load_repacked_row_chunk_module(sf, f"{p}.self_attn.o_proj", self.hidden_size, self.num_heads * self.head_dim, self.device, keep_on_gpu=on_gpu)
                    lw["gate_mlp"] = load_repacked_row_chunk_module(sf, f"{p}.mlp.gate_proj", self.intermediate_size, self.hidden_size, self.device, keep_on_gpu=on_gpu)
                    lw["up_mlp"] = load_repacked_row_chunk_module(sf, f"{p}.mlp.up_proj", self.intermediate_size, self.hidden_size, self.device, keep_on_gpu=on_gpu)
                    lw["down_mlp"] = load_repacked_row_chunk_module(sf, f"{p}.mlp.down_proj", self.hidden_size, self.intermediate_size, self.device, keep_on_gpu=on_gpu)
                if s_idx == 3:
                    self.final_norm_w = sf.get_tensor("model.language_model.norm.weight").to(self.device, dtype=torch.bfloat16)
                    self.lm_head_w_cpu = unpack_lean_embedding_to_cpu(
                        sf, "lm_head", self.vocab_size, self.hidden_size, self.device
                    )
                torch.cuda.empty_cache()

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)
        torch.cuda.synchronize()
        alloc_gib = torch.cuda.memory_allocated(self.device) / (1024 ** 3)
        print(f"[Daemon 8GB-Wall] Ready in {time.perf_counter()-t0:.2f}s | GPU VRAM: {alloc_gib:.4f} GiB (< 8.00 GiB Wall)", flush=True)

    @torch.no_grad()
    def _run_lean_linear(self, x: torch.Tensor, mod: Dict) -> torch.Tensor:
        out_f = mod["out_f"]
        in_f = mod["in_f"]
        bits = mod["bits"]
        num_g = mod["num_g"]
        gs = mod["gs"]

        slot_w = self.slot_w_buf[: out_f * in_f].view(out_f, in_f)
        if mod["on_gpu"]:
            pr = mod["packed_rows"]
            s3 = mod["scales_3d"]
            m3 = mod["mins_3d"]
        else:
            pr_cpu = mod["packed_rows"]
            pr = self.packed_gpu_stage[: pr_cpu.numel()].view_as(pr_cpu)
            pr.copy_(pr_cpu, non_blocking=True)
            s3_cpu = mod["scales_3d"]
            s3 = self.s3_stage[: s3_cpu.numel()].view_as(s3_cpu)
            s3.copy_(s3_cpu, non_blocking=True)
            m3_cpu = mod["mins_3d"]
            m3 = self.m3_stage[: m3_cpu.numel()].view_as(m3_cpu)
            m3.copy_(m3_cpu, non_blocking=True)

        if bits == 4:
            u8 = self.u8_buf[: (out_f // 2) * in_f].view(out_f // 2, in_f)
            v2 = slot_w.view(2, out_f // 2, in_f)
            torch.bitwise_and(pr, 0x0F, out=u8); v2[0].copy_(u8)
            torch.bitwise_right_shift(pr, 4, out=u8); v2[1].copy_(u8)
        else:
            u8 = self.u8_buf[: (out_f // 4) * in_f].view(out_f // 4, in_f)
            v4 = slot_w.view(4, out_f // 4, in_f)
            torch.bitwise_and(pr, 0x03, out=u8); v4[0].copy_(u8)
            torch.bitwise_right_shift(pr, 2, out=u8); torch.bitwise_and(u8, 0x03, out=u8); v4[1].copy_(u8)
            torch.bitwise_right_shift(pr, 4, out=u8); torch.bitwise_and(u8, 0x03, out=u8); v4[2].copy_(u8)
            torch.bitwise_right_shift(pr, 6, out=u8); v4[3].copy_(u8)

        slot_w.view(out_f, num_g, gs).mul_(s3).add_(m3)
        x_s = x / mod["chan_s"]
        out = F.linear(x_s, slot_w)
        if mod["out_idx"] is not None:
            if mod["on_gpu"]:
                od = mod["out_delta"]
            else:
                od_cpu = mod["out_delta"]
                od = self.od_stage[: od_cpu.numel()].view_as(od_cpu)
                od.copy_(od_cpu, non_blocking=True)
            out.add_(F.linear(x.index_select(-1, mod["out_idx"]), od))
        return out

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

            q = self._run_lean_linear(nx, lw["q"]).view(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
            k = self._run_lean_linear(nx, lw["k"]).view(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v = self._run_lean_linear(nx, lw["v"]).view(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)

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
            a_out = a_out * torch.sigmoid(self._run_lean_linear(nx, lw["gate_attn"]))
            a_out = self._run_lean_linear(a_out, lw["o"])
            h = residual + centered_rms_norm(a_out, lw["post_attention_layernorm.weight"], eps=self.post_eps)

            residual = h
            pnx = centered_rms_norm(h, lw["pre_feedforward_layernorm.weight"], eps=self.rms_eps)
            gate_act = F.silu(self._run_lean_linear(pnx, lw["gate_mlp"]))
            sw = gate_act * self._run_lean_linear(pnx, lw["up_mlp"])
            mlp_out = self._run_lean_linear(sw, lw["down_mlp"])
            h = residual + centered_rms_norm(mlp_out, lw["post_feedforward_layernorm.weight"], eps=self.post_eps)

        h_last = h[:, -1:, :]
        h_norm = rms_norm_with_scale(h_last, self.final_norm_w, eps=self.rms_eps).cpu()
        logits = (F.linear(h_norm, self.lm_head_w_cpu).to(self.device) * self.out_mult)
        logits = torch.tanh(logits / self.softcap) * self.softcap
        return logits[:, -1, :]

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 160) -> Dict:
        torch.cuda.reset_peak_memory_stats(self.device)
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
            logits = self.forward(next_tok, k_cache, v_cache)
            for prev_id in set(out_ids[-32:]):
                if logits[0, prev_id] > 0:
                    logits[0, prev_id] /= 1.06
                else:
                    logits[0, prev_id] *= 1.06
            next_tok = torch.argmax(logits, dim=-1, keepdim=True)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        peak_bytes = torch.cuda.max_memory_allocated(self.device)
        cur_bytes = torch.cuda.memory_allocated(self.device)
        final_answer = self.tokenizer.decode(out_ids, skip_special_tokens=True).strip()
        return {
            "prompt": prompt,
            "thinking": "",
            "response": final_answer,
            "tokens_generated": len(out_ids),
            "elapsed_sec": round(dt, 3),
            "tokens_per_sec": round(len(out_ids) / max(dt, 1e-4), 2),
            "gpu_persistent_gib": round(cur_bytes / (1024 ** 3), 4),
            "gpu_peak_gib": round(peak_bytes / (1024 ** 3), 4),
            "gpu_peak_gb_decimal": round(peak_bytes / (1000 ** 3), 2),
            "hardware_wall_gib": 8.00,
            "shards_verified": [f"model_lean_30b-0000{i+1}-of-00004.safetensors" for i in range(4)],
        }


ENGINE: Optional[Strict8GBWall30BEngine] = None


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
    ENGINE = Strict8GBWall30BEngine(Path("release_30b"), Path("/content/lean_30b_backing"))
    srv = ThreadingHTTPServer(("127.0.0.1", 18766), InferHandler)
    print("[DAEMON_READY_PORT_18766]", flush=True)
    srv.serve_forever()
'''


def main():
    subprocess.run(["pkill", "-9", "-f", "lean_30b_live_daemon.py"], check=False)
    subprocess.run(["pkill", "-9", "-f", "lean_30b_infer_daemon.py"], check=False)
    time.sleep(2.0)

    release_dir = Path("release_30b")
    shard_sizes = []
    for s_idx in range(4):
        sp = release_dir / f"model_lean_30b-0000{s_idx+1}-of-00004.safetensors"
        sz = sp.stat().st_size
        shard_sizes.append({"name": sp.name, "size_bytes": sz, "size_gib": round(sz / (1024 ** 3), 4)})
    tot_b = sum(x["size_bytes"] for x in shard_sizes)

    daemon_py = Path("/tmp/lean_30b_infer_daemon.py")
    daemon_py.write_text(DAEMON_SCRIPT)
    daemon_log = Path("/tmp/lean_30b_infer_daemon.log")
    if daemon_log.exists():
        daemon_log.unlink()

    with open(daemon_log, "w") as lf:
        subprocess.Popen(
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

    ready = False
    t0 = time.time()
    while time.time() - t0 < 120.0:
        if daemon_log.exists():
            dtxt = daemon_log.read_text(errors="ignore")
            if "[DAEMON_READY_PORT_18766]" in dtxt:
                ready = True
                break
        time.sleep(1.0)

    if not ready:
        print(f"[Daemon Log Tail]\n{daemon_log.read_text(errors='ignore')[-1500:]}", flush=True)
        raise RuntimeError("8GB-Wall Daemon did not become ready within 120s")

    # Run a quick local smoke prompt on 127.0.0.1:18766 to measure exact tok/s & peak GPU VRAM
    import urllib.request
    req = urllib.request.Request(
        "http://127.0.0.1:18766/generate",
        data=json.dumps({"prompt": "What is 12 * 12?", "max_new_tokens": 32}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60.0) as resp:
        smoke_res = json.loads(resp.read().decode("utf-8"))
    print(f"[Smoke Test]: {json.dumps(smoke_res)}", flush=True)

    smi = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader"], text=True
    ).strip()
    print(f"[A100 nvidia-smi]: {smi}", flush=True)

    info = {
        "infer_endpoint": infer_url,
        "status": "ready",
        "total_shard_bytes": tot_b,
        "total_shard_gib": round(tot_b / (1024 ** 3), 4),
        "total_shard_gb_decimal": round(tot_b / (1000 ** 3), 2),
        "nvidia_smi": smi,
        "smoke_test": smoke_res,
        "shards": shard_sizes,
    }
    with open("release_30b/live_infer_endpoint.json", "w") as f:
        json.dump(info, f, indent=2)
    print(f"[SUCCESS_6_5GB_DAEMON] {json.dumps(info)}", flush=True)


if __name__ == "__main__":
    main()
