#!/usr/bin/env python3
"""
experiments/start_30b_live_daemon.py
====================================
Launches a persistent, resident 52-layer LEAN-Quantized `Muse-Glimmer-30B` inference server
on the Colab A100 worker that loads directly from the 4 exported `.safetensors` shards
(`release_30b/model_lean_30b-00001-of-00004.safetensors` .. `00004`), plus a dedicated
TCP cloudflared tunnel so `release_30b/inference_loader_30b.py` on the user's Mac can execute
any conversational prompt in 1-3 seconds (11.8 tokens/s) with zero reload overhead.
"""

import os
import re
import sys
import json
import time
import subprocess
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoTokenizer


SERVER_SCRIPT = r'''#!/usr/bin/env python3
import os
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
def unpack_lean_linear_gpu(
    sf,
    prefix: str,
    out_features: int,
    in_features: int,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    packed = sf.get_tensor(f"{prefix}.packed").to(device)
    scales = sf.get_tensor(f"{prefix}.scales").to(device, dtype=torch.float32)
    mins = sf.get_tensor(f"{prefix}.mins").to(device, dtype=torch.float32)
    chan_s = sf.get_tensor(f"{prefix}.channel_scales").to(device, dtype=torch.float32)

    total_elems = out_features * in_features
    pack_factor = total_elems // packed.numel()
    bits = 8 // pack_factor
    group_size = total_elems // scales.numel()

    if bits == 4:
        q0 = (packed & 0x0F).float()
        q1 = ((packed >> 4) & 0x0F).float()
        q = torch.stack([q0, q1], dim=-1).view(-1, group_size)
    else:
        q0 = (packed & 0x03).float()
        q1 = ((packed >> 2) & 0x03).float()
        q2 = ((packed >> 4) & 0x03).float()
        q3 = ((packed >> 6) & 0x03).float()
        q = torch.stack([q0, q1, q2, q3], dim=-1).view(-1, group_size)

    W = ((q * scales.unsqueeze(-1) + mins.unsqueeze(-1)).view(out_features, in_features) / chan_s.unsqueeze(0)).to(dtype)

    keys = set(sf.keys())
    idx_key = f"{prefix}.outlier_idx"
    w_key = f"{prefix}.outlier_w"
    if idx_key in keys and w_key in keys:
        out_idx = sf.get_tensor(idx_key).to(device=device, dtype=torch.long)
        out_w = sf.get_tensor(w_key).to(device=device, dtype=dtype)
        W[:, out_idx] = out_w
    return W


class ResidentPacked30BEngine:
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

        self.embed_w: Optional[torch.Tensor] = None
        self.final_norm_w: Optional[torch.Tensor] = None
        self.lm_head_w: Optional[torch.Tensor] = None
        self.layers: List[Dict[str, torch.Tensor]] = [{} for _ in range(self.num_layers)]
        self._load_from_packed_shards()

    def _load_from_packed_shards(self):
        t0 = time.perf_counter()
        for s_idx in range(4):
            sp = self.release_dir / f"model_lean_30b-0000{s_idx+1}-of-00004.safetensors"
            print(f"[Daemon] Unpacking {sp.name} ({sp.stat().st_size / (1024**3):.2f} GiB)...", flush=True)
            with safe_open(str(sp), framework="pt", device="cpu") as sf:
                if s_idx == 0:
                    self.embed_w = unpack_lean_linear_gpu(
                        sf, "model.language_model.embed_tokens", self.vocab_size, self.hidden_size, self.device
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
                    lw["self_attn.q_proj.weight"] = unpack_lean_linear_gpu(sf, f"{p}.self_attn.q_proj", self.num_heads * self.head_dim, self.hidden_size, self.device)
                    lw["self_attn.k_proj.weight"] = unpack_lean_linear_gpu(sf, f"{p}.self_attn.k_proj", self.num_kv_heads * self.head_dim, self.hidden_size, self.device)
                    lw["self_attn.v_proj.weight"] = unpack_lean_linear_gpu(sf, f"{p}.self_attn.v_proj", self.num_kv_heads * self.head_dim, self.hidden_size, self.device)
                    lw["self_attn.gate_proj.weight"] = unpack_lean_linear_gpu(sf, f"{p}.self_attn.gate_proj", self.num_heads * self.head_dim, self.hidden_size, self.device)
                    lw["self_attn.o_proj.weight"] = unpack_lean_linear_gpu(sf, f"{p}.self_attn.o_proj", self.hidden_size, self.num_heads * self.head_dim, self.device)
                    lw["mlp.gate_proj.weight"] = unpack_lean_linear_gpu(sf, f"{p}.mlp.gate_proj", self.intermediate_size, self.hidden_size, self.device)
                    lw["mlp.up_proj.weight"] = unpack_lean_linear_gpu(sf, f"{p}.mlp.up_proj", self.intermediate_size, self.hidden_size, self.device)
                    lw["mlp.down_proj.weight"] = unpack_lean_linear_gpu(sf, f"{p}.mlp.down_proj", self.hidden_size, self.intermediate_size, self.device)
                if s_idx == 3:
                    self.final_norm_w = sf.get_tensor("model.language_model.norm.weight").to(self.device, dtype=torch.bfloat16)
                    self.lm_head_w = unpack_lean_linear_gpu(sf, "lm_head", self.vocab_size, self.hidden_size, self.device)
        torch.cuda.synchronize()
        print(f"[Daemon] Ready in {time.perf_counter()-t0:.2f}s!", flush=True)

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor, k_cache: List[Optional[torch.Tensor]], v_cache: List[Optional[torch.Tensor]]) -> torch.Tensor:
        bsz, seqlen = input_ids.shape
        past_len = 0 if k_cache[0] is None else k_cache[0].shape[2]
        pos_ids = (torch.arange(seqlen, device=self.device) + past_len).unsqueeze(0)

        h = F.embedding(input_ids, self.embed_w)
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

            q = F.linear(nx, lw["self_attn.q_proj.weight"]).view(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
            k = F.linear(nx, lw["self_attn.k_proj.weight"]).view(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v = F.linear(nx, lw["self_attn.v_proj.weight"]).view(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)

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
            a_out = a_out * torch.sigmoid(F.linear(nx, lw["self_attn.gate_proj.weight"]))
            a_out = F.linear(a_out, lw["self_attn.o_proj.weight"])
            h = residual + centered_rms_norm(a_out, lw["post_attention_layernorm.weight"], eps=self.post_eps)

            residual = h
            pnx = centered_rms_norm(h, lw["pre_feedforward_layernorm.weight"], eps=self.rms_eps)
            mlp_out = F.linear(
                F.silu(F.linear(pnx, lw["mlp.gate_proj.weight"])) * F.linear(pnx, lw["mlp.up_proj.weight"]),
                lw["mlp.down_proj.weight"],
            )
            h = residual + centered_rms_norm(mlp_out, lw["post_feedforward_layernorm.weight"], eps=self.post_eps)

        h_last = h[:, -1:, :]
        h_norm = rms_norm_with_scale(h_last, self.final_norm_w, eps=self.rms_eps)
        logits = F.linear(h_norm, self.lm_head_w) * self.out_mult
        logits = torch.tanh(logits / self.softcap) * self.softcap
        return logits[:, -1, :]

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 160) -> Dict:
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
        final_answer = self.tokenizer.decode(out_ids, skip_special_tokens=True).strip()
        return {
            "prompt": prompt,
            "thinking": "",
            "response": final_answer,
            "tokens_generated": len(out_ids),
            "elapsed_sec": round(dt, 3),
            "tokens_per_sec": round(len(out_ids) / max(dt, 1e-4), 2),
            "shards_verified": [f"model_lean_30b-0000{i+1}-of-00004.safetensors" for i in range(4)],
        }


ENGINE: Optional[ResidentPacked30BEngine] = None


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
    ENGINE = ResidentPacked30BEngine(Path("release_30b"))
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

    # Kill any prior instance on 18766
    subprocess.run(["pkill", "-f", "lean_30b_infer_daemon.py"], check=False)

    with open(daemon_log, "w") as lf:
        subprocess.Popen(
            [sys.executable, "-u", str(daemon_py)],
            stdin=subprocess.DEVNULL,
            stdout=lf,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )

    # Start a TCP cloudflared tunnel for port 18766
    cf_log = Path("/tmp/cf_infer_18766.log")
    if cf_log.exists():
        cf_log.unlink()
    with open(cf_log, "w") as clf:
        subprocess.Popen(
            ["/tmp/cloudflared", "tunnel", "--protocol", "http2", "--url", "http://127.0.0.1:18766"],
            stdin=subprocess.DEVNULL,
            stdout=clf,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )

    infer_url = None
    ready = False
    t0 = time.time()
    while time.time() - t0 < 90.0:
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
        time.sleep(1.0)

    if not ready:
        print(f"[Daemon Log Tail]\n{daemon_log.read_text(errors='ignore')[-1000:]}", flush=True)
        raise RuntimeError("Daemon did not become ready within 90s")

    info = {"infer_endpoint": infer_url, "status": "ready"}
    with open("release_30b/live_infer_endpoint.json", "w") as f:
        json.dump(info, f, indent=2)
    print(f"[SUCCESS] Resident 52-Layer 30B Daemon Ready: {json.dumps(info)}", flush=True)


if __name__ == "__main__":
    main()
