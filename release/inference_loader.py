"""
Project LEAN — Open-Checkpoint Runtime Loader (Qwen2.5-7B-ThinkingCore-KAC)
===========================================================================
Public inference-only runtime for loading the pre-compiled 3.84 GiB AWQ-KAC
checkpoint (`model_kac.safetensors`) and 18 KB BM25 Retrieval Store (`bm25_index.pkl`).

NOTE: This module contains ONLY the dynamic bit-unpacking forward pass
(4-bit GQA Attention + 2-bit SwiGLU MLP) and zero-copy safetensors loader.
The NEXUS / Priced-Quanta Knapsack Compiler that produced the checkpoint is proprietary.
"""

import os
import re
import math
import pickle
import argparse
from collections import Counter
from typing import Dict, Any, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding


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


BASE_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"


class PackedQuantizedLinear(nn.Module):
    """
    Inference-only bit-packed linear layer (Group-128 symmetric quantization
    with pre-calibrated per-channel activation scales).
    Unpacks 2 weights/byte (4-bit Attention) or 4 weights/byte (2-bit MLP) on the fly.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bits: int = 4,
        group_size: int = 128,
        bias: bool = False,
    ):
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
        else:
            raise ValueError(f"Unsupported bit-width: {self.bits}")

        unpacked_grouped = unpacked.view(self.num_groups, self.group_size).float()
        w = (unpacked_grouped * self.scales.unsqueeze(-1).float()).view(-1)[: self.in_features * self.out_features]
        w = w.view(self.out_features, self.in_features).to(dtype=target_dtype)

        bias = self.bias.to(dtype=target_dtype) if self.bias is not None else None
        return F.linear(x_scaled, w, bias).to(dtype=target_dtype)


class StandaloneBM25Lookup:
    """Zero-dependency runtime wrapper for the 18 KB canonical BM25 retrieval store."""
    def __init__(self, index_path: str):
        with open(index_path, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, dict):
            self.k1 = data.get("k1", 1.5)
            self.b = data.get("b", 0.75)
            self.corpus = data["corpus"]
            self.doc_freqs = Counter(data["doc_freqs"])
            self.doc_lens = data["doc_lens"]
            self.avg_doc_len = float(data["avg_doc_len"])
        else:
            self.k1 = getattr(data, "k1", 1.5)
            self.b = getattr(data, "b", 0.75)
            self.corpus = data.corpus
            self.doc_freqs = data.doc_freqs
            self.doc_lens = data.doc_lens
            self.avg_doc_len = float(data.avg_doc_len)

    @staticmethod
    def tokenize(text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    def augment_prompt(self, query: str, top_k: int = 1) -> str:
        q_toks = self.tokenize(query)
        scores = []
        n_docs = len(self.corpus)
        for i, doc in enumerate(self.corpus):
            doc_toks = self.tokenize(doc["text"])
            doc_len = self.doc_lens[i]
            score = 0.0
            doc_counts = Counter(doc_toks)
            for t in q_toks:
                if t in self.doc_freqs:
                    df = self.doc_freqs[t]
                    idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
                    tf = doc_counts.get(t, 0)
                    denom = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / max(1.0, self.avg_doc_len)))
                    score += idf * (tf * (self.k1 + 1.0)) / max(1e-4, denom)
            scores.append((doc, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        if not scores or top_k <= 0:
            return query
        evidence = scores[0][0]["text"]
        return f"Passage: {evidence}\nQuestion: {query}"


def detect_best_device() -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_thinking_core(
    checkpoint_path: str = "model_kac.safetensors",
    bm25_path: Optional[str] = "bm25_index.pkl",
    base_model_id: str = BASE_MODEL_ID,
    device: Optional[str] = None,
) -> Tuple[nn.Module, Any, Optional[StandaloneBM25Lookup]]:
    """
    Streams the 3.84 GiB pre-packed checkpoint directly into GPU / Apple Silicon / CPU
    without ever downloading or instantiating the 15.2 GB FP16 base weights.
    """
    target_device = device or detect_best_device()
    config_source = os.path.dirname(checkpoint_path) if os.path.exists(os.path.join(os.path.dirname(checkpoint_path), "config.json")) else base_model_id

    config = AutoConfig.from_pretrained(config_source, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(config_source, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16)

    for layer in model.model.layers:
        for proj_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            old = getattr(layer.self_attn, proj_name)
            has_bias = getattr(old, "bias", None) is not None
            setattr(
                layer.self_attn,
                proj_name,
                PackedQuantizedLinear(old.in_features, old.out_features, bits=4, group_size=128, bias=has_bias),
            )
        for proj_name in ["gate_proj", "up_proj", "down_proj"]:
            old = getattr(layer.mlp, proj_name)
            has_bias = getattr(old, "bias", None) is not None
            setattr(
                layer.mlp,
                proj_name,
                PackedQuantizedLinear(old.in_features, old.out_features, bits=2, group_size=128, bias=has_bias),
            )

    load_dev = target_device if target_device.startswith("cuda") else "cpu"
    with safe_open(checkpoint_path, framework="pt", device=load_dev) as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            if target_device != load_dev:
                tensor = tensor.to(target_device)
            set_module_tensor_to_device(model, key, device=target_device, value=tensor)

    model.model.rotary_emb = Qwen2RotaryEmbedding(config=config, device=target_device)
    model.eval()

    retriever = StandaloneBM25Lookup(bm25_path) if (bm25_path and os.path.exists(bm25_path)) else None
    return model, tokenizer, retriever


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Qwen2.5-7B-ThinkingCore-KAC (3.84 GiB) Inference")
    parser.add_argument("--checkpoint", type=str, default="model_kac.safetensors")
    parser.add_argument("--bm25", type=str, default="bm25_index.pkl")
    parser.add_argument("--prompt", type=str, default="The capital of France is")
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--no-rag", action="store_true")
    args = parser.parse_args()

    dev = detect_best_device()
    print(f"[ThinkingCore-KAC] Loading 3.84 GiB checkpoint onto {dev}...")
    model, tok, retriever = load_thinking_core(args.checkpoint, None if args.no_rag else args.bm25, device=dev)

    query = args.prompt if (args.no_rag or retriever is None) else retriever.augment_prompt(args.prompt)
    inputs = tok(query, return_tensors="pt").to(dev)
    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    print("\n--- Output ---")
    print(tok.decode(out_ids[0], skip_special_tokens=True))
