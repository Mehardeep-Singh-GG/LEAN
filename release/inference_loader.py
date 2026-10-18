"""
release/inference_loader.py
===========================
Standalone Public Inference Loader for Qwen2.5-7B-ThinkingCore-KAC.

Zero imports from proprietary compiler (zero imports from `src.lean`).
Runs the 3.84 GiB baked checkpoint on consumer 8 GB VRAM with dynamic on-the-fly
decompression and optional canonical BM25 retrieval routing.

Usage:
  python release/inference_loader.py --prompt "The capital of France is"
  python release/inference_loader.py --prompt "What is the speed of light in vacuum?" --rag
"""

import os
import sys
import re
import math
import time
import json
import pickle
import argparse
from pathlib import Path
from collections import Counter
from typing import Dict, Any, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM


class StandaloneBM25:
    """Fast, zero-dependency in-memory BM25 index for external knowledge store."""
    def __init__(self, data: Optional[Dict[str, Any]] = None):
        if data:
            self.k1 = data.get("k1", 1.5)
            self.b = data.get("b", 0.75)
            self.corpus = data.get("corpus", [])
            self.doc_freqs = Counter(data.get("doc_freqs", {}))
            self.doc_lens = data.get("doc_lens", [])
            self.avg_doc_len = data.get("avg_doc_len", 0.0)
        else:
            self.k1, self.b = 1.5, 0.75
            self.corpus, self.doc_freqs, self.doc_lens, self.avg_doc_len = [], Counter(), [], 0.0

    @staticmethod
    def tokenize(text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    def search(self, query: str, top_k: int = 1) -> List[Tuple[Dict[str, Any], float]]:
        q_toks = self.tokenize(query)
        scores = []
        n_docs = len(self.corpus)
        for i, doc in enumerate(self.corpus):
            doc_toks = self.tokenize(doc["text"])
            doc_len = self.doc_lens[i] if i < len(self.doc_lens) else len(doc_toks)
            score = 0.0
            doc_counts = Counter(doc_toks)
            for t in q_toks:
                if t in self.doc_freqs:
                    df = self.doc_freqs[t]
                    idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
                    tf = doc_counts[t]
                    num = tf * (self.k1 + 1.0)
                    denom = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / max(1e-6, self.avg_doc_len)))
                    score += idf * (num / max(1e-6, denom))
            scores.append((doc, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


class _BM25SafeUnpickler(pickle.Unpickler):
    """Allows unpickling legacy BM25 index objects without needing source modules."""
    def find_class(self, module, name):
        if name in ("BM25Index", "StandaloneBM25"):
            return StandaloneBM25
        return super().find_class(module, name)


class PackedQuantizedLinear(nn.Module):
    """
    Public Consumer Runtime Layer:
    Stores packed integer weights in VRAM and decompresses on-the-fly during forward pass.
    """
    def __init__(self, in_features: int, out_features: int, bits: int = 4, group_size: int = 128, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.group_size = group_size
        self.num_groups = (in_features * out_features + group_size - 1) // group_size
        self.pack_factor = 8 // bits
        
        packed_bytes = (in_features * out_features + self.pack_factor - 1) // self.pack_factor
        self.register_buffer("packed", torch.zeros(packed_bytes, dtype=torch.uint8))
        self.register_buffer("scales", torch.zeros(self.num_groups, dtype=torch.float16))
        self.register_buffer("channel_scales", torch.ones(in_features, dtype=torch.float16))
        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=torch.float16))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        target_dtype = x.dtype
        device = x.device
        
        # Invert channel salience: (x / s) * W_scaled^T
        x_scaled = (x / self.channel_scales.view(1, 1, -1).to(dtype=target_dtype, device=device))
        
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
            raise NotImplementedError(f"Unsupported bit precision: {self.bits}")

        unpacked_grouped = unpacked.view(self.num_groups, self.group_size).float()
        w = (unpacked_grouped * self.scales.unsqueeze(-1).float()).view(-1)[:self.in_features * self.out_features]
        w = w.view(self.out_features, self.in_features).to(dtype=target_dtype, device=device)
        
        return F.linear(x_scaled, w, self.bias.to(dtype=target_dtype, device=device) if self.bias is not None else None)


def load_kac_model(
    checkpoint_dir: str = "release",
    base_model_id: str = "Qwen/Qwen2.5-7B-Instruct",
    device: str = "cuda"
) -> nn.Module:
    """
    Instantiates model skeleton and streams weights from safetensor shards.
    """
    ckpt_path = Path(checkpoint_dir)
    config_file = ckpt_path / "config.json"
    if config_file.exists():
        config = AutoConfig.from_pretrained(str(ckpt_path))
    else:
        config = AutoConfig.from_pretrained(base_model_id)

    print(f"Instantiating model skeleton ({config.num_hidden_layers} layers) on meta device...")
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16)

    # Patch quantized linear layers
    for layer in model.model.layers:
        for name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            lin = getattr(layer.self_attn, name)
            has_bias = getattr(lin, "bias", None) is not None or name in ["q_proj", "k_proj", "v_proj"]
            setattr(layer.self_attn, name, PackedQuantizedLinear(lin.in_features, lin.out_features, bits=4, group_size=128, bias=has_bias))
        for name in ["gate_proj", "up_proj", "down_proj"]:
            lin = getattr(layer.mlp, name)
            has_bias = getattr(lin, "bias", None) is not None
            setattr(layer.mlp, name, PackedQuantizedLinear(lin.in_features, lin.out_features, bits=2, group_size=128, bias=has_bias))

    # Move skeleton to target device
    model = model.to_empty(device=device)

    # Discover Shards
    index_file = ckpt_path / "model_kac.safetensors.index.json"
    if index_file.exists():
        with open(index_file, "r") as f:
            idx_data = json.load(f)
        shard_names = sorted(list(set(idx_data["weight_map"].values())))
        shard_files = [ckpt_path / s for s in shard_names]
    else:
        shard_files = sorted(list(ckpt_path.glob("*.safetensors")))

    if not shard_files:
        raise FileNotFoundError(f"No safetensors shards found in {ckpt_path}")

    print(f"Loading {len(shard_files)} safetensor shards directly into {device.upper()} VRAM...")
    param_dict = dict(model.named_parameters())
    buffer_dict = dict(model.named_buffers())

    for sf in shard_files:
        print(f"  Streaming shard: {sf.name} ({sf.stat().st_size / (1024**3):.2f} GiB)...")
        with safe_open(str(sf), framework="pt", device=device) as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                if key in param_dict:
                    param_dict[key].data.copy_(tensor)
                elif key in buffer_dict:
                    buffer_dict[key].data.copy_(tensor)

    model.eval()
    if device == "cuda" and torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        print(f"Model loaded successfully! Live GPU VRAM: {allocated:.2f} GB")
    return model


def load_bm25_index(index_path: str = "release/bm25_index.pkl") -> Optional[StandaloneBM25]:
    """Loads the 18 KB canonical BM25 index."""
    p = Path(index_path)
    if not p.exists():
        return None
    with open(p, "rb") as f:
        try:
            data = pickle.load(f)
        except Exception:
            f.seek(0)
            data = _BM25SafeUnpickler(f).load()

    if isinstance(data, dict):
        return StandaloneBM25(data)
    elif isinstance(data, StandaloneBM25):
        return data
    else:
        return StandaloneBM25({
            "corpus": getattr(data, "corpus", []),
            "doc_freqs": getattr(data, "doc_freqs", {}),
            "doc_lens": getattr(data, "doc_lens", []),
            "avg_doc_len": getattr(data, "avg_doc_len", 0.0),
        })


def main():
    parser = argparse.ArgumentParser(description="Qwen2.5-7B-ThinkingCore-KAC Public Inference Loader")
    parser.add_argument("--prompt", type=str, default="The capital of France is", help="Generation prompt")
    parser.add_argument("--checkpoint_dir", type=str, default="release", help="Path to checkpoint directory")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Base HuggingFace model ID for tokenizer")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Target device (cuda/cpu)")
    parser.add_argument("--max_new_tokens", type=int, default=20, help="Number of tokens to generate")
    parser.add_argument("--rag", action="store_true", help="Enable BM25 retrieval augmentation")
    args = parser.parse_args()

    print("=" * 80)
    print("Qwen2.5-7B-ThinkingCore-KAC: Standalone Public Runtime")
    print(f"Device: {args.device.upper()} | Precision: 4b-Attn / 2b-MLP (3.84 GiB)")
    print("=" * 80)

    # 1. Load Tokenizer
    tok_source = args.checkpoint_dir if (Path(args.checkpoint_dir) / "tokenizer.json").exists() else args.model_id
    print(f"Loading tokenizer from {tok_source}...")
    tokenizer = AutoTokenizer.from_pretrained(tok_source, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Optional BM25 Retrieval Augmentation
    active_prompt = args.prompt
    if args.rag:
        bm25_file = Path(args.checkpoint_dir) / "bm25_index.pkl"
        bm25 = load_bm25_index(str(bm25_file))
        if bm25 and bm25.corpus:
            results = bm25.search(args.prompt, top_k=1)
            if results:
                passage = results[0][0]["text"]
                active_prompt = f"Context: {passage}\nQuestion: {args.prompt}\nAnswer:"
                print(f"[RAG Injected Context]: {passage}")

    # 3. Load Model
    model = load_kac_model(checkpoint_dir=args.checkpoint_dir, base_model_id=args.model_id, device=args.device)

    # 4. Generate
    print(f"\nPrompt: '{active_prompt}'")
    print("Generating tokens...", flush=True)

    inputs = tokenizer(active_prompt, return_tensors="pt").to(args.device)
    input_ids = inputs["input_ids"]

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(args.max_new_tokens):
            out = model(input_ids)
            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            if next_token.item() == tokenizer.eos_token_id:
                break
    t_gen = time.perf_counter() - t0

    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    print("\n" + "=" * 80)
    print("OUTPUT RESULT:")
    print("=" * 80)
    print(generated_text)
    print("=" * 80)
    new_tokens_count = input_ids.shape[1] - inputs["input_ids"].shape[1]
    print(f"Generated {new_tokens_count} tokens in {t_gen:.2f}s ({new_tokens_count / max(1e-6, t_gen):.2f} tokens/sec).")


if __name__ == "__main__":
    main()
