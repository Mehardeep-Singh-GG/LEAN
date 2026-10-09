"""
Exact Description Length (DL) Accounting Engine for Project LEAN.

Measures the single physical currency of model representation:
Total bits required to completely specify parameters, codebooks, scales, and index structures.
Prevents deceptive reporting of raw sparsity or bit-width without structural overhead.
"""

import math
from typing import Dict, Any, Optional
import torch
import torch.nn as nn


def dl_dense_tensor(numel: int, bit_width: int = 16) -> int:
    """Exact bits required for a standard uncompressed tensor."""
    return numel * bit_width


def dl_quantized_tensor(numel: int, bit_width: int, group_size: int = 128) -> int:
    """
    Exact bits for a group-quantized tensor (int-b).
    Includes:
    - Quantized weight values (numel * bit_width)
    - Per-group scale constants (fp16 = 16 bits per group)
    - Per-group zero-point constants (int-b per group)
    """
    num_groups = math.ceil(numel / group_size)
    weight_bits = numel * bit_width
    scale_bits = num_groups * 16
    zero_bits = num_groups * bit_width
    return weight_bits + scale_bits + zero_bits


def empirical_entropy(tensor: torch.Tensor) -> float:
    """
    Calculate the empirical Shannon entropy H(X) in bits per element.
    H(X) = -sum p(x) * log2(p(x))
    """
    flat = tensor.detach().flatten()
    if flat.numel() == 0:
        return 0.0
    _, counts = torch.unique(flat, return_counts=True)
    probs = counts.float() / flat.numel()
    entropy = -torch.sum(probs * torch.log2(probs)).item()
    return max(0.0, float(entropy))


def dl_entropy_coded_tensor(tensor: torch.Tensor, group_size: int = 128) -> int:
    """
    Description length under arithmetic/entropy coding.
    Calculates empirical Shannon entropy of quantized symbols + frequency table overhead + scale/zero overhead.
    """
    numel = tensor.numel()
    if numel == 0:
        return 0
    h = empirical_entropy(tensor)
    data_bits = math.ceil(numel * h)
    
    unique_vals = len(torch.unique(tensor.detach().flatten()))
    codebook_overhead = unique_vals * 16  # 16-bit frequency counts for decoder
    
    num_groups = math.ceil(numel / group_size)
    scale_overhead = num_groups * 16
    return data_bits + codebook_overhead + scale_overhead


def dl_low_rank(d_in: int, d_out: int, rank: int, bit_width: int = 16) -> int:
    """
    Exact bits for a low-rank factored matrix W ~= U @ V^T.
    U is (d_out, rank), V is (d_in, rank).
    """
    factor_elements = rank * (d_in + d_out)
    return factor_elements * bit_width


def dl_sparse_tensor(
    numel: int,
    num_nonzero: int,
    value_bits: int = 16,
    index_encoding: str = "bitmask"
) -> int:
    """
    Exact bits for sparse matrix representation.
    - bitmask: 1 bit per total matrix element + value_bits per non-zero value.
    - coo: ceil(log2(numel)) index bits per non-zero value + value_bits.
    """
    val_bits = num_nonzero * value_bits
    if index_encoding == "bitmask":
        idx_bits = numel
    elif index_encoding == "coo":
        idx_bits = num_nonzero * math.ceil(math.log2(max(numel, 2)))
    else:
        raise ValueError(f"Unknown index encoding: {index_encoding}")
    return val_bits + idx_bits


class ModelDescriptionLengthAccountant:
    """Aggregates and verifies complete description length of models."""
    
    def __init__(self):
        self.layer_breakdown: Dict[str, Dict[str, Any]] = {}
        
    def add_dense_layer(self, name: str, tensor: torch.Tensor, bit_width: int = 16):
        numel = tensor.numel()
        bits = dl_dense_tensor(numel, bit_width)
        self.layer_breakdown[name] = {
            "type": "dense",
            "numel": numel,
            "bits": bits,
            "bit_width": bit_width
        }
        
    def add_quantized_layer(self, name: str, numel: int, bit_width: int, group_size: int = 128):
        bits = dl_quantized_tensor(numel, bit_width, group_size)
        self.layer_breakdown[name] = {
            "type": "quantized",
            "numel": numel,
            "bits": bits,
            "bit_width": bit_width,
            "group_size": group_size
        }
        
    def add_low_rank_layer(self, name: str, d_in: int, d_out: int, rank: int, bit_width: int = 16):
        bits = dl_low_rank(d_in, d_out, rank, bit_width)
        numel = rank * (d_in + d_out)
        self.layer_breakdown[name] = {
            "type": "low_rank",
            "numel": numel,
            "bits": bits,
            "rank": rank,
            "original_numel": d_in * d_out
        }
        
    def add_sparse_layer(self, name: str, numel: int, num_nonzero: int, value_bits: int = 16):
        bits = dl_sparse_tensor(numel, num_nonzero, value_bits, index_encoding="bitmask")
        self.layer_breakdown[name] = {
            "type": "sparse",
            "numel": numel,
            "num_nonzero": num_nonzero,
            "bits": bits,
            "sparsity": 1.0 - (num_nonzero / numel)
        }
        
    def total_bits(self) -> int:
        return sum(layer["bits"] for layer in self.layer_breakdown.values())
        
    def total_bytes(self) -> float:
        return self.total_bits() / 8.0
        
    def total_megabytes(self) -> float:
        return self.total_bytes() / (1024.0 * 1024.0)
        
    def bits_per_parameter(self, original_numel: int) -> float:
        return self.total_bits() / float(original_numel)
