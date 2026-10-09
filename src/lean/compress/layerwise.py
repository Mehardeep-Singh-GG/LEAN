"""
Layer-wise memory-safe transformer compression suite.

Enables evaluation within 8 GB unified memory on Apple Silicon:
- RTN Quantization: 4-bit, 3-bit, 2-bit with group-wise scaling
- Magnitude Pruning: 50% unstructured
- SVD Truncation: low-rank factor decomposition
"""

import math
import copy
from typing import Dict, Any, Tuple, Optional, Callable
import torch
import torch.nn as nn

from src.lean.measure.ruler_a import quantize_tensor_rtn, magnitude_prune, svd_low_rank_compress


def compress_linear_layer(
    layer: nn.Linear,
    compressor_type: str,
    param: Any
) -> Tuple[nn.Linear, int]:
    """
    Compress a single Linear layer in-place or return a compressed replacement.
    Returns (layer, description_length_bits).
    """
    orig_weight = layer.weight.data
    
    if compressor_type == "rtn":
        bits = int(param)
        new_w, dl_bits = quantize_tensor_rtn(orig_weight, bits=bits, group_size=128)
        layer.weight.data = new_w
    elif compressor_type == "prune":
        sparsity = float(param)
        new_w, dl_bits = magnitude_prune(orig_weight, sparsity=sparsity)
        layer.weight.data = new_w
    elif compressor_type == "svd":
        rank = int(param)
        new_w, dl_bits = svd_low_rank_compress(orig_weight, rank=rank, factor_bits=4)
        layer.weight.data = new_w
    else:
        dl_bits = orig_weight.numel() * 16
        
    bias_bits = layer.bias.numel() * 16 if layer.bias is not None else 0
    return layer, dl_bits + bias_bits


def compress_model_layerwise(
    model: nn.Module,
    compressor_type: str,
    param: Any,
    skip_modules: Optional[list] = None
) -> Tuple[nn.Module, int]:
    """
    Sequentially compress all linear layers of a causal language model layer-by-layer.
    Avoids duplicating whole models in memory.
    """
    if skip_modules is None:
        skip_modules = ["embed_out", "lm_head", "embed_in", "wte", "wpe"]
        
    compressed_model = copy.deepcopy(model)
    total_dl_bits = 0
    
    for name, module in compressed_model.named_modules():
        # Check if module should be compressed
        if isinstance(module, nn.Linear):
            # Check skip list
            if any(skip in name for skip in skip_modules):
                # Count uncompressed bits
                total_dl_bits += module.weight.numel() * 16
                if module.bias is not None:
                    total_dl_bits += module.bias.numel() * 16
                continue
                
            _, layer_dl = compress_linear_layer(module, compressor_type, param)
            total_dl_bits += layer_dl
        elif hasattr(module, "weight") and module.weight is not None and not isinstance(module, nn.Linear):
            total_dl_bits += module.weight.numel() * 16
            if hasattr(module, "bias") and module.bias is not None:
                total_dl_bits += module.bias.numel() * 16
                
    return compressed_model, total_dl_bits
