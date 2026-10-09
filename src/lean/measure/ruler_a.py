"""
Ruler A: Post-Hoc Model Compression Description Length Estimator.

Evaluates the minimal description length DL_min(tau) required to preserve
at least a fraction tau of the uncompressed model capability.
Explores the Pareto frontier across:
1. Quantization: int16, int8, int4, int3, int2, int1 (RTN with group scaling)
2. Low-rank SVD decomposition
3. Structured/unstructured sparsity with bitmask encoding
4. Arithmetic entropy coding of quantized states
"""

import copy
import math
from typing import Dict, Any, Callable, List, Tuple
import torch
import torch.nn as nn

from src.lean.measure.dl import (
    dl_quantized_tensor,
    dl_low_rank,
    dl_sparse_tensor,
    dl_entropy_coded_tensor,
    empirical_entropy
)


def quantize_tensor_rtn(weight: torch.Tensor, bits: int, group_size: int = 128) -> Tuple[torch.Tensor, int]:
    """
    Round-to-Nearest (RTN) symmetric group quantization.
    Returns (dequantized_weight, description_length_bits).
    """
    if bits >= 16:
        return weight.clone(), weight.numel() * 16
    
    orig_shape = weight.shape
    flat = weight.detach().clone().flatten()
    numel = flat.numel()
    
    num_groups = math.ceil(numel / group_size)
    padded_len = num_groups * group_size
    if padded_len > numel:
        flat = torch.nn.functional.pad(flat, (0, padded_len - numel))
        
    grouped = flat.view(num_groups, group_size)
    max_val = torch.max(torch.abs(grouped), dim=-1, keepdim=True).values.clamp(min=1e-8)
    
    # Int range: [-2^(b-1), 2^(b-1) - 1]
    qmax = (2 ** (bits - 1)) - 1
    qmin = -(2 ** (bits - 1))
    
    scale = max_val / qmax
    q_weights = torch.round(grouped / scale).clamp(qmin, qmax)
    dequant = (q_weights * scale).view(-1)[:numel].view(orig_shape)
    
    # Description length: quantized weights + scales
    # Use entropy coding if distribution is non-uniform, otherwise standard group bits
    dl_bits = dl_entropy_coded_tensor(q_weights.flatten()[:numel], group_size=group_size)
    return dequant, dl_bits


def svd_low_rank_compress(weight: torch.Tensor, rank: int, factor_bits: int = 4) -> Tuple[torch.Tensor, int]:
    """
    Truncated SVD low-rank factor compression: W ~= U @ V^T with quantized factors.
    Returns (reconstructed_weight, description_length_bits).
    """
    orig_shape = weight.shape
    if len(orig_shape) != 2:
        return weight.clone(), weight.numel() * 16
        
    d_out, d_in = orig_shape
    max_rank = min(d_out, d_in)
    actual_rank = min(rank, max_rank)
    
    if actual_rank <= 0:
        return torch.zeros_like(weight), 0
        
    U, S, Vh = torch.linalg.svd(weight.detach().float(), full_matrices=False)
    
    U_r = U[:, :actual_rank] @ torch.diag(torch.sqrt(S[:actual_rank]))
    V_r = (torch.diag(torch.sqrt(S[:actual_rank])) @ Vh[:actual_rank, :]).T
    
    # Quantize factors U_r and V_r to factor_bits
    if factor_bits < 16:
        q_U, dl_u = quantize_tensor_rtn(U_r, bits=factor_bits, group_size=32)
        q_V, dl_v = quantize_tensor_rtn(V_r, bits=factor_bits, group_size=32)
        reconstructed = (q_U @ q_V.T).to(weight.dtype)
        dl_bits = dl_u + dl_v
    else:
        reconstructed = (U_r @ V_r.T).to(weight.dtype)
        dl_bits = dl_low_rank(d_in, d_out, actual_rank, bit_width=16)
        
    return reconstructed, dl_bits


def magnitude_prune(weight: torch.Tensor, sparsity: float) -> Tuple[torch.Tensor, int]:
    """
    Magnitude pruning with bitmask description length accounting.
    Returns (pruned_weight, description_length_bits).
    """
    if sparsity <= 0.0:
        return weight.clone(), weight.numel() * 16
    if sparsity >= 1.0:
        return torch.zeros_like(weight), weight.numel()
        
    flat = torch.abs(weight.detach().flatten())
    k = int(flat.numel() * (1.0 - sparsity))
    if k == 0:
        return torch.zeros_like(weight), weight.numel()
        
    threshold = torch.kthvalue(flat, flat.numel() - k + 1).values
    mask = (torch.abs(weight) >= threshold).float()
    pruned = weight * mask
    
    num_nonzero = int(mask.sum().item())
    dl_bits = dl_sparse_tensor(weight.numel(), num_nonzero, value_bits=16, index_encoding="bitmask")
    return pruned, dl_bits


class RulerA:
    """
    Estimator A: Post-Hoc Compression Description Length Search.
    Finds the minimal bit length DL_min(tau) such that metric >= tau * baseline_metric.
    """
    
    def __init__(self, model: nn.Module, eval_fn: Callable[[nn.Module], float], higher_is_better: bool = True):
        self.base_model = model
        self.eval_fn = eval_fn
        self.higher_is_better = higher_is_better
        
        # Uncompressed baseline
        self.baseline_score = eval_fn(self.base_model)
        self.uncompressed_dl_bits = sum(p.numel() * 16 for p in self.base_model.parameters())
        
    def evaluate_compression(self, compressor_type: str, param_val: Any) -> Tuple[float, int, bool]:
        """
        Apply a compressor to all 2D linear weight matrices and evaluate score and DL.
        """
        compressed_model = copy.deepcopy(self.base_model)
        total_dl = 0
        
        for name, module in compressed_model.named_modules():
            if isinstance(module, nn.Linear):
                w = module.weight.data
                if compressor_type == "rtn":
                    new_w, bits = quantize_tensor_rtn(w, bits=param_val)
                elif compressor_type == "svd":
                    rank, factor_bits = param_val
                    new_w, bits = svd_low_rank_compress(w, rank=rank, factor_bits=factor_bits)
                elif compressor_type == "prune":
                    new_w, bits = magnitude_prune(w, sparsity=param_val)
                else:
                    new_w, bits = w, w.numel() * 16
                module.weight.data = new_w
                total_dl += bits
            elif hasattr(module, "weight") and module.weight is not None:
                total_dl += module.weight.numel() * 16
            if hasattr(module, "bias") and module.bias is not None:
                total_dl += module.bias.numel() * 16
                
        score = self.eval_fn(compressed_model)
        return score, total_dl
        
    def find_dl_min(self, tau: float = 0.95) -> Dict[str, Any]:
        """
        Execute Pareto sweep across compression types to locate DL_min(tau).
        """
        target_score = self.baseline_score * tau if self.higher_is_better else self.baseline_score / tau
        
        candidates = []
        
        # 1. Sweep RTN bit-widths: [8, 4, 3, 2, 1]
        for bits in [8, 4, 3, 2, 1]:
            score, dl = self.evaluate_compression("rtn", bits)
            passed = (score >= target_score) if self.higher_is_better else (score <= target_score)
            candidates.append({"type": f"rtn_{bits}b", "param": bits, "score": score, "dl_bits": dl, "passed": passed})
            
        # 2. Sweep low-rank SVD ranks with quantized factors
        for rank in [1, 2, 4, 8, 16, 32, 64]:
            for f_bits in [4, 8, 16]:
                score, dl = self.evaluate_compression("svd", (rank, f_bits))
                passed = (score >= target_score) if self.higher_is_better else (score <= target_score)
                candidates.append({"type": f"svd_r{rank}_{f_bits}b", "param": (rank, f_bits), "score": score, "dl_bits": dl, "passed": passed})
                
        # 3. Sweep magnitude pruning: [0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.98]
        for s in [0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.98]:
            score, dl = self.evaluate_compression("prune", s)
            passed = (score >= target_score) if self.higher_is_better else (score <= target_score)
            candidates.append({"type": f"prune_{int(s*100)}%", "param": s, "score": score, "dl_bits": dl, "passed": passed})
            
        # Filter candidates meeting quality threshold tau
        passing = [c for c in candidates if c["passed"]]
        
        if not passing:
            best = {"type": "uncompressed", "param": 16, "score": self.baseline_score, "dl_bits": self.uncompressed_dl_bits, "passed": True}
        else:
            best = min(passing, key=lambda c: c["dl_bits"])
            
        return {
            "tau": tau,
            "baseline_score": self.baseline_score,
            "target_score": target_score,
            "dl_min_bits": best["dl_bits"],
            "dl_min_bytes": best["dl_bits"] / 8.0,
            "compression_ratio": self.uncompressed_dl_bits / float(best["dl_bits"]),
            "optimal_config": best,
            "all_candidates": candidates
        }
