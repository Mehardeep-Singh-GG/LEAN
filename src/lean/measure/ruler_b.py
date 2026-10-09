"""
Ruler B: Prequential (Online) Minimum Description Length (MDL) Estimator.

Measures the intrinsic information content required for a model to learn a capability,
independent of any post-hoc compression heuristic.

Formula:
  L_prequential(D) = cost(theta_0) + sum_{t=1}^T -log2 P_{theta_{t-1}}(D_t)

where the dataset D is divided into sequential blocks D_1, ..., D_T,
and each block's transmission cost in bits is measured under the model trained on prior blocks.
"""

import math
from typing import Dict, Any, List, Callable, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def compute_nll_bits(model: nn.Module, batch_x: torch.Tensor, batch_y: torch.Tensor) -> float:
    """Compute negative log-likelihood of batch in bits (log base 2)."""
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    with torch.no_grad():
        logits = model(batch_x)
        # Handle sequence or classification outputs
        if logits.dim() == 3:
            logits = logits.view(-1, logits.size(-1))
            batch_y = batch_y.view(-1)
        loss_nats = criterion(logits, batch_y).item()
    # 1 nat = 1 / ln(2) bits
    loss_bits = loss_nats / math.log(2)
    return loss_bits


def calculate_prequential_code_length(
    model_fn: Callable[[], nn.Module],
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    num_blocks: int = 5,
    epochs_per_block: int = 15,
    lr: float = 1e-3,
    prior_param_bits: int = 1000
) -> Dict[str, Any]:
    """
    Compute online prequential code length over sequential data blocks.
    """
    total_samples = train_x.size(0)
    block_size = max(1, total_samples // num_blocks)
    
    model = model_fn()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    cumulative_code_bits = float(prior_param_bits)
    block_records = []
    
    for b in range(num_blocks):
        start_idx = b * block_size
        end_idx = min(start_idx + block_size, total_samples)
        if start_idx >= end_idx:
            break
            
        cur_x = train_x[start_idx:end_idx]
        cur_y = train_y[start_idx:end_idx]
        
        # 1. Transmission cost of new block under model trained so far
        block_nll_bits = compute_nll_bits(model, cur_x, cur_y)
        cumulative_code_bits += block_nll_bits
        
        block_records.append({
            "block": b + 1,
            "samples": cur_x.size(0),
            "nll_bits": block_nll_bits,
            "bits_per_sample": block_nll_bits / max(1, cur_x.size(0)),
            "cumulative_bits": cumulative_code_bits
        })
        
        # 2. Update model online with current block to convergence
        dataset = TensorDataset(cur_x, cur_y)
        loader = DataLoader(dataset, batch_size=min(32, cur_x.size(0)), shuffle=True)
        
        model.train()
        for epoch in range(epochs_per_block):
            total_loss = 0.0
            for bx, by in loader:
                optimizer.zero_grad()
                out = model(bx)
                if out.dim() == 3:
                    out = out.view(-1, out.size(-1))
                    by = by.view(-1)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            # If model achieves near-zero error on block, can exit early
            if total_loss / len(loader) < 0.01 and epoch >= 20:
                break
                
    return {
        "total_samples": total_samples,
        "num_blocks": len(block_records),
        "total_prequential_bits": cumulative_code_bits,
        "total_prequential_bytes": cumulative_code_bits / 8.0,
        "bits_per_sample": cumulative_code_bits / float(total_samples),
        "blocks": block_records
    }
