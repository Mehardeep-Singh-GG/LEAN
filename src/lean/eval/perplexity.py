"""
Evaluation module: Exact Perplexity Calculation.

Computes sliding-window cross-entropy perplexity on target evaluation corpora:
  PPL = exp( 1/N * sum_{i=1}^N L_i )
"""

import math
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from tqdm import tqdm


def calculate_perplexity(
    model: nn.Module,
    tokenizer: Any,
    text: str,
    max_length: int = 1024,
    stride: int = 512,
    device: Optional[str] = None
) -> Dict[str, Any]:
    """
    Compute sliding window perplexity for a given text and causal language model.
    """
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        
    model.eval()
    model.to(device)
    
    encodings = tokenizer(text, return_tensors="pt")
    input_ids_all = encodings.input_ids if hasattr(encodings, "input_ids") else encodings["input_ids"]
    seq_len = input_ids_all.size(1)
    
    nlls = []
    total_tokens = 0
    
    with torch.no_grad():
        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - max(begin_loc, end_loc - stride)
            
            input_ids = input_ids_all[:, begin_loc:end_loc].to(device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100
            
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss * trg_len
            
            nlls.append(neg_log_likelihood)
            total_tokens += trg_len
            
            if end_loc == seq_len:
                break
                
    total_nll = torch.stack(nlls).sum()
    ppl = math.exp((total_nll / total_tokens).item())
    
    return {
        "perplexity": float(ppl),
        "total_tokens": int(total_tokens),
        "mean_loss_nats": float((total_nll / total_tokens).item()),
        "mean_loss_bits": float((total_nll / total_tokens).item() / math.log(2))
    }
