"""
Transformer Programs: Discrete Categorical Attention & Algorithmic Modules.

Implements the Transformer Program framework (Friedman et al., 2023):
- Attention heads parameterized with categorical selection logits
- Gumbel-Softmax discrete annealing to transition continuous routing to discrete RASP selectors
- Discrete extraction method compiling neural attention patterns into RASP selectors
"""

import math
from typing import Dict, Any, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiscreteAttentionHead(nn.Module):
    """
    A single attention head capable of annealing from continuous softmax
    to hard discrete RASP selection matrices.
    """
    def __init__(self, d_model: int, num_predicates: int = 4):
        super().__init__()
        self.d_model = d_model
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.temperature = 1.0

    def forward(self, x: torch.Tensor, hard: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with Gumbel-Softmax / continuous relaxation.
        x: [B, L, D]
        Returns: (output: [B, L, D], attention_weights: [B, L, L])
        """
        B, L, D = x.shape
        q = self.q_proj(x)  # [B, L, D]
        k = self.k_proj(x)  # [B, L, D]
        v = self.v_proj(x)  # [B, L, D]

        # Scaled dot-product logits
        scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(D)  # [B, L, L]

        if hard or not self.training:
            # Deterministic discrete selection (argmax / top-1)
            attn = F.softmax(scores / max(self.temperature, 0.01), dim=-1)
            # Hard 1-hot in eval mode
            index = attn.max(dim=-1, keepdim=True)[1]
            attn_hard = torch.zeros_like(attn).scatter_(-1, index, 1.0)
            attn = attn_hard - attn.detach() + attn  # Straight-through estimator
        else:
            # Gumbel-Softmax relaxation
            attn = F.gumbel_softmax(scores, tau=self.temperature, hard=False, dim=-1)

        out = torch.bmm(attn, v)
        return out, attn

    def extract_discrete_pattern(self, x: torch.Tensor) -> List[List[int]]:
        """Extract the exact deterministic integer key index selected for each query position."""
        self.eval()
        with torch.no_grad():
            _, attn = self.forward(x, hard=True)
            # attn shape: [B, L, L] -> take first batch element
            selected_indices = attn[0].argmax(dim=-1).cpu().tolist()
            return selected_indices


class TransformerProgramBlock(nn.Module):
    """A Transformer Program block consisting of discrete attention and an MLP."""
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.head = DiscreteAttentionHead(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, hard: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        norm_x = self.ln1(x)
        attn_out, attn_weights = self.head(norm_x, hard=hard)
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x, attn_weights


class TransformerProgram(nn.Module):
    """
    Distillable Transformer Program model that maps token sequences to algorithmic outputs
    and compiles into discrete RASP ASTs.
    """
    def __init__(self, vocab_size: int, d_model: int = 32, d_ff: int = 64, num_layers: int = 1):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(64, d_model)
        self.blocks = nn.ModuleList([TransformerProgramBlock(d_model, d_ff) for _ in range(num_layers)])
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, tokens: torch.Tensor, hard: bool = False) -> torch.Tensor:
        B, L = tokens.shape
        pos = torch.arange(L, device=tokens.device).unsqueeze(0).expand(B, L)
        x = self.token_emb(tokens) + self.pos_emb(pos)
        
        for block in self.blocks:
            x, _ = block(x, hard=hard)
            
        logits = self.head(x)
        return logits

    def set_temperature(self, temp: float):
        """Set temperature across all attention heads."""
        for block in self.blocks:
            block.head.temperature = max(temp, 0.01)

    def get_attention_patterns(self, tokens: torch.Tensor) -> List[torch.Tensor]:
        """Extract all attention weight matrices."""
        patterns = []
        B, L = tokens.shape
        pos = torch.arange(L, device=tokens.device).unsqueeze(0).expand(B, L)
        x = self.token_emb(tokens) + self.pos_emb(pos)
        for block in self.blocks:
            norm_x = block.ln1(x)
            attn_out, attn = block.head(norm_x, hard=True)
            x = x + attn_out + block.ffn(block.ln2(x + attn_out))
            patterns.append(attn)
        return patterns
