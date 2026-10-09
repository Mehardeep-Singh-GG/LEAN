"""
World-Randomized MDL Factorization (WRMF) Architectures.

Implements:
1. ThinkingCore: Shared procedural algorithmic engine containing skill sub-circuits
   and shared projection heads.
2. Memory Triad:
   - LoRAMemory: Per-world rank-r adaptation matrices (W = W_core + (alpha/r) * B @ A).
   - PKMMemory: Per-world Product-Key Memory with key-value slot addressing.
   - EntityEmbeddingMemory: Per-world entity and relation embedding tables.
3. WRMFNetwork: Multi-world factorized network theta_m = theta_core + Mem_m supporting
   joint training, frozen-core transfer, and memory swapping.
"""

import math
from typing import Dict, List, Tuple, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.lean.synthetic.skills import SKILL_METADATA


class ThinkingCore(nn.Module):
    """
    Shared Algorithmic Thinking Core.
    Houses the computational invariant across worlds:
    - Shared base projection trunk & normalization.
    - Dedicated procedural skill circuits scaled by RASP complexity K(s).
    - Shared categorical output head.
    """
    def __init__(
        self,
        d_model: int = 16,
        vocab_size: int = 64,
        skill_ids: Optional[List[str]] = None
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.skill_ids = skill_ids or ["S1", "S2", "S3", "S5", "S6"]
        
        # Shared projection trunk & normalization
        self.base_trunk = nn.Linear(d_model, d_model)
        self.base_norm = nn.LayerNorm(d_model)
        self.shared_head = nn.Linear(d_model, vocab_size)
        
        # Procedural skill circuits proportional to RASP complexity K(s)
        self.skill_circuits = nn.ModuleDict()
        for s_id in self.skill_ids:
            meta = next((v for k, v in SKILL_METADATA.items() if v.get("id") == s_id or k == s_id), {})
            rasp_toks = meta.get("rasp_tokens", 32)
            d_s = max(4, int(math.ceil(rasp_toks / 8.0)))
            self.skill_circuits[s_id] = nn.Sequential(
                nn.Linear(d_model, d_s, bias=False),
                nn.GELU(),
                nn.Linear(d_s, d_model, bias=False)
            )
            
    def forward_skill(self, skill_id: str, x: torch.Tensor) -> torch.Tensor:
        """Executes procedural skill circuit through shared trunk and head."""
        circuit = self.skill_circuits[skill_id]
        h = self.base_norm(self.base_trunk(x) + circuit(x))
        return self.shared_head(h)
        
    def forward_head(self, h: torch.Tensor) -> torch.Tensor:
        """Projects latent hidden state to vocabulary logits."""
        return self.shared_head(self.base_norm(h))


class LoRAMemory(nn.Module):
    """
    Per-World LoRA Memory (Memory Type A).
    Low-rank adaptation matrices: Delta W = (alpha/r) * B @ A.
    """
    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        d_model: int = 16,
        r: int = 4,
        alpha: float = 8.0
    ):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.d_model = d_model
        self.r = r
        self.scaling = alpha / r
        
        # Base entity & relation table
        self.ent_emb = nn.Embedding(num_entities, d_model // 2)
        self.rel_emb = nn.Embedding(num_relations, d_model // 2)
        
        # Low-rank factor adaptation matrices
        self.lora_A = nn.Parameter(torch.randn(r, d_model) * (1.0 / math.sqrt(d_model)))
        self.lora_B = nn.Parameter(torch.zeros(d_model, r))
        self.proj = nn.Linear(d_model, d_model)
        
    def forward(self, ent_ids: torch.Tensor, rel_ids: torch.Tensor) -> torch.Tensor:
        e = self.ent_emb(ent_ids)
        r = self.rel_emb(rel_ids)
        x = torch.cat([e, r], dim=-1)
        # Base projection + LoRA delta
        h_base = self.proj(x)
        lora_delta = (x @ self.lora_A.t()) @ self.lora_B.t() * self.scaling
        return h_base + lora_delta


class PKMMemory(nn.Module):
    """
    Per-World Product-Key Memory (Memory Type B).
    Associative key-value memory addressing per-world slots.
    """
    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        d_model: int = 16,
        num_slots: int = 64,
        top_k: int = 2
    ):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.d_model = d_model
        self.num_slots = num_slots
        self.top_k = top_k
        
        self.ent_emb = nn.Embedding(num_entities, d_model // 2)
        self.rel_emb = nn.Embedding(num_relations, d_model // 2)
        self.query_proj = nn.Linear(d_model, d_model)
        
        # Key-Value slots per world
        self.keys = nn.Parameter(torch.randn(num_slots, d_model) * (1.0 / math.sqrt(d_model)))
        self.values = nn.Parameter(torch.randn(num_slots, d_model) * 0.1)
        
    def forward(self, ent_ids: torch.Tensor, rel_ids: torch.Tensor) -> torch.Tensor:
        e = self.ent_emb(ent_ids)
        r = self.rel_emb(rel_ids)
        query = self.query_proj(torch.cat([e, r], dim=-1))
        
        # Cosine / dot-product routing to memory slots
        scores = (query @ self.keys.t()) * (1.0 / math.sqrt(self.d_model))
        weights = F.softmax(scores, dim=-1)
        
        # Soft attention over value slots
        mem_out = weights @ self.values
        return query + mem_out


class EntityEmbeddingMemory(nn.Module):
    """
    Per-World Entity Embedding Memory (Memory Type C).
    Pure tabular associative memory representing world-specific entity-relation knowledge.
    """
    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        d_model: int = 16
    ):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.d_model = d_model
        
        self.ent_emb = nn.Embedding(num_entities, d_model // 2)
        self.rel_emb = nn.Embedding(num_relations, d_model // 2)
        self.adapter = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        
    def forward(self, ent_ids: torch.Tensor, rel_ids: torch.Tensor) -> torch.Tensor:
        e = self.ent_emb(ent_ids)
        r = self.rel_emb(rel_ids)
        x = torch.cat([e, r], dim=-1)
        return self.adapter(x)


def create_memory_module(
    memory_type: str,
    num_entities: int,
    num_relations: int,
    d_model: int = 16
) -> nn.Module:
    """Factory creating memory module corresponding to memory type."""
    if memory_type == "lora":
        return LoRAMemory(num_entities, num_relations, d_model=d_model, r=4)
    elif memory_type == "pkm":
        return PKMMemory(num_entities, num_relations, d_model=d_model, num_slots=max(32, num_entities // 2))
    elif memory_type == "entity_emb":
        return EntityEmbeddingMemory(num_entities, num_relations, d_model=d_model)
    else:
        raise ValueError(f"Unknown memory type: {memory_type}")


class WRMFNetwork(nn.Module):
    """
    World-Randomized MDL Factorization (WRMF) Container Network:
    theta_m = theta_core + Mem_m
    
    Houses:
    - Single shared ThinkingCore (invariant across worlds).
    - Collection of M distinct World Memories (world-specific).
    """
    def __init__(
        self,
        num_worlds: int,
        num_entities: int,
        num_relations: int,
        vocab_size: int = 64,
        d_model: int = 16,
        memory_type: str = "lora",
        skill_ids: Optional[List[str]] = None
    ):
        super().__init__()
        self.num_worlds = num_worlds
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.memory_type = memory_type
        self.skill_ids = skill_ids or ["S1", "S2", "S3", "S5", "S6"]
        
        # 1. Shared invariant Thinking Core
        self.core = ThinkingCore(d_model=d_model, vocab_size=vocab_size, skill_ids=self.skill_ids)
        
        # 2. Per-world memories
        self.memories = nn.ModuleList([
            create_memory_module(memory_type, num_entities, num_relations, d_model=d_model)
            for _ in range(num_worlds)
        ])
        
    def freeze_core(self) -> None:
        """Strictly freezes all parameters in the Thinking Core (for transfer testing)."""
        for p in self.core.parameters():
            p.requires_grad = False
            
    def unfreeze_core(self) -> None:
        """Unfreezes all parameters in the Thinking Core."""
        for p in self.core.parameters():
            p.requires_grad = True
            
    def forward_fact(self, world_idx: int, ent_ids: torch.Tensor, rel_ids: torch.Tensor) -> torch.Tensor:
        """Executes factual query for a specific world m."""
        mem = self.memories[world_idx]
        h = mem(ent_ids, rel_ids)
        return self.core.forward_head(h)
        
    def forward_skill(self, skill_id: str, x: torch.Tensor) -> torch.Tensor:
        """Executes procedural skill circuit in the shared thinking core."""
        return self.core.forward_skill(skill_id, x)
