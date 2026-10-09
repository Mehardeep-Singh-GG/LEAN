"""
Multi-World WRMF Trainer and Evaluation Engine.

Executes:
1. Joint WRMF training across M synthetic worlds with MDL complexity penalty.
2. Decisive Transfer Test: Frozen core + fresh memory on unseen worlds (Acceptance: >= 95%).
3. Memory Swap Test: Cross-evaluating memories across worlds.
4. Null Control Test: Shuffled skill mappings across worlds causing core structural collapse.
"""

import time
import math
import copy
from typing import Dict, List, Tuple, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.lean.synthetic.world import SyntheticWorld, ATTRIBUTE_VOCABS
from src.lean.wrmf.models import WRMFNetwork, create_memory_module
from src.lean.wrmf.loss import compute_wrmf_loss, compute_module_physical_dl, compute_module_dl_proxy


def generate_canonical_skill_data(
    skill_ids: List[str],
    vocab_size: int = 64,
    d_model: int = 24,
    num_samples: int = 50,
    seed: int = 42
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Generates canonical deterministic procedural skill input-target pairs.
    Every world shares the identical procedural logic.
    """
    rng = np.random.RandomState(seed)
    data = {}
    for s_id in skill_ids:
        x_raw = rng.randn(num_samples, d_model)
        x = torch.tensor(x_raw, dtype=torch.float32)
        if s_id == "S1":
            # Copy identity
            y = torch.tensor(np.argmax(x_raw, axis=-1) % vocab_size, dtype=torch.long)
        elif s_id == "S2":
            # Sequence inversion
            y = torch.tensor(np.argmax(x_raw[:, ::-1], axis=-1) % vocab_size, dtype=torch.long)
        elif s_id == "S3":
            # Numerical rank sort
            y = torch.tensor(np.argsort(x_raw[:, :8], axis=-1)[:, -1] % vocab_size, dtype=torch.long)
        elif s_id == "S5":
            # Key-value associative lookup
            k = np.argmax(x_raw[:, :4], axis=-1)
            v = np.argmax(x_raw[:, 4:8], axis=-1)
            y = torch.tensor((k * 5 + v) % vocab_size, dtype=torch.long)
        elif s_id == "S6":
            # Modular addition
            a = np.argmax(x_raw[:, :8], axis=-1)
            b = np.argmax(x_raw[:, 8:16], axis=-1)
            y = torch.tensor((a + b) % vocab_size, dtype=torch.long)
        else:
            y = torch.tensor(np.argmax(x_raw, axis=-1) % vocab_size, dtype=torch.long)
        data[s_id] = (x, y)
    return data


def build_world_data(
    k_entities: int,
    seed: int,
    skills: List[str],
    vocab_size: int = 64,
    d_model: int = 24,
    num_samples_per_skill: int = 50,
    skill_seed: int = 42
) -> Dict[str, Any]:
    """Generates synthetic world factual triples and canonical shared skill inputs."""
    all_vals = []
    for attr, vals in ATTRIBUTE_VOCABS.items():
        all_vals.extend(vals)
    val_to_id = {v: i % vocab_size for i, v in enumerate(all_vals)}
    
    # 1. World-specific independent facts
    sw = SyntheticWorld(k_entities=k_entities, seed=seed)
    triples = []
    for ent_idx, ent in enumerate(sw.entities):
        for r_idx, r in enumerate(sw.attribute_names):
            val_str = sw.facts[ent][r]
            triples.append((ent_idx, r_idx, val_to_id[val_str]))
            
    # 2. Universally shared procedural skills
    skill_inputs = generate_canonical_skill_data(
        skill_ids=skills,
        vocab_size=vocab_size,
        d_model=d_model,
        num_samples=num_samples_per_skill,
        seed=skill_seed
    )
        
    return {
        "seed": seed,
        "k_entities": k_entities,
        "triples": triples,
        "skill_inputs": skill_inputs,
        "fact_bits": sw.theoretical_fact_bits
    }


def train_wrmf_joint(
    network: WRMFNetwork,
    worlds_data: List[Dict[str, Any]],
    epochs: int = 70,
    lr: float = 0.03,
    beta: float = 1e-5,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Jointly trains WRMFNetwork across M worlds using AdamW and MDL proxy penalty.
    """
    network.to(device)
    network.train()
    optimizer = torch.optim.AdamW(network.parameters(), lr=lr, weight_decay=1e-4)
    
    num_worlds = len(worlds_data)
    start_time = time.time()
    losses = []
    
    for ep in range(epochs):
        optimizer.zero_grad()
        nll_losses = []
        
        for m in range(num_worlds):
            w_data = worlds_data[m]
            triples = w_data["triples"]
            e = torch.tensor([t[0] for t in triples], dtype=torch.long, device=device)
            r = torch.tensor([t[1] for t in triples], dtype=torch.long, device=device)
            v = torch.tensor([t[2] for t in triples], dtype=torch.long, device=device)
            
            # Factual loss on world m
            out_f = network.forward_fact(m, e, r)
            l_f = F.cross_entropy(out_f, v)
            
            # Skill losses
            l_s = torch.tensor(0.0, device=device)
            for s_id, (sx, sy) in w_data["skill_inputs"].items():
                out_s = network.forward_skill(s_id, sx.to(device))
                l_s = l_s + F.cross_entropy(out_s, sy.to(device))
                
            nll_losses.append(l_f + l_s)
            
        total_loss, metrics = compute_wrmf_loss(
            nll_losses, network.core, [m for m in network.memories], beta=beta
        )
        total_loss.backward()
        optimizer.step()
        losses.append(float(total_loss.item()))
        
    wall_clock = time.time() - start_time
    
    # Evaluate final accuracy
    network.eval()
    world_accs = []
    with torch.no_grad():
        for m in range(num_worlds):
            triples = worlds_data[m]["triples"]
            e = torch.tensor([t[0] for t in triples], dtype=torch.long, device=device)
            r = torch.tensor([t[1] for t in triples], dtype=torch.long, device=device)
            v = torch.tensor([t[2] for t in triples], dtype=torch.long, device=device)
            preds_f = torch.argmax(network.forward_fact(m, e, r), dim=-1)
            acc_f = float((preds_f == v).float().mean().item())
            
            skill_accs = []
            for s_id, (sx, sy) in worlds_data[m]["skill_inputs"].items():
                preds_s = torch.argmax(network.forward_skill(s_id, sx.to(device)), dim=-1)
                skill_accs.append(float((preds_s == sy.to(device)).float().mean().item()))
                
            mean_s = float(np.mean(skill_accs)) if skill_accs else 1.0
            world_accs.append((acc_f + mean_s) / 2.0)
            
    mean_accuracy = float(np.mean(world_accs))
    
    # Exact physical description lengths
    core_physical_dl = compute_module_physical_dl(network.core)
    core_circuits_dl = compute_module_physical_dl(network.core.skill_circuits)
    mem_physical_dls = [compute_module_physical_dl(m) for m in network.memories]
    
    return {
        "final_loss": losses[-1] if losses else 0.0,
        "mean_accuracy": mean_accuracy,
        "world_accuracies": world_accs,
        "core_physical_dl": core_physical_dl,
        "core_circuits_dl": core_circuits_dl,
        "mem_physical_dls": mem_physical_dls,
        "mean_mem_dl": float(np.mean(mem_physical_dls)),
        "epochs_trained": epochs,
        "wall_clock_sec": wall_clock,
        "converged": mean_accuracy >= 0.85
    }


def evaluate_transfer_test(
    network: WRMFNetwork,
    unseen_world_data: Dict[str, Any],
    epochs: int = 80,
    lr: float = 0.03,
    beta: float = 1e-5,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Decisive Transfer Test:
    1. Freeze thinking core theta_core strictly.
    2. Attach freshly initialized memory module for an unseen world.
    3. Train ONLY the fresh memory on unseen facts.
    4. Verify performance reaches >= 95% of joint baseline.
    """
    network.to(device)
    network.eval()
    
    # 1. Strictly freeze core parameters
    network.freeze_core()
    
    # 2. Create fresh memory module
    k_entities = unseen_world_data["k_entities"]
    fresh_mem = create_memory_module(
        memory_type=network.memory_type,
        num_entities=k_entities,
        num_relations=6,
        d_model=network.d_model
    ).to(device)
    fresh_mem.train()
    
    # Train only fresh memory
    optimizer = torch.optim.AdamW(fresh_mem.parameters(), lr=lr, weight_decay=1e-4)
    triples = unseen_world_data["triples"]
    e = torch.tensor([t[0] for t in triples], dtype=torch.long, device=device)
    r = torch.tensor([t[1] for t in triples], dtype=torch.long, device=device)
    v = torch.tensor([t[2] for t in triples], dtype=torch.long, device=device)
    
    start_time = time.time()
    for ep in range(epochs):
        optimizer.zero_grad()
        h = fresh_mem(e, r)
        out_f = network.core.forward_head(h)
        l_f = F.cross_entropy(out_f, v)
        
        mem_proxy = compute_module_dl_proxy(fresh_mem)
        loss = l_f + beta * mem_proxy
        loss.backward()
        optimizer.step()
        
    wall_clock = time.time() - start_time
    
    # Evaluate accuracy on unseen world
    fresh_mem.eval()
    with torch.no_grad():
        h = fresh_mem(e, r)
        out_f = network.core.forward_head(h)
        preds_f = torch.argmax(out_f, dim=-1)
        fact_acc = float((preds_f == v).float().mean().item())
        
        # Skill accuracy executed on unseen world using frozen core
        skill_accs = []
        for s_id, (sx, sy) in unseen_world_data["skill_inputs"].items():
            preds_s = torch.argmax(network.forward_skill(s_id, sx.to(device)), dim=-1)
            skill_accs.append(float((preds_s == sy.to(device)).float().mean().item()))
        mean_skill_acc = float(np.mean(skill_accs)) if skill_accs else 1.0
        
    transfer_accuracy = (fact_acc + mean_skill_acc) / 2.0
    mem_physical_dl = compute_module_physical_dl(fresh_mem)
    
    # Unfreeze core after test
    network.unfreeze_core()
    
    return {
        "transfer_accuracy": transfer_accuracy,
        "fact_accuracy": fact_acc,
        "skill_accuracy": mean_skill_acc,
        "mem_physical_dl": mem_physical_dl,
        "wall_clock_sec": wall_clock
    }


def evaluate_null_control(
    k_entities: int,
    num_worlds: int,
    skills: List[str],
    memory_type: str = "entity_emb",
    epochs: int = 60,
    lr: float = 0.03,
    beta: float = 1e-3,
    seed: int = 42,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Null Control Test:
    Randomizes / uncorrelates skills across worlds so no invariant computational skill exists.
    Under MDL penalty beta, the core cannot find common structure and collapses structurally.
    """
    vocab_size = 64
    d_model = 24
    worlds_data = []
    
    canonical_data = generate_canonical_skill_data(skills, vocab_size=vocab_size, d_model=d_model, num_samples=50, seed=seed)
    
    for m in range(num_worlds):
        w_seed = seed + m * 100
        sw = SyntheticWorld(k_entities=k_entities, seed=w_seed)
        triples = []
        for ent_idx, ent in enumerate(sw.entities):
            for r_idx, r in enumerate(sw.attribute_names):
                triples.append((ent_idx, r_idx, (ent_idx + r_idx * 3 + m * 7) % vocab_size))
                
        # Scrambled/permuted skills across worlds on the SAME inputs x
        shuffled_skills = {}
        for i, orig_s in enumerate(skills):
            target_s = skills[(i + m + 1) % len(skills)]
            shuffled_skills[orig_s] = (canonical_data[orig_s][0], canonical_data[target_s][1])
            
        worlds_data.append({
            "triples": triples,
            "skill_inputs": shuffled_skills,
            "k_entities": k_entities
        })
        
    net_null = WRMFNetwork(
        num_worlds=num_worlds,
        num_entities=k_entities,
        num_relations=6,
        vocab_size=vocab_size,
        d_model=d_model,
        memory_type=memory_type,
        skill_ids=skills
    ).to(device)
    
    # Train under higher beta penalty
    train_res = train_wrmf_joint(
        network=net_null,
        worlds_data=worlds_data,
        epochs=epochs,
        lr=lr,
        beta=beta,
        device=device
    )
    
    # Evaluate simultaneous cross-world skill satisfaction (must satisfy all worlds to count as learned)
    net_null.eval()
    joint_passes = []
    with torch.no_grad():
        for orig_s in skills:
            x = canonical_data[orig_s][0].to(device)
            preds = torch.argmax(net_null.forward_skill(orig_s, x), dim=-1)
            accs = [
                float((preds == worlds_data[m]["skill_inputs"][orig_s][1].to(device)).float().mean().item())
                for m in range(num_worlds)
            ]
            joint_passes.append(min(accs))
    simultaneous_skill_acc = float(np.mean(joint_passes)) if joint_passes else 0.0
    
    core_null_dl = train_res["core_physical_dl"]
    circuits_null_dl = train_res["core_circuits_dl"]
    return {
        "core_null_dl": core_null_dl,
        "circuits_null_dl": circuits_null_dl,
        "skill_accuracy": simultaneous_skill_acc,
        "skill_circuits": net_null.core.skill_circuits,
        "mean_accuracy": train_res["mean_accuracy"],
        "epochs_trained": train_res["epochs_trained"]
    }
