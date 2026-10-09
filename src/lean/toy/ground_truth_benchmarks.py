"""
Ground Truth Calibration Benchmarks (R1 - R4) for Milestone M1.

R1 (Random Table): Incompressible key-value associations with known bits I = K * log2(|V|).
R2 (Pure Rule): Algorithmic modular addition rule (a + b) % p with O(1) description complexity.
R3 (Rule + Table): Composite capability measuring additivity DL(R3) ~= DL(R1) + DL(R2).
R4 (Pure Noise): Arbitrary random labels measuring memorization overhead.
"""

import math
import random
from typing import Dict, Any, Tuple
import torch
import torch.nn as nn


class BenchmarkLinear(nn.Module):
    """Single linear layer for exact lookup table memory representation."""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        
    def forward(self, x):
        return self.fc(x)


class BenchmarkMLP(nn.Module):
    """Simple 2-layer MLP for ground truth learning and calibration."""
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        
    def forward(self, x):
        h = self.relu(self.fc1(x))
        return self.fc2(h)


class BenchmarkComposite(nn.Module):
    """Multi-capability composite model combining independent table memory and rule modules."""
    def __init__(self, k_facts: int, vocab_size: int, prime_p: int, hidden_dim: int = 64):
        super().__init__()
        self.table = BenchmarkLinear(k_facts, vocab_size)
        self.rule = BenchmarkMLP(2 * prime_p, hidden_dim, prime_p)
        
    def forward(self, x_key, x_rule):
        return self.table(x_key), self.rule(x_rule)


def generate_r1_random_table(k_facts: int = 512, vocab_size: int = 128, seed: int = 42) -> Dict[str, Any]:
    """
    R1: Pure Random Lookup Table.
    Exact theoretical information: I = k_facts * log2(vocab_size) bits.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    
    # One-hot or embedding input representation for keys
    keys = torch.arange(k_facts)
    # Inputs: one-hot or normalized key index
    x = torch.zeros(k_facts, k_facts)
    x.scatter_(1, keys.unsqueeze(1), 1.0)
    
    # Random values drawn uniformly
    y = torch.randint(0, vocab_size, (k_facts,))
    
    theoretical_bits = k_facts * math.log2(vocab_size)
    return {
        "name": "R1_Random_Table",
        "k_facts": k_facts,
        "vocab_size": vocab_size,
        "theoretical_bits": theoretical_bits,
        "x": x,
        "y": y
    }


def generate_r2_pure_rule(num_samples: int = 1000, prime_p: int = 97, seed: int = 42) -> Dict[str, Any]:
    """
    R2: Pure Algorithmic Rule - Modular Addition (a + b) % prime_p.
    Theoretical information: O(1) bits (rule complexity is independent of sample size).
    """
    random.seed(seed)
    torch.manual_seed(seed)
    
    a = torch.randint(0, prime_p, (num_samples,))
    b = torch.randint(0, prime_p, (num_samples,))
    y = (a + b) % prime_p
    
    # One-hot inputs for modular addition
    x_a = torch.zeros(num_samples, prime_p)
    x_a.scatter_(1, a.unsqueeze(1), 1.0)
    x_b = torch.zeros(num_samples, prime_p)
    x_b.scatter_(1, b.unsqueeze(1), 1.0)
    x = torch.cat([x_a, x_b], dim=1)
    
    return {
        "name": "R2_Pure_Rule",
        "num_samples": num_samples,
        "prime_p": prime_p,
        "theoretical_rule_bits": 48.0,  # Constant RASP program length in bits
        "x": x,
        "y": y
    }


def generate_r3_rule_plus_table(k_facts: int = 256, vocab_size: int = 64, num_samples: int = 1000, prime_p: int = 64, seed: int = 42) -> Dict[str, Any]:
    """
    R3: Composite Task - Table Lookup + Modular Addition.
    Target = (Table[key] + a + b) % prime_p.
    Theoretical information: I_table + I_rule.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    
    table = torch.randint(0, vocab_size, (k_facts,))
    
    key_idx = torch.randint(0, k_facts, (num_samples,))
    a = torch.randint(0, prime_p, (num_samples,))
    b = torch.randint(0, prime_p, (num_samples,))
    
    # Target computation
    table_vals = table[key_idx]
    y = (table_vals + a + b) % prime_p
    
    # Inputs: one-hot key + one-hot (a, b)
    x_keys = torch.zeros(num_samples, k_facts)
    x_keys.scatter_(1, key_idx.unsqueeze(1), 1.0)
    x_a = torch.zeros(num_samples, prime_p)
    x_a.scatter_(1, a.unsqueeze(1), 1.0)
    x_b = torch.zeros(num_samples, prime_p)
    x_b.scatter_(1, b.unsqueeze(1), 1.0)
    x = torch.cat([x_keys, x_a, x_b], dim=1)
    
    theoretical_table_bits = k_facts * math.log2(vocab_size)
    return {
        "name": "R3_Rule_Plus_Table",
        "k_facts": k_facts,
        "vocab_size": vocab_size,
        "theoretical_table_bits": theoretical_table_bits,
        "theoretical_total_bits": theoretical_table_bits + 48.0,
        "x": x,
        "y": y
    }


def generate_r4_noise_labels(num_samples: int = 500, num_classes: int = 50, seed: int = 42) -> Dict[str, Any]:
    """
    R4: Pure Noise Memorization.
    Random input vectors paired with uniform random class labels.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    
    x = torch.randn(num_samples, 64)
    y = torch.randint(0, num_classes, (num_samples,))
    
    theoretical_bits = num_samples * math.log2(num_classes)
    return {
        "name": "R4_Pure_Noise",
        "num_samples": num_samples,
        "num_classes": num_classes,
        "theoretical_bits": theoretical_bits,
        "x": x,
        "y": y
    }
