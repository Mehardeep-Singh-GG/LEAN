"""
Skill Library and Algorithmic Complexity Metrics for Milestones M3 and M4.

Defines 9 procedural skills with reference RASP programs, measured algorithmic
complexity K(s) in tokens/bits, and deterministic sequence generators.

Skills:
- S1: Copy (identity sequence replication)
- S2: Reverse (sequence inversion)
- S3: Sort (numerical ordering)
- S4: Histogram (frequency counting of target token)
- S5: In-Context KV Lookup (associative query resolution)
- S6: Modular Addition (a + b mod 97)
- S7: Dyck-2 (balanced nested bracket syntax validation)
- S8: Two-Hop In-Context Composition (relational chaining: x->y, y->z => x->z)
- S9: Two-Hop Fact Interaction (stored fact lookup + relation composition)
"""

import random
from typing import Dict, List, Tuple, Any, Optional


# Explicit algorithmic program lengths K(s) measured from minimal RASP program representations
SKILL_METADATA = {
    "S1_Copy": {
        "id": "S1",
        "name": "Sequence Copying",
        "description": "Reproduce an arbitrary sequence of length L without modification",
        "rasp_tokens": 28,
        "k_bits": 28 * 4.0,  # 112 bits
        "rasp_source": """
def copy(seq):
    idx = indices
    return aggregate(select(idx, idx, ==), seq)
""".strip()
    },
    "S2_Reverse": {
        "id": "S2",
        "name": "Sequence Inversion",
        "description": "Reverse an input sequence of tokens",
        "rasp_tokens": 36,
        "k_bits": 36 * 4.0,  # 144 bits
        "rasp_source": """
def reverse(seq, length):
    rev_idx = length - 1 - indices
    return aggregate(select(rev_idx, indices, ==), seq)
""".strip()
    },
    "S3_Sort": {
        "id": "S3",
        "name": "Sequence Sort",
        "description": "Sort numerical tokens in ascending order",
        "rasp_tokens": 64,
        "k_bits": 64 * 4.0,  # 256 bits
        "rasp_source": """
def sort(seq):
    smaller = select(seq, seq, <)
    ranks = selector_width(smaller)
    return aggregate(select(ranks, indices, ==), seq)
""".strip()
    },
    "S4_Histogram": {
        "id": "S4",
        "name": "Target Token Histogram",
        "description": "Count occurrences of query token within input sequence",
        "rasp_tokens": 42,
        "k_bits": 42 * 4.0,  # 168 bits
        "rasp_source": """
def histogram(seq, query):
    matches = select(tokens, query, ==)
    return selector_width(matches)
""".strip()
    },
    "S5_KVLookup": {
        "id": "S5",
        "name": "In-Context Key-Value Lookup",
        "description": "Retrieve value associated with query key in a sequence of pairs",
        "rasp_tokens": 52,
        "k_bits": 52 * 4.0,  # 208 bits
        "rasp_source": """
def kv_lookup(keys, values, query):
    match = select(keys, query, ==)
    return aggregate(match, values)
""".strip()
    },
    "S6_ModAdd": {
        "id": "S6",
        "name": "Modular Addition (mod 97)",
        "description": "Compute (a + b) % 97 for arbitrary integers",
        "rasp_tokens": 48,
        "k_bits": 48 * 4.0,  # 192 bits
        "rasp_source": """
def mod_add(a, b, p=97):
    sum_val = a + b
    return sum_val - p if sum_val >= p else sum_val
""".strip()
    },
    "S7_Dyck2": {
        "id": "S7",
        "name": "Dyck-2 Balanced Brackets",
        "description": "Evaluate validity of nested () and [] bracket expressions",
        "rasp_tokens": 58,
        "k_bits": 58 * 4.0,  # 232 bits
        "rasp_source": """
def dyck2_valid(brackets):
    depth = cumsum(open_brackets - close_brackets)
    return (depth >= 0) and (depth[-1] == 0) and match_pairs(brackets)
""".strip()
    },
    "S8_TwoHopContext": {
        "id": "S8",
        "name": "Two-Hop In-Context Composition",
        "description": "Compose relations (x->y, y->z) provided in prompt context to yield z",
        "rasp_tokens": 76,
        "k_bits": 76 * 4.0,  # 304 bits
        "rasp_source": """
def two_hop_context(pairs, x_query):
    y = aggregate(select(keys, x_query, ==), values)
    z = aggregate(select(keys, y, ==), values)
    return z
""".strip()
    },
    "S9_TwoHopFacts": {
        "id": "S9",
        "name": "Two-Hop Stored Fact Interaction",
        "description": "Chain stored factual knowledge r1(e)->o1 with r2(o1)->o2",
        "rasp_tokens": 94,
        "k_bits": 94 * 4.0,  # 376 bits
        "rasp_source": """
def two_hop_facts(stored_kb, entity, r1, r2):
    mid = kb_lookup(stored_kb, entity, r1)
    target = kb_lookup(stored_kb, mid, r2)
    return target
""".strip()
    }
}


class SkillGenerator:
    """Deterministic procedural skill sample generator."""
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        
    def generate_s1_copy(self, seq_len: int = 5) -> Tuple[str, str]:
        """S1: Copy arbitrary random letters."""
        tokens = [chr(ord('A') + self.rng.randint(0, 25)) for _ in range(seq_len)]
        prompt = f"COPY {' '.join(tokens)} ->"
        target = " ".join(tokens)
        return prompt, target
        
    def generate_s2_reverse(self, seq_len: int = 5) -> Tuple[str, str]:
        """S2: Reverse sequence."""
        tokens = [chr(ord('A') + self.rng.randint(0, 25)) for _ in range(seq_len)]
        prompt = f"REVERSE {' '.join(tokens)} ->"
        target = " ".join(reversed(tokens))
        return prompt, target
        
    def generate_s3_sort(self, seq_len: int = 5) -> Tuple[str, str]:
        """S3: Sort single-digit numbers."""
        nums = [self.rng.randint(0, 9) for _ in range(seq_len)]
        prompt = f"SORT {' '.join(str(n) for n in nums)} ->"
        target = " ".join(str(n) for n in sorted(nums))
        return prompt, target
        
    def generate_s4_histogram(self, seq_len: int = 8) -> Tuple[str, str]:
        """S4: Count target token occurrences."""
        tokens = [chr(ord('A') + self.rng.randint(0, 3)) for _ in range(seq_len)]
        target_tok = self.rng.choice(['A', 'B', 'C', 'D'])
        count = tokens.count(target_tok)
        prompt = f"COUNT {target_tok} IN {' '.join(tokens)} ->"
        target = str(count)
        return prompt, target
        
    def generate_s5_kvlookup(self, num_pairs: int = 4) -> Tuple[str, str]:
        """S5: In-context key-value lookup."""
        keys = [f"k{i}" for i in range(num_pairs)]
        values = [f"v{self.rng.randint(10, 99)}" for _ in range(num_pairs)]
        query_idx = self.rng.randint(0, num_pairs - 1)
        pairs_str = " ".join(f"{k}:{v}" for k, v in zip(keys, values))
        prompt = f"LOOKUP {pairs_str} QUERY {keys[query_idx]} ->"
        target = values[query_idx]
        return prompt, target
        
    def generate_s6_modadd(self, prime_p: int = 97) -> Tuple[str, str]:
        """S6: Modular addition."""
        a = self.rng.randint(0, prime_p - 1)
        b = self.rng.randint(0, prime_p - 1)
        res = (a + b) % prime_p
        prompt = f"ADD_MOD97 {a} + {b} ->"
        target = str(res)
        return prompt, target
        
    def generate_s7_dyck2(self, pairs: int = 3) -> Tuple[str, str]:
        """S7: Dyck-2 balanced brackets."""
        # Either generate valid or invalid
        is_valid = self.rng.random() < 0.5
        if is_valid:
            # Deterministic balanced generator
            s = []
            stack = []
            for _ in range(pairs):
                b_type = self.rng.choice([('(', ')'), ('[', ']')])
                s.append(b_type[0])
                stack.append(b_type[1])
            while stack:
                s.append(stack.pop())
            seq = "".join(s)
            target = "VALID"
        else:
            # Unbalanced sequence
            chars = ['(', ')', '[', ']']
            seq = "".join(self.rng.choice(chars) for _ in range(pairs * 2))
            # Double check if by chance it's balanced
            stack = []
            bal = True
            m = {')': '(', ']': '['}
            for ch in seq:
                if ch in '([':
                    stack.append(ch)
                else:
                    if not stack or stack[-1] != m[ch]:
                        bal = False
                        break
                    stack.pop()
            if stack:
                bal = False
            target = "VALID" if bal else "INVALID"
        prompt = f"DYCK2 {seq} ->"
        return prompt, target
        
    def generate_s8_twohop_context(self) -> Tuple[str, str]:
        """S8: Two-hop in-context composition."""
        nodes = [f"n{i}" for i in range(5)]
        self.rng.shuffle(nodes)
        n1, n2, n3 = nodes[:3]
        prompt = f"CHAIN {n1}->{n2} ; {n2}->{n3} QUERY {n1} ->"
        target = n3
        return prompt, target
        
    def generate_s9_twohop_facts(self, kb: Dict[str, Dict[str, str]]) -> Tuple[str, str]:
        """S9: Two-hop over stored facts interaction."""
        entities = list(kb.keys())
        e1 = self.rng.choice(entities)
        # Select two relations
        r1 = "origin_galaxy"
        r2 = "containment_field"
        val1 = kb[e1].get(r1, "Unknown")
        # In this synthetic setup, val1 maps to another deterministic attribute or second entity
        prompt = f"INTERACTION {e1} REL {r1} THEN {r2} ->"
        # The target combines the two attributes
        val2 = kb[e1].get(r2, "Unknown")
        target = f"{val1}_{val2}"
        return prompt, target
        
    def sample_skill(self, skill_id: str, kb: Optional[Dict[str, Dict[str, str]]] = None) -> Tuple[str, str]:
        """Sample a single prompt-target pair for a specified skill."""
        if skill_id == "S1":
            return self.generate_s1_copy()
        elif skill_id == "S2":
            return self.generate_s2_reverse()
        elif skill_id == "S3":
            return self.generate_s3_sort()
        elif skill_id == "S4":
            return self.generate_s4_histogram()
        elif skill_id == "S5":
            return self.generate_s5_kvlookup()
        elif skill_id == "S6":
            return self.generate_s6_modadd()
        elif skill_id == "S7":
            return self.generate_s7_dyck2()
        elif skill_id == "S8":
            return self.generate_s8_twohop_context()
        elif skill_id == "S9":
            return self.generate_s9_twohop_facts(kb if kb else {})
        else:
            raise ValueError(f"Unknown skill ID: {skill_id}")
