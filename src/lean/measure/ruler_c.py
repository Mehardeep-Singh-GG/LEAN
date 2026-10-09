"""
Ruler C: Algorithmic Program Description Length (Kolmogorov Upper Bound).

Measures the programmatic complexity K(s) of skills and data generators
by parsing their minimal formal programs (RASP / Python AST) into token description length.

Formula:
  K(program) = len(tokens) * log2(|V_grammar|)
"""

import ast
import math
from typing import Dict, Any


GRAMMAR_VOCAB_SIZE = 64  # Compact DSL vocabulary of primitives (Select, Aggregate, Map, Add, Mod, etc.)


def count_ast_complexity(code_str: str) -> Dict[str, Any]:
    """Parse python code string and count AST nodes and token complexity."""
    tree = ast.parse(code_str)
    num_nodes = sum(1 for _ in ast.walk(tree))
    
    # Calculate token length approximation
    tokens = code_str.split()
    num_tokens = len(tokens)
    
    bits_per_token = math.log2(GRAMMAR_VOCAB_SIZE)
    total_program_bits = math.ceil(num_tokens * bits_per_token)
    
    return {
        "num_ast_nodes": num_nodes,
        "num_tokens": num_tokens,
        "program_bits": total_program_bits,
        "program_bytes": total_program_bits / 8.0
    }


# Reference programs for algorithmic ground truth benchmarks
MODULAR_ADD_RASP = """
def modular_add(a, b, p):
    return (a + b) % p
"""

COPY_SEQUENCE_RASP = """
def copy_seq(tokens):
    return [t for t in tokens]
"""

LOOKUP_TABLE_RASP = """
def lookup_table(key, table):
    return table.get(key, None)
"""


def get_benchmark_program_complexity(benchmark_name: str) -> Dict[str, Any]:
    """Return Ruler C complexity for standard reference algorithms."""
    mapping = {
        "modular_add": MODULAR_ADD_RASP,
        "R2_Pure_Rule": MODULAR_ADD_RASP,
        "copy_seq": COPY_SEQUENCE_RASP,
        "lookup_table": LOOKUP_TABLE_RASP,
        "R1_Random_Table": LOOKUP_TABLE_RASP
    }
    code = mapping.get(benchmark_name, "def identity(x): return x")
    res = count_ast_complexity(code)
    res["benchmark"] = benchmark_name
    res["code"] = code
    return res
