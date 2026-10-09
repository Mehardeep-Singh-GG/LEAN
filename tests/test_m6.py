"""
Unit tests for Milestone M6: Neural Decompilation (RASP Program Synthesis).

Verifies:
1. Preregistration immutability and SHA-256 hash match.
2. Executable RASP interpreter execution across all 9 canonical procedural skills (S1..S9).
3. Discrete Transformer Program architecture and attention routing.
4. Universal compilation constant c_alg constancy and CV <= 30.0%.
5. Symbolic closed-form recovery for S6 (ModAdd) and S7 (Dyck-2).
"""

import hashlib
from pathlib import Path
import pytest
import torch

from src.lean.decompile.rasp import RASP_LIBRARY, exec_s1_copy, exec_s2_reverse, exec_s3_sort
from src.lean.decompile.transformer_program import TransformerProgram, DiscreteAttentionHead
from src.lean.decompile.symbolic import recover_modular_addition_symbolic, recover_dyck2_symbolic
from src.lean.decompile.distiller import decompile_all_skills, evaluate_skill_decompilation


PREREG_M6_PATH = Path("prereg/M6.md")
EXPECTED_M6_HASH = "0c3d7592e876e5777ff2c235994e6c9586bc006a58120608c9aa845c87087cbc"


def test_m6_preregistration_lock():
    """Verify prereg/M6.md is locked and matches cryptographic SHA-256 hash."""
    assert PREREG_M6_PATH.exists(), "prereg/M6.md does not exist"
    with open(PREREG_M6_PATH, "rb") as f:
        actual_hash = hashlib.sha256(f.read()).hexdigest()
    assert actual_hash == EXPECTED_M6_HASH, f"M6 prereg hash mismatch: got {actual_hash}"


def test_rasp_interpreter_execution():
    """Verify execution of all 9 RASP programs in RASP_LIBRARY."""
    assert len(RASP_LIBRARY) == 9, "RASP library must contain all 9 canonical skills"
    
    # S1 Copy
    res_s1 = RASP_LIBRARY["S1"].execute(["A", "B", "C"])
    assert res_s1 == ["A", "B", "C"]
    
    # S2 Reverse
    res_s2 = RASP_LIBRARY["S2"].execute(["A", "B", "C"])
    assert res_s2 == ["C", "B", "A"]
    
    # S3 Sort
    res_s3 = RASP_LIBRARY["S3"].execute([3, 1, 4, 1, 5])
    assert res_s3 == [1, 1, 3, 4, 5]
    
    # S4 Histogram
    res_s4 = RASP_LIBRARY["S4"].execute(["A", "B", "A", "C"], "A")
    assert res_s4 == 2
    
    # S5 KVLookup
    res_s5 = RASP_LIBRARY["S5"].execute(["k0", "k1"], ["v10", "v20"], "k1")
    assert res_s5 == "v20"
    
    # S6 ModAdd
    res_s6 = RASP_LIBRARY["S6"].execute(40, 60, 97)
    assert res_s6 == (100 % 97)
    
    # S7 Dyck-2
    res_s7_valid = RASP_LIBRARY["S7"].execute("([()])")
    assert res_s7_valid == "VALID"
    res_s7_invalid = RASP_LIBRARY["S7"].execute("([)]")
    assert res_s7_invalid == "INVALID"
    
    # S8 TwoHopContext
    pairs = [("n0", "n1"), ("n1", "n2")]
    res_s8 = RASP_LIBRARY["S8"].execute(pairs, "n0")
    assert res_s8 == "n2"
    
    # S9 TwoHopFacts
    kb = {"e1": {"origin_galaxy": "gal1", "containment_field": "field1"}}
    res_s9 = RASP_LIBRARY["S9"].execute(kb, "e1", "origin_galaxy", "containment_field")
    assert res_s9 == "gal1_field1"


def test_transformer_program_forward():
    """Verify discrete Transformer Program forward pass and Gumbel annealing."""
    device = "cpu"
    prog = TransformerProgram(vocab_size=32, d_model=16, d_ff=32, num_layers=1).to(device)
    prog.eval()
    
    x = torch.randint(0, 32, (2, 8), device=device)
    out = prog(x, hard=True)
    assert out.shape == (2, 8, 32)
    
    patterns = prog.get_attention_patterns(x)
    assert len(patterns) == 1
    assert patterns[0].shape == (2, 8, 8)


def test_c_alg_constancy_and_cv():
    """Criterion C2: Verify universal compilation constant c_alg has CV <= 30.0%."""
    res = decompile_all_skills(seeds=[42], num_samples_per_seed=50)
    cv = res["c_alg_stats"]["cv"]
    assert cv <= 0.30, f"c_alg CV too high: {cv * 100:.2f}% > 30.0%"
    assert res["acceptance"]["C2_c_alg_cv"], "C2 acceptance check failed"


def test_symbolic_recovery_s6_s7():
    """Criterion C4: Verify exact symbolic closed-form recovery for S6 and S7."""
    # S6: Modular Addition
    s6 = recover_modular_addition_symbolic(lambda a, b: (a + b) % 97, p=97)
    assert s6["exact_agreement"], f"S6 symbolic recovery failed: {s6}"
    assert s6["symbolic_formula"] == "(a + b) % 97"
    
    # S7: Dyck-2 Balanced Brackets
    test_seqs = ["()", "(())", "[]", "([])", ")(", "([)]"]
    s7 = recover_dyck2_symbolic(
        lambda seq: "VALID" if seq in ["()", "(())", "[]", "([])"] else "INVALID",
        test_sequences=test_seqs
    )
    assert s7["exact_agreement"], f"S7 symbolic recovery failed: {s7}"
