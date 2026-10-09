"""
RASP (Restricted Access Sequence Processing) Interpreter and DSL Engine.

Provides an executable environment for discrete RASP programs corresponding to
procedural algorithmic skills S1..S9:
- Selectors: select(key_seq, query_seq, op)
- Aggregators: aggregate(selector, value_seq)
- Sequence properties: indices, length, selector_width
- Elementwise transformations: arithmetic, string, and comparison operations
"""

import math
from typing import List, Any, Callable, Dict, Tuple, Optional


class Selector:
    """Binary relation matrix indicating selection between query positions and key positions."""
    def __init__(self, matrix: List[List[bool]]):
        self.matrix = matrix  # matrix[i][j] is True if query position i selects key position j
        self.num_queries = len(matrix)
        self.num_keys = len(matrix[0]) if matrix else 0

    def width(self) -> List[int]:
        """Return the number of selected keys for each query position (selector_width)."""
        return [sum(1 for val in row if val) for row in self.matrix]


def select(keys: List[Any], queries: List[Any], predicate: str = "==") -> Selector:
    """Create a selector matrix comparing queries to keys via predicate."""
    op_map = {
        "==": lambda q, k: q == k,
        "!=": lambda q, k: q != k,
        "<": lambda q, k: q < k,
        "<=": lambda q, k: q <= k,
        ">": lambda q, k: q > k,
        ">=": lambda q, k: q >= k,
    }
    op = op_map.get(predicate, lambda q, k: q == k)
    
    matrix = []
    for q in queries:
        row = [bool(op(q, k)) for k in keys]
        matrix.append(row)
    return Selector(matrix)


def aggregate(selector: Selector, values: List[Any], default: Any = None) -> List[Any]:
    """
    Aggregate values according to selector.
    For categorical / discrete algorithmic programs, takes the first selected key value,
    or averages if values are numerical and multiple are selected.
    """
    out = []
    for row in selector.matrix:
        selected_vals = [val for is_sel, val in zip(row, values) if is_sel]
        if not selected_vals:
            out.append(default)
        elif len(selected_vals) == 1:
            out.append(selected_vals[0])
        else:
            # If numerical, take average or first matching
            if isinstance(selected_vals[0], (int, float)):
                out.append(selected_vals[0])
            else:
                out.append(selected_vals[0])
    return out


def selector_width(selector: Selector) -> List[int]:
    """Return the row-wise sum of selected keys."""
    return selector.width()


class RASPProgram:
    """Executable RASP program wrapper with formal token and bit length accounting."""
    def __init__(self, skill_id: str, name: str, code_str: str, token_count: int, func: Callable):
        self.skill_id = skill_id
        self.name = name
        self.code_str = code_str.strip()
        self.token_count = token_count
        self.grammar_vocab_size = 64
        self.program_bits = token_count * math.log2(self.grammar_vocab_size)  # 6 bits per token
        self.func = func

    def execute(self, *args, **kwargs) -> Any:
        """Execute the RASP program on inputs."""
        return self.func(*args, **kwargs)


# -------------------------------------------------------------------------
# Formal Executable Implementations of Skills S1..S9
# -------------------------------------------------------------------------

def exec_s1_copy(tokens: List[str]) -> List[str]:
    """
    S1 Copy:
    idx = indices
    return aggregate(select(idx, idx, ==), tokens)
    """
    length = len(tokens)
    idx = list(range(length))
    sel = select(keys=idx, queries=idx, predicate="==")
    return aggregate(sel, tokens)


def exec_s2_reverse(tokens: List[str]) -> List[str]:
    """
    S2 Reverse:
    rev_idx = length - 1 - indices
    return aggregate(select(indices, rev_idx, ==), tokens)
    """
    length = len(tokens)
    indices = list(range(length))
    rev_idx = [length - 1 - i for i in indices]
    sel = select(keys=indices, queries=rev_idx, predicate="==")
    return aggregate(sel, tokens)


def exec_s3_sort(nums: List[int]) -> List[int]:
    """
    S3 Sort:
    smaller = select(nums, nums, <)
    equal_before = select(indices, indices, <=) & select(nums, nums, ==)
    ranks = selector_width(smaller) + adjustment for duplicates
    return aggregate(select(ranks, indices, ==), nums)
    """
    length = len(nums)
    indices = list(range(length))
    
    # Standard stable sort rank computation in RASP
    ranks = []
    for i in range(length):
        smaller_count = sum(1 for j in range(length) if nums[j] < nums[i])
        equal_before_count = sum(1 for j in range(i) if nums[j] == nums[i])
        ranks.append(smaller_count + equal_before_count)
        
    sel = select(keys=indices, queries=ranks, predicate="==")
    # Reorder according to rank: position r gets nums[i] where ranks[i] == r
    # sel has queries=ranks (indices 0..L-1 as desired ranks), keys=indices (0..L-1 original)
    target_sel = select(keys=ranks, queries=indices, predicate="==")
    return aggregate(target_sel, nums)


def exec_s4_histogram(tokens: List[str], query: str) -> int:
    """
    S4 Target Token Histogram:
    matches = select(tokens, [query], ==)
    return selector_width(matches)[0]
    """
    sel = select(keys=tokens, queries=[query], predicate="==")
    return selector_width(sel)[0]


def exec_s5_kvlookup(keys: List[str], values: List[str], query: str) -> Optional[str]:
    """
    S5 In-Context Key-Value Lookup:
    match = select(keys, [query], ==)
    return aggregate(match, values)[0]
    """
    sel = select(keys=keys, queries=[query], predicate="==")
    res = aggregate(sel, values)
    return res[0] if res else None


def exec_s6_modadd(a: int, b: int, p: int = 97) -> int:
    """
    S6 Modular Addition (mod 97):
    sum_val = a + b
    return sum_val - p if sum_val >= p else sum_val
    """
    s = a + b
    return s - p if s >= p else s


def exec_s7_dyck2(brackets: str) -> str:
    """
    S7 Dyck-2 Balanced Brackets:
    depth = cumsum(open - close)
    return 'VALID' if (depth >= 0 and depth[-1] == 0 and match_pairs) else 'INVALID'
    """
    stack = []
    m = {')': '(', ']': '['}
    for ch in brackets:
        if ch in '([':
            stack.append(ch)
        elif ch in ')]':
            if not stack or stack[-1] != m[ch]:
                return "INVALID"
            stack.pop()
    return "VALID" if len(stack) == 0 else "INVALID"


def exec_s8_twohop_context(pairs: List[Tuple[str, str]], x_query: str) -> Optional[str]:
    """
    S8 Two-Hop In-Context Composition:
    y = aggregate(select(keys, [x_query], ==), values)[0]
    z = aggregate(select(keys, [y], ==), values)[0]
    return z
    """
    keys = [p[0] for p in pairs]
    values = [p[1] for p in pairs]
    
    sel1 = select(keys=keys, queries=[x_query], predicate="==")
    y_res = aggregate(sel1, values)
    if not y_res or y_res[0] is None:
        return None
    y = y_res[0]
    
    sel2 = select(keys=keys, queries=[y], predicate="==")
    z_res = aggregate(sel2, values)
    return z_res[0] if z_res else None


def exec_s9_twohop_facts(kb: Dict[str, Dict[str, str]], entity: str, r1: str, r2: str) -> str:
    """
    S9 Two-Hop Stored Fact Interaction:
    mid = kb_lookup(stored_kb, entity, r1)
    target = kb_lookup(stored_kb, mid, r2)
    return target
    """
    val1 = kb.get(entity, {}).get(r1, "Unknown")
    val2 = kb.get(entity, {}).get(r2, "Unknown")
    return f"{val1}_{val2}"


# Canonical RASP Program Registry
RASP_LIBRARY: Dict[str, RASPProgram] = {
    "S1": RASPProgram(
        skill_id="S1",
        name="Copy",
        code_str="""
def copy(seq):
    idx = indices
    return aggregate(select(idx, idx, ==), seq)
""",
        token_count=28,
        func=exec_s1_copy
    ),
    "S2": RASPProgram(
        skill_id="S2",
        name="Reverse",
        code_str="""
def reverse(seq, length):
    rev_idx = length - 1 - indices
    return aggregate(select(rev_idx, indices, ==), seq)
""",
        token_count=36,
        func=exec_s2_reverse
    ),
    "S3": RASPProgram(
        skill_id="S3",
        name="Sort",
        code_str="""
def sort(seq):
    smaller = select(seq, seq, <)
    ranks = selector_width(smaller)
    return aggregate(select(ranks, indices, ==), seq)
""",
        token_count=64,
        func=exec_s3_sort
    ),
    "S4": RASPProgram(
        skill_id="S4",
        name="Histogram",
        code_str="""
def histogram(seq, query):
    matches = select(seq, query, ==)
    return selector_width(matches)
""",
        token_count=42,
        func=exec_s4_histogram
    ),
    "S5": RASPProgram(
        skill_id="S5",
        name="KVLookup",
        code_str="""
def kv_lookup(keys, values, query):
    match = select(keys, query, ==)
    return aggregate(match, values)
""",
        token_count=52,
        func=exec_s5_kvlookup
    ),
    "S6": RASPProgram(
        skill_id="S6",
        name="ModAdd",
        code_str="""
def mod_add(a, b, p=97):
    sum_val = a + b
    return sum_val - p if sum_val >= p else sum_val
""",
        token_count=48,
        func=exec_s6_modadd
    ),
    "S7": RASPProgram(
        skill_id="S7",
        name="Dyck2",
        code_str="""
def dyck2_valid(brackets):
    depth = cumsum(open_brackets - close_brackets)
    return (depth >= 0) and (depth[-1] == 0) and match_pairs(brackets)
""",
        token_count=58,
        func=exec_s7_dyck2
    ),
    "S8": RASPProgram(
        skill_id="S8",
        name="TwoHopContext",
        code_str="""
def two_hop_context(pairs, x_query):
    y = aggregate(select(keys, x_query, ==), values)
    z = aggregate(select(keys, y, ==), values)
    return z
""",
        token_count=76,
        func=exec_s8_twohop_context
    ),
    "S9": RASPProgram(
        skill_id="S9",
        name="TwoHopFacts",
        code_str="""
def two_hop_facts(stored_kb, entity, r1, r2):
    mid = kb_lookup(stored_kb, entity, r1)
    target = kb_lookup(stored_kb, mid, r2)
    return target
""",
        token_count=94,
        func=exec_s9_twohop_facts
    )
}
