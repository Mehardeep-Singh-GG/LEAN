"""
Unit tests for Exact Description Length (DL) Accounting Engine.
"""

import pytest
import torch
from src.lean.measure.dl import (
    dl_dense_tensor,
    dl_quantized_tensor,
    dl_low_rank,
    dl_sparse_tensor,
    ModelDescriptionLengthAccountant
)


def test_dl_dense():
    """Verify dense fp16 and int8 description length math."""
    numel = 10_000
    assert dl_dense_tensor(numel, bit_width=16) == 160_000
    assert dl_dense_tensor(numel, bit_width=8) == 80_000


def test_dl_quantized_group_overhead():
    """Verify group-quantized tensors accurately count scale and zero-point overhead."""
    numel = 1024
    group_size = 128
    bit_width = 4
    # 8 groups
    expected_weight_bits = 1024 * 4  # 4096
    expected_scale_bits = 8 * 16     # 128
    expected_zero_bits = 8 * 4       # 32
    expected_total = 4096 + 128 + 32  # 4256
    
    assert dl_quantized_tensor(numel, bit_width=bit_width, group_size=group_size) == expected_total


def test_dl_low_rank():
    """Verify low rank SVD parameter description length."""
    d_in = 512
    d_out = 512
    rank = 16
    bit_width = 16
    expected = 16 * (512 + 512) * 16  # 262,144 bits
    assert dl_low_rank(d_in, d_out, rank, bit_width) == expected


def test_dl_sparse_encodings():
    """Verify bitmask and COO sparse index representations."""
    numel = 1000
    num_nonzero = 100
    # Bitmask: 100 * 16 + 1000 = 2600
    assert dl_sparse_tensor(numel, num_nonzero, value_bits=16, index_encoding="bitmask") == 2600


def test_accountant_aggregation():
    """Verify ModelDescriptionLengthAccountant aggregates diverse layer types."""
    accountant = ModelDescriptionLengthAccountant()
    accountant.add_dense_layer("embed", torch.empty((100, 64)), bit_width=16)
    accountant.add_quantized_layer("mlp", numel=4096, bit_width=4, group_size=128)
    accountant.add_low_rank_layer("attn", d_in=64, d_out=64, rank=8, bit_width=16)
    
    total_bits = accountant.total_bits()
    assert total_bits > 0
    assert accountant.total_bytes() == total_bits / 8.0
    assert accountant.total_megabytes() == (total_bits / 8.0) / (1024 * 1024)
