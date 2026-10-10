import pytest
import torch
import torch.nn as nn
from experiments.run_7b_grand_challenge import PackedQuantizedLinear


def test_packed_quantized_linear_4bit():
    """Verify 4-bit packed linear layer initialization, packing, and forward pass."""
    torch.manual_seed(42)
    in_feat, out_feat = 128, 256
    layer = PackedQuantizedLinear(in_feat, out_feat, bits=4, group_size=128, bias=True)
    
    # Weight to pack
    weight = torch.randn(out_feat, in_feat)
    layer.pack_weights(weight)
    
    # Check packed buffer sizes
    expected_packed_bytes = (in_feat * out_feat) // 2
    assert layer.packed.numel() == expected_packed_bytes
    assert layer.scales.numel() == (in_feat * out_feat) // 128
    
    # Test forward pass
    x = torch.randn(2, 8, in_feat)
    out = layer(x)
    assert out.shape == (2, 8, out_feat)
    assert not torch.isnan(out).any()


def test_packed_quantized_linear_2bit():
    """Verify 2-bit packed linear layer initialization, packing, and forward pass."""
    torch.manual_seed(42)
    in_feat, out_feat = 128, 256
    layer = PackedQuantizedLinear(in_feat, out_feat, bits=2, group_size=128, bias=False)
    
    weight = torch.randn(out_feat, in_feat)
    layer.pack_weights(weight)
    
    expected_packed_bytes = (in_feat * out_feat) // 4
    assert layer.packed.numel() == expected_packed_bytes
    
    x = torch.randn(1, 4, in_feat)
    out = layer(x)
    assert out.shape == (1, 4, out_feat)
    assert not torch.isnan(out).any()
