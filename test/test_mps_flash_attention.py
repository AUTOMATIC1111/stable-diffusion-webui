import torch

from modules import mps_flash_attention
from modules.mps_flash_attention import should_use_mfa_shape


def test_dimension_40_routes_cross_attention_to_mfa():
    assert should_use_mfa_shape(4096, 77, 40)


def test_fallback_omits_unsupported_enable_gqa_keyword():
    previous_availability = mps_flash_attention._availability
    mps_flash_attention._availability = False
    try:
        query = torch.randn(1, 2, 8, 4)
        actual = mps_flash_attention.scaled_dot_product_attention(query, query, query)
        expected = torch.nn.functional.scaled_dot_product_attention(query, query, query)
    finally:
        mps_flash_attention._availability = previous_availability

    assert torch.allclose(actual, expected)


def test_dimension_40_routes_self_attention_to_mfa():
    assert should_use_mfa_shape(4096, 4096, 40)


def test_measured_sd1_dimensions_route_to_mfa():
    assert not should_use_mfa_shape(4096, 4096, 64)
    assert should_use_mfa_shape(1024, 1024, 80)
    assert should_use_mfa_shape(1024, 77, 80)
    assert should_use_mfa_shape(256, 256, 160)


def test_short_attention_stays_on_pytorch_sdpa():
    assert not should_use_mfa_shape(128, 128, 40)
