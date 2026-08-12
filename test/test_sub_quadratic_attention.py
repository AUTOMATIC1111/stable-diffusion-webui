import torch
import torch.nn.functional as F

from modules.sub_quadratic_attention import efficient_dot_product_attention


def test_streaming_online_softmax_matches_sdpa():
    torch.manual_seed(123)
    query = torch.randn(4, 37, 16)
    key = torch.randn(4, 53, 16)
    value = torch.randn(4, 53, 16)

    expected = F.scaled_dot_product_attention(query, key, value)
    actual = efficient_dot_product_attention(
        query,
        key,
        value,
        query_chunk_size=13,
        kv_chunk_size=11,
        use_checkpoint=False,
    )

    assert torch.allclose(actual, expected, atol=2e-5, rtol=2e-5)


def test_streaming_online_softmax_gradients_match_sdpa():
    torch.manual_seed(321)
    query = torch.randn(2, 17, 8, dtype=torch.float64, requires_grad=True)
    key = torch.randn(2, 23, 8, dtype=torch.float64, requires_grad=True)
    value = torch.randn(2, 23, 8, dtype=torch.float64, requires_grad=True)
    gradient = torch.randn_like(query)

    expected = F.scaled_dot_product_attention(query, key, value)
    actual = efficient_dot_product_attention(
        query,
        key,
        value,
        query_chunk_size=7,
        kv_chunk_size=5,
        use_checkpoint=False,
    )

    expected_gradients = torch.autograd.grad(expected, (query, key, value), gradient, retain_graph=True)
    actual_gradients = torch.autograd.grad(actual, (query, key, value), gradient)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        assert torch.allclose(actual_gradient, expected_gradient, atol=1e-10, rtol=1e-8)
