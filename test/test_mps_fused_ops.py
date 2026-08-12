import torch
import torch.nn.functional as F

from modules import mps_fused_ops


def test_cpu_fallback_matches_pytorch():
    torch.manual_seed(1)
    norm = torch.nn.GroupNorm(4, 8)
    source = torch.randn(1, 8, 6, 10)

    actual = mps_fused_ops.group_norm_silu(source, norm)
    expected = F.silu(norm(source))

    assert torch.equal(actual, expected)


def test_native_fusion_matches_pytorch():
    if not torch.backends.mps.is_available():
        return
    torch.manual_seed(1)
    norm = torch.nn.GroupNorm(32, 320).eval().half().to("mps")
    source = torch.randn(1, 320, 48, 80, device="mps", dtype=torch.float16)

    with torch.no_grad():
        expected = F.silu(norm(source))
        actual = mps_fused_ops.group_norm_silu(source, norm) + 0
        torch.mps.synchronize()

    difference = (actual.float() - expected.float()).abs()
    assert torch.isfinite(actual).all().item()
    assert difference.max().item() < 0.02
    assert difference.mean().item() < 0.001


def test_geglu_cpu_fallback_matches_pytorch():
    torch.manual_seed(2)
    projection = torch.nn.Linear(8, 32)
    source = torch.randn(2, 6, 8)

    actual = mps_fused_ops.geglu(source, projection)
    projected = projection(source)
    value, gate = projected.chunk(2, dim=-1)
    expected = value * F.gelu(gate)

    assert torch.equal(actual, expected)


def test_native_geglu_matches_pytorch():
    if not torch.backends.mps.is_available():
        return
    torch.manual_seed(2)
    projection = torch.nn.Linear(320, 2560).eval().half().to("mps")
    source = torch.randn(2, 256, 320, device="mps", dtype=torch.float16)

    with torch.no_grad():
        projected = projection(source)
        value, gate = projected.chunk(2, dim=-1)
        expected = value * F.gelu(gate)
        actual = mps_fused_ops.geglu(source, projection) + 0
        torch.mps.synchronize()

    assert torch.isfinite(actual).all().item()
    assert torch.equal(actual, expected)
