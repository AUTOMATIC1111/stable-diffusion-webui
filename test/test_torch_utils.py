import types

import pytest

torch = pytest.importorskip("torch", reason="torch is an optional runtime dependency for these unit tests")


@pytest.mark.parametrize("wrapped", [True, False])
def test_get_param(wrapped):
    from modules import torch_utils

    mod = torch.nn.Linear(1, 1)
    cpu = torch.device("cpu")
    mod.to(dtype=torch.float16, device=cpu)
    if wrapped:
        # more or less how spandrel wraps a thing
        mod = types.SimpleNamespace(model=mod)
    p = torch_utils.get_param(mod)
    assert p.dtype == torch.float16
    assert p.device == cpu
