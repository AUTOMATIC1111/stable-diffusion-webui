import inspect
import torch

from annotator.oneformer.detectron2.utils.env import TORCH_VERSION

try:
    from torch.fx._symbolic_trace import is_fx_tracing as is_fx_tracing_current

    tracing_current_exists = True
except ImportError:
    tracing_current_exists = False

try:
    from torch.fx._symbolic_trace import _orig_module_call

    tracing_legacy_exists = True
except ImportError:
    tracing_legacy_exists = False


@torch.jit.ignore
def is_fx_tracing_legacy() -> bool:
    """
    Returns a bool indicating whether torch.fx is currently symbolically tracing a module.
    Can be useful for gating module logic that is incompatible with symbolic tracing.
    """
    return torch.nn.Module.__call__ is not _orig_module_call


@torch.jit.ignore
def is_fx_tracing() -> bool:
    """Returns whether execution is currently in
    Torch FX tracing mode"""
    if TORCH_VERSION >= (1, 10) and tracing_current_exists:
        return is_fx_tracing_current()
    elif tracing_legacy_exists:
        return is_fx_tracing_legacy()
    else:
        # Can't find either current or legacy tracing indication code.
        # Enabling this assert_fx_safe() call regardless of tracing status.
        return False


@torch.jit.ignore
def assert_fx_safe(condition: bool, message: str) -> torch.Tensor:
    """An FX-tracing safe version of assert.
    Avoids erroneous type assertion triggering when types are masked inside
    an fx.proxy.Proxy object during tracing.
    Args: condition - either a boolean expression or a string representing
    the condition to test. If this assert triggers an exception when tracing
    due to dynamic control flow, try encasing the expression in quotation
    marks and supplying it as a string."""
    # Must return a concrete tensor for compatibility with PyTorch <=1.8.
    # If <=1.8 compatibility is not needed, return type can be converted to None
    if not is_fx_tracing():
        try:
            if isinstance(condition, str):
                caller_frame = inspect.currentframe().f_back
                torch._assert(
                    eval(condition, caller_frame.f_globals, caller_frame.f_locals), message
                )
                return torch.ones(1)
            else:
                torch._assert(condition, message)
                return torch.ones(1)
        except torch.fx.proxy.TraceError as e:
            print(
                "Found a non-FX compatible assertion. Skipping the check. Failure is shown below"
                + str(e)
            )
    return torch.zeros(1)
