import warnings

import torch

from annotator.mmpkg.mmcv.utils import digit_version


def is_jit_tracing() -> bool:
    if (torch.__version__ != 'parrots'
            and digit_version(torch.__version__) >= digit_version('1.6.0')):
        on_trace = torch.jit.is_tracing()
        # In PyTorch 1.6, torch.jit.is_tracing has a bug.
        # Refers to https://github.com/pytorch/pytorch/issues/42448
        if isinstance(on_trace, bool):
            return on_trace
        else:
            return torch._C._is_tracing()
    else:
        warnings.warn(
            'torch.jit.is_tracing is only supported after v1.6.0. '
            'Therefore is_tracing returns False automatically. Please '
            'set on_trace manually if you are using trace.', UserWarning)
        return False
