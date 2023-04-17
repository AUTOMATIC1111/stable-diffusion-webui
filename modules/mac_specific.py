import torch
import platform
from modules import paths
from modules.sd_hijack_utils import CondFunc
from packaging import version


# has_mps is only available in nightly pytorch (for now) and macOS 12.3+.
# check `getattr` and try it for compatibility
def check_for_mps() -> bool:
    if not getattr(torch, 'has_mps', False):
        return False
    try:
        torch.zeros(1).to(torch.device("mps"))
        return True
    except Exception:
        return False
has_mps = check_for_mps()


# MPS workaround for https://github.com/pytorch/pytorch/issues/89784
def cumsum_fix(input, cumsum_func, *args, **kwargs):
    if input.device.type == 'mps':
        output_dtype = kwargs.get('dtype', input.dtype)
        if output_dtype == torch.int64:
            return cumsum_func(input.cpu(), *args, **kwargs).to(input.device)
        elif output_dtype == torch.bool or cumsum_needs_int_fix and (output_dtype == torch.int8 or output_dtype == torch.int16):
            return cumsum_func(input.to(torch.int32), *args, **kwargs).to(torch.int64)
    return cumsum_func(input, *args, **kwargs)


if has_mps:
    # MPS fix for randn in torchsde
    CondFunc('torchsde._brownian.brownian_interval._randn', lambda _, size, dtype, device, seed: torch.randn(size, dtype=dtype, device=torch.device("cpu"), generator=torch.Generator(torch.device("cpu")).manual_seed(int(seed))).to(device), lambda _, size, dtype, device, seed: device.type == 'mps')

    if platform.mac_ver()[0].startswith("13.2."):
        # MPS workaround for https://github.com/pytorch/pytorch/issues/95188, thanks to danieldk (https://github.com/explosion/curated-transformers/pull/124)
        CondFunc('torch.nn.functional.linear', lambda _, input, weight, bias: (torch.matmul(input, weight.t()) + bias) if bias is not None else torch.matmul(input, weight.t()), lambda _, input, weight, bias: input.numel() > 10485760)

    if version.parse(torch.__version__) < version.parse("1.13"):
        # PyTorch 1.13 doesn't need these fixes but unfortunately is slower and has regressions that prevent training from working

        # MPS workaround for https://github.com/pytorch/pytorch/issues/79383
        CondFunc('torch.Tensor.to', lambda orig_func, self, *args, **kwargs: orig_func(self.contiguous(), *args, **kwargs),
                                                          lambda _, self, *args, **kwargs: self.device.type != 'mps' and (args and isinstance(args[0], torch.device) and args[0].type == 'mps' or isinstance(kwargs.get('device'), torch.device) and kwargs['device'].type == 'mps'))
        # MPS workaround for https://github.com/pytorch/pytorch/issues/80800 
        CondFunc('torch.nn.functional.layer_norm', lambda orig_func, *args, **kwargs: orig_func(*([args[0].contiguous()] + list(args[1:])), **kwargs),
                                                                                        lambda _, *args, **kwargs: args and isinstance(args[0], torch.Tensor) and args[0].device.type == 'mps')
        # MPS workaround for https://github.com/pytorch/pytorch/issues/90532
        CondFunc('torch.Tensor.numpy', lambda orig_func, self, *args, **kwargs: orig_func(self.detach(), *args, **kwargs), lambda _, self, *args, **kwargs: self.requires_grad)
    elif version.parse(torch.__version__) > version.parse("1.13.1"):
        cumsum_needs_int_fix = not torch.Tensor([1,2]).to(torch.device("mps")).equal(torch.ShortTensor([1,1]).to(torch.device("mps")).cumsum(0))
        cumsum_fix_func = lambda orig_func, input, *args, **kwargs: cumsum_fix(input, orig_func, *args, **kwargs)
        CondFunc('torch.cumsum', cumsum_fix_func, None)
        CondFunc('torch.Tensor.cumsum', cumsum_fix_func, None)
        CondFunc('torch.narrow', lambda orig_func, *args, **kwargs: orig_func(*args, **kwargs).clone(), None)
    if version.parse(torch.__version__) == version.parse("2.0"):
        # MPS workaround for https://github.com/pytorch/pytorch/issues/96113
        CondFunc('torch.nn.functional.layer_norm', lambda orig_func, x, normalized_shape, weight, bias, eps, **kwargs: orig_func(x.float(), normalized_shape, weight.float() if weight is not None else None, bias.float() if bias is not None else bias, eps).to(x.dtype), lambda *args, **kwargs: len(args) == 6)
