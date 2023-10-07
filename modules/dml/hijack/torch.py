import torch

from modules.sd_hijack_utils import CondFunc

CondFunc('torchsde._brownian.brownian_interval._randn', lambda _, size, dtype, device, seed: torch.randn(size, dtype=dtype, device=torch.device("cpu"), generator=torch.Generator(torch.device("cpu")).manual_seed(int(seed))).to(device), lambda _, size, dtype, device, seed: device.type == 'privateuseone')

# https://github.com/microsoft/DirectML/issues/400
CondFunc('torch.Tensor.new', lambda orig, self, *args, **kwargs: orig(self.cpu(), *args, **kwargs).to(self.device), lambda orig, self, *args, **kwargs: torch.dml.is_directml_device(self.device))

_lerp = torch.lerp
def lerp(*args, **kwargs) -> torch.Tensor:
    rep = None
    for i in range(0, len(args)):
        if torch.is_tensor(args[i]):
            rep = args[i]
            break
    if rep is None:
        for key in kwargs:
            if torch.is_tensor(kwargs[key]):
                rep = kwargs[key]
                break
    if torch.dml.is_directml_device(rep.device):
        args = list(args)

        if rep.dtype == torch.float16:
            for i in range(len(args)):
                if torch.is_tensor(args[i]):
                    args[i] = args[i].float()
        for i in range(len(args)):
            if torch.is_tensor(args[i]):
                args[i] = args[i].cpu()

        if rep.dtype == torch.float16:
            for kwarg in kwargs:
                if torch.is_tensor(kwargs[kwarg]):
                    kwargs[kwarg] = kwargs[kwarg].float()
        for kwarg in kwargs:
            if torch.is_tensor(kwargs[kwarg]):
                kwargs[kwarg] = kwargs[kwarg].cpu()
        return _lerp(*args, **kwargs).to(rep.device).type(rep.dtype)
    return _lerp(*args, **kwargs)
torch.lerp = lerp
