import torch

from modules.sd_hijack_utils import CondFunc

def to_sub(orig, self: torch.Tensor, *args, **kwargs):
    def validate(device: torch.device | str):
        if torch.dml.is_directml_device(torch.device(device)):
            raise NotImplementedError("Cannot copy out of meta tensor; no data!")
    for arg in args:
        validate(arg)
    if "device" in kwargs:
        validate(kwargs["device"])
    return orig(self, *args, **kwargs)

CondFunc('torchsde._brownian.brownian_interval._randn', lambda _, size, dtype, device, seed: torch.randn(size, dtype=dtype, device=torch.device("cpu"), generator=torch.Generator(torch.device("cpu")).manual_seed(int(seed))).to(device), lambda _, size, dtype, device, seed: device.type == 'privateuseone')

# https://github.com/microsoft/DirectML/issues/400
CondFunc('torch.Tensor.new', lambda orig, self, *args, **kwargs: orig(self.cpu(), *args, **kwargs), lambda orig, self, *args, **kwargs: torch.dml.is_directml_device(self.device))
# https://github.com/microsoft/DirectML/issues/477
CondFunc('torch.Tensor.to', to_sub, lambda orig, self, *args, **kwargs: self.device.type == "meta")
