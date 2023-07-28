import torch

from modules.sd_hijack_utils import CondFunc

CondFunc('torchsde._brownian.brownian_interval._randn', lambda _, size, dtype, device, seed: torch.randn(size, dtype=dtype, device=torch.device("cpu"), generator=torch.Generator(torch.device("cpu")).manual_seed(int(seed))).to(device), lambda _, size, dtype, device, seed: device.type == 'privateuseone')

# https://github.com/microsoft/DirectML/issues/400
CondFunc('torch.Tensor.new', lambda orig, self, *args, **kwargs: orig(self.cpu(), *args, **kwargs), lambda orig, self, *args, **kwargs: torch.dml.is_directml_device(self.device))
