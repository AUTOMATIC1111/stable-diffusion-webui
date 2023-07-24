import torch

from modules.sd_hijack_utils import CondFunc

CondFunc('torchsde._brownian.brownian_interval._randn', lambda _, size, dtype, device, seed: torch.randn(size, dtype=dtype, device=torch.device("cpu"), generator=torch.Generator(torch.device("cpu")).manual_seed(int(seed))).to(device), lambda _, size, dtype, device, seed: device.type == 'privateuseone')

_new = torch.Tensor.new
def new(self: torch.Tensor, *args, **kwargs):
    if torch.dml.is_directml_device(self.device):
        return _new(self.cpu(), *args, **kwargs).to(self.device)
    return _new(self, *args, **kwargs)

torch.Tensor.new = new
