import torch

import network
from modules import devices


class ModuleTypeLora(network.ModuleType):
    def create_module(self, net: network.Network, weights: network.NetworkWeights):
        if all(x in weights.w for x in ["lora_up.weight", "lora_down.weight"]):
            return NetworkModuleLora(net, weights)

        return None


class NetworkModuleLora(network.NetworkModule):
    def __init__(self,  net: network.Network, weights: network.NetworkWeights):
        super().__init__(net, weights)

        self.up = self.create_module(weights.w["lora_up.weight"])
        self.down = self.create_module(weights.w["lora_down.weight"])
        self.alpha = weights.w["alpha"] if "alpha" in weights.w else None

    def create_module(self, weight, none_ok=False):
        if weight is None and none_ok:
            return None

        if type(self.sd_module) == torch.nn.Linear:
            module = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False)
        elif type(self.sd_module) == torch.nn.modules.linear.NonDynamicallyQuantizableLinear:
            module = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False)
        elif type(self.sd_module) == torch.nn.MultiheadAttention:
            module = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False)
        elif type(self.sd_module) == torch.nn.Conv2d and weight.shape[2:] == (1, 1):
            module = torch.nn.Conv2d(weight.shape[1], weight.shape[0], (1, 1), bias=False)
        elif type(self.sd_module) == torch.nn.Conv2d and weight.shape[2:] == (3, 3):
            module = torch.nn.Conv2d(weight.shape[1], weight.shape[0], (3, 3), bias=False)
        else:
            print(f'Network layer {self.network_key} matched a layer with unsupported type: {type(self.sd_module).__name__}')
            return None

        with torch.no_grad():
            module.weight.copy_(weight)

        module.to(device=devices.cpu, dtype=devices.dtype)
        module.weight.requires_grad_(False)

        return module

    def calc_updown(self, target):
        up = self.up.weight.to(target.device, dtype=target.dtype)
        down = self.down.weight.to(target.device, dtype=target.dtype)

        if up.shape[2:] == (1, 1) and down.shape[2:] == (1, 1):
            updown = (up.squeeze(2).squeeze(2) @ down.squeeze(2).squeeze(2)).unsqueeze(2).unsqueeze(3)
        elif up.shape[2:] == (3, 3) or down.shape[2:] == (3, 3):
            updown = torch.nn.functional.conv2d(down.permute(1, 0, 2, 3), up).permute(1, 0, 2, 3)
        else:
            updown = up @ down

        updown = updown * self.network.multiplier * (self.alpha / self.up.weight.shape[1] if self.alpha else 1.0)

        return updown

    def forward(self, x, y):
        self.up.to(device=devices.device)
        self.down.to(device=devices.device)

        return y + self.up(self.down(x)) * self.network.multiplier * (self.alpha / self.up.weight.shape[1] if self.alpha else 1.0)


