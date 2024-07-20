import torch

import lyco_helpers
import modules.models.sd3.mmdit
import network
from modules import devices


class ModuleTypeLora(network.ModuleType):
    def create_module(self, net: network.Network, weights: network.NetworkWeights):
        if all(x in weights.w for x in ["lora_up.weight", "lora_down.weight"]):
            return NetworkModuleLora(net, weights)

        if all(x in weights.w for x in ["lora_A.weight", "lora_B.weight"]):
            w = weights.w.copy()
            weights.w.clear()
            weights.w.update({"lora_up.weight": w["lora_B.weight"], "lora_down.weight": w["lora_A.weight"]})

            return NetworkModuleLora(net, weights)

        return None


class NetworkModuleLora(network.NetworkModule):
    def __init__(self,  net: network.Network, weights: network.NetworkWeights):
        super().__init__(net, weights)

        self.up_model = self.create_module(weights.w, "lora_up.weight")
        self.down_model = self.create_module(weights.w, "lora_down.weight")
        self.mid_model = self.create_module(weights.w, "lora_mid.weight", none_ok=True)

        self.dim = weights.w["lora_down.weight"].shape[0]

    def create_module(self, weights, key, none_ok=False):
        weight = weights.get(key)

        if weight is None and none_ok:
            return None

        is_linear = type(self.sd_module) in [torch.nn.Linear, torch.nn.modules.linear.NonDynamicallyQuantizableLinear, torch.nn.MultiheadAttention, modules.models.sd3.mmdit.QkvLinear]
        is_conv = type(self.sd_module) in [torch.nn.Conv2d]

        if is_linear:
            weight = weight.reshape(weight.shape[0], -1)
            module = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False)
        elif is_conv and key == "lora_down.weight" or key == "dyn_up":
            if len(weight.shape) == 2:
                weight = weight.reshape(weight.shape[0], -1, 1, 1)

            if weight.shape[2] != 1 or weight.shape[3] != 1:
                module = torch.nn.Conv2d(weight.shape[1], weight.shape[0], self.sd_module.kernel_size, self.sd_module.stride, self.sd_module.padding, bias=False)
            else:
                module = torch.nn.Conv2d(weight.shape[1], weight.shape[0], (1, 1), bias=False)
        elif is_conv and key == "lora_mid.weight":
            module = torch.nn.Conv2d(weight.shape[1], weight.shape[0], self.sd_module.kernel_size, self.sd_module.stride, self.sd_module.padding, bias=False)
        elif is_conv and key == "lora_up.weight" or key == "dyn_down":
            module = torch.nn.Conv2d(weight.shape[1], weight.shape[0], (1, 1), bias=False)
        else:
            raise AssertionError(f'Lora layer {self.network_key} matched a layer with unsupported type: {type(self.sd_module).__name__}')

        with torch.no_grad():
            if weight.shape != module.weight.shape:
                weight = weight.reshape(module.weight.shape)
            module.weight.copy_(weight)

        module.to(device=devices.cpu, dtype=devices.dtype)
        module.weight.requires_grad_(False)

        return module

    def calc_updown(self, orig_weight):
        up = self.up_model.weight.to(orig_weight.device)
        down = self.down_model.weight.to(orig_weight.device)

        output_shape = [up.size(0), down.size(1)]
        if self.mid_model is not None:
            # cp-decomposition
            mid = self.mid_model.weight.to(orig_weight.device)
            updown = lyco_helpers.rebuild_cp_decomposition(up, down, mid)
            output_shape += mid.shape[2:]
        else:
            if len(down.shape) == 4:
                output_shape += down.shape[2:]
            updown = lyco_helpers.rebuild_conventional(up, down, output_shape, self.network.dyn_dim)

        return self.finalize_updown(updown, orig_weight, output_shape)

    def forward(self, x, y):
        self.up_model.to(device=devices.device)
        self.down_model.to(device=devices.device)

        return y + self.up_model(self.down_model(x)) * self.multiplier() * self.calc_scale()


