import torch
import diffusers.models.lora as diffusers_lora
import network
from modules import devices

class ModuleTypeOFT(network.ModuleType):
    def create_module(self, net: network.Network, weights: network.NetworkWeights):
        """
        weights.w.items()

        alpha  :  tensor(0.0010, dtype=torch.bfloat16)
        oft_blocks  :  tensor([[[ 0.0000e+00,  1.4400e-04,  1.7319e-03,  ..., -8.8882e-04,
           5.7373e-03, -4.4250e-03],
         [-1.4400e-04,  0.0000e+00,  8.6594e-04,  ...,  1.5945e-03,
          -8.5449e-04,  1.9684e-03], ...etc...
         , dtype=torch.bfloat16)"""

        if "oft_blocks" in weights.w.keys():
            module = NetworkModuleOFT(net, weights)
            return module
        else:
            return None


class NetworkModuleOFT(network.NetworkModule):
    def __init__(self, net: network.Network, weights: network.NetworkWeights):
        super().__init__(net, weights)

        self.weights = weights.w.get("oft_blocks").to(device=devices.device)
        self.dim = self.weights.shape[0]  # num blocks
        self.alpha = self.multiplier()
        self.block_size = self.weights.shape[-1]

    def get_weight(self):
        block_Q = self.weights - self.weights.transpose(1, 2)
        I = torch.eye(self.block_size, device=devices.device).unsqueeze(0).repeat(self.dim, 1, 1)
        block_R = torch.matmul(I + block_Q, (I - block_Q).inverse())
        block_R_weighted = self.alpha * block_R + (1 - self.alpha) * I
        R = torch.block_diag(*block_R_weighted)
        return R

    def calc_updown(self, orig_weight):
        R = self.get_weight().to(device=devices.device, dtype=orig_weight.dtype)
        if orig_weight.dim() == 4:
            updown = torch.einsum("oihw, op -> pihw", orig_weight, R) * self.calc_scale()
        else:
            updown = torch.einsum("oi, op -> pi", orig_weight, R) * self.calc_scale()

        return self.finalize_updown(updown, orig_weight, orig_weight.shape)
