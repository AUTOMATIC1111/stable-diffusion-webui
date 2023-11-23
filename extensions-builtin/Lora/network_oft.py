import torch
import network
from lyco_helpers import factorization
from einops import rearrange


class ModuleTypeOFT(network.ModuleType):
    def create_module(self, net: network.Network, weights: network.NetworkWeights):
        if all(x in weights.w for x in ["oft_blocks"]) or all(x in weights.w for x in ["oft_diag"]):
            return NetworkModuleOFT(net, weights)

        return None

# Supports both kohya-ss' implementation of COFT  https://github.com/kohya-ss/sd-scripts/blob/main/networks/oft.py
# and KohakuBlueleaf's implementation of OFT/COFT https://github.com/KohakuBlueleaf/LyCORIS/blob/dev/lycoris/modules/diag_oft.py
class NetworkModuleOFT(network.NetworkModule):
    def __init__(self,  net: network.Network, weights: network.NetworkWeights):

        super().__init__(net, weights)

        self.lin_module = None
        self.org_module: list[torch.Module] = [self.sd_module]

        # kohya-ss
        if "oft_blocks" in weights.w.keys():
            self.is_kohya = True
            self.oft_blocks = weights.w["oft_blocks"] # (num_blocks, block_size, block_size)
            self.alpha = weights.w["alpha"] # alpha is constraint
            self.dim = self.oft_blocks.shape[0] # lora dim
        # LyCORIS
        elif "oft_diag" in weights.w.keys():
            self.is_kohya = False
            self.oft_blocks = weights.w["oft_diag"]
            # self.alpha is unused
            self.dim = self.oft_blocks.shape[1] # (num_blocks, block_size, block_size)

        is_linear = type(self.sd_module) in [torch.nn.Linear, torch.nn.modules.linear.NonDynamicallyQuantizableLinear]
        is_conv = type(self.sd_module) in [torch.nn.Conv2d]
        is_other_linear = type(self.sd_module) in [torch.nn.MultiheadAttention] # unsupported

        if is_linear:
            self.out_dim = self.sd_module.out_features
        elif is_conv:
            self.out_dim = self.sd_module.out_channels
        elif is_other_linear:
            self.out_dim = self.sd_module.embed_dim

        if self.is_kohya:
            self.constraint = self.alpha * self.out_dim
            self.num_blocks = self.dim
            self.block_size = self.out_dim // self.dim
        else:
            self.constraint = None
            self.block_size, self.num_blocks = factorization(self.out_dim, self.dim)

    def calc_updown_kb(self, orig_weight, multiplier):
        oft_blocks = self.oft_blocks.to(orig_weight.device, dtype=orig_weight.dtype)
        oft_blocks = oft_blocks - oft_blocks.transpose(1, 2) # ensure skew-symmetric orthogonal matrix

        R = oft_blocks.to(orig_weight.device, dtype=orig_weight.dtype)
        R = R * multiplier + torch.eye(self.block_size, device=orig_weight.device)

        # This errors out for MultiheadAttention, might need to be handled up-stream
        merged_weight = rearrange(orig_weight, '(k n) ... -> k n ...', k=self.num_blocks, n=self.block_size)
        merged_weight = torch.einsum(
            'k n m, k n ... -> k m ...',
            R,
            merged_weight
        )
        merged_weight = rearrange(merged_weight, 'k m ... -> (k m) ...')

        updown = merged_weight.to(orig_weight.device, dtype=orig_weight.dtype) - orig_weight
        output_shape = orig_weight.shape
        return self.finalize_updown(updown, orig_weight, output_shape)

    def calc_updown(self, orig_weight):
        # if alpha is a very small number as in coft, calc_scale() will return a almost zero number so we ignore it
        multiplier = self.multiplier()
        return self.calc_updown_kb(orig_weight, multiplier)

    # override to remove the multiplier/scale factor; it's already multiplied in get_weight
    def finalize_updown(self, updown, orig_weight, output_shape, ex_bias=None):
        if self.bias is not None:
            updown = updown.reshape(self.bias.shape)
            updown += self.bias.to(orig_weight.device, dtype=orig_weight.dtype)
            updown = updown.reshape(output_shape)

        if len(output_shape) == 4:
            updown = updown.reshape(output_shape)

        if orig_weight.size().numel() == updown.size().numel():
            updown = updown.reshape(orig_weight.shape)

        if ex_bias is not None:
            ex_bias = ex_bias * self.multiplier()

        return updown, ex_bias
