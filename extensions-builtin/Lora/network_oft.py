import torch
import network
from einops import rearrange


class ModuleTypeOFT(network.ModuleType):
    def create_module(self, net: network.Network, weights: network.NetworkWeights):
        if all(x in weights.w for x in ["oft_blocks"]):
            return NetworkModuleOFT(net, weights)

        return None

# adapted from kohya's implementation https://github.com/kohya-ss/sd-scripts/blob/main/networks/oft.py
class NetworkModuleOFT(network.NetworkModule):
    def __init__(self,  net: network.Network, weights: network.NetworkWeights):

        super().__init__(net, weights)

        self.oft_blocks = weights.w["oft_blocks"]
        self.alpha = weights.w["alpha"]
        self.dim = self.oft_blocks.shape[0]
        self.num_blocks = self.dim

        if "Linear" in self.sd_module.__class__.__name__:
            self.out_dim = self.sd_module.out_features
        elif "Conv" in self.sd_module.__class__.__name__:
            self.out_dim = self.sd_module.out_channels

        self.constraint = self.alpha * self.out_dim
        self.block_size = self.out_dim // self.num_blocks

        self.org_module: list[torch.Module] = [self.sd_module]

    # def merge_weight(self, R_weight, org_weight):
    #     R_weight = R_weight.to(org_weight.device, dtype=org_weight.dtype)
    #     if org_weight.dim() == 4:
    #         weight = torch.einsum("oihw, op -> pihw", org_weight, R_weight)
    #     else:
    #         weight = torch.einsum("oi, op -> pi", org_weight, R_weight)
    #     weight = torch.einsum(
    #         "k n m, k n ... -> k m ...", 
    #         self.oft_diag * scale + torch.eye(self.block_size, device=device), 
    #         org_weight
    #     )
    #     return weight

    def get_weight(self, oft_blocks, multiplier=None):
        # constraint = self.constraint.to(oft_blocks.device, dtype=oft_blocks.dtype)

        # block_Q = oft_blocks - oft_blocks.transpose(1, 2)
        # norm_Q = torch.norm(block_Q.flatten())
        # new_norm_Q = torch.clamp(norm_Q, max=constraint)
        # block_Q = block_Q * ((new_norm_Q + 1e-8) / (norm_Q + 1e-8))
        # m_I = torch.eye(self.block_size, device=oft_blocks.device).unsqueeze(0).repeat(self.num_blocks, 1, 1)
        # block_R = torch.matmul(m_I + block_Q, (m_I - block_Q).inverse())

        # block_R_weighted = multiplier * block_R + (1 - multiplier) * m_I
        # R = torch.block_diag(*block_R_weighted)
        #return R
        return self.oft_blocks


    def calc_updown(self, orig_weight):
        multiplier = self.multiplier() * self.calc_scale()
        #R = self.get_weight(self.oft_blocks, multiplier)
        R = self.oft_blocks.to(orig_weight.device, dtype=orig_weight.dtype)
        #merged_weight = self.merge_weight(R, orig_weight)

        orig_weight = rearrange(orig_weight, '(k n) ... -> k n ...', k=self.num_blocks, n=self.block_size)
        weight = torch.einsum(
            'k n m, k n ... -> k m ...',
            R * multiplier + torch.eye(self.block_size, device=orig_weight.device),
            orig_weight
        )
        weight = rearrange(weight, 'k m ... -> (k m) ...')

        #updown = merged_weight.to(orig_weight.device, dtype=orig_weight.dtype) - orig_weight
        updown = weight.to(orig_weight.device, dtype=orig_weight.dtype) - orig_weight
        output_shape = orig_weight.shape
        orig_weight = orig_weight

        return self.finalize_updown(updown, orig_weight, output_shape)

    # override to remove the multiplier/scale factor; it's already multiplied in get_weight
    def finalize_updown(self, updown, orig_weight, output_shape, ex_bias=None):
        #return super().finalize_updown(updown, orig_weight, output_shape, ex_bias)

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
