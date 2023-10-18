import torch
import network


class ModuleTypeOFT(network.ModuleType):
    def create_module(self, net: network.Network, weights: network.NetworkWeights):
        if all(x in weights.w for x in ["oft_blocks"]):
            return NetworkModuleOFT(net, weights)

        return None

# adapted from https://github.com/kohya-ss/sd-scripts/blob/main/networks/oft.py
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

        self.constraint = self.alpha
        #self.constraint = self.alpha * self.out_dim
        self.block_size = self.out_dim // self.num_blocks

        self.org_module: list[torch.Module] = [self.sd_module]

        self.R = self.get_weight()

        self.apply_to()

    # replace forward method of original linear rather than replacing the module
    def apply_to(self):
        self.org_forward = self.org_module[0].forward
        self.org_module[0].forward = self.forward
    
    def get_weight(self, multiplier=None):
        if not multiplier:
            multiplier = self.multiplier()
        block_Q = self.oft_blocks - self.oft_blocks.transpose(1, 2)
        norm_Q = torch.norm(block_Q.flatten())
        new_norm_Q = torch.clamp(norm_Q, max=self.constraint)
        block_Q = block_Q * ((new_norm_Q + 1e-8) / (norm_Q + 1e-8))
        I = torch.eye(self.block_size, device=self.oft_blocks.device).unsqueeze(0).repeat(self.num_blocks, 1, 1)
        block_R = torch.matmul(I + block_Q, (I - block_Q).inverse())

        block_R_weighted = multiplier * block_R + (1 - multiplier) * I
        R = torch.block_diag(*block_R_weighted)

        return R

    def calc_updown(self, orig_weight):
        # this works
        # R = self.R
        self.R = self.get_weight(self.multiplier())

        # sending R to device causes major deepfrying i.e. just doesn't work
        # R = self.R.to(orig_weight.device, dtype=orig_weight.dtype)

        # if orig_weight.dim() == 4:
        #     weight = torch.einsum("oihw, op -> pihw", orig_weight, R)
        # else:
        #     weight = torch.einsum("oi, op -> pi", orig_weight, R)

        updown = orig_weight @ self.R
        output_shape = self.oft_blocks.shape

        ## this works
        # updown = orig_weight @ R
        # output_shape = [orig_weight.size(0), R.size(1)]

        return self.finalize_updown(updown, orig_weight, output_shape)
    
    def forward(self, x, y=None):
        x = self.org_forward(x)
        if self.multiplier() == 0.0:
            return x
        #R = self.get_weight().to(x.device, dtype=x.dtype)
        R = self.R.to(x.device, dtype=x.dtype)
        if x.dim() == 4:
            x = x.permute(0, 2, 3, 1)
            x = torch.matmul(x, R)
            x = x.permute(0, 3, 1, 2)
        else:
            x = torch.matmul(x, R)
        return x
