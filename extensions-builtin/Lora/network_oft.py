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

        self.scale = 1.0
        self.is_kohya = False
        self.is_boft = False

        # kohya-ss
        if "oft_blocks" in weights.w.keys():
            self.is_kohya = True
            self.oft_blocks = weights.w["oft_blocks"] # (num_blocks, block_size, block_size)
            self.alpha = weights.w["alpha"] # alpha is constraint
            self.dim = self.oft_blocks.shape[0] # lora dim
        # LyCORIS OFT
        elif "oft_diag" in weights.w.keys():
            self.oft_blocks = weights.w["oft_diag"]
            # self.alpha is unused
            self.dim = self.oft_blocks.shape[1] # (num_blocks, block_size, block_size)

            # LyCORIS BOFT
            if weights.w["oft_diag"].dim() == 4:
                self.is_boft = True
        self.rescale = weights.w.get('rescale', None)
        if self.rescale is not None:
            self.rescale = self.rescale.reshape(-1, *[1]*(self.org_module[0].weight.dim() - 1))

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
        elif self.is_boft:
            self.constraint = None
            self.boft_m = weights.w["oft_diag"].shape[0]
            self.block_num = weights.w["oft_diag"].shape[1]
            self.block_size = weights.w["oft_diag"].shape[2]
            self.boft_b = self.block_size
            #self.block_size, self.block_num = butterfly_factor(self.out_dim, self.dim)
        else:
            self.constraint = None
            self.block_size, self.num_blocks = factorization(self.out_dim, self.dim)

    def calc_updown(self, orig_weight):
        oft_blocks = self.oft_blocks.to(orig_weight.device)
        eye = torch.eye(self.block_size, device=oft_blocks.device)

        if self.is_kohya:
            block_Q = oft_blocks - oft_blocks.transpose(1, 2) # ensure skew-symmetric orthogonal matrix
            norm_Q = torch.norm(block_Q.flatten())
            new_norm_Q = torch.clamp(norm_Q, max=self.constraint.to(oft_blocks.device))
            block_Q = block_Q * ((new_norm_Q + 1e-8) / (norm_Q + 1e-8))
            oft_blocks = torch.matmul(eye + block_Q, (eye - block_Q).float().inverse())

        R = oft_blocks.to(orig_weight.device)

        if not self.is_boft:
            # This errors out for MultiheadAttention, might need to be handled up-stream
            merged_weight = rearrange(orig_weight, '(k n) ... -> k n ...', k=self.num_blocks, n=self.block_size)
            merged_weight = torch.einsum(
                'k n m, k n ... -> k m ...',
                R,
                merged_weight
            )
            merged_weight = rearrange(merged_weight, 'k m ... -> (k m) ...')
        else:
            # TODO: determine correct value for scale
            scale = 1.0
            m = self.boft_m
            b = self.boft_b
            r_b = b // 2
            inp = orig_weight
            for i in range(m):
                bi = R[i] # b_num, b_size, b_size
                if i == 0:
                    # Apply multiplier/scale and rescale into first weight
                    bi = bi * scale + (1 - scale) * eye
                inp = rearrange(inp, "(c g k) ... -> (c k g) ...", g=2, k=2**i * r_b)
                inp = rearrange(inp, "(d b) ... -> d b ...", b=b)
                inp = torch.einsum("b i j, b j ... -> b i ...", bi, inp)
                inp = rearrange(inp, "d b ... -> (d b) ...")
                inp = rearrange(inp, "(c k g) ... -> (c g k) ...", g=2, k=2**i * r_b)
            merged_weight = inp

        # Rescale mechanism
        if self.rescale is not None:
            merged_weight = self.rescale.to(merged_weight) * merged_weight

        updown = merged_weight.to(orig_weight.device) - orig_weight.to(merged_weight.dtype)
        output_shape = orig_weight.shape
        return self.finalize_updown(updown, orig_weight, output_shape)
