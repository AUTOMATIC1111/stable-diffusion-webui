import torch
import network
from einops import rearrange
from modules import devices


class ModuleTypeOFT(network.ModuleType):
    def create_module(self, net: network.Network, weights: network.NetworkWeights):
        if all(x in weights.w for x in ["oft_blocks"]) or all(x in weights.w for x in ["oft_diag"]):
            return NetworkModuleOFT(net, weights)

        return None

# adapted from kohya's implementation https://github.com/kohya-ss/sd-scripts/blob/main/networks/oft.py
class NetworkModuleOFT(network.NetworkModule):
    def __init__(self,  net: network.Network, weights: network.NetworkWeights):

        super().__init__(net, weights)

        self.lin_module = None
        self.org_module: list[torch.Module] = [self.sd_module]
        # kohya-ss
        if "oft_blocks" in weights.w.keys():
            self.is_kohya = True
            self.oft_blocks = weights.w["oft_blocks"]
            self.alpha = weights.w["alpha"]
            self.dim = self.oft_blocks.shape[0]
        elif "oft_diag" in weights.w.keys():
            self.is_kohya = False
            self.oft_blocks = weights.w["oft_diag"]
            # alpha is rank if alpha is 0 or None
            if self.alpha is None:
                pass
            self.dim = self.oft_blocks.shape[1] # FIXME: almost certainly incorrect, assumes tensor is shape [*, m, n]
        else:
            raise ValueError("oft_blocks or oft_diag must be in weights dict")

        is_linear = type(self.sd_module) in [torch.nn.Linear, torch.nn.modules.linear.NonDynamicallyQuantizableLinear]
        is_conv = type(self.sd_module) in [torch.nn.Conv2d]
        is_other_linear = type(self.sd_module) in [ torch.nn.MultiheadAttention]
        #if "Linear" in self.sd_module.__class__.__name__ or is_linear:
        if is_linear:
            self.out_dim = self.sd_module.out_features
            #elif hasattr(self.sd_module, "embed_dim"):
            #    self.out_dim = self.sd_module.embed_dim
            #else:
            #    raise ValueError("Linear sd_module must have out_features or embed_dim")
        elif is_other_linear:
            self.out_dim = self.sd_module.embed_dim
            #self.org_weight = self.org_module[0].weight
#            if hasattr(self.sd_module, "in_proj_weight"):
#                self.in_proj_dim = self.sd_module.in_proj_weight.shape[1]
#            if hasattr(self.sd_module, "out_proj_weight"):
#                self.out_proj_dim = self.sd_module.out_proj_weight.shape[0]
#            self.in_proj_dim = self.sd_module.in_proj_weight.shape[1]
        elif is_conv:
            self.out_dim = self.sd_module.out_channels
        else:
            raise ValueError("sd_module must be Linear or Conv")


        if self.is_kohya:
            self.num_blocks = self.dim
            self.block_size = self.out_dim // self.num_blocks
            self.constraint = self.alpha * self.out_dim
        #elif is_linear or is_conv:
        else:
            self.block_size, self.num_blocks = factorization(self.out_dim, self.dim)
            self.constraint = None


        # if is_other_linear:
        #     weight = self.oft_blocks.reshape(self.oft_blocks.shape[0], -1)
        #     module = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False)
        #     with torch.no_grad():
        #         if weight.shape != module.weight.shape:
        #             weight = weight.reshape(module.weight.shape)
        #         module.weight.copy_(weight)
        #     module.to(device=devices.cpu, dtype=devices.dtype)
        #     module.weight.requires_grad_(False)
        #     self.lin_module = module
            #return module

    def merge_weight(self, R_weight, org_weight):
        R_weight = R_weight.to(org_weight.device, dtype=org_weight.dtype)
        if org_weight.dim() == 4:
            weight = torch.einsum("oihw, op -> pihw", org_weight, R_weight)
        else:
            weight = torch.einsum("oi, op -> pi", org_weight, R_weight)
        #weight = torch.einsum(
        #    "k n m, k n ... -> k m ...", 
        #    self.oft_diag * scale + torch.eye(self.block_size, device=device), 
        #    org_weight
        #)
        return weight

    def get_weight(self, oft_blocks, multiplier=None):
        if self.constraint is not None:
            constraint = self.constraint.to(oft_blocks.device, dtype=oft_blocks.dtype)

        block_Q = oft_blocks - oft_blocks.transpose(1, 2)
        norm_Q = torch.norm(block_Q.flatten())
        if self.constraint is not None:
            new_norm_Q = torch.clamp(norm_Q, max=constraint)
        else:
            new_norm_Q = norm_Q
        block_Q = block_Q * ((new_norm_Q + 1e-8) / (norm_Q + 1e-8))
        m_I = torch.eye(self.block_size, device=oft_blocks.device).unsqueeze(0).repeat(self.num_blocks, 1, 1)
        block_R = torch.matmul(m_I + block_Q, (m_I - block_Q).inverse())

        block_R_weighted = multiplier * block_R + (1 - multiplier) * m_I
        R = torch.block_diag(*block_R_weighted)
        return R
        #return self.oft_blocks


    def calc_updown(self, orig_weight):
        multiplier = self.multiplier() * self.calc_scale()
        is_other_linear = type(self.sd_module) in [ torch.nn.MultiheadAttention]
        if self.is_kohya and not is_other_linear:
            R = self.get_weight(self.oft_blocks, multiplier)
            #R = self.oft_blocks.to(orig_weight.device, dtype=orig_weight.dtype)
            merged_weight = self.merge_weight(R, orig_weight)
        elif not self.is_kohya and not is_other_linear:
            if is_other_linear and orig_weight.shape[0] != orig_weight.shape[1]:
                orig_weight=orig_weight.permute(1, 0)
            R = self.oft_blocks.to(orig_weight.device, dtype=orig_weight.dtype)
            merged_weight = rearrange(orig_weight, '(k n) ... -> k n ...', k=self.num_blocks, n=self.block_size)
            #orig_weight = rearrange(orig_weight, '(k n) ... -> k n ...', k=self.block_size, n=self.num_blocks)
            merged_weight = torch.einsum(
                'k n m, k n ... -> k m ...',
                R * multiplier + torch.eye(self.block_size, device=orig_weight.device),
                merged_weight 
            )
            merged_weight = rearrange(merged_weight, 'k m ... -> (k m) ...')
            if is_other_linear and orig_weight.shape[0] != orig_weight.shape[1]:
                orig_weight=orig_weight.permute(1, 0)
                #merged_weight=merged_weight.permute(1, 0)
            updown = merged_weight.to(orig_weight.device, dtype=orig_weight.dtype) - orig_weight
            #updown = weight.to(orig_weight.device, dtype=orig_weight.dtype) - orig_weight
            output_shape = orig_weight.shape
        else:
            # skip for now
            updown = torch.zeros([orig_weight.shape[1], orig_weight.shape[1]], device=orig_weight.device, dtype=orig_weight.dtype)
            output_shape = (orig_weight.shape[1], orig_weight.shape[1])

        #if self.lin_module is not None:
        #    R = self.lin_module.weight.to(orig_weight.device, dtype=orig_weight.dtype)
        #    weight = torch.mul(torch.mul(R, multiplier), orig_weight)
        #else:

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
    
# copied from https://github.com/KohakuBlueleaf/LyCORIS/blob/dev/lycoris/modules/lokr.py
def factorization(dimension: int, factor:int=-1) -> tuple[int, int]:
    '''
    return a tuple of two value of input dimension decomposed by the number closest to factor
    second value is higher or equal than first value.
    
    In LoRA with Kroneckor Product, first value is a value for weight scale.
    secon value is a value for weight.
    
    Becuase of non-commutative property, A⊗B ≠ B⊗A. Meaning of two matrices is slightly different.
    
    examples)
    factor
        -1               2                4               8               16               ...
    127 -> 1, 127   127 -> 1, 127    127 -> 1, 127   127 -> 1, 127   127 -> 1, 127
    128 -> 8, 16    128 -> 2, 64     128 -> 4, 32    128 -> 8, 16    128 -> 8, 16
    250 -> 10, 25   250 -> 2, 125    250 -> 2, 125   250 -> 5, 50    250 -> 10, 25
    360 -> 8, 45    360 -> 2, 180    360 -> 4, 90    360 -> 8, 45    360 -> 12, 30
    512 -> 16, 32   512 -> 2, 256    512 -> 4, 128   512 -> 8, 64    512 -> 16, 32
    1024 -> 32, 32  1024 -> 2, 512   1024 -> 4, 256  1024 -> 8, 128  1024 -> 16, 64
    '''
    
    if factor > 0 and (dimension % factor) == 0:
        m = factor
        n = dimension // factor
        if m > n:
            n, m = m, n
        return m, n
    if factor < 0:
        factor = dimension
    m, n = 1, dimension
    length = m + n
    while m<n:
        new_m = m + 1
        while dimension%new_m != 0:
            new_m += 1
        new_n = dimension // new_m
        if new_m + new_n > length or new_m>factor:
            break
        else:
            m, n = new_m, new_n
    if m > n:
        n, m = m, n
    return m, n

