import torch
import network
from modules import devices


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
        self.org_weight = self.org_module[0].weight.to(self.org_module[0].weight.device, copy=True)
        #self.org_weight = self.org_module[0].weight.to(devices.cpu, copy=True)
        init_multiplier = self.multiplier() * self.calc_scale()
        self.last_multiplier = init_multiplier
        self.R = self.get_weight(self.oft_blocks, init_multiplier)

        self.merged_weight = self.merge_weight()
        self.apply_to()
        self.merged = False

        # weights_backup = getattr(self.org_module[0], 'network_weights_backup', None)
        # if weights_backup is None:
        #     self.org_module[0].network_weights_backup = self.org_weight


    def merge_weight(self):
        #org_sd = self.org_module[0].state_dict()
        R = self.R.to(self.org_weight.device, dtype=self.org_weight.dtype)
        if self.org_weight.dim() == 4:
            weight = torch.einsum("oihw, op -> pihw", self.org_weight, R)
        else:
            weight = torch.einsum("oi, op -> pi", self.org_weight, R)
        #org_sd['weight'] = weight
        # replace weight
        #self.org_module[0].load_state_dict(org_sd)
        return weight
        pass
    
    def replace_weight(self, new_weight):
        org_sd = self.org_module[0].state_dict()
        org_sd['weight'] = new_weight
        self.org_module[0].load_state_dict(org_sd)
        self.merged = True

    def restore_weight(self):
        org_sd = self.org_module[0].state_dict()
        org_sd['weight'] = self.org_weight
        self.org_module[0].load_state_dict(org_sd)
        self.merged = False


    # replace forward method of original linear rather than replacing the module
    # how do we revert this to unload the weights?
    def apply_to(self):
        self.org_forward = self.org_module[0].forward
        #self.org_module[0].forward = self.forward
        self.org_module[0].register_forward_pre_hook(self.pre_forward_hook)
        self.org_module[0].register_forward_hook(self.forward_hook)

    def get_weight(self, oft_blocks, multiplier=None):
        multiplier = multiplier.to(oft_blocks.device, dtype=oft_blocks.dtype)
        constraint = self.constraint.to(oft_blocks.device, dtype=oft_blocks.dtype)
        block_Q = oft_blocks - oft_blocks.transpose(1, 2)
        norm_Q = torch.norm(block_Q.flatten())
        new_norm_Q = torch.clamp(norm_Q, max=constraint)
        block_Q = block_Q * ((new_norm_Q + 1e-8) / (norm_Q + 1e-8))
        m_I = torch.eye(self.block_size, device=oft_blocks.device).unsqueeze(0).repeat(self.num_blocks, 1, 1)
        block_R = torch.matmul(m_I + block_Q, (m_I - block_Q).inverse())
        block_R_weighted = multiplier * block_R + (1 - multiplier) * m_I
        R = torch.block_diag(*block_R_weighted)
        #R = torch.block_diag(*block_R)

        return R

    def calc_updown(self, orig_weight):
        #oft_blocks = self.oft_blocks.to(orig_weight.device, dtype=orig_weight.dtype)

        #R = self.R.to(orig_weight.device, dtype=orig_weight.dtype)
        ##self.R = R

        #R = self.R.to(orig_weight.device, dtype=orig_weight.dtype)
        ##self.R = R
        #if orig_weight.dim() == 4:
        #    weight = torch.einsum("oihw, op -> pihw", orig_weight, R)
        #else:
        #    weight = torch.einsum("oi, op -> pi", orig_weight, R)

        #updown = orig_weight @ R
        #updown = weight
        updown = torch.zeros_like(orig_weight, device=orig_weight.device, dtype=orig_weight.dtype)
        #updown = orig_weight
        output_shape = orig_weight.shape
        orig_weight = self.merged_weight.to(orig_weight.device, dtype=orig_weight.dtype)
        #output_shape = self.oft_blocks.shape

        return self.finalize_updown(updown, orig_weight, output_shape)
    
    def pre_forward_hook(self, module, input):
        multiplier = self.multiplier() * self.calc_scale()
        if not multiplier==self.last_multiplier or not self.merged:

        #if multiplier != self.last_multiplier or not self.merged:
            self.R = self.get_weight(self.oft_blocks, multiplier)
            self.last_multiplier = multiplier
            self.merged_weight = self.merge_weight()
            self.replace_weight(self.merged_weight)
        #elif not self.merged:
        #    self.replace_weight(self.merged_weight)

    
    def forward_hook(self, module, args, output):
        pass
        #output = output * self.multiplier() * self.calc_scale()
        #if len(args) > 0:
        #    y = args[0]
        #    output = output + y
        #return output
        #if self.merged:
        #    pass
            #self.restore_weight()
        #print(f'Forward hook in {self.network_key} called')

        #x = output
        #R = self.R.to(x.device, dtype=x.dtype)

        #if x.dim() == 4:
        #    x = x.permute(0, 2, 3, 1)
        #    x = torch.matmul(x, R)
        #    x = x.permute(0, 3, 1, 2)
        #else:
        #    x = torch.matmul(x, R)
        #return x

    # def forward(self, x, y=None):
    #     x = self.org_forward(x)
    #     if self.multiplier() == 0.0:
    #         return x

    #     # calculating R here is excruciatingly slow
    #     #R = self.get_weight().to(x.device, dtype=x.dtype)
    #     R = self.R.to(x.device, dtype=x.dtype)

    #     if x.dim() == 4:
    #         x = x.permute(0, 2, 3, 1)
    #         x = torch.matmul(x, R)
    #         x = x.permute(0, 3, 1, 2)
    #     else:
    #         x = torch.matmul(x, R)
    #     return x
