import torch
import networks
from modules import patches, shared


class LoraPatches:
    def __init__(self):
        self.active = False
        self.Linear_forward = None
        self.Linear_load_state_dict = None
        self.Conv2d_forward = None
        self.Conv2d_load_state_dict = None
        self.GroupNorm_forward = None
        self.GroupNorm_load_state_dict = None
        self.LayerNorm_forward = None
        self.LayerNorm_load_state_dict = None
        self.MultiheadAttention_forward = None
        self.MultiheadAttention_load_state_dict = None

    def apply(self):
        if self.active or shared.opts.lora_force_diffusers:
            return
        self.Linear_forward = patches.patch(__name__, torch.nn.Linear, 'forward', networks.network_Linear_forward)
        self.Linear_load_state_dict = patches.patch(__name__, torch.nn.Linear, '_load_from_state_dict', networks.network_Linear_load_state_dict)
        self.Conv2d_forward = patches.patch(__name__, torch.nn.Conv2d, 'forward', networks.network_Conv2d_forward)
        self.Conv2d_load_state_dict = patches.patch(__name__, torch.nn.Conv2d, '_load_from_state_dict', networks.network_Conv2d_load_state_dict)
        self.GroupNorm_forward = patches.patch(__name__, torch.nn.GroupNorm, 'forward', networks.network_GroupNorm_forward)
        self.GroupNorm_load_state_dict = patches.patch(__name__, torch.nn.GroupNorm, '_load_from_state_dict', networks.network_GroupNorm_load_state_dict)
        self.LayerNorm_forward = patches.patch(__name__, torch.nn.LayerNorm, 'forward', networks.network_LayerNorm_forward)
        self.LayerNorm_load_state_dict = patches.patch(__name__, torch.nn.LayerNorm, '_load_from_state_dict', networks.network_LayerNorm_load_state_dict)
        self.MultiheadAttention_forward = patches.patch(__name__, torch.nn.MultiheadAttention, 'forward', networks.network_MultiheadAttention_forward)
        self.MultiheadAttention_load_state_dict = patches.patch(__name__, torch.nn.MultiheadAttention, '_load_from_state_dict', networks.network_MultiheadAttention_load_state_dict)
        networks.timer['load'] = 0
        networks.timer['apply'] = 0
        networks.timer['restore'] = 0
        self.active = True

    def undo(self):
        if not self.active or shared.opts.lora_force_diffusers:
            return
        self.Linear_forward = patches.undo(__name__, torch.nn.Linear, 'forward') # pylint: disable=E1128
        self.Linear_load_state_dict = patches.undo(__name__, torch.nn.Linear, '_load_from_state_dict') # pylint: disable=E1128
        self.Conv2d_forward = patches.undo(__name__, torch.nn.Conv2d, 'forward') # pylint: disable=E1128
        self.Conv2d_load_state_dict = patches.undo(__name__, torch.nn.Conv2d, '_load_from_state_dict') # pylint: disable=E1128
        self.GroupNorm_forward = patches.undo(__name__, torch.nn.GroupNorm, 'forward') # pylint: disable=E1128
        self.GroupNorm_load_state_dict = patches.undo(__name__, torch.nn.GroupNorm, '_load_from_state_dict') # pylint: disable=E1128
        self.LayerNorm_forward = patches.undo(__name__, torch.nn.LayerNorm, 'forward') # pylint: disable=E1128
        self.LayerNorm_load_state_dict = patches.undo(__name__, torch.nn.LayerNorm, '_load_from_state_dict') # pylint: disable=E1128
        self.MultiheadAttention_forward = patches.undo(__name__, torch.nn.MultiheadAttention, 'forward') # pylint: disable=E1128
        self.MultiheadAttention_load_state_dict = patches.undo(__name__, torch.nn.MultiheadAttention, '_load_from_state_dict') # pylint: disable=E1128
        patches.originals.pop(__name__, None)
        self.active = False
