import torch

import networks
from modules import patches


class LoraPatches:
    def __init__(self):
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

    def undo(self):
        self.Linear_forward = patches.undo(__name__, torch.nn.Linear, 'forward')
        self.Linear_load_state_dict = patches.undo(__name__, torch.nn.Linear, '_load_from_state_dict')
        self.Conv2d_forward = patches.undo(__name__, torch.nn.Conv2d, 'forward')
        self.Conv2d_load_state_dict = patches.undo(__name__, torch.nn.Conv2d, '_load_from_state_dict')
        self.GroupNorm_forward = patches.undo(__name__, torch.nn.GroupNorm, 'forward')
        self.GroupNorm_load_state_dict = patches.undo(__name__, torch.nn.GroupNorm, '_load_from_state_dict')
        self.LayerNorm_forward = patches.undo(__name__, torch.nn.LayerNorm, 'forward')
        self.LayerNorm_load_state_dict = patches.undo(__name__, torch.nn.LayerNorm, '_load_from_state_dict')
        self.MultiheadAttention_forward = patches.undo(__name__, torch.nn.MultiheadAttention, 'forward')
        self.MultiheadAttention_load_state_dict = patches.undo(__name__, torch.nn.MultiheadAttention, '_load_from_state_dict')

