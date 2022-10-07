import glob
import os
import sys
import traceback

import torch
from modules import devices


class HypernetworkModule(torch.nn.Module):
    def __init__(self, dim, state_dict):
        super().__init__()

        self.linear1 = torch.nn.Linear(dim, dim * 2)
        self.linear2 = torch.nn.Linear(dim * 2, dim)

        self.load_state_dict(state_dict, strict=True)
        self.to(devices.device)

    def forward(self, x):
        return x + (self.linear2(self.linear1(x)))


class Hypernetwork:
    filename = None
    name = None

    def __init__(self, filename):
        self.filename = filename
        self.name = os.path.splitext(os.path.basename(filename))[0]
        self.layers = {}

        state_dict = torch.load(filename, map_location='cpu')
        for size, sd in state_dict.items():
            self.layers[size] = (HypernetworkModule(size, sd[0]), HypernetworkModule(size, sd[1]))


def load_hypernetworks(path):
    res = {}

    for filename in glob.iglob(path + '**/*.pt', recursive=True):
        try:
            hn = Hypernetwork(filename)
            res[hn.name] = hn
        except Exception:
            print(f"Error loading hypernetwork {filename}", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)

    return res

def apply(self, x, context=None, mask=None, original=None):


    if CrossAttention.hypernetwork is not None and context.shape[2] in CrossAttention.hypernetwork:
        if context.shape[1] == 77 and CrossAttention.noise_cond:
            context = context + (torch.randn_like(context) * 0.1)
        h_k, h_v = CrossAttention.hypernetwork[context.shape[2]]
        k = self.to_k(h_k(context))
        v = self.to_v(h_v(context))
    else:
        k = self.to_k(context)
        v = self.to_v(context)
