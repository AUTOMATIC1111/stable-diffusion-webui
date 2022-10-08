import glob
import os
import sys
import traceback

import torch

from ldm.util import default
from modules import devices, shared
import torch
from torch import einsum
from einops import rearrange, repeat


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

    for filename in glob.iglob(os.path.join(path, '**/*.pt'), recursive=True):
        try:
            hn = Hypernetwork(filename)
            res[hn.name] = hn
        except Exception:
            print(f"Error loading hypernetwork {filename}", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)

    return res


def attention_CrossAttention_forward(self, x, context=None, mask=None):
    h = self.heads

    q = self.to_q(x)
    context = default(context, x)

    hypernetwork = shared.selected_hypernetwork()
    hypernetwork_layers = (hypernetwork.layers if hypernetwork is not None else {}).get(context.shape[2], None)

    if hypernetwork_layers is not None:
        k = self.to_k(hypernetwork_layers[0](context))
        v = self.to_v(hypernetwork_layers[1](context))
    else:
        k = self.to_k(context)
        v = self.to_v(context)

    q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

    sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

    if mask is not None:
        mask = rearrange(mask, 'b ... -> b (...)')
        max_neg_value = -torch.finfo(sim.dtype).max
        mask = repeat(mask, 'b j -> (b h) () j', h=h)
        sim.masked_fill_(~mask, max_neg_value)

    # attention, what we cannot get enough of
    attn = sim.softmax(dim=-1)

    out = einsum('b i j, b j d -> b i d', attn, v)
    out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
    return self.to_out(out)
