import math
import sys
import traceback

import torch
from torch import einsum

from ldm.util import default
from einops import rearrange

from modules import shared

if shared.cmd_opts.xformers or shared.cmd_opts.force_enable_xformers:
    try:
        import xformers.ops
        import functorch
        xformers._is_functorch_available = True
        shared.xformers_available = True
    except Exception:
        print("Cannot import xformers", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)


# see https://github.com/basujindal/stable-diffusion/pull/117 for discussion
def split_cross_attention_forward_v1(self, x, context=None, mask=None):
    h = self.heads

    q_in = self.to_q(x)
    context = default(context, x)

    hypernetwork = shared.loaded_hypernetwork
    hypernetwork_layers = (hypernetwork.layers if hypernetwork is not None else {}).get(context.shape[2], None)

    if hypernetwork_layers is not None:
        k_in = self.to_k(hypernetwork_layers[0](context))
        v_in = self.to_v(hypernetwork_layers[1](context))
    else:
        k_in = self.to_k(context)
        v_in = self.to_v(context)
    del context, x

    q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q_in, k_in, v_in))
    del q_in, k_in, v_in

    r1 = torch.zeros(q.shape[0], q.shape[1], v.shape[2], device=q.device)
    for i in range(0, q.shape[0], 2):
        end = i + 2
        s1 = einsum('b i d, b j d -> b i j', q[i:end], k[i:end])
        s1 *= self.scale

        s2 = s1.softmax(dim=-1)
        del s1

        r1[i:end] = einsum('b i j, b j d -> b i d', s2, v[i:end])
        del s2
    del q, k, v

    r2 = rearrange(r1, '(b h) n d -> b n (h d)', h=h)
    del r1

    return self.to_out(r2)


# taken from https://github.com/Doggettx/stable-diffusion
def split_cross_attention_forward(self, x, context=None, mask=None):
    h = self.heads

    q_in = self.to_q(x)
    context = default(context, x)

    hypernetwork = shared.loaded_hypernetwork
    hypernetwork_layers = (hypernetwork.layers if hypernetwork is not None else {}).get(context.shape[2], None)

    if hypernetwork_layers is not None:
        k_in = self.to_k(hypernetwork_layers[0](context))
        v_in = self.to_v(hypernetwork_layers[1](context))
    else:
        k_in = self.to_k(context)
        v_in = self.to_v(context)

    k_in *= self.scale

    del context, x

    q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q_in, k_in, v_in))
    del q_in, k_in, v_in

    r1 = torch.zeros(q.shape[0], q.shape[1], v.shape[2], device=q.device, dtype=q.dtype)

    stats = torch.cuda.memory_stats(q.device)
    mem_active = stats['active_bytes.all.current']
    mem_reserved = stats['reserved_bytes.all.current']
    mem_free_cuda, _ = torch.cuda.mem_get_info(torch.cuda.current_device())
    mem_free_torch = mem_reserved - mem_active
    mem_free_total = mem_free_cuda + mem_free_torch

    gb = 1024 ** 3
    tensor_size = q.shape[0] * q.shape[1] * k.shape[1] * q.element_size()
    modifier = 3 if q.element_size() == 2 else 2.5
    mem_required = tensor_size * modifier
    steps = 1

    if mem_required > mem_free_total:
        steps = 2 ** (math.ceil(math.log(mem_required / mem_free_total, 2)))
        # print(f"Expected tensor size:{tensor_size/gb:0.1f}GB, cuda free:{mem_free_cuda/gb:0.1f}GB "
        #       f"torch free:{mem_free_torch/gb:0.1f} total:{mem_free_total/gb:0.1f} steps:{steps}")

    if steps > 64:
        max_res = math.floor(math.sqrt(math.sqrt(mem_free_total / 2.5)) / 8) * 64
        raise RuntimeError(f'Not enough memory, use lower resolution (max approx. {max_res}x{max_res}). '
                           f'Need: {mem_required / 64 / gb:0.1f}GB free, Have:{mem_free_total / gb:0.1f}GB free')

    slice_size = q.shape[1] // steps if (q.shape[1] % steps) == 0 else q.shape[1]
    for i in range(0, q.shape[1], slice_size):
        end = i + slice_size
        s1 = einsum('b i d, b j d -> b i j', q[:, i:end], k)

        s2 = s1.softmax(dim=-1, dtype=q.dtype)
        del s1

        r1[:, i:end] = einsum('b i j, b j d -> b i d', s2, v)
        del s2

    del q, k, v

    r2 = rearrange(r1, '(b h) n d -> b n (h d)', h=h)
    del r1

    return self.to_out(r2)

def xformers_attention_forward(self, x, context=None, mask=None):
    h = self.heads
    q_in = self.to_q(x)
    context = default(context, x)
    hypernetwork = shared.loaded_hypernetwork
    hypernetwork_layers = (hypernetwork.layers if hypernetwork is not None else {}).get(context.shape[2], None)
    if hypernetwork_layers is not None:
        k_in = self.to_k(hypernetwork_layers[0](context))
        v_in = self.to_v(hypernetwork_layers[1](context))
    else:
        k_in = self.to_k(context)
        v_in = self.to_v(context)
    q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b n h d', h=h), (q_in, k_in, v_in))
    del q_in, k_in, v_in
    out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None)

    out = rearrange(out, 'b n h d -> b n (h d)', h=h)
    return self.to_out(out)

def cross_attention_attnblock_forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q1 = self.q(h_)
        k1 = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q1.shape

        q2 = q1.reshape(b, c, h*w)
        del q1

        q = q2.permute(0, 2, 1)   # b,hw,c
        del q2

        k = k1.reshape(b, c, h*w) # b,c,hw
        del k1

        h_ = torch.zeros_like(k, device=q.device)

        stats = torch.cuda.memory_stats(q.device)
        mem_active = stats['active_bytes.all.current']
        mem_reserved = stats['reserved_bytes.all.current']
        mem_free_cuda, _ = torch.cuda.mem_get_info(torch.cuda.current_device())
        mem_free_torch = mem_reserved - mem_active
        mem_free_total = mem_free_cuda + mem_free_torch

        tensor_size = q.shape[0] * q.shape[1] * k.shape[2] * q.element_size()
        mem_required = tensor_size * 2.5
        steps = 1

        if mem_required > mem_free_total:
            steps = 2**(math.ceil(math.log(mem_required / mem_free_total, 2)))

        slice_size = q.shape[1] // steps if (q.shape[1] % steps) == 0 else q.shape[1]
        for i in range(0, q.shape[1], slice_size):
            end = i + slice_size

            w1 = torch.bmm(q[:, i:end], k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
            w2 = w1 * (int(c)**(-0.5))
            del w1
            w3 = torch.nn.functional.softmax(w2, dim=2, dtype=q.dtype)
            del w2

            # attend to values
            v1 = v.reshape(b, c, h*w)
            w4 = w3.permute(0, 2, 1)   # b,hw,hw (first hw of k, second of q)
            del w3

            h_[:, :, i:end] = torch.bmm(v1, w4)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
            del v1, w4

        h2 = h_.reshape(b, c, h, w)
        del h_

        h3 = self.proj_out(h2)
        del h2

        h3 += x

        return h3
    
def xformers_attnblock_forward(self, x):
    try:
        h_ = x
        h_ = self.norm(h_)
        q1 = self.q(h_).contiguous()
        k1 = self.k(h_).contiguous()
        v = self.v(h_).contiguous()
        out = xformers.ops.memory_efficient_attention(q1, k1, v)
        out = self.proj_out(out)
        return x + out
    except NotImplementedError:
        return cross_attention_attnblock_forward(self, x)
