import math
import sys
import traceback
import importlib

import torch
from torch import einsum

from ldm.util import default
from einops import rearrange

from modules import shared
from modules.hypernetworks import hypernetwork


if shared.cmd_opts.xformers or shared.cmd_opts.force_enable_xformers:
    try:
        import xformers.ops
        shared.xformers_available = True
    except Exception:
        print("Cannot import xformers", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)


# see https://github.com/basujindal/stable-diffusion/pull/117 for discussion
def split_cross_attention_forward_v1(self, x, context=None, mask=None):
    h = self.heads

    q_in = self.to_q(x)
    context = default(context, x)

    context_k, context_v = hypernetwork.apply_hypernetwork(shared.loaded_hypernetwork, context)
    k_in = self.to_k(context_k)
    v_in = self.to_v(context_v)
    del context, context_k, context_v, x

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


# taken from https://github.com/Doggettx/stable-diffusion and modified
def split_cross_attention_forward(self, x, context=None, mask=None):
    h = self.heads

    q_in = self.to_q(x)
    context = default(context, x)

    context_k, context_v = hypernetwork.apply_hypernetwork(shared.loaded_hypernetwork, context)
    k_in = self.to_k(context_k)
    v_in = self.to_v(context_v)

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


def check_for_psutil():
    try:
        spec = importlib.util.find_spec('psutil')
        return spec is not None
    except ModuleNotFoundError:
        return False

invokeAI_mps_available = check_for_psutil()

# -- Taken from https://github.com/invoke-ai/InvokeAI --
if invokeAI_mps_available:
    import psutil
    mem_total_gb = psutil.virtual_memory().total // (1 << 30)

def einsum_op_compvis(q, k, v):
    s = einsum('b i d, b j d -> b i j', q, k)
    s = s.softmax(dim=-1, dtype=s.dtype)
    return einsum('b i j, b j d -> b i d', s, v)

def einsum_op_slice_0(q, k, v, slice_size):
    r = torch.zeros(q.shape[0], q.shape[1], v.shape[2], device=q.device, dtype=q.dtype)
    for i in range(0, q.shape[0], slice_size):
        end = i + slice_size
        r[i:end] = einsum_op_compvis(q[i:end], k[i:end], v[i:end])
    return r

def einsum_op_slice_1(q, k, v, slice_size):
    r = torch.zeros(q.shape[0], q.shape[1], v.shape[2], device=q.device, dtype=q.dtype)
    for i in range(0, q.shape[1], slice_size):
        end = i + slice_size
        r[:, i:end] = einsum_op_compvis(q[:, i:end], k, v)
    return r

def einsum_op_mps_v1(q, k, v):
    if q.shape[1] <= 4096: # (512x512) max q.shape[1]: 4096
        return einsum_op_compvis(q, k, v)
    else:
        slice_size = math.floor(2**30 / (q.shape[0] * q.shape[1]))
        return einsum_op_slice_1(q, k, v, slice_size)

def einsum_op_mps_v2(q, k, v):
    if mem_total_gb > 8 and q.shape[1] <= 4096:
        return einsum_op_compvis(q, k, v)
    else:
        return einsum_op_slice_0(q, k, v, 1)

def einsum_op_tensor_mem(q, k, v, max_tensor_mb):
    size_mb = q.shape[0] * q.shape[1] * k.shape[1] * q.element_size() // (1 << 20)
    if size_mb <= max_tensor_mb:
        return einsum_op_compvis(q, k, v)
    div = 1 << int((size_mb - 1) / max_tensor_mb).bit_length()
    if div <= q.shape[0]:
        return einsum_op_slice_0(q, k, v, q.shape[0] // div)
    return einsum_op_slice_1(q, k, v, max(q.shape[1] // div, 1))

def einsum_op_cuda(q, k, v):
    stats = torch.cuda.memory_stats(q.device)
    mem_active = stats['active_bytes.all.current']
    mem_reserved = stats['reserved_bytes.all.current']
    mem_free_cuda, _ = torch.cuda.mem_get_info(q.device)
    mem_free_torch = mem_reserved - mem_active
    mem_free_total = mem_free_cuda + mem_free_torch
    # Divide factor of safety as there's copying and fragmentation
    return self.einsum_op_tensor_mem(q, k, v, mem_free_total / 3.3 / (1 << 20))

def einsum_op(q, k, v):
    if q.device.type == 'cuda':
        return einsum_op_cuda(q, k, v)

    if q.device.type == 'mps':
        if mem_total_gb >= 32:
            return einsum_op_mps_v1(q, k, v)
        return einsum_op_mps_v2(q, k, v)

    # Smaller slices are faster due to L2/L3/SLC caches.
    # Tested on i7 with 8MB L3 cache.
    return einsum_op_tensor_mem(q, k, v, 32)

def split_cross_attention_forward_invokeAI(self, x, context=None, mask=None):
    h = self.heads

    q = self.to_q(x)
    context = default(context, x)

    context_k, context_v = hypernetwork.apply_hypernetwork(shared.loaded_hypernetwork, context)
    k = self.to_k(context_k) * self.scale
    v = self.to_v(context_v)
    del context, context_k, context_v, x

    q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
    r = einsum_op(q, k, v)
    return self.to_out(rearrange(r, '(b h) n d -> b n (h d)', h=h))

# -- End of code from https://github.com/invoke-ai/InvokeAI --

def xformers_attention_forward(self, x, context=None, mask=None):
    h = self.heads
    q_in = self.to_q(x)
    context = default(context, x)

    context_k, context_v = hypernetwork.apply_hypernetwork(shared.loaded_hypernetwork, context)
    k_in = self.to_k(context_k)
    v_in = self.to_v(context_v)

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
