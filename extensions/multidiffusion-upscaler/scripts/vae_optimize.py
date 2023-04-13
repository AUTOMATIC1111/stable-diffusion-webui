#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
# ------------------------------------------------------------------------
#
#   Ultimate VAE Tile Optimization
#
#   Introducing a revolutionary new optimization designed to make
#   the VAE work with giant images on limited VRAM!
#   Say goodbye to the frustration of OOM and hello to seamless output!
#
# ------------------------------------------------------------------------
#
#   This script is a wild hack that splits the image into tiles,
#   encodes each tile separately, and merges the result back together.
#
#   Advantages:
#   - The VAE can now work with giant images on limited VRAM
#       (~10 GB for 8K images!)
#   - The merged output is completely seamless without any post-processing.
#
#   Drawbacks:
#   - Giant RAM needed. To store the intermediate results for a 4096x4096
#       images, you need 32 GB RAM it consumes ~20GB); for 8192x8192
#       you need 128 GB RAM machine (it consumes ~100 GB)
#   - NaNs always appear in for 8k images when you use fp16 (half) VAE
#       You must use --no-half-vae to disable half VAE for that giant image.
#   - Slow speed. With default tile size, it takes around 50/200 seconds
#       to encode/decode a 4096x4096 image; and 200/900 seconds to encode/decode
#       a 8192x8192 image. (The speed is limited by both the GPU and the CPU.)
#   - The gradient calculation is not compatible with this hack. It
#       will break any backward() or torch.autograd.grad() that passes VAE.
#       (But you can still use the VAE to generate training data.)
#
#   How it works:
#   1) The image is split into tiles.
#       - To ensure perfect results, each tile is padded with 32 pixels
#           on each side.
#       - Then the conv2d/silu/upsample/downsample can produce identical
#           results to the original image without splitting.
#   2) The original forward is decomposed into a task queue and a task worker.
#       - The task queue is a list of functions that will be executed in order.
#       - The task worker is a loop that executes the tasks in the queue.
#   3) The task queue is executed for each tile.
#       - Current tile is sent to GPU.
#       - local operations are directly executed.
#       - Group norm calculation is temporarily suspended until the mean
#           and var of all tiles are calculated.
#       - The residual is pre-calculated and stored and addded back later.
#       - When need to go to the next tile, the current tile is send to cpu.
#   4) After all tiles are processed, tiles are merged on cpu and return.
#
#   Enjoy!
#
#   @author: LI YI @ Nanyang Technological University - Singapore
#   @date: 2023-03-02
#   @license: MIT License
#
#   Please give me a star if you like this project!
#
# -------------------------------------------------------------------------
'''

import gc
from time import time
import math
from tqdm import tqdm

import torch
import torch.version
import torch.nn.functional as F
from einops import rearrange
import gradio as gr

import modules.scripts as scripts
import modules.devices as devices
from modules.shared import state
from ldm.modules.diffusionmodules.model import AttnBlock, MemoryEfficientAttnBlock

try:
    import xformers
    import xformers.ops
except ImportError:
    pass


def get_recommend_encoder_tile_size():
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(
            devices.device).total_memory // 2**20
        if total_memory > 16*1000:
            ENCODER_TILE_SIZE = 3072
        elif total_memory > 12*1000:
            ENCODER_TILE_SIZE = 2048
        elif total_memory > 8*1000:
            ENCODER_TILE_SIZE = 1536
        else:
            ENCODER_TILE_SIZE = 960
    else:
        ENCODER_TILE_SIZE = 512
    return ENCODER_TILE_SIZE


def get_recommend_decoder_tile_size():
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(
            devices.device).total_memory // 2**20
        if total_memory > 30*1000:
            DECODER_TILE_SIZE = 256
        elif total_memory > 16*1000:
            DECODER_TILE_SIZE = 192
        elif total_memory > 12*1000:
            DECODER_TILE_SIZE = 128
        elif total_memory > 8*1000:
            DECODER_TILE_SIZE = 96
        else:
            DECODER_TILE_SIZE = 64
    else:
        DECODER_TILE_SIZE = 64
    return DECODER_TILE_SIZE


# inplace version of silu
def inplace_nonlinearity(x):
    # Test: fix for Nans
    return F.silu(x, inplace=True)

# extracted from ldm.modules.diffusionmodules.model


def attn_forward(self, h_):
    q = self.q(h_)
    k = self.k(h_)
    v = self.v(h_)

    # compute attention
    b, c, h, w = q.shape
    q = q.reshape(b, c, h*w)
    q = q.permute(0, 2, 1)   # b,hw,c
    k = k.reshape(b, c, h*w)  # b,c,hw
    w_ = torch.bmm(q, k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
    w_ = w_ * (int(c)**(-0.5))
    w_ = torch.nn.functional.softmax(w_, dim=2)

    # attend to values
    v = v.reshape(b, c, h*w)
    w_ = w_.permute(0, 2, 1)   # b,hw,hw (first hw of k, second of q)
    # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
    h_ = torch.bmm(v, w_)
    h_ = h_.reshape(b, c, h, w)

    h_ = self.proj_out(h_)

    return h_


def xformer_attn_forward(self, h_):
    q = self.q(h_)
    k = self.k(h_)
    v = self.v(h_)

    # compute attention
    B, C, H, W = q.shape
    q, k, v = map(lambda x: rearrange(x, 'b c h w -> b (h w) c'), (q, k, v))

    q, k, v = map(
        lambda t: t.unsqueeze(3)
        .reshape(B, t.shape[1], 1, C)
        .permute(0, 2, 1, 3)
        .reshape(B * 1, t.shape[1], C)
        .contiguous(),
        (q, k, v),
    )
    out = xformers.ops.memory_efficient_attention(
        q, k, v, attn_bias=None, op=self.attention_op)

    out = (
        out.unsqueeze(0)
        .reshape(B, 1, out.shape[1], C)
        .permute(0, 2, 1, 3)
        .reshape(B, out.shape[1], C)
    )
    out = rearrange(out, 'b (h w) c -> b c h w', b=B, h=H, w=W, c=C)
    out = self.proj_out(out)
    return out


def attn2task(task_queue, net):
    if isinstance(net, AttnBlock):
        task_queue.append(('store_res', lambda x: x))
        task_queue.append(('pre_norm', net.norm))
        task_queue.append(('attn', lambda x, net=net: attn_forward(net, x)))
        task_queue.append(['add_res', None])
    elif isinstance(net, MemoryEfficientAttnBlock):
        task_queue.append(('store_res', lambda x: x))
        task_queue.append(('pre_norm', net.norm))
        task_queue.append(
            ('attn', lambda x, net=net: xformer_attn_forward(net, x)))
        task_queue.append(['add_res', None])


def resblock2task(queue, block):
    """
    Turn a ResNetBlock into a sequence of tasks and append to the task queue

    @param queue: the target task queue
    @param block: ResNetBlock

    """
    if block.in_channels != block.out_channels:
        if block.use_conv_shortcut:
            queue.append(('store_res', block.conv_shortcut))
        else:
            queue.append(('store_res', block.nin_shortcut))
    else:
        queue.append(('store_res', lambda x: x))
    queue.append(('pre_norm', block.norm1))
    queue.append(('silu', inplace_nonlinearity))
    queue.append(('conv1', block.conv1))
    queue.append(('pre_norm', block.norm2))
    queue.append(('silu', inplace_nonlinearity))
    queue.append(('conv2', block.conv2))
    queue.append(['add_res', None])


def build_sampling(task_queue, net, is_decoder):
    """
    Build the sampling part of a task queue
    @param task_queue: the target task queue
    @param net: the network
    @param is_decoder: currently building decoder or encoder
    """
    if is_decoder:
        resblock2task(task_queue, net.mid.block_1)
        attn2task(task_queue, net.mid.attn_1)
        resblock2task(task_queue, net.mid.block_2)
        resolution_iter = reversed(range(net.num_resolutions))
        block_ids = net.num_res_blocks + 1
        condition = 0
        module = net.up
        func_name = 'upsample'
    else:
        resolution_iter = range(net.num_resolutions)
        block_ids = net.num_res_blocks
        condition = net.num_resolutions - 1
        module = net.down
        func_name = 'downsample'

    for i_level in resolution_iter:
        for i_block in range(block_ids):
            resblock2task(task_queue, module[i_level].block[i_block])
        if i_level != condition:
            task_queue.append((func_name, getattr(module[i_level], func_name)))

    if not is_decoder:
        resblock2task(task_queue, net.mid.block_1)
        attn2task(task_queue, net.mid.attn_1)
        resblock2task(task_queue, net.mid.block_2)


def build_task_queue(net, is_decoder):
    """
    Build a single task queue for the encoder or decoder
    @param net: the VAE decoder or encoder network
    @param is_decoder: currently building decoder or encoder
    @return: the task queue
    """
    task_queue = []
    task_queue.append(('conv_in', net.conv_in))

    # construct the sampling part of the task queue
    # because encoder and decoder share the same architecture, we extract the sampling part
    build_sampling(task_queue, net, is_decoder)

    if not is_decoder or not net.give_pre_end:
        task_queue.append(('pre_norm', net.norm_out))
        task_queue.append(('silu', inplace_nonlinearity))
        task_queue.append(('conv_out', net.conv_out))
        if is_decoder and net.tanh_out:
            task_queue.append(('tanh', torch.tanh))

    return task_queue


def clone_task_queue(task_queue):
    """
    Clone a task queue
    @param task_queue: the task queue to be cloned
    @return: the cloned task queue
    """
    return [[item for item in task] for task in task_queue]


def get_var_mean(input, num_groups, eps=1e-6):
    """
    Get mean and var for group norm
    """
    b, c = input.size(0), input.size(1)
    channel_in_group = int(c/num_groups)
    input_reshaped = input.contiguous().view(
        1, int(b * num_groups), channel_in_group, *input.size()[2:])
    var, mean = torch.var_mean(
        input_reshaped, dim=[0, 2, 3, 4], unbiased=False)
    return var, mean


def custom_group_norm(input, num_groups, mean, var, weight=None, bias=None, eps=1e-6):
    """
    Custom group norm with fixed mean and var

    @param input: input tensor
    @param num_groups: number of groups. by default, num_groups = 32
    @param mean: mean, must be pre-calculated by get_var_mean
    @param var: var, must be pre-calculated by get_var_mean
    @param weight: weight, should be fetched from the original group norm
    @param bias: bias, should be fetched from the original group norm
    @param eps: epsilon, by default, eps = 1e-6 to match the original group norm

    @return: normalized tensor
    """
    b, c = input.size(0), input.size(1)
    channel_in_group = int(c/num_groups)
    input_reshaped = input.contiguous().view(
        1, int(b * num_groups), channel_in_group, *input.size()[2:])

    out = F.batch_norm(input_reshaped, mean, var, weight=None, bias=None,
                       training=False, momentum=0, eps=eps)

    out = out.view(b, c, *input.size()[2:])

    # post affine transform
    if weight is not None:
        out *= weight.view(1, -1, 1, 1)
    if bias is not None:
        out += bias.view(1, -1, 1, 1)
    return out


def crop_valid_region(x, input_bbox, target_bbox, is_decoder):
    """
    Crop the valid region from the tile
    @param x: input tile
    @param input_bbox: original input bounding box
    @param target_bbox: output bounding box
    @param scale: scale factor
    @return: cropped tile
    """
    padded_bbox = [i * 8 if is_decoder else i//8 for i in input_bbox]
    margin = [target_bbox[i] - padded_bbox[i] for i in range(4)]
    return x[:, :, margin[2]:x.size(2)+margin[3], margin[0]:x.size(3)+margin[1]]

# ↓↓↓ https://github.com/Kahsolt/stable-diffusion-webui-vae-tile-infer ↓↓↓


def perfcount(fn):
    def wrapper(*args, **kwargs):
        ts = time()

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(devices.device)
        devices.torch_gc()
        gc.collect()

        ret = fn(*args, **kwargs)

        devices.torch_gc()
        gc.collect()
        if torch.cuda.is_available():
            vram = torch.cuda.max_memory_allocated(devices.device) / 2**20
            torch.cuda.reset_peak_memory_stats(devices.device)
            print(
                f'[Tiled VAE]: Done in {time() - ts:.3f}s, max VRAM alloc {vram:.3f} MB')
        else:
            print(f'[Tiled VAE]: Done in {time() - ts:.3f}s')

        return ret
    return wrapper

# copy end :)


class GroupNormParam:
    def __init__(self):
        self.var_list = []
        self.mean_list = []
        self.pixel_list = []
        self.weight = None
        self.bias = None

    def add_tile(self, tile, layer):
        var, mean = get_var_mean(tile, 32)
        # For giant images, the variance can be larger than max float16
        # In this case we create a copy to float32
        if var.dtype == torch.float16 and var.isinf().any():
            fp32_tile = tile.float()
            var, mean = get_var_mean(fp32_tile, 32)
        # ============= DEBUG: test for infinite =============
        # if torch.isinf(var).any():
        #    print('var: ', var)
        # ====================================================
        self.var_list.append(var)
        self.mean_list.append(mean)
        self.pixel_list.append(
            tile.shape[2]*tile.shape[3])
        if hasattr(layer, 'weight'):
            self.weight = layer.weight
            self.bias = layer.bias
        else:
            self.weight = None
            self.bias = None

    def summary(self):
        """
        summarize the mean and var and return a function
        that apply group norm on each tile
        """
        if len(self.var_list) == 0:
            return None
        var = torch.vstack(self.var_list)
        mean = torch.vstack(self.mean_list)
        max_value = max(self.pixel_list)
        pixels = torch.tensor(
            self.pixel_list, dtype=torch.float32, device=devices.device) / max_value
        sum_pixels = torch.sum(pixels)
        pixels = pixels.unsqueeze(
            1) / sum_pixels
        var = torch.sum(
            var * pixels, dim=0)
        mean = torch.sum(
            mean * pixels, dim=0)
        return lambda x:  custom_group_norm(x, 32, mean, var, self.weight, self.bias)

    @staticmethod
    def from_tile(tile, norm):
        """
        create a function from a single tile without summary
        """
        var, mean = get_var_mean(tile, 32)
        if var.dtype == torch.float16 and var.isinf().any():
            fp32_tile = tile.float()
            var, mean = get_var_mean(fp32_tile, 32)
            # if it is a macbook, we need to convert back to float16
            if var.device.type == 'mps':
                # clamp to avoid overflow
                var = torch.clamp(var, 0, 60000)
                var = var.half()
                mean = mean.half()
        if hasattr(norm, 'weight'):
            weight = norm.weight
            bias = norm.bias
        else:
            weight = None
            bias = None

        def group_norm_func(x, mean=mean, var=var, weight=weight, bias=bias):
            return custom_group_norm(x, 32, mean, var, weight, bias, 1e-6)
        return group_norm_func


class VAEHook:

    def __init__(self, net, tile_size, is_decoder, fast_decoder, fast_encoder, color_fix, to_gpu=False):
        self.net = net                  # encoder | decoder
        self.tile_size = tile_size
        self.is_decoder = is_decoder
        self.fast_mode = (fast_encoder and not is_decoder) or (
            fast_decoder and is_decoder)
        self.color_fix = color_fix and not is_decoder
        self.to_gpu = to_gpu
        self.pad = 11 if is_decoder else 32

    def __call__(self, x):
        B, C, H, W = x.shape
        original_device = next(self.net.parameters()).device
        try:
            if self.to_gpu:
                self.net.to(devices.get_optimal_device())
            if max(H, W) <= self.pad * 2 + self.tile_size:
                print("[Tiled VAE]: the input size is tiny and unnecessary to tile.")
                return self.net.original_forward(x)
            else:
                return self.vae_tile_forward(x)
        finally:
            self.net.to(original_device)

    def get_best_tile_size(self, lowerbound, upperbound):
        """
        Get the best tile size for GPU memory
        """
        divider = 32
        while divider >= 2:
            remainer = lowerbound % divider
            if remainer == 0:
                return lowerbound
            candidate = lowerbound - remainer + divider
            if candidate <= upperbound:
                return candidate
            divider //= 2
        return lowerbound

    def split_tiles(self, h, w):
        """
        Tool function to split the image into tiles
        @param h: height of the image
        @param w: width of the image
        @return: tile_input_bboxes, tile_output_bboxes
        """
        tile_input_bboxes, tile_output_bboxes = [], []
        tile_size = self.tile_size
        pad = self.pad
        num_height_tiles = math.ceil((h - 2 * pad) / tile_size)
        num_width_tiles = math.ceil((w - 2 * pad) / tile_size)
        # If any of the numbers are 0, we let it be 1
        # This is to deal with long and thin images
        num_height_tiles = max(num_height_tiles, 1)
        num_width_tiles = max(num_width_tiles, 1)

        # Suggestions from https://github.com/Kahsolt: auto shrink the tile size
        real_tile_height = math.ceil((h - 2 * pad) / num_height_tiles)
        real_tile_width = math.ceil((w - 2 * pad) / num_width_tiles)
        real_tile_height = self.get_best_tile_size(real_tile_height, tile_size)
        real_tile_width = self.get_best_tile_size(real_tile_width, tile_size)

        print(f'[Tiled VAE]: split to {num_height_tiles}x{num_width_tiles} = {num_height_tiles*num_width_tiles} tiles. ' +
              f'Optimal tile size {real_tile_width}x{real_tile_height}, original tile size {tile_size}x{tile_size}')

        for i in range(num_height_tiles):
            for j in range(num_width_tiles):
                # bbox: [x1, x2, y1, y2]
                # the padding is is unnessary for image borders. So we directly start from (32, 32)
                input_bbox = [
                    pad + j * real_tile_width,
                    min(pad + (j + 1) * real_tile_width, w),
                    pad + i * real_tile_height,
                    min(pad + (i + 1) * real_tile_height, h),
                ]

                # if the output bbox is close to the image boundary, we extend it to the image boundary
                output_bbox = [
                    input_bbox[0] if input_bbox[0] > pad else 0,
                    input_bbox[1] if input_bbox[1] < w - pad else w,
                    input_bbox[2] if input_bbox[2] > pad else 0,
                    input_bbox[3] if input_bbox[3] < h - pad else h,
                ]

                # scale to get the final output bbox
                output_bbox = [x * 8 if self.is_decoder else x // 8 for x in output_bbox]
                tile_output_bboxes.append(output_bbox)

                # indistinguishable expand the input bbox by pad pixels
                tile_input_bboxes.append([
                    max(0, input_bbox[0] - pad),
                    min(w, input_bbox[1] + pad),
                    max(0, input_bbox[2] - pad),
                    min(h, input_bbox[3] + pad),
                ])

        return tile_input_bboxes, tile_output_bboxes

    @torch.no_grad()
    def estimate_group_norm(self, z, task_queue, color_fix):
        device = z.device
        tile = z
        last_id = len(task_queue) - 1
        while last_id >= 0 and task_queue[last_id][0] != 'pre_norm':
            last_id -= 1
        if last_id <= 0 or task_queue[last_id][0] != 'pre_norm':
            raise ValueError('No group norm found in the task queue')
        # estimate until the last group norm
        for i in range(last_id + 1):
            task = task_queue[i]
            if task[0] == 'pre_norm':
                group_norm_func = GroupNormParam.from_tile(tile, task[1])
                task_queue[i] = ('apply_norm', group_norm_func)
                if i == last_id:
                    return True
                tile = group_norm_func(tile)
            elif task[0] == 'store_res':
                task_id = i + 1
                while task_id < last_id and task_queue[task_id][0] != 'add_res':
                    task_id += 1
                if task_id >= last_id:
                    continue
                task_queue[task_id][1] = task[1](tile)
            elif task[0] == 'add_res':
                tile += task[1].to(device)
                task[1] = None
            elif color_fix and task[0] == 'downsample':
                for j in range(i, last_id + 1):
                    if task_queue[j][0] == 'store_res':
                        task_queue[j] = ('store_res_cpu', task_queue[j][1])
                return True
            else:
                tile = task[1](tile)
            try:
                devices.test_for_nans(tile, "vae")
            except:
                print(f'Nan detected in fast mode estimation. Fast mode disabled.')
                return False

        raise IndexError('Should not reach here')

    @perfcount
    @torch.no_grad()
    def vae_tile_forward(self, z):
        """
        Decode a latent vector z into an image in a tiled manner.
        @param z: latent vector
        @return: image
        """
        device = next(self.net.parameters()).device
        net = self.net
        tile_size = self.tile_size
        is_decoder = self.is_decoder

        z = z.detach() # detach the input to avoid backprop

        N, height, width = z.shape[0], z.shape[2], z.shape[3]
        net.last_z_shape = z.shape

        # Split the input into tiles and build a task queue for each tile
        print(
            f'[Tiled VAE]: input_size: {z.shape}, tile_size: {tile_size}, padding: {self.pad}')

        in_bboxes, out_bboxes = self.split_tiles(height, width)

        # Prepare tiles by split the input latents
        tiles = []
        for input_bbox in in_bboxes:
            tile = z[:, :, input_bbox[2]:input_bbox[3],
                     input_bbox[0]:input_bbox[1]].cpu()
            tiles.append(tile)

        num_tiles = len(tiles)
        num_completed = 0

        single_task_queue = build_task_queue(net, is_decoder)

        if self.fast_mode:
            # Fast mode: downsample the input image to the tile size,
            # then estimate the group norm parameters on the downsampled image
            scale_factor = tile_size / max(height, width)
            z = z.to(device)
            downsampled_z = F.interpolate(
                z, scale_factor=scale_factor, mode='nearest-exact')
            # use nearest-exact to keep statictics as close as possible
            print(
                f'[Tiled VAE]: Fast mode enabled, estimating group norm parameters on {downsampled_z.shape[3]} x {downsampled_z.shape[2]} image')

            # ======= Special thanks to @Kahsolt for distribution shift issue ======= #
            # The downsampling will heavily distort its mean and std, so we need to recover it.
            std_old, mean_old = torch.std_mean(z, dim=[0, 2, 3], keepdim=True)
            std_new, mean_new = torch.std_mean(
                downsampled_z, dim=[0, 2, 3], keepdim=True)
            downsampled_z = (downsampled_z - mean_new) / \
                std_new * std_old + mean_old
            del std_old, mean_old, std_new, mean_new
            # occasionally the std_new is too small or too large, which exceeds the range of float16
            # so we need to clamp it to max z's range.
            downsampled_z = torch.clamp_(
                downsampled_z, min=z.min(), max=z.max())
            estimate_task_queue = clone_task_queue(single_task_queue)
            if self.estimate_group_norm(downsampled_z, estimate_task_queue, color_fix=self.color_fix):
                single_task_queue = estimate_task_queue
            del downsampled_z

        task_queues = [clone_task_queue(single_task_queue)
                       for _ in range(num_tiles)]
        # Free memory of input latent tensor
        del z
        result = None

        # Build task queues

        # Task queue execution
        desc = f"[Tiled VAE]: Executing {'Decoder' if is_decoder else 'Encoder'} Task Queue: "
        pbar = tqdm(total=num_tiles * len(task_queues[0]), desc=desc)

        # execute the task back and forth when switch tiles so that we always
        # keep one tile on the GPU to reduce unnecessary data transfer
        forward = True
        while True:
            group_norm_param = GroupNormParam()
            for i in range(num_tiles) if forward else reversed(range(num_tiles)):
                if state.interrupted:
                    return

                tile = tiles[i].to(device)
                input_bbox = in_bboxes[i]
                task_queue = task_queues[i]

                while len(task_queue) > 0:
                    if state.interrupted:
                        return
                    # DEBUG: current task
                    # print('Running task: ', task_queue[0][0], ' on tile ', i, '/', num_tiles, ' with shape ', tile.shape)
                    task = task_queue.pop(0)
                    if task[0] == 'pre_norm':
                        group_norm_param.add_tile(tile, task[1])
                        break
                    elif task[0] == 'store_res' or task[0] == 'store_res_cpu':
                        task_id = 0
                        res = task[1](tile)
                        if not self.fast_mode or task[0] == 'store_res_cpu':
                            res = res.cpu()
                        while task_queue[task_id][0] != 'add_res':
                            task_id += 1
                        task_queue[task_id][1] = res
                    elif task[0] == 'add_res':
                        tile += task[1].to(device)
                        task[1] = None
                    else:
                        tile = task[1](tile)
                    pbar.update(1)

                # check for NaNs in the tile.
                # If there are NaNs, we abort the process to save user's time
                devices.test_for_nans(tile, "vae")

                if len(task_queue) == 0:
                    tiles[i] = None
                    num_completed += 1
                    if result is None:
                        result = torch.zeros((N, tile.shape[1], height * 8 if is_decoder else height // 8, width * 8 if is_decoder else width // 8), device=device, requires_grad=False)
                    result[:, :, out_bboxes[i][2]:out_bboxes[i][3], out_bboxes[i][0]:out_bboxes[i][1]] = crop_valid_region(tile, in_bboxes[i], out_bboxes[i], is_decoder)
                    del tile
                elif i == num_tiles - 1 and forward:
                    forward = False
                    tiles[i] = tile
                elif i == 0 and not forward:
                    forward = True
                    tiles[i] = tile
                else:
                    tiles[i] = tile.cpu()
                    del tile

            if num_completed == num_tiles:
                break

            # insert the group norm task to the head of each task queue
            group_norm_func = group_norm_param.summary()
            if group_norm_func is not None:
                for i in range(num_tiles):
                    task_queue = task_queues[i]
                    task_queue.insert(0, ('apply_norm', group_norm_func))

        # Done!
        pbar.close()
        return result


class Script(scripts.Script):

    def title(self):
        return "Tiled VAE"

    def show(self, is_img2img):
        if devices.get_optimal_device_name() == 'mps':
            print(f'[Tiled VAE]: Tiled VAE is not needed on Mac. Skip loading...')
            return False
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion('Tiled VAE', open=False):
            with gr.Row():
                enabled = gr.Checkbox(label='Enable', value=False)
                vae_to_gpu = gr.Checkbox(label='Move VAE to GPU', value=False)

            encoder_size_tips = gr.HTML(
                '<p style="margin-bottom:0.8em">Please use smaller tile size when see CUDA error: out of memory.</p>')
            with gr.Row():
                encoder_tile_size = gr.Slider(
                    label='Encoder Tile Size', minimum=256, maximum=4096, step=16, value=get_recommend_encoder_tile_size())
                decoder_tile_size = gr.Slider(
                    label='Decoder Tile Size', minimum=48,  maximum=512,  step=16, value=get_recommend_decoder_tile_size())
                reset = gr.Button(value='↻ Reset', variant='tool')
                reset.click(fn=lambda: [get_recommend_encoder_tile_size(), get_recommend_decoder_tile_size()], outputs=[
                    encoder_tile_size, decoder_tile_size])

            with gr.Row():
                fast_encoder = gr.Checkbox(
                    label='Fast Encoder', value=True)
                fast_decoder = gr.Checkbox(
                    label='Fast Decoder', value=True)

            with gr.Row():
                fast_encoder_tips = gr.HTML(
                    '<p style="margin-bottom:0.8em">Fast Encoder may change colors; Can fix it with more RAM and lower speed.</p>')
                color_fix = gr.Checkbox(
                    label='Encoder Color Fix', value=False)

            def on_fast_encoder(value):
                if value:
                    return gr.update(visible=True, interactive=True), gr.update(visible=True)
                else:
                    return gr.update(visible=False, interactive=False), gr.update(visible=False)

            fast_encoder.change(fn=on_fast_encoder, inputs=[fast_encoder], outputs=[
                                color_fix, fast_encoder_tips])

        return [enabled, vae_to_gpu, fast_decoder, fast_encoder, color_fix, encoder_tile_size, decoder_tile_size]

    def process(self, p, enabled, vae_to_gpu, fast_decoder, fast_encoder, color_fix, encoder_tile_size, decoder_tile_size):

        vae = p.sd_model.first_stage_model
        # for shorthand
        encoder = vae.encoder
        decoder = vae.decoder

        # save original forward (only once)
        if not hasattr(encoder, 'original_forward'):
            setattr(encoder, 'original_forward', encoder.forward)
        if not hasattr(decoder, 'original_forward'):
            setattr(decoder, 'original_forward', decoder.forward)

        # undo hijack if disabled
        if not enabled:
            if isinstance(encoder.forward, VAEHook): encoder.forward = encoder.original_forward
            if isinstance(decoder.forward, VAEHook): decoder.forward = decoder.original_forward
            return

        if devices.get_optimal_device == torch.device('cpu'):
            print("[Tiled VAE] Tiled VAE is not needed as your device has no GPU VRAM.")
            return
        if vae.device == torch.device('cpu') and not vae_to_gpu:
            print(
                "[Tiled VAE] VAE is on CPU. Please enable 'Move VAE to GPU' to use Tiled VAE.")
            return

        # do hijack
        encoder.forward = VAEHook(
            encoder, encoder_tile_size, is_decoder=False, fast_decoder=fast_decoder, fast_encoder=fast_encoder, color_fix=color_fix, to_gpu=vae_to_gpu)
        decoder.forward = VAEHook(
            decoder, decoder_tile_size, is_decoder=True, fast_decoder=fast_decoder, fast_encoder=fast_encoder, color_fix=color_fix, to_gpu=vae_to_gpu)

    def postprocess(self, p, processed, *args):
        # release vram for the next run
        vae = p.sd_model.first_stage_model
        encoder = vae.encoder
        decoder = vae.decoder
        if isinstance(encoder.forward, VAEHook):
            encoder.forward.net = None
            encoder.forward = encoder.original_forward
        if isinstance(decoder.forward, VAEHook):
            decoder.forward.net = None
            decoder.forward = decoder.original_forward
        gc.collect()
        devices.torch_gc()
