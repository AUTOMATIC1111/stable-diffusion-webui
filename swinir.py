import sys
import traceback
import cv2
from collections import OrderedDict
import os
import requests
from collections import namedtuple
import numpy as np
from PIL import Image
import torch
import modules.images
from modules.shared import cmd_opts, opts, device
from modules.swinir_arch import SwinIR as net
precision_scope = torch.autocast if cmd_opts.precision == "autocast" else contextlib.nullcontext
def load_model(task = "realsr", large_model = True, model_path=next(os.listdir(cmd_opts.esrgan_models_path))):
    if not large_model:
    # use 'nearest+conv' to avoid block artifacts
        model = net(upscale=scale, in_chans=3, img_size=64, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='nearest+conv', resi_connection='1conv')
    else:
        # larger model size; use '3conv' to save parameters and memory; use ema for GAN training
        model = net(upscale=scale, in_chans=3, img_size=64, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6, 6, 6, 6, 6, 6], embed_dim=240,
                    num_heads=[8, 8, 8, 8, 8, 8, 8, 8, 8],
                    mlp_ratio=2, upsampler='nearest+conv', resi_connection='3conv')
    
    pretrained_model = torch.load(model_path)
    model.load_state_dict(pretrained_model, strict=True)

    return model.half().to(device)
    
def upscale(img, tile=opts.ESRGAN_tile, tile_overlap=opts.ESRGAN_tile_overlap, window_size = 8, scale = 4):
    img = cv2.imread(img, cv2.IMREAD_COLOR).astype(np.float16) / 255.
    model = load_model()
    with torch.no_grad(), precision_scope("cuda"):
        _, _, h_old, w_old = img.size()
        h_pad = (h_old // window_size + 1) * window_size - h_old
        w_pad = (w_old // window_size + 1) * window_size - w_old
        img = torch.cat([img, torch.flip(img, [2])], 2)[:, :, :h_old + h_pad, :]
        img = torch.cat([img, torch.flip(img, [3])], 3)[:, :, :, :w_old + w_pad]
        output = inference(img, model, tile, tile_overlap, window_size, scale)
        output = output[..., :h_old * scale, :w_old * scale]
        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        if output.ndim == 3:
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
        output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8
        return output
    
    
def inference(img, model, tile, tile_overlap, window_size, scale):
    # test the image tile by tile
    b, c, h, w = img.size()
    tile = min(tile, h, w)
    assert tile % window_size == 0, "tile size should be a multiple of window_size"
    sf = scale

    stride = tile - tile_overlap
    h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
    w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
    E = torch.zeros(b, c, h*sf, w*sf, dtype=torch.half, device=device).type_as(img)
    W = torch.zeros_like(E, dtype=torch.half, device=device)

    for h_idx in h_idx_list:
        for w_idx in w_idx_list:
            in_patch = img[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
            out_patch = model(in_patch)
            out_patch_mask = torch.ones_like(out_patch)

            E[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch)
            W[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch_mask)
    output = E.div_(W)

    return output