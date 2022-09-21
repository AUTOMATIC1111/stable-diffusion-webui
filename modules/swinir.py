import sys
import traceback
import cv2
import os
import contextlib
import numpy as np
from PIL import Image
import torch
import modules.images
from modules.shared import cmd_opts, opts, device
from modules.swinir_arch import SwinIR as net

precision_scope = (
    torch.autocast if cmd_opts.precision == "autocast" else contextlib.nullcontext
)


def load_model(filename, scale=4):
    model = net(
        upscale=scale,
        in_chans=3,
        img_size=64,
        window_size=8,
        img_range=1.0,
        depths=[6, 6, 6, 6, 6, 6, 6, 6, 6],
        embed_dim=240,
        num_heads=[8, 8, 8, 8, 8, 8, 8, 8, 8],
        mlp_ratio=2,
        upsampler="nearest+conv",
        resi_connection="3conv",
    )

    pretrained_model = torch.load(filename)
    model.load_state_dict(pretrained_model["params_ema"], strict=True)
    if not cmd_opts.no_half:
        model = model.half()
    return model


def load_models(dirname):
    for file in os.listdir(dirname):
        path = os.path.join(dirname, file)
        model_name, extension = os.path.splitext(file)

        if extension != ".pt" and extension != ".pth":
            continue

        try:
            modules.shared.sd_upscalers.append(UpscalerSwin(path, model_name))
        except Exception:
            print(f"Error loading SwinIR model: {path}", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)


def upscale(
    img,
    model,
    tile=opts.SWIN_tile,
    tile_overlap=opts.SWIN_tile_overlap,
    window_size=8,
    scale=4,
):
    img = np.array(img)
    img = img[:, :, ::-1]
    img = np.moveaxis(img, 2, 0) / 255
    img = torch.from_numpy(img).float()
    img = img.unsqueeze(0).to(device)
    with torch.no_grad(), precision_scope("cuda"):
        _, _, h_old, w_old = img.size()
        h_pad = (h_old // window_size + 1) * window_size - h_old
        w_pad = (w_old // window_size + 1) * window_size - w_old
        img = torch.cat([img, torch.flip(img, [2])], 2)[:, :, : h_old + h_pad, :]
        img = torch.cat([img, torch.flip(img, [3])], 3)[:, :, :, : w_old + w_pad]
        output = inference(img, model, tile, tile_overlap, window_size, scale)
        output = output[..., : h_old * scale, : w_old * scale]
        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        if output.ndim == 3:
            output = np.transpose(
                output[[2, 1, 0], :, :], (1, 2, 0)
            )  # CHW-RGB to HCW-BGR
        output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8
        return Image.fromarray(output, "RGB")


def inference(img, model, tile, tile_overlap, window_size, scale):
    # test the image tile by tile
    b, c, h, w = img.size()
    tile = min(tile, h, w)
    assert tile % window_size == 0, "tile size should be a multiple of window_size"
    sf = scale

    stride = tile - tile_overlap
    h_idx_list = list(range(0, h - tile, stride)) + [h - tile]
    w_idx_list = list(range(0, w - tile, stride)) + [w - tile]
    E = torch.zeros(b, c, h * sf, w * sf, dtype=torch.half, device=device).type_as(img)
    W = torch.zeros_like(E, dtype=torch.half, device=device)

    for h_idx in h_idx_list:
        for w_idx in w_idx_list:
            in_patch = img[..., h_idx : h_idx + tile, w_idx : w_idx + tile]
            out_patch = model(in_patch)
            out_patch_mask = torch.ones_like(out_patch)

            E[
                ..., h_idx * sf : (h_idx + tile) * sf, w_idx * sf : (w_idx + tile) * sf
            ].add_(out_patch)
            W[
                ..., h_idx * sf : (h_idx + tile) * sf, w_idx * sf : (w_idx + tile) * sf
            ].add_(out_patch_mask)
    output = E.div_(W)

    return output


class UpscalerSwin(modules.images.Upscaler):
    def __init__(self, filename, title):
        self.name = title
        self.model = load_model(filename)

    def do_upscale(self, img):
        model = self.model.to(device)
        img = upscale(img, model)
        return img
