import logging
from typing import Callable

import numpy as np
import torch
import tqdm
from PIL import Image

from modules import devices, images, shared, torch_utils

logger = logging.getLogger(__name__)


def pil_image_to_torch_bgr(img: Image.Image) -> torch.Tensor:
    img = np.array(img.convert("RGB"))
    img = img[:, :, ::-1]  # flip RGB to BGR
    img = np.transpose(img, (2, 0, 1))  # HWC to CHW
    img = np.ascontiguousarray(img) / 255  # Rescale to [0, 1]
    return torch.from_numpy(img)


def torch_bgr_to_pil_image(tensor: torch.Tensor) -> Image.Image:
    if tensor.ndim == 4:
        # If we're given a tensor with a batch dimension, squeeze it out
        # (but only if it's a batch of size 1).
        if tensor.shape[0] != 1:
            raise ValueError(f"{tensor.shape} does not describe a BCHW tensor")
        tensor = tensor.squeeze(0)
    assert tensor.ndim == 3, f"{tensor.shape} does not describe a CHW tensor"
    # TODO: is `tensor.float().cpu()...numpy()` the most efficient idiom?
    arr = tensor.float().cpu().clamp_(0, 1).numpy()  # clamp
    arr = 255.0 * np.moveaxis(arr, 0, 2)  # CHW to HWC, rescale
    arr = arr.round().astype(np.uint8)
    arr = arr[:, :, ::-1]  # flip BGR to RGB
    return Image.fromarray(arr, "RGB")


def upscale_pil_patch(model, img: Image.Image) -> Image.Image:
    """
    Upscale a given PIL image using the given model.
    """
    param = torch_utils.get_param(model)

    with torch.inference_mode():
        tensor = pil_image_to_torch_bgr(img).unsqueeze(0)  # add batch dimension
        tensor = tensor.to(device=param.device, dtype=param.dtype)
        with devices.without_autocast():
            return torch_bgr_to_pil_image(model(tensor))


def upscale_with_model(
    model: Callable[[torch.Tensor], torch.Tensor],
    img: Image.Image,
    *,
    tile_size: int,
    tile_overlap: int = 0,
    desc="tiled upscale",
) -> Image.Image:
    if tile_size <= 0:
        logger.debug("Upscaling %s without tiling", img)
        output = upscale_pil_patch(model, img)
        logger.debug("=> %s", output)
        return output

    grid = images.split_grid(img, tile_size, tile_size, tile_overlap)
    newtiles = []

    with tqdm.tqdm(total=grid.tile_count, desc=desc, disable=not shared.opts.enable_upscale_progressbar) as p:
        for y, h, row in grid.tiles:
            newrow = []
            for x, w, tile in row:
                if shared.state.interrupted:
                    return img
                output = upscale_pil_patch(model, tile)
                scale_factor = output.width // tile.width
                newrow.append([x * scale_factor, w * scale_factor, output])
                p.update(1)
            newtiles.append([y * scale_factor, h * scale_factor, newrow])

    newgrid = images.Grid(
        newtiles,
        tile_w=grid.tile_w * scale_factor,
        tile_h=grid.tile_h * scale_factor,
        image_w=grid.image_w * scale_factor,
        image_h=grid.image_h * scale_factor,
        overlap=grid.overlap * scale_factor,
    )
    return images.combine_grid(newgrid)


def tiled_upscale_2(
    img: torch.Tensor,
    model,
    *,
    tile_size: int,
    tile_overlap: int,
    scale: int,
    device: torch.device,
    desc="Tiled upscale",
):
    # Alternative implementation of `upscale_with_model` originally used by
    # SwinIR and ScuNET.  It differs from `upscale_with_model` in that tiling and
    # weighting is done in PyTorch space, as opposed to `images.Grid` doing it in
    # Pillow space without weighting.

    b, c, h, w = img.size()
    tile_size = min(tile_size, h, w)

    if tile_size <= 0:
        logger.debug("Upscaling %s without tiling", img.shape)
        return model(img)

    stride = tile_size - tile_overlap
    h_idx_list = list(range(0, h - tile_size, stride)) + [h - tile_size]
    w_idx_list = list(range(0, w - tile_size, stride)) + [w - tile_size]
    result = torch.zeros(
        b,
        c,
        h * scale,
        w * scale,
        device=device,
        dtype=img.dtype,
    )
    weights = torch.zeros_like(result)
    logger.debug("Upscaling %s to %s with tiles", img.shape, result.shape)
    with tqdm.tqdm(total=len(h_idx_list) * len(w_idx_list), desc=desc, disable=not shared.opts.enable_upscale_progressbar) as pbar:
        for h_idx in h_idx_list:
            if shared.state.interrupted or shared.state.skipped:
                break

            for w_idx in w_idx_list:
                if shared.state.interrupted or shared.state.skipped:
                    break

                # Only move this patch to the device if it's not already there.
                in_patch = img[
                    ...,
                    h_idx : h_idx + tile_size,
                    w_idx : w_idx + tile_size,
                ].to(device=device)

                out_patch = model(in_patch)

                result[
                    ...,
                    h_idx * scale : (h_idx + tile_size) * scale,
                    w_idx * scale : (w_idx + tile_size) * scale,
                ].add_(out_patch)

                out_patch_mask = torch.ones_like(out_patch)

                weights[
                    ...,
                    h_idx * scale : (h_idx + tile_size) * scale,
                    w_idx * scale : (w_idx + tile_size) * scale,
                ].add_(out_patch_mask)

                pbar.update(1)

    output = result.div_(weights)

    return output


def upscale_2(
    img: Image.Image,
    model,
    *,
    tile_size: int,
    tile_overlap: int,
    scale: int,
    desc: str,
):
    """
    Convenience wrapper around `tiled_upscale_2` that handles PIL images.
    """
    param = torch_utils.get_param(model)
    tensor = pil_image_to_torch_bgr(img).to(dtype=param.dtype).unsqueeze(0)  # add batch dimension

    with torch.no_grad():
        output = tiled_upscale_2(
            tensor,
            model,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            scale=scale,
            desc=desc,
            device=param.device,
        )
    return torch_bgr_to_pil_image(output)
