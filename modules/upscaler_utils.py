import logging
from typing import Callable

import numpy as np
import torch
import tqdm
from PIL import Image

from modules import devices, images

logger = logging.getLogger(__name__)


def upscale_without_tiling(model, img: Image.Image):
    img = np.array(img)
    img = img[:, :, ::-1]
    img = np.ascontiguousarray(np.transpose(img, (2, 0, 1))) / 255
    img = torch.from_numpy(img).float()

    model_weight = next(iter(model.model.parameters()))
    img = img.unsqueeze(0).to(device=model_weight.device, dtype=model_weight.dtype)

    with torch.no_grad():
        output = model(img)

    output = output.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = 255. * np.moveaxis(output, 0, 2)
    output = output.astype(np.uint8)
    output = output[:, :, ::-1]
    return Image.fromarray(output, 'RGB')


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
        output = upscale_without_tiling(model, img)
        logger.debug("=> %s", output)
        return output

    grid = images.split_grid(img, tile_size, tile_size, tile_overlap)
    newtiles = []

    with tqdm.tqdm(total=grid.tile_count, desc=desc) as p:
        for y, h, row in grid.tiles:
            newrow = []
            for x, w, tile in row:
                logger.debug("Tile (%d, %d) %s...", x, y, tile)
                output = upscale_without_tiling(model, tile)
                scale_factor = output.width // tile.width
                logger.debug("=> %s (scale factor %s)", output, scale_factor)
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
