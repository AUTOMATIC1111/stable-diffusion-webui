import contextlib
import math
import os
import threading
import time
from typing import Optional

import numpy as np
import uvicorn
import yaml
from fastapi import FastAPI
from PIL import Image
from pydantic import BaseModel
from webui import *

from .utils import (
    collect_prompt,
    fix_aspect_ratio,
    get_sampler_index,
    get_upscaler_index,
    load_config,
    save_img,
    set_face_restorer,
)

app = FastAPI()


# TODO:
# - common attributes of Img2Img and Txt2Img can be refactored out
# - seed should be int, is likely str as legacy from being based on hlky originally


class Txt2ImgRequest(BaseModel):
    """Text2Img API request. If optional attributes aren't set, the defaults
    from `krita_config.yaml` will be used.
    """

    orig_width: int
    """Requested image width."""
    orig_height: int
    """Requested image height."""

    prompt: Optional[str]
    """Requested prompt."""
    negative_prompt: Optional[str]
    """Requested negative prompt."""
    sampler_name: Optional[str]
    """Exact name of sampler to use. Name should follow exact spelling and capitalization as in the WebUI."""
    steps: Optional[int]
    """Number of steps for diffusion."""
    cfg_scale: Optional[float]
    """Guidance scale for diffusion."""

    batch_count: Optional[int]
    """Number of batches to render."""
    batch_size: Optional[int]
    """Number of images per batch to render."""
    base_size: Optional[int]
    """Native/base resolution of model used."""
    max_size: Optional[int]
    """Max input resolution allowed to prevent image artifacts."""
    seed: Optional[str]
    """Seed used for noise generation. Incremented by 1 for each image rendered."""
    tiling: Optional[bool]
    """Whether to generate a tileable image."""

    use_gfpgan: Optional[bool]
    """Whether to use GFPGAN for face restoration."""
    face_restorer: Optional[str]
    """Exact name of face restorer to use."""
    codeformer_weight: Optional[float]
    """Strength of face restoration if using CodeFormer. 0.0 is the strongest and 1.0 is the weakest."""


class Img2ImgRequest(BaseModel):
    """Img2Img API request. If optional attributes aren't set, the defaults from
    `krita_config.yaml` will be used.
    """

    mode: Optional[int]
    """Img2Img mode. 0 is normal img2img on the selected region, 1 is inpainting, and 2 (unsupported) is batch processing."""

    src_path: str
    """Path to image being used."""
    mask_path: Optional[str]
    """Path to image mask being used."""

    prompt: Optional[str]
    """Requested prompt."""
    negative_prompt: Optional[str]
    """Requested negative prompt."""
    sampler_name: Optional[str]
    """Exact name of sampler to use. Name should follow exact spelling and capitalization as in the WebUI."""
    steps: Optional[int]
    """Number of steps for diffusion."""
    cfg_scale: Optional[float]
    """Guidance scale for diffusion."""
    denoising_strength: Optional[float]
    """Strength of denoising from 0.0 to 1.0."""

    batch_count: Optional[int]
    """Number of batches to render."""
    batch_size: Optional[int]
    """Number of images per batch to render."""
    base_size: Optional[int]
    """Native/base resolution of model used."""
    max_size: Optional[int]
    """Max input resolution allowed to prevent image artifacts."""
    seed: Optional[str]
    """Seed used for noise generation. Incremented by 1 for each image rendered."""
    tiling: Optional[bool]
    """Whether to generate a tileable image."""

    use_gfpgan: Optional[bool]
    """Whether to use GFPGAN for face restoration."""
    face_restorer: Optional[str]
    """Exact name of face restorer to use."""
    codeformer_weight: Optional[float]
    """Strength of face restoration if using CodeFormer. 0.0 is the strongest and 1.0 is the weakest."""

    upscale_overlap: Optional[int]
    """Size of overlap in pixels for upscaling."""
    upscaler_name: Optional[str]
    """Exact name of upscaler to use."""

    inpainting_fill: Optional[int]
    """What to fill inpainted region with. 0 is blur, 1 is empty, 2 is latent noise, and 3 is latent empty."""
    inpaint_full_res: Optional[bool]
    """Whether to use the full resolution for inpainting."""
    mask_blur: Optional[int]
    """Size of blur at boundaries of mask."""
    invert_mask: Optional[bool]
    """Whether to invert the mask."""


class UpscaleRequest(BaseModel):
    """Upscale API request. If optional attributes aren't set, the defaults from
    `krita_config.yaml` will be used.
    """

    src_path: str
    """Path to image being used."""
    upscaler_name: Optional[str]
    """Exact name of upscaler to use."""
    downscale_first: Optional[bool]
    """Whether to downscale the image by x0.5 first."""


@app.get("/config")
async def read_item():
    """Get information about backend API.

    Returns config from `krita_config.yaml`, the list of available upscalers,
    the path to the rendered image and image mask.

    Returns:
        Dict: information.
    """
    # TODO:
    # - function and route name isn't descriptive, feels more like get_state()
    # - response isn't well typed
    # - ensuring the folders for images exist should be refactored out
    opt = load_config()["plugin"]
    sample_path = opt["sample_path"]
    os.makedirs(sample_path, exist_ok=True)
    filename = f"{int(time.time())}"
    path = os.path.join(sample_path, filename)
    src_path = os.path.abspath(path)
    return {
        "new_img": src_path + ".png",
        "new_img_mask": src_path + "_mask.png",
        "upscalers": [upscaler.name for upscaler in shared.sd_upscalers],
        **opt,
    }


@app.post("/txt2img")
async def f_txt2img(req: Txt2ImgRequest):
    """Post request for Txt2Img.

    Args:
        req (Txt2ImgRequest): Request.

    Returns:
        Dict: Outputs and info.
    """
    print(f"txt2img: {req}")

    opt = load_config()["txt2img"]
    set_face_restorer(
        req.face_restorer or opt["face_restorer"],
        req.codeformer_weight or opt["codeformer_weight"],
    )

    sampler_index = get_sampler_index(req.sampler_name or opt["sampler_name"])

    seed = opt["seed"]
    if req.seed is not None and not req.seed == "":
        seed = int(req.seed)

    width, height = fix_aspect_ratio(
        req.base_size or opt["base_size"],
        req.max_size or opt["max_size"],
        req.orig_width,
        req.orig_height,
    )

    output_images, info, html = modules.txt2img.txt2img(
        req.prompt or collect_prompt(opt, "prompts"),
        req.negative_prompt or collect_prompt(opt, "negative_prompt"),
        "None",
        "None",
        req.steps or opt["steps"],
        sampler_index,
        req.use_gfpgan or opt["use_gfpgan"],
        req.tiling or opt["tiling"],
        req.batch_count or opt["n_iter"],
        req.batch_size or opt["batch_size"],
        req.cfg_scale or opt["cfg_scale"],
        seed,
        None,
        0,
        0,
        0,
        False,
        height,
        width,
        False,
        False,
        0,
        0,
    )

    sample_path = opt["sample_path"]
    os.makedirs(sample_path, exist_ok=True)
    resized_images = [
        modules.images.resize_image(0, image, req.orig_width, req.orig_height)
        for image in output_images
    ]
    outputs = [
        save_img(image, sample_path, filename=f"{int(time.time())}_{i}.png")
        for i, image in enumerate(resized_images)
    ]
    print(f"finished: {outputs}\n{info}")
    return {"outputs": outputs, "info": info}


@app.post("/img2img")
async def f_img2img(req: Img2ImgRequest):
    """Post request for Img2Img.

    Args:
        req (Img2ImgRequest): Request.

    Returns:
        Dict: Outputs and info.
    """
    print(f"img2img: {req}")

    opt = load_config()["img2img"]
    set_face_restorer(
        req.face_restorer or opt["face_restorer"],
        req.codeformer_weight or opt["codeformer_weight"],
    )

    sampler_index = get_sampler_index(req.sampler_name or opt["sampler_name"])

    seed = opt["seed"]
    if req.seed is not None and not req.seed == "":
        seed = int(req.seed)

    mode = req.mode or opt["mode"]

    image = Image.open(req.src_path)
    orig_width, orig_height = image.size

    if mode == 1:
        mask = Image.open(req.mask_path).convert("L")
    else:
        mask = None

    # because API in webui changed
    if mode == 3:
        mode = 2

    upscaler_index = get_upscaler_index(req.upscaler_name or opt["upscaler_name"])

    base_size = req.base_size or opt["base_size"]
    if mode == 2:
        width, height = base_size, base_size
        if upscaler_index > 0:
            image = image.convert("RGB")
    else:
        width, height = fix_aspect_ratio(
            base_size, req.max_size or opt["max_size"], orig_width, orig_height
        )

    output_images, info, html = modules.img2img.img2img(
        0,
        req.prompt or collect_prompt(opt, "prompts"),
        req.negative_prompt or collect_prompt(opt, "negative_prompt"),
        "None",
        "None",
        image,
        {"image": image, "mask": mask},
        image,
        mask,
        mode,
        req.steps or opt["steps"],
        sampler_index,
        req.mask_blur or opt["mask_blur"],
        req.inpainting_fill or opt["inpainting_fill"],
        req.use_gfpgan or opt["use_gfpgan"],
        req.tiling or opt["tiling"],
        req.batch_count or opt["n_iter"],
        req.batch_size or opt["batch_size"],
        req.cfg_scale or opt["cfg_scale"],
        req.denoising_strength or opt["denoising_strength"],
        seed,
        None,
        0,
        0,
        0,
        False,
        height,
        width,
        opt["resize_mode"],
        req.inpaint_full_res or opt["inpaint_full_res"],
        32,
        False,  # req.invert_mask or opt['invert_mask'],
        "",
        "",
        # upscaler_index,
        # req.upscale_overlap or opt['upscale_overlap'],
        0,
    )

    resized_images = [
        modules.images.resize_image(0, image, orig_width, orig_height)
        for image in output_images
    ]

    if mode == 1:

        def remove_not_masked(img):
            masked_img = Image.new("RGBA", img.size, (0, 0, 0, 0))
            masked_img.paste(img, (0, 0), mask=mask)
            return masked_img

        resized_images = [remove_not_masked(x) for x in resized_images]

    sample_path = opt["sample_path"]
    os.makedirs(sample_path, exist_ok=True)
    outputs = [
        save_img(image, sample_path, filename=f"{int(time.time())}_{i}.png")
        for i, image in enumerate(resized_images)
    ]
    print(f"finished: {outputs}\n{info}")
    return {"outputs": outputs, "info": info}


@app.post("/upscale")
async def f_upscale(req: UpscaleRequest):
    """Post request for upscaling.

    Args:
        req (UpscaleRequest): Request.

    Returns:
        Dict: Output.
    """
    print(f"upscale: {req}")

    opt = load_config()["upscale"]
    image = Image.open(req.src_path).convert("RGB")
    orig_width, orig_height = image.size

    upscaler_index = get_upscaler_index(req.upscaler_name or opt["upscaler_name"])
    upscaler = shared.sd_upscalers[upscaler_index]

    if upscaler.name == "None":
        print(f"No upscaler selected, will do nothing")
        return

    if req.downscale_first or opt["downscale_first"]:
        image = modules.images.resize_image(0, image, orig_width // 2, orig_height // 2)

    upscaled_image = upscaler.upscale(image, 2 * orig_width, 2 * orig_height)
    resized_image = modules.images.resize_image(
        0, upscaled_image, orig_width, orig_height
    )

    sample_path = opt["sample_path"]
    os.makedirs(sample_path, exist_ok=True)
    output = save_img(resized_image, sample_path, filename=f"{int(time.time())}.png")
    print(f"finished: {output}")
    return {"output": output}


class Server(uvicorn.Server):
    def install_signal_handlers(self):
        pass

    @contextlib.contextmanager
    def run_in_thread(self):
        thread = threading.Thread(target=self.run)
        thread.start()
        try:
            while not self.started:
                time.sleep(1e-3)
            yield
        finally:
            self.should_exit = True
            thread.join()


def start():
    config = uvicorn.Config(
        "krita_server:app", host="127.0.0.1", port=8000, log_level="info"
    )
    server = Server(config=config)

    with server.run_in_thread():
        webui()


if __name__ == "__main__":
    start()
