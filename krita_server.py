import math
import time
import yaml
import os
from typing import Optional

import numpy as np
from pydantic import BaseModel
from fastapi import FastAPI
import uvicorn

from webui import *

from PIL import Image

app = FastAPI()


def load_config():
    with open("../../krita_config.yaml") as file:
        return yaml.safe_load(file)


def save_img(image, sample_path, filename):
    path = os.path.join(sample_path, filename)
    image.save(path)
    return os.path.abspath(path)


def fix_aspect_ratio(base_size, max_size, orig_width, orig_height):
    def rnd(r, x):
        z = 64
        return z * round(r * x / z)

    ratio = orig_width / orig_height

    if orig_width > orig_height:
        width, height = rnd(ratio, base_size), base_size
        if width > max_size:
            width, height = max_size, rnd(1 / ratio, max_size)
    else:
        width, height = base_size, rnd(1 / ratio, base_size)
        if height > max_size:
            width, height = rnd(ratio, max_size), max_size

    new_ratio = width / height

    print(f"img size: {orig_width}x{orig_height} -> {width}x{height}, "
          f"aspect ratio: {ratio:.2f} -> {new_ratio:.2f}, {100 * (new_ratio - ratio) / ratio :.2f}% change")
    return width, height


def collect_prompt(opt):
    prompts = opt['prompts']
    if isinstance(prompts, str):
        return prompts
    if isinstance(prompts, list):
        return ", ".join(prompts)
    if isinstance(prompts, dict):
        prompt = ""
        for item, weight in prompts.items():
            if not prompt == "":
                prompt += " "
            if weight is None:
                prompt += f"{item}"
            else:
                prompt += f"{item}:{weight}"
        return prompt
    raise Exception("wtf man, fix your prompts")


class Txt2ImgRequest(BaseModel):
    orig_width: int
    orig_height: int

    prompt: Optional[str]
    negative_prompt: Optional[str]
    sampler_name: Optional[str]
    steps: Optional[int]
    cfg_scale: Optional[float]

    batch_count: Optional[int]
    batch_size: Optional[int]
    base_size: Optional[int]
    max_size: Optional[int]
    seed: Optional[str]


class Img2ImgRequest(BaseModel):
    mode: Optional[int]

    src_path: str
    mask_path: Optional[str]

    prompt: Optional[str]
    sampler_name: Optional[str]
    steps: Optional[int]
    cfg_scale: Optional[float]
    denoising_strength: Optional[float]

    batch_count: Optional[int]
    batch_size: Optional[int]
    base_size: Optional[int]
    max_size: Optional[int]
    seed: Optional[str]

    normalize_prompt_weights: Optional[bool]
    use_gfpgan: Optional[bool]
    use_realesrgan: Optional[bool]
    realesrgan_model: Optional[str]

    upscale_overlap: Optional[int]

    inpainting_fill: Optional[int]
    inpaint_full_res: Optional[bool]
    mask_blur: Optional[int]


def get_sampler_index(sampler_name):
    for i, sampler in enumerate(modules.sd_samplers.samplers):
        name, constructor, aliases = sampler
        if sampler_name in aliases:
            return i


@app.get("/config")
async def read_item():
    opt = load_config()['plugin']
    sample_path = opt['sample_path']
    os.makedirs(sample_path, exist_ok=True)
    filename = f"{int(time.time())}"
    path = os.path.join(sample_path, filename)
    src_path = os.path.abspath(path)
    print(f"src path: {src_path}")
    return {"new_img": src_path + ".png", "new_img_mask": src_path + "_mask.png", **opt}


@app.post("/txt2img")
async def f_txt2img(req: Txt2ImgRequest):
    print(f"txt2img: {req}")

    opt = load_config()['txt2img']

    sampler_index = get_sampler_index(req.sampler_name or opt['sampler_name'])

    seed = opt['seed']
    if req.seed is not None and not req.seed == '':
        seed = int(req.seed)

    width, height = fix_aspect_ratio(req.base_size or opt['base_size'], req.max_size or opt['max_size'],
                                     req.orig_width, req.orig_height)

    output_images, info, html = modules.txt2img.txt2img(
        req.prompt or collect_prompt(opt),
        req.negative_prompt or opt['negative_prompt'],
        req.steps or opt['steps'],
        sampler_index,
        opt['use_gfpgan'],
        req.batch_count or opt['n_iter'],
        req.batch_size or opt['batch_size'],
        req.cfg_scale or opt['cfg_scale'],
        seed,
        height,
        width,
        0
    )

    sample_path = opt['sample_path']
    os.makedirs(sample_path, exist_ok=True)
    resized_images = [images.resize_image(0, image, req.orig_width, req.orig_height) for image in output_images]
    outputs = [save_img(image, sample_path, filename=f"{int(time.time())}_{i}.png")
               for i, image in enumerate(resized_images)]
    print(f"finished: {outputs}\n{info}")
    return {"outputs": outputs, "info": info}


@app.post("/img2img")
async def f_img2img(req: Img2ImgRequest):
    print(f"img2img: {req}")

    opt = load_config()['img2img']

    sampler_index = get_sampler_index(req.sampler_name or opt['sampler_name'])

    seed = opt['seed']
    if req.seed is not None and not req.seed == '':
        seed = int(req.seed)

    mode = req.mode or opt['mode']

    image = Image.open(req.src_path)
    orig_width, orig_height = image.size

    if mode == 1:
        mask = Image.open(req.mask_path).convert('L')
    else:
        mask = None

    base_size = req.base_size or opt['base_size']
    if mode == 3:
        width, height = base_size, base_size
    else:
        width, height = fix_aspect_ratio(base_size, req.max_size or opt['max_size'],
                                         orig_width, orig_height)

    output_images, info, html = modules.img2img.img2img(
        req.prompt or collect_prompt(opt),
        image,
        {"image": image, "mask": mask},
        req.steps or opt['steps'],
        sampler_index,
        req.mask_blur or opt['mask_blur'],
        req.inpainting_fill or opt['inpainting_fill'],
        req.use_gfpgan or opt['use_gfpgan'],
        mode,
        req.batch_count or opt['n_iter'],
        req.batch_size or opt['batch_size'],
        req.cfg_scale or opt['cfg_scale'],
        req.denoising_strength or opt['denoising_strength'],
        seed,
        height,
        width,
        opt['resize_mode'],
        "None",
        req.upscale_overlap or opt['upscale_overlap'],
        req.inpaint_full_res or opt['inpaint_full_res'],
        0
    )

    resized_images = [images.resize_image(0, image, orig_width, orig_height) for image in output_images]

    if mode == 1:
        def remove_not_masked(img):
            masked_img = Image.new("RGBA", img.size, (0, 0, 0, 0))
            masked_img.paste(img, (0, 0), mask=mask)
            return masked_img

        resized_images = [remove_not_masked(x) for x in resized_images]

    sample_path = opt['sample_path']
    os.makedirs(sample_path, exist_ok=True)
    outputs = [save_img(image, sample_path, filename=f"{int(time.time())}_{i}.png")
               for i, image in enumerate(resized_images)]
    print(f"finished: {outputs}\n{info}")
    return {"outputs": outputs, "info": info}


def main():
    uvicorn.run("krita_server:app", host="127.0.0.1", port=8000, log_level="info")


if __name__ == "__main__":
    main()
