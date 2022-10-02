import logging
import os
import time

from fastapi import FastAPI
from PIL import Image
from webui import modules, shared

from .structs import Img2ImgRequest, Txt2ImgRequest, UpscaleRequest
from .utils import (
    fix_aspect_ratio,
    get_sampler_index,
    get_upscaler_index,
    load_config,
    merge_default_config,
    parse_prompt,
    save_img,
    set_face_restorer,
)

app = FastAPI()

log = logging.getLogger(__name__)


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
    # - response isn't well typed but is deeply tied into the krita_plugin side..
    # - ensuring the folders for images exist should be refactored out
    opt = load_config().plugin

    os.makedirs(opt.sample_path, exist_ok=True)
    path = os.path.join(opt.sample_path, f"{int(time.time())}")
    src_path = os.path.abspath(path)
    return {
        "new_img": src_path + ".png",
        "new_img_mask": src_path + "_mask.png",
        "upscalers": [upscaler.name for upscaler in shared.sd_upscalers],
        **opt.dict(),
    }


@app.post("/txt2img")
async def f_txt2img(req: Txt2ImgRequest):
    """Post request for Txt2Img.

    Args:
        req (Txt2ImgRequest): Request.

    Returns:
        Dict: Outputs and info.
    """
    log.info(f"txt2img: {req}")

    opt = load_config().txt2img
    req = merge_default_config(req, opt)

    set_face_restorer(req.face_restorer, req.codeformer_weight)
    width, height = fix_aspect_ratio(
        req.base_size, req.max_size, req.orig_width, req.orig_height
    )

    output_images, info, html = modules.txt2img.txt2img(
        parse_prompt(req.prompt),  # prompt
        parse_prompt(req.negative_prompt),  # negative_prompt
        "None",  # prompt_style: saved prompt styles (unsupported)
        "None",  # prompt_style2: saved prompt styles (unsupported)
        req.steps,  # steps
        get_sampler_index(req.sampler_name),  # sampler_index
        req.use_gfpgan,  # restore_faces
        req.tiling,  # tiling
        req.batch_count,  # n_iter
        req.batch_size,  # batch_size
        req.cfg_scale,  # cfg_scale
        req.seed,  # seed
        None,  # subseed
        0,  # subseed_strength
        0,  # seed_resize_from_h
        0,  # seed_resize_from_w
        False,  # seed_enable_extras
        height,  # height
        width,  # width
        False,  # enable_hr: high res fix
        False,  # scale_latent
        0,  # denoising_strength: only applicable if high res fix in use
        # *args below
        0,  # selects which script to use. 0 to not run any.
    )

    os.makedirs(opt.sample_path, exist_ok=True)
    resized_images = [
        modules.images.resize_image(0, image, req.orig_width, req.orig_height)
        for image in output_images
    ]
    outputs = [
        save_img(image, opt.sample_path, filename=f"{int(time.time())}_{i}.png")
        for i, image in enumerate(resized_images)
    ]
    log.info(f"finished: {outputs}\n{info}")
    return {"outputs": outputs, "info": info}


@app.post("/img2img")
async def f_img2img(req: Img2ImgRequest):
    """Post request for Img2Img.

    Args:
        req (Img2ImgRequest): Request.

    Returns:
        Dict: Outputs and info.
    """
    log.info(f"img2img: {req}")

    opt = load_config().img2img
    req = merge_default_config(req, opt)

    image = Image.open(req.src_path)
    mask = Image.open(req.mask_path).convert("L") if req.mode == 1 else None

    set_face_restorer(req.face_restorer, req.codeformer_weight)

    orig_width, orig_height = image.size

    # Disabled as SD upscale is now a script, not builtin
    if False and req.mode == 2:
        width, height = req.base_size, req.base_size
        if upscaler_index > 0:
            image = image.convert("RGB")
    else:
        width, height = fix_aspect_ratio(
            req.base_size, req.max_size, orig_width, orig_height
        )

    # NOTE:
    # - image & mask repeated due to Gradio API have separate tabs for each mode...
    # - mask is used only in inpaint mode
    # - mask_mode determines whethere init_img_with_mask or init_img_inpaint is used,
    #   I dont know why
    # - the internal code for img2img is confusing and duplicative...

    output_images, info, html = modules.img2img.img2img(
        req.mode,  # mode
        parse_prompt(req.prompt),  # prompt
        parse_prompt(req.negative_prompt),  # negative_prompt
        "None",  # prompt_style: saved prompt styles (unsupported)
        "None",  # prompt_style2: saved prompt styles (unsupported)
        image,  # init_img
        {"image": image, "mask": mask},  # init_img_with_mask
        image,  # init_img_inpaint
        mask,  # init_mask_inpaint
        -1,  # mask_mode: internally checks if equal 0. enables Alpha Mask but idk what it does.
        req.steps,  # steps
        get_sampler_index(req.sampler_name),  # sampler_index
        req.mask_blur,  # mask_blur
        req.inpainting_fill,  # inpainting_fill
        req.use_gfpgan,  # restore_faces
        req.tiling,  # tiling
        req.batch_count,  # n_iter
        req.batch_size,  # batch_size
        req.cfg_scale,  # cfg_scale
        req.denoising_strength,  # denoising_strength
        req.seed,  # seed
        None,  # subseed
        0,  # subseed_strength
        0,  # seed_resize_from_h
        0,  # seed_resize_from_w
        False,  # seed_enable_extras
        height,  # height
        width,  # width
        req.resize_mode,  # resize_mode
        req.inpaint_full_res,  # inpaint_full_res
        32,  # inpaint_full_res_padding
        req.invert_mask,  # inpainting_mask_invert
        "",  # img2img_batch_input_dir (unspported)
        "",  # img2img_batch_output_dir (unspported)
        # Disabled as SD upscale is now a script, not builtin
        # get_upscaler_index(req.upscaler_name),
        # req.upscale_overlap,
        # *args below
        0,  # selects which script to use. 0 to not run any.
    )

    resized_images = [
        modules.images.resize_image(0, image, orig_width, orig_height)
        for image in output_images
    ]

    if req.mode == 1:

        def remove_not_masked(img):
            masked_img = Image.new("RGBA", img.size, (0, 0, 0, 0))
            masked_img.paste(img, (0, 0), mask=mask)
            return masked_img

        resized_images = [remove_not_masked(x) for x in resized_images]

    os.makedirs(opt.sample_path, exist_ok=True)
    outputs = [
        save_img(image, opt.sample_path, filename=f"{int(time.time())}_{i}.png")
        for i, image in enumerate(resized_images)
    ]
    log.info(f"finished: {outputs}\n{info}")
    return {"outputs": outputs, "info": info}


@app.post("/upscale")
async def f_upscale(req: UpscaleRequest):
    """Post request for upscaling.

    Args:
        req (UpscaleRequest): Request.

    Returns:
        Dict: Output.
    """
    log.info(f"upscale: {req}")

    opt = load_config().upscale
    req = merge_default_config(req, opt)

    image = Image.open(req.src_path).convert("RGB")
    orig_width, orig_height = image.size

    upscaler_index = get_upscaler_index(req.upscaler_name)
    upscaler = shared.sd_upscalers[upscaler_index]

    if upscaler.name == "None":
        log.info(f"No upscaler selected, will do nothing")
        return

    if req.downscale_first:
        image = modules.images.resize_image(0, image, orig_width // 2, orig_height // 2)

    upscaled_image = upscaler.upscale(image, 2 * orig_width, 2 * orig_height)
    resized_image = modules.images.resize_image(
        0, upscaled_image, orig_width, orig_height
    )

    os.makedirs(opt.sample_path, exist_ok=True)
    output = save_img(
        resized_image, opt.sample_path, filename=f"{int(time.time())}.png"
    )
    log.info(f"finished: {output}")
    return {"output": output}
