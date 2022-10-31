import math
import os
import sys
import traceback

import numpy as np
from PIL import Image, ImageOps, ImageChops

from modules import devices
from modules.processing import Processed, StableDiffusionProcessingImg2Img, process_images
from modules.shared import opts, state
import modules.shared as shared
import modules.processing as processing
from modules.ui import plaintext_to_html
import modules.images as images
import modules.scripts


def process_batch(p, input_dir, output_dir, args):
    processing.fix_seed(p)

    images = shared.listfiles(input_dir)

    print(f"Will process {len(images)} images, creating {p.n_iter * p.batch_size} new images for each.")

    save_normally = output_dir == ''

    p.do_not_save_grid = True
    p.do_not_save_samples = not save_normally

    state.job_count = len(images) * p.n_iter

    for i, image in enumerate(images):
        state.job = f"{i+1} out of {len(images)}"
        if state.skipped:
            state.skipped = False

        if state.interrupted:
            break

        img = Image.open(image)
        # Use the EXIF orientation of photos taken by smartphones.
        img = ImageOps.exif_transpose(img) 
        p.init_images = [img] * p.batch_size

        proc = modules.scripts.scripts_img2img.run(p, *args)
        if proc is None:
            proc = process_images(p)

        for n, processed_image in enumerate(proc.images):
            filename = os.path.basename(image)

            if n > 0:
                left, right = os.path.splitext(filename)
                filename = f"{left}-{n}{right}"

            if not save_normally:
                processed_image.save(os.path.join(output_dir, filename))


def img2img(mode: int, prompt: str, negative_prompt: str, prompt_style: str, prompt_style2: str, init_img, init_img_with_mask, init_img_inpaint, init_mask_inpaint, mask_mode, steps: int, sampler_index: int, mask_blur: int, inpainting_fill: int, restore_faces: bool, tiling: bool, n_iter: int, batch_size: int, cfg_scale: float, denoising_strength: float, seed: int, subseed: int, subseed_strength: float, seed_resize_from_h: int, seed_resize_from_w: int, seed_enable_extras: bool, height: int, width: int, resize_mode: int, inpaint_full_res: bool, inpaint_full_res_padding: int, inpainting_mask_invert: int, img2img_batch_input_dir: str, img2img_batch_output_dir: str, *args):
    is_inpaint = mode == 1
    is_batch = mode == 2

    if is_inpaint:
        # Drawn mask
        if mask_mode == 0:
            image = init_img_with_mask['image']
            mask = init_img_with_mask['mask']
            alpha_mask = ImageOps.invert(image.split()[-1]).convert('L').point(lambda x: 255 if x > 0 else 0, mode='1')
            mask = ImageChops.lighter(alpha_mask, mask.convert('L')).convert('L')
            image = image.convert('RGB')
        # Uploaded mask
        else:
            image = init_img_inpaint
            mask = init_mask_inpaint
    # No mask
    else:
        image = init_img
        mask = None

    # Use the EXIF orientation of photos taken by smartphones.
    image = ImageOps.exif_transpose(image) 

    assert 0. <= denoising_strength <= 1., 'can only work with strength in [0.0, 1.0]'

    p = StableDiffusionProcessingImg2Img(
        sd_model=shared.sd_model,
        outpath_samples=opts.outdir_samples or opts.outdir_img2img_samples,
        outpath_grids=opts.outdir_grids or opts.outdir_img2img_grids,
        prompt=prompt,
        negative_prompt=negative_prompt,
        styles=[prompt_style, prompt_style2],
        seed=seed,
        subseed=subseed,
        subseed_strength=subseed_strength,
        seed_resize_from_h=seed_resize_from_h,
        seed_resize_from_w=seed_resize_from_w,
        seed_enable_extras=seed_enable_extras,
        sampler_index=sampler_index,
        batch_size=batch_size,
        n_iter=n_iter,
        steps=steps,
        cfg_scale=cfg_scale,
        width=width,
        height=height,
        restore_faces=restore_faces,
        tiling=tiling,
        init_images=[image],
        mask=mask,
        mask_blur=mask_blur,
        inpainting_fill=inpainting_fill,
        resize_mode=resize_mode,
        denoising_strength=denoising_strength,
        inpaint_full_res=inpaint_full_res,
        inpaint_full_res_padding=inpaint_full_res_padding,
        inpainting_mask_invert=inpainting_mask_invert,
    )

    p.scripts = modules.scripts.scripts_txt2img
    p.script_args = args

    if shared.cmd_opts.enable_console_prompts:
        print(f"\nimg2img: {prompt}", file=shared.progress_print_out)

    p.extra_generation_params["Mask blur"] = mask_blur

    if is_batch:
        assert not shared.cmd_opts.hide_ui_dir_config, "Launched with --hide-ui-dir-config, batch img2img disabled"

        process_batch(p, img2img_batch_input_dir, img2img_batch_output_dir, args)

        processed = Processed(p, [], p.seed, "")
    else:
        processed = modules.scripts.scripts_img2img.run(p, *args)
        if processed is None:
            processed = process_images(p)

    shared.total_tqdm.clear()

    generation_info_js = processed.js()
    if opts.samples_log_stdout:
        print(generation_info_js)

    if opts.do_not_show_images:
        processed.images = []

    return processed.images, generation_info_js, plaintext_to_html(processed.info)
