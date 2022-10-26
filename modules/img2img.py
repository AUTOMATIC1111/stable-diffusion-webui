import math
import os
import sys
import traceback

import numpy as np
from PIL import Image, ImageOps, ImageChops

from scipy import ndimage
import base64
import io

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

    images = [file for file in [os.path.join(input_dir, x) for x in os.listdir(input_dir)] if os.path.isfile(file)]

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


def img2img(mode: int,
            srcimg,vwt,vww,vwh,vct,vcw,vch,
            prompt: str, negative_prompt: str, prompt_style: str, prompt_style2: str,
            init_img, init_img_with_mask, init_img_inpaint, init_mask_inpaint,
            indopaint, vthresh, vloch, vlocw, outpainting_fill,
            mask_mode, steps: int, sampler_index: int, mask_blur: int, inpainting_fill: int,
            restore_faces: bool, tiling: bool, n_iter: int, batch_size: int, cfg_scale: float,
            denoising_strength: float, seed: int, subseed: int, subseed_strength: float,
            seed_resize_from_h: int, seed_resize_from_w: int, seed_enable_extras: bool,
            height: int, width: int, resize_mode: int, inpaint_full_res: bool,
            inpaint_full_res_padding: int, inpainting_mask_invert: int,
            img2img_batch_input_dir: str, img2img_batch_output_dir: str,*args):
    
    outparms = None
    is_inpaint = mode == 1
    is_batch = mode == 2

    if is_inpaint:
        if mask_mode == 0:
            image = init_img_with_mask['image']
            mask = init_img_with_mask['mask']
            alpha_mask = ImageOps.invert(image.split()[-1]).convert('L').point(lambda x: 255 if x > 0 else 0, mode='1')
            mask = ImageChops.lighter(alpha_mask, mask.convert('L')).convert('L')
            image = image.convert('RGB')
        else:
            image = init_img_inpaint
            mask = init_mask_inpaint
    else:
        image = init_img
        mask = None
        if indopaint:
            locmx = np.array(image).max(axis = 2)
            mx = ndimage.grey_dilation(locmx,size = (vloch,vlocw))
            mregion = (mx < vthresh)
            expreg = ndimage.grey_dilation(mregion,size = (vloch,vlocw))
            mask = np.logical_and(expreg,locmx < vthresh)
            mask = Image.fromarray(mask)
            
            inpainting_fill = outpainting_fill
            if srcimg is None:
                print("Cannot overlay outpaint, did not receive source image.")
            else:
                # if srcimg.startswith("data:image/png;base64,"): # Nope.
                #     srcimg = srcimg[len("data:image/png;base64,"):]
                # srcimg_b = base64.decodebytes(srcimg.encode('utf-8'))
                srcimg_b = base64.b64decode(srcimg.split(",")[-1])
                srcimg_pil = Image.open(io.BytesIO(srcimg_b))
                if "translateX" in vwt:
                    fparst = vwt.find("(")
                    fpared = vwt.find(")")
                    vwx = vwt[fparst + 1:fpared]
                else:
                    fparst = -1
                    fpared = -1
                    vwx = "0px"
                if "translateY" in vwt:
                    fparst = vwt.find("(",fparst + 1)
                    fpared = vwt.find(")",fpared + 1)
                    vwy = vwt[fparst + 1:fpared]
                else:
                    fparst = -1
                    fpared = -1
                    vwy = "0px"
                if "translateX" in vct:    
                    fparst = vct.find("(")
                    fpared = vct.find(")")
                    vcx = vct[fparst + 1:fpared]
                else:
                    fparst = -1
                    fpared = -1
                    vcx = "0px"
                if "translateY" in vct:
                    fparst = vct.find("(",fparst + 1)
                    fpared = vct.find(")",fpared + 1)
                    vcy = vct[fparst + 1:fpared]
                else:
                    fparst = -1
                    fpared = -1
                    vcy = "0px"
                fpx = lambda x: float(x.replace("px","")) 
                vwx = fpx(vwx) 
                vwy = fpx(vwy)
                vww = fpx(vww)
                vwh = fpx(vwh)
                vcx = fpx(vcx)
                vcy = fpx(vcy)
                vcw = fpx(vcw)
                vch = fpx(vch)
                outparms = (srcimg_pil,vwx,vwy,vww,vwh,vcx,vcy,vcw,vch)

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
        outparms = outparms,
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
