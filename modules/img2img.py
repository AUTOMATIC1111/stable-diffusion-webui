import math
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

def img2img(prompt: str, negative_prompt: str, prompt_style: str, init_img, init_img_with_mask, init_mask, mask_mode, steps: int, sampler_index: int, mask_blur: int, inpainting_fill: int, restore_faces: bool, tiling: bool, mode: int, n_iter: int, batch_size: int, cfg_scale: float, denoising_strength: float, denoising_strength_change_factor: float, seed: int, subseed: int, subseed_strength: float, seed_resize_from_h: int, seed_resize_from_w: int, height: int, width: int, resize_mode: int, upscaler_index: str, upscale_overlap: int, inpaint_full_res: bool, inpainting_mask_invert: int, *args):
    is_inpaint = mode == 1
    is_loopback = mode == 2
    is_upscale = mode == 3

    if is_inpaint:
        if mask_mode == 0:
            image = init_img_with_mask['image']
            mask = init_img_with_mask['mask']
            alpha_mask = ImageOps.invert(image.split()[-1]).convert('L').point(lambda x: 255 if x > 0 else 0, mode='1')
            mask = ImageChops.lighter(alpha_mask, mask.convert('L')).convert('L')
            image = image.convert('RGB')
        else:
            image = init_img
            mask = init_mask
    else:
        image = init_img
        mask = None

    assert 0. <= denoising_strength <= 1., 'can only work with strength in [0.0, 1.0]'

    p = StableDiffusionProcessingImg2Img(
        sd_model=shared.sd_model,
        outpath_samples=opts.outdir_samples or opts.outdir_img2img_samples,
        outpath_grids=opts.outdir_grids or opts.outdir_img2img_grids,
        prompt=prompt,
        negative_prompt=negative_prompt,
        prompt_style=prompt_style,
        seed=seed,
        subseed=subseed,
        subseed_strength=subseed_strength,
        seed_resize_from_h=seed_resize_from_h,
        seed_resize_from_w=seed_resize_from_w,
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
        inpainting_mask_invert=inpainting_mask_invert,
        extra_generation_params={
            "Denoising strength change factor": (denoising_strength_change_factor if is_loopback else None)
        }
    )
    print(f"\nimg2img: {prompt}", file=shared.progress_print_out)

    if is_loopback:
        output_images, info = None, None
        history = []
        initial_seed = None
        initial_info = None

        state.job_count = n_iter

        for i in range(n_iter):
            p.n_iter = 1
            p.batch_size = 1
            p.do_not_save_grid = True

            state.job = f"Batch {i + 1} out of {n_iter}"
            processed = process_images(p)

            if initial_seed is None:
                initial_seed = processed.seed
                initial_info = processed.info
            
            init_img = processed.images[0]

            p.init_images = [init_img]
            p.seed = processed.seed + 1
            p.denoising_strength = min(max(p.denoising_strength * denoising_strength_change_factor, 0.1), 1)
            history.append(processed.images[0])

        grid = images.image_grid(history, batch_size, rows=1)

        images.save_image(grid, p.outpath_grids, "grid", initial_seed, prompt, opts.grid_format, info=info, short_filename=not opts.grid_extended_filename, p=p)

        processed = Processed(p, history, initial_seed, initial_info)

    elif is_upscale:
        initial_info = None

        processing.fix_seed(p)
        seed = p.seed

        upscaler = shared.sd_upscalers[upscaler_index]
        img = upscaler.upscale(init_img, init_img.width * 2, init_img.height * 2)

        devices.torch_gc()

        grid = images.split_grid(img, tile_w=width, tile_h=height, overlap=upscale_overlap)

        upscale_count = p.n_iter
        p.n_iter = 1
        p.do_not_save_grid = True
        p.do_not_save_samples = True

        work = []

        for y, h, row in grid.tiles:
            for tiledata in row:
                work.append(tiledata[2])

        batch_count = math.ceil(len(work) / p.batch_size)
        state.job_count = batch_count * upscale_count

        print(f"SD upscaling will process a total of {len(work)} images tiled as {len(grid.tiles[0][2])}x{len(grid.tiles)} per upscale in a total of {state.job_count} batches.")

        result_images = []
        for n in range(upscale_count):
            start_seed = seed + n
            p.seed = start_seed

            work_results = []
            for i in range(batch_count):
                p.init_images = work[i*p.batch_size:(i+1)*p.batch_size]

                state.job = f"Batch {i + 1} out of {state.job_count}"
                processed = process_images(p)

                if initial_info is None:
                    initial_info = processed.info

                p.seed = processed.seed + 1
                work_results += processed.images

            image_index = 0
            for y, h, row in grid.tiles:
                for tiledata in row:
                    tiledata[2] = work_results[image_index] if image_index < len(work_results) else Image.new("RGB", (p.width, p.height))
                    image_index += 1

            combined_image = images.combine_grid(grid)
            result_images.append(combined_image)

            if opts.samples_save:
                images.save_image(combined_image, p.outpath_samples, "", start_seed, prompt, opts.samples_format, info=initial_info, p=p)

        processed = Processed(p, result_images, seed, initial_info)

    else:

        processed = modules.scripts.scripts_img2img.run(p, *args)

        if processed is None:
            processed = process_images(p)

    shared.total_tqdm.clear()

    return processed.images, processed.js(), plaintext_to_html(processed.info)
