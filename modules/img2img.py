import math
from PIL import Image

from modules.processing import Processed, StableDiffusionProcessingImg2Img, process_images
from modules.shared import opts, state
import modules.shared as shared
import modules.processing as processing
from modules.ui import plaintext_to_html
import modules.images as images
import modules.scripts

def img2img(prompt: str, init_img, init_img_with_mask, steps: int, sampler_index: int, mask_blur: int, inpainting_fill: int, use_GFPGAN: bool, tiling: bool, mode: int, n_iter: int, batch_size: int, cfg_scale: float, denoising_strength: float, seed: int, height: int, width: int, resize_mode: int, upscaler_index: str, upscale_overlap: int, inpaint_full_res: bool, inpainting_mask_invert: int, *args):
    is_inpaint = mode == 1
    is_loopback = mode == 2
    is_upscale = mode == 3

    if is_inpaint:
        image = init_img_with_mask['image']
        mask = init_img_with_mask['mask']
    else:
        image = init_img
        mask = None

    assert 0. <= denoising_strength <= 1., 'can only work with strength in [0.0, 1.0]'

    p = StableDiffusionProcessingImg2Img(
        sd_model=shared.sd_model,
        outpath_samples=opts.outdir_samples or opts.outdir_img2img_samples,
        outpath_grids=opts.outdir_grids or opts.outdir_img2img_grids,
        prompt=prompt,
        seed=seed,
        sampler_index=sampler_index,
        batch_size=batch_size,
        n_iter=n_iter,
        steps=steps,
        cfg_scale=cfg_scale,
        width=width,
        height=height,
        use_GFPGAN=use_GFPGAN,
        tiling=tiling,
        init_images=[image],
        mask=mask,
        mask_blur=mask_blur,
        inpainting_fill=inpainting_fill,
        resize_mode=resize_mode,
        denoising_strength=denoising_strength,
        inpaint_full_res=inpaint_full_res,
        inpainting_mask_invert=inpainting_mask_invert,
        extra_generation_params={"Denoising Strength": denoising_strength}
    )

    if is_loopback:
        output_images, info = None, None
        history = []
        initial_seed = None
        initial_info = None

        for i in range(n_iter):
            p.n_iter = 1
            p.batch_size = 1
            p.do_not_save_grid = True

            state.job = f"Batch {i + 1} out of {n_iter}"
            processed = process_images(p)

            if initial_seed is None:
                initial_seed = processed.seed
                initial_info = processed.info

            p.init_images = [processed.images[0]]
            p.seed = processed.seed + 1
            p.denoising_strength = max(p.denoising_strength * 0.95, 0.1)
            history.append(processed.images[0])

        grid = images.image_grid(history, batch_size, rows=1)

        images.save_image(grid, p.outpath_grids, "grid", initial_seed, prompt, opts.grid_format, info=info, short_filename=not opts.grid_extended_filename)

        processed = Processed(p, history, initial_seed, initial_info)

    elif is_upscale:
        initial_seed = None
        initial_info = None

        upscaler = shared.sd_upscalers[upscaler_index]
        img = upscaler.upscale(init_img, init_img.width * 2, init_img.height * 2)

        processing.torch_gc()

        grid = images.split_grid(img, tile_w=width, tile_h=height, overlap=upscale_overlap)

        p.n_iter = 1
        p.do_not_save_grid = True
        p.do_not_save_samples = True

        work = []
        work_results = []

        for y, h, row in grid.tiles:
            for tiledata in row:
                work.append(tiledata[2])

        batch_count = math.ceil(len(work) / p.batch_size)
        print(f"SD upscaling will process a total of {len(work)} images tiled as {len(grid.tiles[0][2])}x{len(grid.tiles)} in a total of {batch_count} batches.")

        for i in range(batch_count):
            p.init_images = work[i*p.batch_size:(i+1)*p.batch_size]

            state.job = f"Batch {i + 1} out of {batch_count}"
            processed = process_images(p)

            if initial_seed is None:
                initial_seed = processed.seed
                initial_info = processed.info

            p.seed = processed.seed + 1
            work_results += processed.images

        image_index = 0
        for y, h, row in grid.tiles:
            for tiledata in row:
                tiledata[2] = work_results[image_index] if image_index < len(work_results) else Image.new("RGB", (p.width, p.height))
                image_index += 1

        combined_image = images.combine_grid(grid)

        if opts.samples_save:
            images.save_image(combined_image, p.outpath_samples, "", initial_seed, prompt, opts.grid_format, info=initial_info)

        processed = Processed(p, [combined_image], initial_seed, initial_info)

    else:

        processed = modules.scripts.scripts_img2img.run(p, *args)

        if processed is None:
            processed = process_images(p)


    return processed.images, processed.js(), plaintext_to_html(processed.info)
