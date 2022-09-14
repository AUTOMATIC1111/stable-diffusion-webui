import modules.scripts
from modules.processing import StableDiffusionProcessing, Processed, StableDiffusionProcessingTxt2Img, StableDiffusionProcessingImg2Img, process_images
from modules.shared import opts, cmd_opts
import modules.shared as shared
import modules.processing as processing
from modules.ui import plaintext_to_html


def txt2img(prompt: str, negative_prompt: str, prompt_style: str, steps: int, sampler_index: int, restore_faces: bool, tiling: bool, n_iter: int, batch_size: int, cfg_scale: float, seed: int, subseed: int, subseed_strength: float, seed_resize_from_h: int, seed_resize_from_w: int, height: int, width: int, *args):
    p = StableDiffusionProcessingTxt2Img(
        sd_model=shared.sd_model,
        outpath_samples=opts.outdir_samples or opts.outdir_txt2img_samples,
        outpath_grids=opts.outdir_grids or opts.outdir_txt2img_grids,
        prompt=prompt,
        prompt_style=prompt_style,
        negative_prompt=negative_prompt,
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
    )

    print(f"\ntxt2img: {prompt}", file=shared.progress_print_out)
    processed = modules.scripts.scripts_txt2img.run(p, *args)

    if processed is not None:
        pass
    else:
        processed = process_images(p)

    shared.total_tqdm.clear()

    return processed.images, processed.js(), plaintext_to_html(processed.info)

