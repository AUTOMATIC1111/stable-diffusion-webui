import modules.scripts
from modules.processing import StableDiffusionProcessing, Processed, StableDiffusionProcessingTxt2Img, StableDiffusionProcessingImg2Img, process_images
from modules.shared import opts, cmd_opts
import modules.shared as shared
import modules.processing as processing
from modules.ui import plaintext_to_html


def txt2img(prompt: str, negative_prompt: str, steps: int, sampler_index: int, use_GFPGAN: bool, n_iter: int, batch_size: int, cfg_scale: float, seed: int, height: int, width: int, *args):
    p = StableDiffusionProcessingTxt2Img(
        sd_model=shared.sd_model,
        outpath_samples=opts.outdir_samples or opts.outdir_txt2img_samples,
        outpath_grids=opts.outdir_grids or opts.outdir_txt2img_grids,
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=seed,
        sampler_index=sampler_index,
        batch_size=batch_size,
        n_iter=n_iter,
        steps=steps,
        cfg_scale=cfg_scale,
        width=width,
        height=height,
        use_GFPGAN=use_GFPGAN
    )

    processed = modules.scripts.scripts_txt2img.run(p, *args)

    if processed is not None:
        pass
    else:
        processed = process_images(p)

    return processed.images, processed.js(), plaintext_to_html(processed.info)

