import modules.scripts
from modules import sd_samplers, shared, processing
from modules.generation_parameters_copypaste import create_override_settings_dict
from modules.ui import plaintext_to_html


def txt2img(id_task: str, prompt: str, negative_prompt: str, prompt_styles, steps: int, sampler_index: int, latent_index: int, full_quality: bool, restore_faces: bool, tiling: bool, n_iter: int, batch_size: int, cfg_scale: float, image_cfg_scale: float, diffusers_guidance_rescale: float, clip_skip: int, seed: int, subseed: int, subseed_strength: float, seed_resize_from_h: int, seed_resize_from_w: int, height: int, width: int, enable_hr: bool, denoising_strength: float, hr_scale: float, hr_upscaler: str, hr_force: bool, hr_second_pass_steps: int, hr_resize_x: int, hr_resize_y: int, refiner_steps: int, refiner_start: int, refiner_prompt: str, refiner_negative: str, override_settings_texts, *args): # pylint: disable=unused-argument

    shared.log.debug(f'txt2img: id_task={id_task}|prompt={prompt}|negative_prompt={negative_prompt}|prompt_styles={prompt_styles}|steps={steps}|sampler_index={sampler_index}|latent_index={latent_index}|full_quality={full_quality}|restore_faces={restore_faces}|tiling={tiling}|n_iter={n_iter}|batch_size={batch_size}|cfg_scale={cfg_scale}|clip_skip={clip_skip}|seed={seed}|subseed={subseed}|subseed_strength={subseed_strength}|seed_resize_from_h={seed_resize_from_h}|seed_resize_from_w={seed_resize_from_w}||height={height}|width={width}|enable_hr={enable_hr}|denoising_strength={denoising_strength}|hr_scale={hr_scale}|hr_upscaler={hr_upscaler}|hr_force={hr_force}|hr_second_pass_steps={hr_second_pass_steps}|hr_resize_x={hr_resize_x}|hr_resize_y={hr_resize_y}|image_cfg_scale={image_cfg_scale}|diffusers_guidance_rescale={diffusers_guidance_rescale}|refiner_steps={refiner_steps}|refiner_start={refiner_start}|refiner_prompt={refiner_prompt}|refiner_negative={refiner_negative}|override_settings_texts={override_settings_texts}')

    if shared.sd_model is None:
        shared.log.warning('Model not loaded')
        return [], '', '', 'Error: model not loaded'

    override_settings = create_override_settings_dict(override_settings_texts)
    if sampler_index is None:
        sampler_index = 0
    if latent_index is None:
        latent_index = 0

    p = processing.StableDiffusionProcessingTxt2Img(
        sd_model=shared.sd_model,
        outpath_samples=shared.opts.outdir_samples or shared.opts.outdir_txt2img_samples,
        outpath_grids=shared.opts.outdir_grids or shared.opts.outdir_txt2img_grids,
        prompt=prompt,
        styles=prompt_styles,
        negative_prompt=negative_prompt,
        seed=seed,
        subseed=subseed,
        subseed_strength=subseed_strength,
        seed_resize_from_h=seed_resize_from_h,
        seed_resize_from_w=seed_resize_from_w,
        seed_enable_extras=True,
        sampler_name=sd_samplers.samplers[sampler_index].name,
        latent_sampler=sd_samplers.samplers[latent_index].name,
        batch_size=batch_size,
        n_iter=n_iter,
        steps=steps,
        cfg_scale=cfg_scale,
        image_cfg_scale=image_cfg_scale,
        diffusers_guidance_rescale=diffusers_guidance_rescale,
        clip_skip=clip_skip,
        width=width,
        height=height,
        full_quality=full_quality,
        restore_faces=restore_faces,
        tiling=tiling,
        enable_hr=enable_hr,
        denoising_strength=denoising_strength,
        hr_scale=hr_scale,
        hr_upscaler=hr_upscaler,
        hr_force=hr_force,
        hr_second_pass_steps=hr_second_pass_steps,
        hr_resize_x=hr_resize_x,
        hr_resize_y=hr_resize_y,
        refiner_steps=refiner_steps,
        refiner_start=refiner_start,
        refiner_prompt=refiner_prompt,
        refiner_negative=refiner_negative,
        override_settings=override_settings,
    )
    p.scripts = modules.scripts.scripts_txt2img
    p.script_args = args
    processed = modules.scripts.scripts_txt2img.run(p, *args)
    if processed is None:
        processed = processing.process_images(p)
    p.close()
    if processed is None:
        return [], '', '', 'Error: processing failed'
    generation_info_js = processed.js()
    return processed.images, generation_info_js, processed.info, plaintext_to_html(processed.comments)
