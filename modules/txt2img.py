import os.path

import modules.scripts
from modules import sd_samplers, processing
from modules.generation_parameters_copypaste import create_override_settings_dict
from modules.shared import opts, cmd_opts
import modules.shared as shared
from modules.ui import plaintext_to_html
import qrcode



def txt2img(id_task: str, prompt: str, negative_prompt: str, prompt_styles, steps: int, sampler_index: int, restore_faces: bool, tiling: bool, n_iter: int, batch_size: int, cfg_scale: float, seed: int, subseed: int, subseed_strength: float, seed_resize_from_h: int, seed_resize_from_w: int, seed_enable_extras: bool, height: int, width: int, enable_hr: bool, denoising_strength: float, hr_scale: float, hr_upscaler: str, hr_second_pass_steps: int, hr_resize_x: int, hr_resize_y: int, hr_sampler_index: int, hr_prompt: str, hr_negative_prompt, override_settings_texts, *args):
    override_settings = create_override_settings_dict(override_settings_texts)

    p = processing.StableDiffusionProcessingTxt2Img(
        sd_model=shared.sd_model,
        outpath_samples=opts.outdir_samples or opts.outdir_txt2img_samples,
        outpath_grids=opts.outdir_grids or opts.outdir_txt2img_grids,
        prompt=prompt,
        styles=prompt_styles,
        negative_prompt=negative_prompt,
        seed=seed,
        subseed=subseed,
        subseed_strength=subseed_strength,
        seed_resize_from_h=seed_resize_from_h,
        seed_resize_from_w=seed_resize_from_w,
        seed_enable_extras=seed_enable_extras,
        sampler_name=sd_samplers.samplers[sampler_index].name,
        batch_size=batch_size,
        n_iter=n_iter,
        steps=steps,
        cfg_scale=cfg_scale,
        width=width,
        height=height,
        restore_faces=restore_faces,
        tiling=tiling,
        enable_hr=enable_hr,
        denoising_strength=denoising_strength if enable_hr else None,
        hr_scale=hr_scale,
        hr_upscaler=hr_upscaler,
        hr_second_pass_steps=hr_second_pass_steps,
        hr_resize_x=hr_resize_x,
        hr_resize_y=hr_resize_y,
        hr_sampler_name=sd_samplers.samplers_for_img2img[hr_sampler_index - 1].name if hr_sampler_index != 0 else None,
        hr_prompt=hr_prompt,
        hr_negative_prompt=hr_negative_prompt,
        override_settings=override_settings,
    )

    p.scripts = modules.scripts.scripts_txt2img
    p.script_args = args

    if cmd_opts.enable_console_prompts:
        print(f"\ntxt2img: {prompt}", file=shared.progress_print_out)

    processed = modules.scripts.scripts_txt2img.run(p, *args)

    if processed is None:
        processed = processing.process_images(p)

    p.close()

    shared.total_tqdm.clear()

    if opts.hide_negative_prompt:
        for i, text in enumerate(processed.infotexts):
            processed.infotexts[i] = text.replace(f'Negative prompt: {negative_prompt}', '')
        processed.info = processed.info.replace(f'Negative prompt: {negative_prompt}', '')
        processed.negative_prompt = ''
        processed.all_negative_prompts = ['' for _ in processed.all_negative_prompts]

    generation_info_js = processed.js()
    if opts.samples_log_stdout:
        print(generation_info_js)

    if opts.do_not_show_images:
        processed.images = []

    if opts.add_qr_code:
        # I'm adding images to array I iterate to, better to create new one and replace
        new_processed_images = processed.images[:processed.index_of_first_image]
        for i, img in enumerate(processed.images[processed.index_of_first_image:]):
            image_path = img.already_saved_as.replace(p.outpath_samples, '')
            qr_link = f'{opts.static_server_uri}{image_path.replace(os.path.sep, "/")}'
            qr_img = qrcode.make(qr_link).get_image()
            new_processed_images.append(img)
            new_processed_images.append(qr_img)
        processed.images = new_processed_images

    return processed.images, generation_info_js, plaintext_to_html(processed.info), plaintext_to_html(processed.comments)
