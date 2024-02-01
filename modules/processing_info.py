import os
from installer import git_commit
from modules import shared, sd_samplers_common, sd_vae, generation_parameters_copypaste
from modules.processing_class import StableDiffusionProcessing


if shared.backend == shared.Backend.ORIGINAL:
    from modules import sd_hijack
else:
    sd_hijack = None


def create_infotext(p: StableDiffusionProcessing, all_prompts=None, all_seeds=None, all_subseeds=None, comments=None, iteration=0, position_in_batch=0, index=None, all_negative_prompts=None):
    if p is None:
        shared.log.warning('Processing info: no data')
        return ''
    if not hasattr(shared.sd_model, 'sd_checkpoint_info'):
        return ''
    if index is None:
        index = position_in_batch + iteration * p.batch_size
    if all_prompts is None:
        all_prompts = p.all_prompts or [p.prompt]
    if all_negative_prompts is None:
        all_negative_prompts = p.all_negative_prompts or [p.negative_prompt]
    if all_seeds is None:
        all_seeds = p.all_seeds or [p.seed]
    if all_subseeds is None:
        all_subseeds = p.all_subseeds or [p.subseed]
    while len(all_prompts) <= index:
        all_prompts.append(all_prompts[-1])
    while len(all_seeds) <= index:
        all_seeds.append(all_seeds[-1])
    while len(all_subseeds) <= index:
        all_subseeds.append(all_subseeds[-1])
    while len(all_negative_prompts) <= index:
        all_negative_prompts.append(all_negative_prompts[-1])
    comment = ', '.join(comments) if comments is not None and type(comments) is list else None
    ops = list(set(p.ops))
    ops.reverse()
    args = {
        # basic
        "Steps": p.steps,
        "Seed": all_seeds[index],
        "Sampler": p.sampler_name,
        "CFG scale": p.cfg_scale,
        "Size": f"{p.width}x{p.height}" if hasattr(p, 'width') and hasattr(p, 'height') else None,
        "Batch": f'{p.n_iter}x{p.batch_size}' if p.n_iter > 1 or p.batch_size > 1 else None,
        "Index": f'{p.iteration + 1}x{index + 1}' if (p.n_iter > 1 or p.batch_size > 1) and index >= 0 else None,
        "Parser": shared.opts.prompt_attention,
        "Model": None if (not shared.opts.add_model_name_to_info) or (not shared.sd_model.sd_checkpoint_info.model_name) else shared.sd_model.sd_checkpoint_info.model_name.replace(',', '').replace(':', ''),
        "Model hash": getattr(p, 'sd_model_hash', None if (not shared.opts.add_model_hash_to_info) or (not shared.sd_model.sd_model_hash) else shared.sd_model.sd_model_hash),
        "VAE": (None if not shared.opts.add_model_name_to_info or sd_vae.loaded_vae_file is None else os.path.splitext(os.path.basename(sd_vae.loaded_vae_file))[0]) if p.full_quality else 'TAESD',
        "Seed resize from": None if p.seed_resize_from_w == 0 or p.seed_resize_from_h == 0 else f"{p.seed_resize_from_w}x{p.seed_resize_from_h}",
        "Clip skip": p.clip_skip if p.clip_skip > 1 else None,
        "Prompt2": p.refiner_prompt if len(p.refiner_prompt) > 0 else None,
        "Negative2": p.refiner_negative if len(p.refiner_negative) > 0 else None,
        "Styles": "; ".join(p.styles) if p.styles is not None and len(p.styles) > 0 else None,
        "Tiling": p.tiling if p.tiling else None,
        # sdnext
        "Backend": 'Diffusers' if shared.backend == shared.Backend.DIFFUSERS else 'Original',
        "App": 'SD.Next',
        "Version": git_commit,
        "Comment": comment,
        "Operations": '; '.join(ops).replace('"', '') if len(p.ops) > 0 else 'none',
    }
    if 'txt2img' in p.ops:
        pass
    if shared.backend == shared.Backend.ORIGINAL:
        args["Variation seed"] = all_subseeds[index] if p.subseed_strength > 0 else None
        args["Variation strength"] = p.subseed_strength if p.subseed_strength > 0 else None
    if 'hires' in p.ops or 'upscale' in p.ops:
        args["Second pass"] = p.enable_hr
        args["Hires force"] = p.hr_force
        args["Hires steps"] = p.hr_second_pass_steps
        args["Hires upscaler"] = p.hr_upscaler
        args["Hires upscale"] = p.hr_scale
        args["Hires resize"] = f"{p.hr_resize_x}x{p.hr_resize_y}"
        args["Hires size"] = f"{p.hr_upscale_to_x}x{p.hr_upscale_to_y}"
        args["Denoising strength"] = p.denoising_strength
        args["Hires sampler"] = p.hr_sampler_name
        args["Image CFG scale"] = p.image_cfg_scale
        args["CFG rescale"] = p.diffusers_guidance_rescale
    if 'refine' in p.ops:
        args["Second pass"] = p.enable_hr
        args["Refiner"] = None if (not shared.opts.add_model_name_to_info) or (not shared.sd_refiner) or (not shared.sd_refiner.sd_checkpoint_info.model_name) else shared.sd_refiner.sd_checkpoint_info.model_name.replace(',', '').replace(':', '')
        args['Image CFG scale'] = p.image_cfg_scale
        args['Refiner steps'] = p.refiner_steps
        args['Refiner start'] = p.refiner_start
        args["Hires steps"] = p.hr_second_pass_steps
        args["Hires sampler"] = p.hr_sampler_name
        args["CFG rescale"] = p.diffusers_guidance_rescale
    if 'img2img' in p.ops or 'inpaint' in p.ops:
        args["Init image size"] = f"{getattr(p, 'init_img_width', 0)}x{getattr(p, 'init_img_height', 0)}"
        args["Init image hash"] = getattr(p, 'init_img_hash', None)
        args['Resize scale'] = getattr(p, 'scale_by', None)
        args["Mask weight"] = getattr(p, "inpainting_mask_weight", shared.opts.inpainting_mask_weight) if p.is_using_inpainting_conditioning else None
        args["Denoising strength"] = getattr(p, 'denoising_strength', None)
        if args["Size"] is None:
            args["Size"] = args["Init image size"]
        # lookup by index
        if getattr(p, 'resize_mode', None) is not None:
            args['Resize mode'] = shared.resize_modes[p.resize_mode] if shared.resize_modes[p.resize_mode] != 'None' else None
    if 'face' in p.ops:
        args["Face restoration"] = shared.opts.face_restoration_model
    if 'color' in p.ops:
        args["Color correction"] = True
    # embeddings
    if sd_hijack is not None and hasattr(sd_hijack.model_hijack, 'embedding_db') and len(sd_hijack.model_hijack.embedding_db.embeddings_used) > 0: # this is for original hijaacked models only, diffusers are handled separately
        args["Embeddings"] = ', '.join(sd_hijack.model_hijack.embedding_db.embeddings_used)
    # samplers
    args["Sampler ENSD"] = shared.opts.eta_noise_seed_delta if shared.opts.eta_noise_seed_delta != 0 and sd_samplers_common.is_sampler_using_eta_noise_seed_delta(p) else None
    args["Sampler ENSM"] = p.initial_noise_multiplier if getattr(p, 'initial_noise_multiplier', 1.0) != 1.0 else None
    args['Sampler order'] = shared.opts.schedulers_solver_order if shared.opts.schedulers_solver_order != shared.opts.data_labels.get('schedulers_solver_order').default else None
    if shared.backend == shared.Backend.DIFFUSERS:
        args['Sampler beta schedule'] = shared.opts.schedulers_beta_schedule if shared.opts.schedulers_beta_schedule != shared.opts.data_labels.get('schedulers_beta_schedule').default else None
        args['Sampler beta start'] = shared.opts.schedulers_beta_start if shared.opts.schedulers_beta_start != shared.opts.data_labels.get('schedulers_beta_start').default else None
        args['Sampler beta end'] = shared.opts.schedulers_beta_end if shared.opts.schedulers_beta_end != shared.opts.data_labels.get('schedulers_beta_end').default else None
        args['Sampler DPM solver'] = shared.opts.schedulers_dpm_solver if shared.opts.schedulers_dpm_solver != shared.opts.data_labels.get('schedulers_dpm_solver').default else None
    if shared.backend == shared.Backend.ORIGINAL:
        args['Sampler brownian'] = shared.opts.schedulers_brownian_noise if shared.opts.schedulers_brownian_noise != shared.opts.data_labels.get('schedulers_brownian_noise').default else None
        args['Sampler discard'] = shared.opts.schedulers_discard_penultimate if shared.opts.schedulers_discard_penultimate != shared.opts.data_labels.get('schedulers_discard_penultimate').default else None
        args['Sampler dyn threshold'] = shared.opts.schedulers_use_thresholding if shared.opts.schedulers_use_thresholding != shared.opts.data_labels.get('schedulers_use_thresholding').default else None
        args['Sampler karras'] = shared.opts.schedulers_use_karras if shared.opts.schedulers_use_karras != shared.opts.data_labels.get('schedulers_use_karras').default else None
        args['Sampler low order'] = shared.opts.schedulers_use_loworder if shared.opts.schedulers_use_loworder != shared.opts.data_labels.get('schedulers_use_loworder').default else None
        args['Sampler quantization'] = shared.opts.enable_quantization if shared.opts.enable_quantization != shared.opts.data_labels.get('enable_quantization').default else None
        args['Sampler sigma'] = shared.opts.schedulers_sigma if shared.opts.schedulers_sigma != shared.opts.data_labels.get('schedulers_sigma').default else None
        args['Sampler sigma min'] = shared.opts.s_min if shared.opts.s_min != shared.opts.data_labels.get('s_min').default else None
        args['Sampler sigma max'] = shared.opts.s_max if shared.opts.s_max != shared.opts.data_labels.get('s_max').default else None
        args['Sampler sigma churn'] = shared.opts.s_churn if shared.opts.s_churn != shared.opts.data_labels.get('s_churn').default else None
        args['Sampler sigma uncond'] = shared.opts.s_churn if shared.opts.s_churn != shared.opts.data_labels.get('s_churn').default else None
        args['Sampler sigma noise'] = shared.opts.s_noise if shared.opts.s_noise != shared.opts.data_labels.get('s_noise').default else None
        args['Sampler sigma tmin'] = shared.opts.s_tmin if shared.opts.s_tmin != shared.opts.data_labels.get('s_tmin').default else None
    # tome
    token_merging_ratio = p.get_token_merging_ratio()
    token_merging_ratio_hr = p.get_token_merging_ratio(for_hr=True) if p.enable_hr else None
    args['ToMe'] = token_merging_ratio if token_merging_ratio != 0 else None
    args['ToMe hires'] = token_merging_ratio_hr if token_merging_ratio_hr != 0 else None

    args.update(p.extra_generation_params)
    params_text = ", ".join([k if k == v else f'{k}: {generation_parameters_copypaste.quote(v)}' for k, v in args.items() if v is not None])
    negative_prompt_text = f"\nNegative prompt: {all_negative_prompts[index]}" if all_negative_prompts[index] else ""
    infotext = f"{all_prompts[index]}{negative_prompt_text}\n{params_text}".strip()
    return infotext
