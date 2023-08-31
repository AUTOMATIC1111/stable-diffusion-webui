import inspect
import typing
import torch
import modules.devices as devices
import modules.shared as shared
import modules.sd_samplers as sd_samplers
import modules.sd_models as sd_models
import modules.sd_vae as sd_vae
import modules.taesd.sd_vae_taesd as sd_vae_taesd
import modules.images as images
from modules.lora_diffusers import lora_state, unload_diffusers_lora
from modules.processing import StableDiffusionProcessing
import modules.prompt_parser_diffusers as prompt_parser_diffusers


try:
    import diffusers
except Exception as ex:
    shared.log.error(f'Failed to import diffusers: {ex}')


def process_diffusers(p: StableDiffusionProcessing, seeds, prompts, negative_prompts):
    results = []
    if p.enable_hr and p.hr_upscaler != 'None' and p.denoising_strength > 0 and len(getattr(p, 'init_images', [])) == 0:
        p.is_hr_pass = True
    is_refiner_enabled = p.enable_hr and p.refiner_steps > 0 and shared.sd_refiner is not None

    def hires_resize(latents): # input=latents output=pil
        latent_upscaler = shared.latent_upscale_modes.get(p.hr_upscaler, None)
        shared.log.info(f'Hires: upscaler={p.hr_upscaler} width={p.hr_upscale_to_x} height={p.hr_upscale_to_y} images={latents.shape[0]}')
        if latent_upscaler is not None:
            latents = torch.nn.functional.interpolate(latents, size=(p.hr_upscale_to_y // 8, p.hr_upscale_to_x // 8), mode=latent_upscaler["mode"], antialias=latent_upscaler["antialias"])
        first_pass_images = vae_decode(latents=latents, model=shared.sd_model, full_quality=True, output_type='pil')
        p.init_images = []
        for first_pass_image in first_pass_images:
            init_image = images.resize_image(1, first_pass_image, p.hr_upscale_to_x, p.hr_upscale_to_y, upscaler_name=p.hr_upscaler) if latent_upscaler is None else first_pass_image
            p.init_images.append(init_image)
        p.width = p.hr_upscale_to_x
        p.height = p.hr_upscale_to_y

    def save_intermediate(latents, suffix):
        for i in range(len(latents)):
            from modules.processing import create_infotext
            info=create_infotext(p, p.all_prompts, p.all_seeds, p.all_subseeds, [], iteration=p.iteration, position_in_batch=i)
            decoded = vae_decode(latents=latents, model=shared.sd_model, output_type='pil', full_quality=p.full_quality)
            for i in range(len(decoded)):
                images.save_image(decoded[i], path=p.outpath_samples, basename="", seed=seeds[i], prompt=prompts[i], extension=shared.opts.samples_format, info=info, p=p, suffix=suffix)

    def diffusers_callback(_step: int, _timestep: int, latents: torch.FloatTensor):
        shared.state.sampling_step += 1
        shared.state.sampling_steps = p.steps
        if p.is_hr_pass:
            shared.state.sampling_steps += p.hr_second_pass_steps
        shared.state.current_latent = latents
        if shared.state.interrupted or shared.state.skipped:
            raise AssertionError('Interrupted...')

    def full_vae_decode(latents, model):
        shared.log.debug(f'VAE decode: name={sd_vae.loaded_vae_file if sd_vae.loaded_vae_file is not None else "baked"} dtype={model.vae.dtype} upcast={model.vae.config.get("force_upcast", None)} images={latents.shape[0]}')
        if shared.opts.diffusers_move_unet and not model.has_accelerate:
            shared.log.debug('Moving to CPU: model=UNet')
            unet_device = model.unet.device
            model.unet.to(devices.cpu)
            devices.torch_gc()
        if not shared.cmd_opts.lowvram and not shared.opts.diffusers_seq_cpu_offload:
            model.vae.to(devices.device)
        latents.to(model.vae.device)
        decoded = model.vae.decode(latents / model.vae.config.scaling_factor, return_dict=False)[0]
        if shared.opts.diffusers_move_unet and not model.has_accelerate:
            model.unet.to(unet_device)
        return decoded

    def taesd_vae_decode(latents):
        shared.log.debug(f'VAE decode: name=TAESD images={latents.shape[0]}')
        decoded = torch.zeros((len(latents), 3, p.height, p.width), dtype=devices.dtype_vae, device=devices.device)
        for i in range(len(output.images)):
            decoded[i] = (sd_vae_taesd.decode(latents[i]) * 2.0) - 1.0
        return decoded

    def vae_decode(latents, model, output_type='np', full_quality=True):
        if not torch.is_tensor(latents): # already decoded
            return latents
        if latents.shape[0] == 0:
            shared.log.error(f'VAE nothing to decode: {latents.shape}')
            return []
        if shared.state.interrupted or shared.state.skipped:
            return []
        if not hasattr(model, 'vae'):
            shared.log.error('VAE not found in model')
            return []
        if len(latents.shape) == 3: # lost a batch dim in hires
            latents = latents.unsqueeze(0)
        if full_quality:
            decoded = full_vae_decode(latents=latents, model=shared.sd_model)
        else:
            decoded = taesd_vae_decode(latents=latents)
        imgs = model.image_processor.postprocess(decoded, output_type=output_type)
        return imgs

    def fix_prompts(prompts, negative_prompts, prompts_2, negative_prompts_2):
        if type(prompts) is str:
            prompts = [prompts]
        if type(negative_prompts) is str:
            negative_prompts = [negative_prompts]
        while len(negative_prompts) < len(prompts):
            negative_prompts.append(negative_prompts[-1])
        if type(prompts_2) is str:
            prompts_2 = [prompts_2]
        if type(prompts_2) is list:
            while len(prompts_2) < len(prompts):
                prompts_2.append(prompts_2[-1])
        if type(negative_prompts_2) is str:
            negative_prompts_2 = [negative_prompts_2]
        if type(negative_prompts_2) is list:
            while len(negative_prompts_2) < len(prompts_2):
                negative_prompts_2.append(negative_prompts_2[-1])
        return prompts, negative_prompts, prompts_2, negative_prompts_2

    def set_pipeline_args(model, prompts: list, negative_prompts: list, prompts_2: typing.Optional[list]=None, negative_prompts_2: typing.Optional[list]=None, is_refiner: bool=False, desc:str='', **kwargs):
        if hasattr(model, "set_progress_bar_config"):
            model.set_progress_bar_config(bar_format='Progress {rate_fmt}{postfix} {bar} {percentage:3.0f}% {n_fmt}/{total_fmt} {elapsed} {remaining} '+desc, ncols=80, colour='#327fba')
        args = {}
        pipeline = model
        signature = inspect.signature(type(pipeline).__call__)
        possible = signature.parameters.keys()
        generator_device = devices.cpu if shared.opts.diffusers_generator_device == "cpu" else shared.device
        generator = [torch.Generator(generator_device).manual_seed(s) for s in seeds]
        prompt_embed = None
        pooled = None
        negative_embed = None
        negative_pooled = None
        prompts, negative_prompts, prompts_2, negative_prompts_2 = fix_prompts(prompts, negative_prompts, prompts_2, negative_prompts_2)
        if shared.opts.prompt_attention in {'Compel parser', 'Full parser'}:
            prompt_embed, pooled, negative_embed, negative_pooled = prompt_parser_diffusers.compel_encode_prompts(model, prompts, negative_prompts, prompts_2, negative_prompts_2, is_refiner, kwargs.pop("clip_skip", None))
        if 'prompt' in possible:
            if hasattr(model, 'text_encoder') and 'prompt_embeds' in possible and prompt_embed is not None:
                args['prompt_embeds'] = prompt_embed
                if not is_refiner and shared.sd_model_type == "sdxl":
                    args['pooled_prompt_embeds'] = pooled
                    # args['prompt_2'] = None # Cannot pass prompts when passing embeds
                if is_refiner and shared.sd_refiner_type == "sdxl":
                    args['pooled_prompt_embeds'] = pooled
                    # args['prompt_2'] = None # Cannot pass prompts when passing embeds
            else:
                args['prompt'] = prompts
        if 'negative_prompt' in possible:
            if hasattr(model, 'text_encoder') and 'negative_prompt_embeds' in possible and negative_embed is not None:
                args['negative_prompt_embeds'] = negative_embed
                if not is_refiner and shared.sd_model_type == "sdxl":
                    args['negative_pooled_prompt_embeds'] = negative_pooled
                    # args['negative_prompt_2'] = None
                if is_refiner and shared.sd_refiner_type == "sdxl":
                    args['negative_pooled_prompt_embeds'] = negative_pooled
                    # args['negative_prompt_2'] = None
            else:
                args['negative_prompt'] = negative_prompts
        if 'guidance_scale' in possible:
            args['guidance_scale'] = p.cfg_scale
        if 'generator' in possible:
            args['generator'] = generator
        if 'output_type' in possible:
            args['output_type'] = 'np'
        if 'callback_steps' in possible:
            args['callback_steps'] = 1
        if 'callback' in possible:
            args['callback'] = diffusers_callback
        for arg in kwargs:
            if arg in possible:
                args[arg] = kwargs[arg]
            else:
                pass
                # shared.log.debug(f'Diffuser not supported: pipeline={pipeline.__class__.__name__} task={sd_models.get_diffusers_task(model)} arg={arg}')
        # shared.log.debug(f'Diffuser pipeline: {pipeline.__class__.__name__} possible={possible}')
        clean = args.copy()
        clean.pop('callback', None)
        clean.pop('callback_steps', None)
        if 'image' in clean:
            clean['image'] = type(clean['image'])
        if 'mask_image' in clean:
            clean['mask_image'] = type(clean['mask_image'])
        if 'prompt' in clean:
            clean['prompt'] = len(clean['prompt'])
        if 'negative_prompt' in clean:
            clean['negative_prompt'] = len(clean['negative_prompt'])
        if 'prompt_embeds' in clean:
            clean['prompt_embeds'] = clean['prompt_embeds'].shape if torch.is_tensor(clean['prompt_embeds']) else type(clean['prompt_embeds'])
        if 'pooled_prompt_embeds' in clean:
            clean['pooled_prompt_embeds'] = clean['pooled_prompt_embeds'].shape if torch.is_tensor(clean['pooled_prompt_embeds']) else type(clean['pooled_prompt_embeds'])
        if 'negative_prompt_embeds' in clean:
            clean['negative_prompt_embeds'] = clean['negative_prompt_embeds'].shape if torch.is_tensor(clean['negative_prompt_embeds']) else type(clean['negative_prompt_embeds'])
        if 'negative_pooled_prompt_embeds' in clean:
            clean['negative_pooled_prompt_embeds'] = clean['negative_pooled_prompt_embeds'].shape if torch.is_tensor(clean['negative_pooled_prompt_embeds']) else type(clean['negative_pooled_prompt_embeds'])
        clean['generator'] = generator_device
        shared.log.debug(f'Diffuser pipeline: {pipeline.__class__.__name__} task={sd_models.get_diffusers_task(model)} set={clean}')
        return args

    def recompile_model(hires=False):
        if shared.opts.cuda_compile and shared.opts.cuda_compile_backend != 'none':
            if shared.opts.cuda_compile_backend == "openvino_fx":
                compile_height = p.height if not hires else p.hr_upscale_to_y
                compile_width = p.width if not hires else p.hr_upscale_to_x
                if (not hasattr(shared.sd_model, "compiled_model_state") or (not shared.sd_model.compiled_model_state.first_pass
                and (shared.sd_model.compiled_model_state.height != compile_height or shared.sd_model.compiled_model_state.width != compile_width
                or shared.sd_model.compiled_model_state.batch_size != p.batch_size))):
                    shared.log.info("OpenVINO: Resolution change detected")
                    shared.log.info("OpenVINO: Recompiling base model")
                    sd_models.unload_model_weights(op='model')
                    sd_models.reload_model_weights(op='model')
                    if is_refiner_enabled:
                        shared.log.info("OpenVINO: Recompiling refiner")
                        sd_models.unload_model_weights(op='refiner')
                        sd_models.reload_model_weights(op='refiner')
                shared.sd_model.compiled_model_state.height = compile_height
                shared.sd_model.compiled_model_state.width = compile_width
                shared.sd_model.compiled_model_state.batch_size = p.batch_size
                shared.sd_model.compiled_model_state.first_pass = False
            else:
                pass #Can be implemented for TensorRT or Olive
        else:
            pass #Do nothing if compile is disabled

    is_karras_compatible = shared.sd_model.__class__.__init__.__annotations__.get("scheduler", None) == diffusers.schedulers.scheduling_utils.KarrasDiffusionSchedulers
    use_sampler = p.sampler_name if not p.is_hr_pass else p.latent_sampler
    if (not hasattr(shared.sd_model.scheduler, 'name')) or (shared.sd_model.scheduler.name != use_sampler) and (use_sampler != 'Default') and is_karras_compatible:
        sampler = sd_samplers.all_samplers_map.get(use_sampler, None)
        if sampler is None:
            sampler = sd_samplers.all_samplers_map.get("UniPC")
        sd_samplers.create_sampler(sampler.name, shared.sd_model) # TODO(Patrick): For wrapped pipelines this is currently a no-op
        sampler_options = f'type:{shared.opts.schedulers_prediction_type} ' if shared.opts.schedulers_prediction_type != 'default' else ''
        sampler_options += 'no_karras ' if not shared.opts.schedulers_use_karras else ''
        sampler_options += 'no_low_order' if not shared.opts.schedulers_use_loworder else ''
        sampler_options += 'dynamic_thresholding' if shared.opts.schedulers_use_thresholding else ''
        sampler_options += f'solver:{shared.opts.schedulers_dpm_solver}' if shared.opts.schedulers_dpm_solver != 'sde-dpmsolver++' else ''
        sampler_options += f'beta:{shared.opts.schedulers_beta_schedule}:{shared.opts.schedulers_beta_start}:{shared.opts.schedulers_beta_end}' if shared.opts.schedulers_beta_schedule != 'default' else ''
        p.extra_generation_params['Sampler options'] = sampler_options if len(sampler_options) > 0 else None
        p.extra_generation_params['Pipeline'] = shared.sd_model.__class__.__name__

    cross_attention_kwargs={}
    if len(getattr(p, 'init_images', [])) > 0:
        while len(p.init_images) < len(prompts):
            p.init_images.append(p.init_images[-1])
    if lora_state['active']:
        cross_attention_kwargs['scale'] = lora_state['multiplier']
    task_specific_kwargs={}
    if sd_models.get_diffusers_task(shared.sd_model) == sd_models.DiffusersTaskType.TEXT_2_IMAGE:
        p.ops.append('txt2img')
        task_specific_kwargs = {"height": p.height, "width": p.width}
    elif sd_models.get_diffusers_task(shared.sd_model) == sd_models.DiffusersTaskType.IMAGE_2_IMAGE:
        p.ops.append('img2img')
        task_specific_kwargs = {"image": p.init_images, "strength": p.denoising_strength}
    elif sd_models.get_diffusers_task(shared.sd_model) == sd_models.DiffusersTaskType.INPAINTING:
        p.ops.append('inpaint')
        task_specific_kwargs = {"image": p.init_images, "mask_image": p.mask, "strength": p.denoising_strength, "height": p.height, "width": p.width}

    if shared.state.interrupted or shared.state.skipped:
        unload_diffusers_lora()
        return results

    recompile_model()

    if shared.opts.diffusers_move_base and not shared.sd_model.has_accelerate:
        shared.sd_model.to(devices.device)

    is_img2img = (sd_models.get_diffusers_task(shared.sd_model) == sd_models.DiffusersTaskType.IMAGE_2_IMAGE or sd_models.get_diffusers_task(shared.sd_model) == sd_models.DiffusersTaskType.INPAINTING)
    use_refiner_start = (is_refiner_enabled and not p.is_hr_pass and not is_img2img and p.refiner_start > 0 and p.refiner_start < 1)
    use_denoise_start = (is_img2img and p.refiner_start > 0 and p.refiner_start < 1)

    base_args = set_pipeline_args(
        model=shared.sd_model,
        prompts=prompts,
        negative_prompts=negative_prompts,
        prompts_2=[p.refiner_prompt] if len(p.refiner_prompt) > 0 else prompts,
        negative_prompts_2=[p.refiner_negative] if len(p.refiner_negative) > 0 else negative_prompts,
        num_inference_steps=int(p.steps // p.refiner_start + 1) if use_refiner_start else int(p.steps // (1 - p.refiner_start)) if use_denoise_start else int(p.steps // p.denoising_strength + 1) if is_img2img else p.steps,
        eta=shared.opts.eta_ddim,
        guidance_rescale=p.diffusers_guidance_rescale,
        denoising_start=0 if use_refiner_start else p.refiner_start if use_denoise_start else None,
        denoising_end=p.refiner_start if use_refiner_start else 1 if use_denoise_start else None,
        output_type='latent' if hasattr(shared.sd_model, 'vae') else 'np',
        is_refiner=False,
        clip_skip=p.clip_skip,
        desc='Base',
        **task_specific_kwargs
    )
    p.extra_generation_params['CFG rescale'] = p.diffusers_guidance_rescale
    p.extra_generation_params["Eta DDIM"] = shared.opts.eta_ddim if shared.opts.eta_ddim is not None and shared.opts.eta_ddim > 0 else None
    try:
        output = shared.sd_model(**base_args) # pylint: disable=not-callable
    except AssertionError as e:
        shared.log.info(e)

    if lora_state['active']:
        p.extra_generation_params['LoRA method'] = shared.opts.diffusers_lora_loader
        unload_diffusers_lora()

    if shared.state.interrupted or shared.state.skipped:
        return results

    # optional hires pass
    if p.is_hr_pass:
        p.init_hr()
        recompile_model(hires=True)
        if p.width != p.hr_upscale_to_x or p.height != p.hr_upscale_to_y:
            if shared.opts.save and not p.do_not_save_samples and shared.opts.save_images_before_highres_fix and hasattr(shared.sd_model, 'vae'):
                save_intermediate(latents=output.images, suffix="-before-hires")
            hires_resize(latents=output.images)
            sd_models.set_diffuser_pipe(shared.sd_model, sd_models.DiffusersTaskType.IMAGE_2_IMAGE)
            p.ops.append('hires')
            hires_args = set_pipeline_args(
                model=shared.sd_model,
                prompts=prompts,
                negative_prompts=negative_prompts,
                prompts_2=[p.refiner_prompt] if len(p.refiner_prompt) > 0 else prompts,
                negative_prompts_2=[p.refiner_negative] if len(p.refiner_negative) > 0 else negative_prompts,
                num_inference_steps=int(p.hr_second_pass_steps // p.denoising_strength + 1),
                eta=shared.opts.eta_ddim,
                guidance_scale=p.image_cfg_scale if p.image_cfg_scale is not None else p.cfg_scale,
                guidance_rescale=p.diffusers_guidance_rescale,
                output_type='latent' if hasattr(shared.sd_model, 'vae') else 'np',
                is_refiner=False,
                clip_skip=p.clip_skip,
                image=p.init_images,
                strength=p.denoising_strength,
                desc='Hires',
            )
            try:
                output = shared.sd_model(**hires_args) # pylint: disable=not-callable
            except AssertionError as e:
                shared.log.info(e)

    # optional refiner pass or decode
    if is_refiner_enabled:
        if shared.opts.save and not p.do_not_save_samples and shared.opts.save_images_before_refiner and hasattr(shared.sd_model, 'vae'):
            save_intermediate(latents=output.images, suffix="-before-refiner")
        if shared.opts.diffusers_move_base and not shared.sd_model.has_accelerate:
            shared.log.debug('Moving to CPU: model=base')
            shared.sd_model.to(devices.cpu)
            devices.torch_gc()

        if (not hasattr(shared.sd_refiner.scheduler, 'name')) or (shared.sd_refiner.scheduler.name != p.latent_sampler) and (p.sampler_name != 'Default'):
            sampler = sd_samplers.all_samplers_map.get(p.latent_sampler, None)
            if sampler is None:
                sampler = sd_samplers.all_samplers_map.get("UniPC")
            sd_samplers.create_sampler(sampler.name, shared.sd_refiner) # TODO(Patrick): For wrapped pipelines this is currently a no-op

        if shared.state.interrupted or shared.state.skipped:
            return results

        if shared.opts.diffusers_move_refiner and not shared.sd_refiner.has_accelerate:
            shared.sd_refiner.to(devices.device)
        p.ops.append('refine')
        for i in range(len(output.images)):
            refiner_args = set_pipeline_args(
                model=shared.sd_refiner,
                prompts=[p.refiner_prompt] if len(p.refiner_prompt) > 0 else prompts[i],
                negative_prompts=[p.refiner_negative] if len(p.refiner_negative) > 0 else negative_prompts[i],
                num_inference_steps=int(p.refiner_steps // (1 - p.refiner_start)) if p.refiner_start > 0 and p.refiner_start < 1 else int(p.refiner_steps // p.denoising_strength + 1),
                eta=shared.opts.eta_ddim,
                strength=p.denoising_strength,
                guidance_scale=p.image_cfg_scale if p.image_cfg_scale is not None else p.cfg_scale,
                guidance_rescale=p.diffusers_guidance_rescale,
                denoising_start=p.refiner_start if p.refiner_start > 0 and p.refiner_start < 1 else None,
                denoising_end=1 if p.refiner_start > 0 and p.refiner_start < 1 else None,
                image=output.images[i],
                output_type='latent' if hasattr(shared.sd_refiner, 'vae') else 'np',
                is_refiner=True,
                clip_skip=p.clip_skip,
                desc='Refiner',
            )
            try:
                refiner_output = shared.sd_refiner(**refiner_args) # pylint: disable=not-callable
            except AssertionError as e:
                shared.log.info(e)

            p.extra_generation_params['Image CFG scale'] = p.image_cfg_scale if p.image_cfg_scale is not None else None
            p.extra_generation_params['Refiner steps'] = p.refiner_steps
            p.extra_generation_params['Refiner start'] = p.refiner_start
            p.extra_generation_params["Hires steps"] = p.hr_second_pass_steps
            p.extra_generation_params["Secondary sampler"] = p.latent_sampler

            if not shared.state.interrupted and not shared.state.skipped:
                refiner_images = vae_decode(latents=refiner_output.images, model=shared.sd_refiner, full_quality=True)
                for refiner_image in refiner_images:
                    results.append(refiner_image)

        if shared.opts.diffusers_move_refiner and not shared.sd_refiner.has_accelerate:
            shared.log.debug('Moving to CPU: model=refiner')
            shared.sd_refiner.to(devices.cpu)
            devices.torch_gc()

    # final decode since there is no refiner
    if not is_refiner_enabled:
        results = vae_decode(latents=output.images, model=shared.sd_model, full_quality=p.full_quality)

    return results
