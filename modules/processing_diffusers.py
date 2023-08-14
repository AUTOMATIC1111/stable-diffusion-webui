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

    def diffusers_callback(step: int, _timestep: int, latents: torch.FloatTensor):
        shared.state.sampling_step = step
        shared.state.sampling_steps = p.steps
        shared.state.current_latent = latents

    def vae_decode(latents, model, output_type='np'):
        if hasattr(model, 'vae') and torch.is_tensor(latents):
            if latents.shape[0] == 0:
                shared.log.error(f'VAE nothing to decode: {latents.shape}')
                return []
            shared.log.debug(f'Diffusers VAE decode: name={sd_vae.loaded_vae_file} dtype={model.vae.dtype} upcast={model.vae.config.get("force_upcast", None)} images={latents.shape[0]}')
            if shared.opts.diffusers_move_unet and not model.has_accelerate:
                shared.log.debug('Diffusers: Moving UNet to CPU')
                unet_device = model.unet.device
                model.unet.to(devices.cpu)
                devices.torch_gc()
            latents.to(model.vae.device)
            decoded = model.vae.decode(latents / model.vae.config.scaling_factor, return_dict=False)[0]
            imgs = model.image_processor.postprocess(decoded, output_type=output_type)
            if shared.opts.diffusers_move_unet and not model.has_accelerate:
                model.unet.to(unet_device)
            return imgs
        else:
            return latents

    def taesd_vae_decode(latents, model, output_type='np'):
        shared.log.debug('Diffusers VAE decode: name=TAESD')
        decoded = torch.zeros((len(latents), 3, p.height, p.width), dtype=devices.dtype_vae, device=devices.device)
        for i in range(len(output.images)):
            decoded[i] = (sd_vae_taesd.decode(latents[i]) * 2.0) - 1.0
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

    def set_pipeline_args(model, prompts: list, negative_prompts: list, prompts_2: typing.Optional[list]=None, negative_prompts_2: typing.Optional[list]=None, is_refiner: bool=False, **kwargs):
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
        if shared.opts.data['prompt_attention'] in {'Compel parser', 'Full parser'}:
            prompt_embed, pooled, negative_embed, negative_pooled = prompt_parser_diffusers.compel_encode_prompts(model,
                                                                                                                  prompts,
                                                                                                                  negative_prompts,
                                                                                                                  prompts_2,
                                                                                                                  negative_prompts_2,
                                                                                                                  is_refiner,
                                                                                                                  kwargs.pop("clip_skip", None))
        if 'prompt' in possible:
            if hasattr(model, 'text_encoder') and 'prompt_embeds' in possible and prompt_embed is not None:
                args['prompt_embeds'] = prompt_embed
                if shared.sd_model_type == "sdxl":
                    args['pooled_prompt_embeds'] = pooled
                    args['prompt_2'] = None #Cannot pass prompts when passing embeds
            else:
                args['prompt'] = prompts
        if 'negative_prompt' in possible:
            if hasattr(model, 'text_encoder') and 'negative_prompt_embeds' in possible and negative_embed is not None:
                args['negative_prompt_embeds'] = negative_embed
                if shared.sd_model_type == "sdxl":
                    args['negative_pooled_prompt_embeds'] = negative_pooled
                    args['negative_prompt_2'] = None
            else:
                args['negative_prompt'] = negative_prompts
        if 'num_inference_steps' in possible:
            args['num_inference_steps'] = p.steps
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
        if 'cross_attention_kwargs' in possible and lora_state['active'] and shared.opts.diffusers_lora_loader == "diffusers default":
            args['cross_attention_kwargs'] = { 'scale': lora_state['multiplier'][0]}
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
            clean['prompt_embeds'] = clean['prompt_embeds'].shape
        if 'pooled_prompt_embeds' in clean:
            clean['pooled_prompt_embeds'] = clean['pooled_prompt_embeds'].shape
        if 'negative_prompt_embeds' in clean:
            clean['negative_prompt_embeds'] = clean['negative_prompt_embeds'].shape
        if 'negative_pooled_prompt_embeds' in clean:
            clean['negative_pooled_prompt_embeds'] = clean['negative_pooled_prompt_embeds'].shape
        clean['generator'] = generator_device
        shared.log.debug(f'Diffuser pipeline: {pipeline.__class__.__name__} task={sd_models.get_diffusers_task(model)} set={clean}')
        return args

    is_karras_compatible = shared.sd_model.__class__.__init__.__annotations__.get("scheduler", None) == diffusers.schedulers.scheduling_utils.KarrasDiffusionSchedulers
    if (not hasattr(shared.sd_model.scheduler, 'name')) or (shared.sd_model.scheduler.name != p.sampler_name) and (p.sampler_name != 'Default') and is_karras_compatible:
        sampler = sd_samplers.all_samplers_map.get(p.sampler_name, None)
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

    if shared.opts.diffusers_move_base and not shared.sd_model.has_accelerate:
        shared.sd_model.to(devices.device)

    refiner_enabled = shared.sd_refiner is not None and p.enable_hr
    pipe_args = set_pipeline_args(
        model=shared.sd_model,
        prompts=prompts,
        negative_prompts=negative_prompts,
        prompts_2=[p.refiner_prompt] if len(p.refiner_prompt) > 0 else prompts,
        negative_prompts_2=[p.refiner_negative] if len(p.refiner_negative) > 0 else negative_prompts,
        eta=shared.opts.eta_ddim,
        guidance_rescale=p.diffusers_guidance_rescale,
        denoising_start=0 if refiner_enabled and p.refiner_start > 0 and p.refiner_start < 1 else None,
        denoising_end=p.refiner_start if refiner_enabled and p.refiner_start > 0 and p.refiner_start < 1 else None,
        output_type='latent' if hasattr(shared.sd_model, 'vae') else 'np',
        is_refiner=False,
        clip_skip=p.clip_skip,
        **task_specific_kwargs
    )
    p.extra_generation_params['CFG rescale'] = p.diffusers_guidance_rescale
    p.extra_generation_params["Eta DDIM"] = shared.opts.eta_ddim if shared.opts.eta_ddim is not None and shared.opts.eta_ddim > 0 else None
    output = shared.sd_model(**pipe_args) # pylint: disable=not-callable
    if shared.state.interrupted or shared.state.skipped:
        unload_diffusers_lora()
        return results

    if shared.sd_refiner is None or not p.enable_hr:
        output.images = vae_decode(output.images, shared.sd_model) if p.full_quality else taesd_vae_decode(output.images, shared.sd_model)

    if lora_state['active']:
        p.extra_generation_params['Lora method'] = shared.opts.diffusers_lora_loader
        unload_diffusers_lora()

    if refiner_enabled:
        for i in range(len(output.images)):
            if shared.opts.save and not p.do_not_save_samples and shared.opts.save_images_before_refiner and hasattr(shared.sd_model, 'vae'):
                from modules.processing import create_infotext
                info=create_infotext(p, p.all_prompts, p.all_seeds, p.all_subseeds, [], iteration=p.iteration, position_in_batch=i)
                decoded = vae_decode(output.images, shared.sd_model, output_type='pil')
                for i in range(len(decoded)):
                    images.save_image(decoded[i], path=p.outpath_samples, basename="", seed=seeds[i], prompt=prompts[i], extension=shared.opts.samples_format, info=info, p=p, suffix="-before-refiner")

        if (shared.opts.diffusers_move_base or shared.cmd_opts.medvram or shared.opts.diffusers_model_cpu_offload) and not (shared.cmd_opts.lowvram or shared.opts.diffusers_seq_cpu_offload):
            shared.log.debug('Diffusers: Moving base model to CPU')
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
            pipe_args = set_pipeline_args(
                model=shared.sd_refiner,
                prompts=[p.refiner_prompt] if len(p.refiner_prompt) > 0 else prompts[i],
                negative_prompts=[p.refiner_negative] if len(p.refiner_negative) > 0 else negative_prompts[i],
                num_inference_steps=p.hr_second_pass_steps,
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
            )
            refiner_output = shared.sd_refiner(**pipe_args) # pylint: disable=not-callable
            p.extra_generation_params['Refiner CFG scale'] = p.image_cfg_scale if p.image_cfg_scale is not None else None
            p.extra_generation_params['Refiner start'] = p.refiner_start
            p.extra_generation_params["Hires steps"] = p.hr_second_pass_steps

            if not shared.state.interrupted and not shared.state.skipped:
                refiner_images = vae_decode(refiner_output.images, shared.sd_refiner)
                results.append(refiner_images[0])

        if shared.opts.diffusers_move_refiner and not shared.sd_refiner.has_accelerate:
            shared.log.debug('Diffusers: Moving refiner model to CPU')
            shared.sd_refiner.to(devices.cpu)
            devices.torch_gc()
    else:
        results = output.images

    if p.is_hr_pass:
        shared.log.warning('Diffusers not implemented: hires fix')

    return results
