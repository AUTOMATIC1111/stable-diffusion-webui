import inspect
import torch
import modules.devices as devices
import modules.shared as shared
import modules.sd_samplers as sd_samplers
import modules.sd_models as sd_models
import modules.images as images
from modules.lora_diffusers import lora_state, unload_diffusers_lora
from modules.processing import StableDiffusionProcessing


def encode_prompt(encoder, prompt):
    cfg = encoder.config
    # TODO implement similar hijack for diffusers text encoder but following diffusers pipeline.encode_prompt concepts
    # from modules import sd_hijack_clip
    # model.text_encoder = sd_hijack_clip.FrozenCLIPEmbedderWithCustomWords(model.text_encoder, None)
    shared.log.debug(f'Diffuser encoder: {encoder.__class__.__name__} dict={getattr(cfg, "vocab_size", None)} layers={getattr(cfg, "num_hidden_layers", None)} tokens={getattr(cfg, "max_position_embeddings", None)}')
    embeds = prompt
    return embeds


def process_diffusers(p: StableDiffusionProcessing, seeds, prompts, negative_prompts):
    results = []

    def diffusers_callback(step: int, _timestep: int, latents: torch.FloatTensor):
        shared.state.sampling_step = step
        shared.state.sampling_steps = p.steps
        shared.state.current_latent = latents

    def vae_decode(latents, model, output_type='np'):
        if hasattr(model, 'vae') and torch.is_tensor(latents):
            shared.log.debug(f'Diffusers VAE decode: name={model.vae.config.get("_name_or_path", "default")} dtype={model.vae.dtype} upcast={model.vae.config.get("force_upcast", None)}')
            latents.to(model.vae.device)
            decoded = model.vae.decode(latents / model.vae.config.scaling_factor, return_dict=False)[0]
            imgs = model.image_processor.postprocess(decoded, output_type=output_type)
            return imgs
        else:
            return latents


    def set_pipeline_args(model, prompt, negative_prompt, **kwargs):
        args = {}
        pipeline = model.main if model.__class__.__name__ == 'PriorPipeline' else model
        signature = inspect.signature(type(pipeline).__call__)
        possible = signature.parameters.keys()
        generator_device = 'cpu' if shared.opts.diffusers_generator_device == "cpu" else shared.device
        generator = [torch.Generator(generator_device).manual_seed(s) for s in seeds]
        if 'prompt' in possible:
            if hasattr(model, 'text_encoder') and 'prompt_embeds' in possible:
                # args['prompt_embeds'] = encode_prompt(model, prompt)
                args['prompt'] = prompt
            else:
                args['prompt'] = prompt
        if 'negative_prompt' in possible:
            if hasattr(model, 'text_encoder') and 'negative_prompt_embeds' in possible:
                # args['negative_prompt_embeds'] = encode_prompt(model, negative_prompt)
                args['negative_prompt'] = negative_prompt
            else:
                args['negative_prompt'] = negative_prompt
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
        if 'cross_attention_kwargs' in possible and lora_state['active']:
            args['cross_attention_kwargs'] = { 'scale': lora_state['multiplier']}
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
        clean['generator'] = generator_device
        shared.log.debug(f'Diffuser pipeline: {pipeline.__class__.__name__} task={sd_models.get_diffusers_task(model)} set={clean}')
        return args

    if (not hasattr(shared.sd_model.scheduler, 'name')) or (shared.sd_model.scheduler.name != p.sampler_name) and (p.sampler_name != 'Default'):
        sampler = sd_samplers.all_samplers_map.get(p.sampler_name, None)
        if sampler is None:
            sampler = sd_samplers.all_samplers_map.get("UniPC")
        sd_samplers.create_sampler(sampler.name, shared.sd_model) # TODO(Patrick): For wrapped pipelines this is currently a no-op

    cross_attention_kwargs={}
    if lora_state['active']:
        cross_attention_kwargs['scale'] = lora_state['multiplier']
    task_specific_kwargs={}
    if sd_models.get_diffusers_task(shared.sd_model) == sd_models.DiffusersTaskType.TEXT_2_IMAGE:
        task_specific_kwargs = {"height": p.height, "width": p.width}
    elif sd_models.get_diffusers_task(shared.sd_model) == sd_models.DiffusersTaskType.IMAGE_2_IMAGE:
        task_specific_kwargs = {"image": p.init_images, "strength": p.denoising_strength}
    elif sd_models.get_diffusers_task(shared.sd_model) == sd_models.DiffusersTaskType.INPAINTING:
        task_specific_kwargs = {"image": p.init_images, "mask_image": p.mask, "strength": p.denoising_strength}

    # TODO diffusers use transformers for prompt parsing
    # from modules.prompt_parser import parse_prompt_attention
    # parsed_prompt = [parse_prompt_attention(prompt) for prompt in prompts]

    if shared.state.interrupted or shared.state.skipped:
        return results

    if shared.opts.diffusers_move_base:
        shared.sd_model.to(devices.device)

    pipe_args = set_pipeline_args(
        model=shared.sd_model,
        prompt=prompts,
        negative_prompt=negative_prompts,
        prompt_2=[p.refiner_prompt] if len(p.refiner_prompt) > 0 else prompts,
        negative_prompt_2=[p.refiner_negative] if len(p.refiner_negative) > 0 else negative_prompts,
        eta=shared.opts.eta_ddim,
        guidance_rescale=p.diffusers_guidance_rescale,
        denoising_start=p.refiner_denoise_start,
        denoising_end=p.refiner_denoise_end,
        # aesthetic_score=shared.opts.diffusers_aesthetics_score,
        output_type='latent' if hasattr(shared.sd_model, 'vae') else 'np',
        **task_specific_kwargs
    )
    output = shared.sd_model(**pipe_args) # pylint: disable=not-callable

    if shared.state.interrupted or shared.state.skipped:
        return results

    if shared.sd_refiner is None or not p.enable_hr:
        output.images = vae_decode(output.images, shared.sd_model)

    if shared.sd_refiner is not None and p.enable_hr:
        for i in range(len(output.images)):
            if shared.opts.save and not p.do_not_save_samples and shared.opts.save_images_before_refiner and hasattr(shared.sd_model, 'vae'):
                from modules.processing import create_infotext
                info=create_infotext(p, p.all_prompts, p.all_seeds, p.all_subseeds, [], iteration=p.iteration, position_in_batch=i)
                decoded = vae_decode(output.images, shared.sd_model, output_type='pil')
                for i in range(len(decoded)):
                    images.save_image(decoded[i], path=p.outpath_samples, basename="", seed=seeds[i], prompt=prompts[i], extension=shared.opts.samples_format, info=info, p=p, suffix="-before-refiner")

        if shared.opts.diffusers_move_base:
            shared.log.debug('Diffusers: Moving base model to CPU')
            shared.sd_model.to('cpu')
            devices.torch_gc()

        if (not hasattr(shared.sd_refiner.scheduler, 'name')) or (shared.sd_refiner.scheduler.name != p.latent_sampler) and (p.sampler_name != 'Default'):
            sampler = sd_samplers.all_samplers_map.get(p.latent_sampler, None)
            if sampler is None:
                sampler = sd_samplers.all_samplers_map.get("UniPC")
            sd_samplers.create_sampler(sampler.name, shared.sd_refiner) # TODO(Patrick): For wrapped pipelines this is currently a no-op

        if shared.state.interrupted or shared.state.skipped:
            return results

        if shared.opts.diffusers_move_refiner:
            shared.sd_refiner.to(devices.device)

        for i in range(len(output.images)):
            pipe_args = set_pipeline_args(
                model=shared.sd_refiner,
                prompt=[p.refiner_prompt] if len(p.refiner_prompt) > 0 else prompts[i],
                negative_prompt=[p.refiner_negative] if len(p.refiner_negative) > 0 else negative_prompts[i],
                num_inference_steps=p.hr_second_pass_steps,
                eta=shared.opts.eta_ddim,
                strength=p.denoising_strength,
                guidance_scale=p.image_cfg_scale if p.image_cfg_scale is not None else p.cfg_scale,
                guidance_rescale=p.diffusers_guidance_rescale,
                # aesthetic_score=shared.opts.diffusers_aesthetics_score,
                denoising_start=p.refiner_denoise_start,
                denoising_end=p.refiner_denoise_end,
                image=output.images[i],
                output_type='latent' if hasattr(shared.sd_refiner, 'vae') else 'np',
            )
            refiner_output = shared.sd_refiner(**pipe_args) # pylint: disable=not-callable
            if not shared.state.interrupted and not shared.state.skipped:
                refiner_images = vae_decode(refiner_output.images, shared.sd_refiner)
                results.append(refiner_images[0])

        if shared.opts.diffusers_move_refiner:
            shared.log.debug('Diffusers: Moving refiner model to CPU')
            shared.sd_refiner.to('cpu')
    else:
        results = output.images

    if p.is_hr_pass:
        shared.log.warning('Diffusers not implemented: hires fix')

    if lora_state['active']:
        unload_diffusers_lora()

    return results
