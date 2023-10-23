import os
import time
import math
import inspect
import typing
import torch
import torchvision.transforms.functional as TF
import diffusers
import modules.devices as devices
import modules.shared as shared
import modules.sd_samplers as sd_samplers
import modules.sd_models as sd_models
import modules.sd_vae as sd_vae
import modules.taesd.sd_vae_taesd as sd_vae_taesd
import modules.images as images
import modules.errors as errors
from modules.processing import StableDiffusionProcessing
import modules.prompt_parser_diffusers as prompt_parser_diffusers
from modules.sd_hijack_hypertile import hypertile_set


def process_diffusers(p: StableDiffusionProcessing, seeds, prompts, negative_prompts):
    results = []
    is_refiner_enabled = p.enable_hr and p.refiner_steps > 0 and p.refiner_start > 0 and p.refiner_start < 1 and shared.sd_refiner is not None

    if hasattr(p, 'init_images') and len(p.init_images) > 0:
        tgt_width, tgt_height = 8 * math.ceil(p.init_images[0].width / 8), 8 * math.ceil(p.init_images[0].height / 8)
        if p.init_images[0].width != tgt_width or p.init_images[0].height != tgt_height:
            shared.log.debug(f'Resizing init images: original={p.init_images[0].width}x{p.init_images[0].height} target={tgt_width}x{tgt_height}')
            p.init_images = [images.resize_image(1, image, tgt_width, tgt_height, upscaler_name=None) for image in p.init_images]
            p.height = tgt_height
            p.width = tgt_width
            hypertile_set(p)
            if getattr(p, 'mask', None) is not None:
                p.mask = images.resize_image(1, p.mask, tgt_width, tgt_height, upscaler_name=None)
            if getattr(p, 'mask_for_overlay', None) is not None:
                p.mask_for_overlay = images.resize_image(1, p.mask_for_overlay, tgt_width, tgt_height, upscaler_name=None)

    def hires_resize(latents): # input=latents output=pil
        latent_upscaler = shared.latent_upscale_modes.get(p.hr_upscaler, None)
        shared.log.info(f'Hires: upscaler={p.hr_upscaler} width={p.hr_upscale_to_x} height={p.hr_upscale_to_y} images={latents.shape[0]}')
        if latent_upscaler is not None:
            latents = torch.nn.functional.interpolate(latents, size=(p.hr_upscale_to_y // 8, p.hr_upscale_to_x // 8), mode=latent_upscaler["mode"], antialias=latent_upscaler["antialias"])
        first_pass_images = vae_decode(latents=latents, model=shared.sd_model, full_quality=p.full_quality, output_type='pil')
        p.init_images = []
        for img in first_pass_images:
            if latent_upscaler is None:
                init_image = images.resize_image(1, img, p.hr_upscale_to_x, p.hr_upscale_to_y, upscaler_name=p.hr_upscaler)
            else:
                init_image = img
            # if is_refiner_enabled:
            #    init_image = vae_encode(init_image, model=shared.sd_model, full_quality=p.full_quality)
            p.init_images.append(init_image)
        return p.init_images

    def save_intermediate(latents, suffix):
        for i in range(len(latents)):
            from modules.processing import create_infotext
            info=create_infotext(p, p.all_prompts, p.all_seeds, p.all_subseeds, [], iteration=p.iteration, position_in_batch=i)
            decoded = vae_decode(latents=latents, model=shared.sd_model, output_type='pil', full_quality=p.full_quality)
            for j in range(len(decoded)):
                images.save_image(decoded[j], path=p.outpath_samples, basename="", seed=seeds[i], prompt=prompts[i], extension=shared.opts.samples_format, info=info, p=p, suffix=suffix)

    def diffusers_callback(step: int, _timestep: int, latents: torch.FloatTensor):
        shared.state.sampling_step = step
        if p.is_hr_pass:
            shared.state.job = 'hires'
            shared.state.sampling_steps = p.hr_second_pass_steps # add optional hires
        elif p.is_refiner_pass:
            shared.state.job = 'refine'
            shared.state.sampling_steps = calculate_refiner_steps() # add optional refiner
        else:
            shared.state.sampling_steps = p.steps # base steps
        shared.state.current_latent = latents
        if shared.state.interrupted or shared.state.skipped:
            raise AssertionError('Interrupted...')
        if shared.state.paused:
            shared.log.debug('Sampling paused')
            while shared.state.paused:
                if shared.state.interrupted or shared.state.skipped:
                    raise AssertionError('Interrupted...')
                time.sleep(0.1)

    def full_vae_decode(latents, model):
        t0 = time.time()
        if shared.opts.diffusers_move_unet and not getattr(model, 'has_accelerate', False):
            shared.log.debug('Moving to CPU: model=UNet')
            unet_device = model.unet.device
            model.unet.to(devices.cpu)
            devices.torch_gc()
        if not shared.cmd_opts.lowvram and not shared.opts.diffusers_seq_cpu_offload:
            model.vae.to(devices.device)
        latents.to(model.vae.device)

        upcast = (model.vae.dtype == torch.float16) and model.vae.config.force_upcast and hasattr(model, 'upcast_vae')
        if upcast: # this is done by diffusers automatically if output_type != 'latent'
            model.upcast_vae()
            latents = latents.to(next(iter(model.vae.post_quant_conv.parameters())).dtype)

        decoded = model.vae.decode(latents / model.vae.config.scaling_factor, return_dict=False)[0]
        if shared.opts.diffusers_move_unet and not getattr(model, 'has_accelerate', False):
            model.unet.to(unet_device)
        t1 = time.time()
        shared.log.debug(f'VAE decode: name={sd_vae.loaded_vae_file if sd_vae.loaded_vae_file is not None else "baked"} dtype={model.vae.dtype} upcast={upcast} images={latents.shape[0]} latents={latents.shape} time={round(t1-t0, 3)}s')
        return decoded

    def full_vae_encode(image, model):
        shared.log.debug(f'VAE encode: name={sd_vae.loaded_vae_file if sd_vae.loaded_vae_file is not None else "baked"} dtype={model.vae.dtype} upcast={model.vae.config.get("force_upcast", None)}')
        if shared.opts.diffusers_move_unet and not getattr(model, 'has_accelerate', False):
            shared.log.debug('Moving to CPU: model=UNet')
            unet_device = model.unet.device
            model.unet.to(devices.cpu)
            devices.torch_gc()
        if not shared.cmd_opts.lowvram and not shared.opts.diffusers_seq_cpu_offload:
            model.vae.to(devices.device)
        encoded = model.vae.encode(image.to(model.vae.device, model.vae.dtype))
        if shared.opts.diffusers_move_unet and not getattr(model, 'has_accelerate', False):
            model.unet.to(unet_device)
        return encoded

    def taesd_vae_decode(latents):
        shared.log.debug(f'VAE decode: name=TAESD images={len(latents)} latents={latents.shape}')
        if len(latents) == 0:
            return []
        decoded = torch.zeros((len(latents), 3, latents.shape[2] * 8, latents.shape[3] * 8), dtype=devices.dtype_vae, device=devices.device)
        for i in range(len(output.images)):
            decoded[i] = (sd_vae_taesd.decode(latents[i]) * 2.0) - 1.0
        return decoded

    def taesd_vae_encode(image):
        shared.log.debug(f'VAE encode: name=TAESD image={image.shape}')
        encoded = sd_vae_taesd.encode(image)
        return encoded

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

    def vae_encode(image, model, full_quality=True): # pylint: disable=unused-variable
        if shared.state.interrupted or shared.state.skipped:
            return []
        if not hasattr(model, 'vae'):
            shared.log.error('VAE not found in model')
            return []
        tensor = TF.to_tensor(image.convert("RGB")).unsqueeze(0).to(devices.device, devices.dtype_vae)
        if full_quality:
            latents = full_vae_encode(image=tensor, model=shared.sd_model)
        else:
            latents = taesd_vae_encode(image=tensor)
        return latents

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

    def task_specific_kwargs(model):
        task_args = {}
        if sd_models.get_diffusers_task(model) == sd_models.DiffusersTaskType.TEXT_2_IMAGE:
            p.ops.append('txt2img')
            task_args = {"height": 8 * math.ceil(p.height / 8), "width": 8 * math.ceil(p.width / 8)}
        elif sd_models.get_diffusers_task(model) == sd_models.DiffusersTaskType.IMAGE_2_IMAGE and len(getattr(p, 'init_images' ,[])) > 0:
            p.ops.append('img2img')
            task_args = {"image": p.init_images, "strength": p.denoising_strength}
        elif sd_models.get_diffusers_task(model) == sd_models.DiffusersTaskType.INSTRUCT and len(getattr(p, 'init_images' ,[])) > 0:
            p.ops.append('instruct')
            task_args = {"height": 8 * math.ceil(p.height / 8), "width": 8 * math.ceil(p.width / 8), "image": p.init_images, "strength": p.denoising_strength}
        elif sd_models.get_diffusers_task(model) == sd_models.DiffusersTaskType.INPAINTING and len(getattr(p, 'init_images' ,[])) > 0:
            p.ops.append('inpaint')
            if getattr(p, 'mask', None) is None:
                p.mask = TF.to_pil_image(torch.ones_like(TF.to_tensor(p.init_images[0]))).convert("L")
            width = 8 * math.ceil(p.init_images[0].width / 8)
            height = 8 * math.ceil(p.init_images[0].height / 8)
            task_args = {"image": p.init_images, "mask_image": p.mask, "strength": p.denoising_strength, "height": height, "width": width}
        return task_args

    def set_pipeline_args(model, prompts: list, negative_prompts: list, prompts_2: typing.Optional[list]=None, negative_prompts_2: typing.Optional[list]=None, desc:str='', **kwargs):

        if hasattr(model, "set_progress_bar_config"):
            model.set_progress_bar_config(bar_format='Progress {rate_fmt}{postfix} {bar} {percentage:3.0f}% {n_fmt}/{total_fmt} {elapsed} {remaining} ' + '\x1b[38;5;71m' + desc, ncols=80, colour='#327fba')
        args = {}
        signature = inspect.signature(type(model).__call__)
        possible = signature.parameters.keys()
        generator_device = devices.cpu if shared.opts.diffusers_generator_device == "cpu" else shared.device
        generator = [torch.Generator(generator_device).manual_seed(s) for s in seeds]
        prompt_embed = None
        pooled = None
        negative_embed = None
        negative_pooled = None
        prompts, negative_prompts, prompts_2, negative_prompts_2 = fix_prompts(prompts, negative_prompts, prompts_2, negative_prompts_2)
        parser = 'Fixed attention'
        if shared.opts.prompt_attention != 'Fixed attention' and 'StableDiffusion' in model.__class__.__name__:
            try:
                prompt_embed, pooled, negative_embed, negative_pooled = prompt_parser_diffusers.encode_prompts(model, prompts, negative_prompts, kwargs.pop("clip_skip", None))
                parser = shared.opts.prompt_attention
            except Exception as e:
                shared.log.error(f'Prompt parser encode: {e}')
                if os.environ.get('SD_PROMPT_DEBUG', None) is not None:
                    errors.display(e, 'Prompt parser encode')
        if 'prompt' in possible:
            if hasattr(model, 'text_encoder') and 'prompt_embeds' in possible and prompt_embed is not None:
                if type(pooled) == list:
                    pooled = pooled[0]
                if type(negative_pooled) == list:
                    negative_pooled = negative_pooled[0]
                args['prompt_embeds'] = prompt_embed
                if 'XL' in model.__class__.__name__:
                    args['pooled_prompt_embeds'] = pooled
            else:
                args['prompt'] = prompts
        if 'negative_prompt' in possible:
            if hasattr(model, 'text_encoder') and 'negative_prompt_embeds' in possible and negative_embed is not None:
                args['negative_prompt_embeds'] = negative_embed
                if 'XL' in model.__class__.__name__:
                    args['negative_pooled_prompt_embeds'] = negative_pooled
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
            if arg in possible: # add kwargs
                args[arg] = kwargs[arg]
            else:
                pass
        task_kwargs = task_specific_kwargs(model)
        for arg in task_kwargs:
            if arg in possible and arg not in args: # task specific args should not override args
                args[arg] = task_kwargs[arg]
            else:
                pass
                # shared.log.debug(f'Diffuser not supported: pipeline={pipeline.__class__.__name__} task={sd_models.get_diffusers_task(model)} arg={arg}')
        # shared.log.debug(f'Diffuser pipeline: {pipeline.__class__.__name__} possible={possible}')
        hypertile_set(p, hr=len(getattr(p, 'init_images', [])))
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
        clean['parser'] = parser
        shared.log.debug(f'Diffuser pipeline: {model.__class__.__name__} task={sd_models.get_diffusers_task(model)} set={clean}')
        # components = [{ k: getattr(v, 'device', None) } for k, v in model.components.items()]
        # shared.log.debug(f'Diffuser pipeline components: {components}')
        return args

    def recompile_model(hires=False):
        if shared.opts.cuda_compile and shared.opts.cuda_compile_backend != 'none':
            if shared.opts.cuda_compile_backend == "openvino_fx":
                compile_height = p.height if not hires else p.hr_upscale_to_y
                compile_width = p.width if not hires else p.hr_upscale_to_x
                if (shared.compiled_model_state is None or
                (not shared.compiled_model_state.first_pass
                and (shared.compiled_model_state.height != compile_height
                or shared.compiled_model_state.width != compile_width
                or shared.compiled_model_state.batch_size != p.batch_size))):
                    shared.log.info("OpenVINO: Parameter change detected")
                    shared.log.info("OpenVINO: Recompiling base model")
                    sd_models.unload_model_weights(op='model')
                    sd_models.reload_model_weights(op='model')
                    if is_refiner_enabled:
                        shared.log.info("OpenVINO: Recompiling refiner")
                        sd_models.unload_model_weights(op='refiner')
                        sd_models.reload_model_weights(op='refiner')
                shared.compiled_model_state.height = compile_height
                shared.compiled_model_state.width = compile_width
                shared.compiled_model_state.batch_size = p.batch_size
                shared.compiled_model_state.first_pass = False
            else:
                pass #Can be implemented for TensorRT or Olive
        else:
            pass #Do nothing if compile is disabled

    def update_sampler(sd_model, second_pass=False):
        sampler_selection = p.latent_sampler if second_pass else p.sampler_name
        is_karras_compatible = sd_model.__class__.__init__.__annotations__.get("scheduler", None) == diffusers.schedulers.scheduling_utils.KarrasDiffusionSchedulers
        if hasattr(sd_model, 'scheduler') and sampler_selection != 'Default' and is_karras_compatible:
            sampler = sd_samplers.all_samplers_map.get(sampler_selection, None)
            if sampler is None:
                sampler = sd_samplers.all_samplers_map.get("UniPC")
            sd_samplers.create_sampler(sampler.name, sd_model)
            # TODO extra_generation_params add sampler options
            # p.extra_generation_params['Sampler options'] = ''

    recompile_model()
    update_sampler(shared.sd_model)
    p.extra_generation_params['Pipeline'] = shared.sd_model.__class__.__name__

    if len(getattr(p, 'init_images', [])) > 0:
        while len(p.init_images) < len(prompts):
            p.init_images.append(p.init_images[-1])

    if shared.state.interrupted or shared.state.skipped:
        return results

    if shared.opts.diffusers_move_base and not getattr(shared.sd_model, 'has_accelerate', False):
        shared.sd_model.to(devices.device)

    is_img2img = bool(sd_models.get_diffusers_task(shared.sd_model) == sd_models.DiffusersTaskType.IMAGE_2_IMAGE or sd_models.get_diffusers_task(shared.sd_model) == sd_models.DiffusersTaskType.INPAINTING)
    use_refiner_start = bool(is_refiner_enabled and not p.is_hr_pass and not is_img2img and p.refiner_start > 0 and p.refiner_start < 1)
    use_denoise_start = bool(is_img2img and p.refiner_start > 0 and p.refiner_start < 1)

    def calculate_base_steps():
        if use_refiner_start:
            return int(p.steps // p.refiner_start + 1) if shared.sd_model_type == 'sdxl' else p.steps
        elif use_denoise_start and shared.sd_model_type == 'sdxl':
            return int(p.steps // (1 - p.refiner_start))
        elif is_img2img:
            return int(p.steps // p.denoising_strength + 1)
        else:
            return p.steps

    def calculate_refiner_steps():
        refiner_is_sdxl = bool("StableDiffusionXL" in shared.sd_refiner.__class__.__name__)
        if p.refiner_start > 0 and p.refiner_start < 1 and refiner_is_sdxl:
            refiner_steps = int(p.refiner_steps // (1 - p.refiner_start))
        else:
            refiner_steps = int(p.refiner_steps // p.denoising_strength + 1) if refiner_is_sdxl else p.refiner_steps
        p.refiner_steps = min(99, refiner_steps)
        return p.refiner_steps

    # pipeline type is set earlier in processing, but check for sanity
    if sd_models.get_diffusers_task(shared.sd_model) != sd_models.DiffusersTaskType.TEXT_2_IMAGE and len(getattr(p, 'init_images' ,[])) == 0: # reset pipeline
        shared.sd_model = sd_models.set_diffuser_pipe(shared.sd_model, sd_models.DiffusersTaskType.TEXT_2_IMAGE)
    base_args = set_pipeline_args(
        model=shared.sd_model,
        prompts=prompts,
        negative_prompts=negative_prompts,
        prompts_2=[p.refiner_prompt] if len(p.refiner_prompt) > 0 else prompts,
        negative_prompts_2=[p.refiner_negative] if len(p.refiner_negative) > 0 else negative_prompts,
        num_inference_steps=calculate_base_steps(),
        eta=shared.opts.scheduler_eta,
        guidance_rescale=p.diffusers_guidance_rescale,
        denoising_start=0 if use_refiner_start else p.refiner_start if use_denoise_start else None,
        denoising_end=p.refiner_start if use_refiner_start else 1 if use_denoise_start else None,
        output_type='latent' if hasattr(shared.sd_model, 'vae') else 'np',
        clip_skip=p.clip_skip,
        desc='Base',
    )
    p.extra_generation_params['CFG rescale'] = p.diffusers_guidance_rescale
    p.extra_generation_params["Sampler Eta"] = shared.opts.scheduler_eta if shared.opts.scheduler_eta is not None and shared.opts.scheduler_eta > 0 and shared.opts.scheduler_eta < 1 else None
    try:
        output = shared.sd_model(**base_args) # pylint: disable=not-callable
    except AssertionError as e:
        shared.log.info(e)
    except ValueError as e:
        shared.state.interrupted = True
        shared.log.error(f'Processing: {e}')
        if shared.cmd_opts.debug:
            errors.display(e, 'Processing')

    if hasattr(shared.sd_model, 'embedding_db') and len(shared.sd_model.embedding_db.embeddings_used) > 0:
        p.extra_generation_params['Embeddings'] = ', '.join(shared.sd_model.embedding_db.embeddings_used)

    if shared.state.interrupted or shared.state.skipped:
        return results

    # optional hires pass
    if p.enable_hr and p.hr_upscaler != 'None' and p.denoising_strength > 0 and len(getattr(p, 'init_images', [])) == 0:
        p.is_hr_pass = True
    latent_scale_mode = shared.latent_upscale_modes.get(p.hr_upscaler, None) if (hasattr(p, "hr_upscaler") and p.hr_upscaler is not None) else shared.latent_upscale_modes.get(shared.latent_upscale_default_mode, "None")
    if p.is_hr_pass:
        p.init_hr()
        if p.width != p.hr_upscale_to_x or p.height != p.hr_upscale_to_y:
            p.ops.append('upscale')
            if shared.opts.save and not p.do_not_save_samples and shared.opts.save_images_before_highres_fix and hasattr(shared.sd_model, 'vae'):
                save_intermediate(latents=output.images, suffix="-before-hires")
            output.images = hires_resize(latents=output.images)
            if latent_scale_mode is not None or p.hr_force:
                p.ops.append('hires')
                shared.sd_model = sd_models.set_diffuser_pipe(shared.sd_model, sd_models.DiffusersTaskType.IMAGE_2_IMAGE)
                recompile_model(hires=True)
                update_sampler(shared.sd_model, second_pass=True)
                hires_args = set_pipeline_args(
                    model=shared.sd_model,
                    prompts=[p.refiner_prompt] if len(p.refiner_prompt) > 0 else prompts,
                    negative_prompts=[p.refiner_negative] if len(p.refiner_negative) > 0 else negative_prompts,
                    prompts_2=[p.refiner_prompt] if len(p.refiner_prompt) > 0 else prompts,
                    negative_prompts_2=[p.refiner_negative] if len(p.refiner_negative) > 0 else negative_prompts,
                    num_inference_steps=int(p.hr_second_pass_steps // p.denoising_strength + 1),
                    eta=shared.opts.scheduler_eta,
                    guidance_scale=p.image_cfg_scale if p.image_cfg_scale is not None else p.cfg_scale,
                    guidance_rescale=p.diffusers_guidance_rescale,
                    output_type='latent' if hasattr(shared.sd_model, 'vae') else 'np',
                    clip_skip=p.clip_skip,
                    image=p.init_images,
                    strength=p.denoising_strength,
                    desc='Hires',
                )
                try:
                    output = shared.sd_model(**hires_args) # pylint: disable=not-callable
                except AssertionError as e:
                    shared.log.info(e)
                p.init_images = []
        p.is_hr_pass = False

    # optional refiner pass or decode
    if is_refiner_enabled:
        if shared.opts.save and not p.do_not_save_samples and shared.opts.save_images_before_refiner and hasattr(shared.sd_model, 'vae'):
            save_intermediate(latents=output.images, suffix="-before-refiner")
        if shared.opts.diffusers_move_base and not getattr(shared.sd_model, 'has_accelerate', False):
            shared.log.debug('Moving to CPU: model=base')
            shared.sd_model.to(devices.cpu)
            devices.torch_gc()

        update_sampler(shared.sd_refiner, second_pass=True)

        if shared.state.interrupted or shared.state.skipped:
            return results

        if shared.opts.diffusers_move_refiner and not getattr(shared.sd_refiner, 'has_accelerate', False):
            shared.sd_refiner.to(devices.device)
        p.ops.append('refine')
        p.is_refiner_pass = True
        shared.sd_model = sd_models.set_diffuser_pipe(shared.sd_model, sd_models.DiffusersTaskType.TEXT_2_IMAGE)
        shared.sd_refiner = sd_models.set_diffuser_pipe(shared.sd_refiner, sd_models.DiffusersTaskType.IMAGE_2_IMAGE)
        for i in range(len(output.images)):
            image = output.images[i]
            noise_level = round(350 * p.denoising_strength)
            output_type='latent' if hasattr(shared.sd_refiner, 'vae') else 'np'
            if shared.sd_refiner.__class__.__name__ == 'StableDiffusionUpscalePipeline':
                image = vae_decode(latents=image, model=shared.sd_model, full_quality=p.full_quality, output_type='pil')
                p.extra_generation_params['Noise level'] = noise_level
                output_type = 'np'
            calculate_refiner_steps()
            refiner_args = set_pipeline_args(
                model=shared.sd_refiner,
                prompts=[p.refiner_prompt] if len(p.refiner_prompt) > 0 else prompts[i],
                negative_prompts=[p.refiner_negative] if len(p.refiner_negative) > 0 else negative_prompts[i],
                num_inference_steps=p.refiner_steps,
                eta=shared.opts.scheduler_eta,
                # strength=p.denoising_strength,
                noise_level=noise_level, # StableDiffusionUpscalePipeline only
                guidance_scale=p.image_cfg_scale if p.image_cfg_scale is not None else p.cfg_scale,
                guidance_rescale=p.diffusers_guidance_rescale,
                denoising_start=p.refiner_start if p.refiner_start > 0 and p.refiner_start < 1 else None,
                denoising_end=1 if p.refiner_start > 0 and p.refiner_start < 1 else None,
                image=image,
                output_type=output_type,
                clip_skip=p.clip_skip,
                desc='Refiner',
            )
            try:
                refiner_output = shared.sd_refiner(**refiner_args) # pylint: disable=not-callable
            except AssertionError as e:
                shared.log.info(e)

            if not shared.state.interrupted and not shared.state.skipped:
                refiner_images = vae_decode(latents=refiner_output.images, model=shared.sd_refiner, full_quality=True)
                for refiner_image in refiner_images:
                    results.append(refiner_image)

        if shared.opts.diffusers_move_refiner and not getattr(shared.sd_refiner, 'has_accelerate', False):
            shared.log.debug('Moving to CPU: model=refiner')
            shared.sd_refiner.to(devices.cpu)
            devices.torch_gc()
        p.is_refiner_pass = True

    # final decode since there is no refiner
    if not is_refiner_enabled:
        if output is not None and output.images is not None and len(output.images) > 0:
            results = vae_decode(latents=output.images, model=shared.sd_model, full_quality=p.full_quality)
        else:
            shared.log.warning('Processing returned no results')
            results = []

    return results
