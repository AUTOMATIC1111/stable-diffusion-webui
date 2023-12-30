import os
import time
import math
import inspect
import typing
import torch
import torchvision.transforms.functional as TF
import modules.devices as devices
import modules.shared as shared
import modules.sd_samplers as sd_samplers
import modules.sd_models as sd_models
import modules.images as images
import modules.errors as errors
from modules.processing import StableDiffusionProcessing, create_random_tensors
import modules.prompt_parser_diffusers as prompt_parser_diffusers
from modules.sd_hijack_hypertile import hypertile_set
from modules.processing_correction import correction_callback
from modules.processing_vae import vae_encode, vae_decode


debug = shared.log.trace if os.environ.get('SD_DIFFUSERS_DEBUG', None) is not None else lambda *args, **kwargs: None
debug('Trace: DIFFUSERS')
debug_steps = shared.log.trace if os.environ.get('SD_STEPS_DEBUG', None) is not None else lambda *args, **kwargs: None
debug_steps('Trace: STEPS')


def process_diffusers(p: StableDiffusionProcessing):
    debug(f'Process diffusers args: {vars(p)}')
    results = []

    def is_txt2img():
        return sd_models.get_diffusers_task(shared.sd_model) == sd_models.DiffusersTaskType.TEXT_2_IMAGE

    def is_refiner_enabled():
        return p.enable_hr and p.refiner_steps > 0 and p.refiner_start > 0 and p.refiner_start < 1 and shared.sd_refiner is not None

    if getattr(p, 'init_images', None) is not None and len(p.init_images) > 0:
        tgt_width, tgt_height = 8 * math.ceil(p.init_images[0].width / 8), 8 * math.ceil(p.init_images[0].height / 8)
        if p.init_images[0].width != tgt_width or p.init_images[0].height != tgt_height:
            shared.log.debug(f'Resizing init images: original={p.init_images[0].width}x{p.init_images[0].height} target={tgt_width}x{tgt_height}')
            p.init_images = [images.resize_image(1, image, tgt_width, tgt_height, upscaler_name=None) for image in p.init_images]
            p.height = tgt_height
            p.width = tgt_width
            hypertile_set(p)
        if getattr(p, 'mask', None) is not None and p.mask.size != (tgt_width, tgt_height):
            p.mask = images.resize_image(1, p.mask, tgt_width, tgt_height, upscaler_name=None)
        if getattr(p, 'mask_for_overlay', None) is not None and p.mask_for_overlay.size != (tgt_width, tgt_height):
            p.mask_for_overlay = images.resize_image(1, p.mask_for_overlay, tgt_width, tgt_height, upscaler_name=None)

    def hires_resize(latents): # input=latents output=pil
        if not torch.is_tensor(latents):
            shared.log.warning('Hires: input is not tensor')
            first_pass_images = vae_decode(latents=latents, model=shared.sd_model, full_quality=p.full_quality, output_type='pil')
            return first_pass_images
        latent_upscaler = shared.latent_upscale_modes.get(p.hr_upscaler, None)
        shared.log.info(f'Hires: upscaler={p.hr_upscaler} width={p.hr_upscale_to_x} height={p.hr_upscale_to_y} images={latents.shape[0]}')
        if latent_upscaler is not None:
            latents = torch.nn.functional.interpolate(latents, size=(p.hr_upscale_to_y // 8, p.hr_upscale_to_x // 8), mode=latent_upscaler["mode"], antialias=latent_upscaler["antialias"])
        first_pass_images = vae_decode(latents=latents, model=shared.sd_model, full_quality=p.full_quality, output_type='pil')
        resized_images = []
        for img in first_pass_images:
            if latent_upscaler is None:
                resized_image = images.resize_image(1, img, p.hr_upscale_to_x, p.hr_upscale_to_y, upscaler_name=p.hr_upscaler)
            else:
                resized_image = img
            resized_images.append(resized_image)
        return resized_images

    def save_intermediate(latents, suffix):
        for i in range(len(latents)):
            from modules.processing import create_infotext
            info=create_infotext(p, p.all_prompts, p.all_seeds, p.all_subseeds, [], iteration=p.iteration, position_in_batch=i)
            decoded = vae_decode(latents=latents, model=shared.sd_model, output_type='pil', full_quality=p.full_quality)
            for j in range(len(decoded)):
                images.save_image(decoded[j], path=p.outpath_samples, basename="", seed=p.seeds[i], prompt=p.prompts[i], extension=shared.opts.samples_format, info=info, p=p, suffix=suffix)

    def diffusers_callback_legacy(step: int, timestep: int, latents: torch.FloatTensor):
        shared.state.sampling_step = step
        shared.state.current_latent = latents
        latents = correction_callback(p, timestep, {'latents': latents})
        if shared.state.interrupted or shared.state.skipped:
            raise AssertionError('Interrupted...')
        if shared.state.paused:
            shared.log.debug('Sampling paused')
            while shared.state.paused:
                if shared.state.interrupted or shared.state.skipped:
                    raise AssertionError('Interrupted...')
                time.sleep(0.1)

    def diffusers_callback(_pipe, step: int, timestep: int, kwargs: dict):
        shared.state.sampling_step = step
        if shared.state.interrupted or shared.state.skipped:
            raise AssertionError('Interrupted...')
        if shared.state.paused:
            shared.log.debug('Sampling paused')
            while shared.state.paused:
                if shared.state.interrupted or shared.state.skipped:
                    raise AssertionError('Interrupted...')
                time.sleep(0.1)
        if kwargs.get('latents', None) is None:
            return kwargs
        kwargs = correction_callback(p, timestep, kwargs)
        if p.scheduled_prompt and hasattr(kwargs, 'prompt_embeds') and hasattr(kwargs, 'negative_prompt_embeds'):
            try:
                i = (step + 1) % len(p.prompt_embeds)
                kwargs["prompt_embeds"] = p.prompt_embeds[i][0:1].repeat(1, kwargs["prompt_embeds"].shape[0], 1).view(
                      kwargs["prompt_embeds"].shape[0], kwargs["prompt_embeds"].shape[1], -1)
                j = (step + 1) % len(p.negative_embeds)
                kwargs["negative_prompt_embeds"] = p.negative_embeds[j][0:1].repeat(1, kwargs["negative_prompt_embeds"].shape[0], 1).view(
                      kwargs["negative_prompt_embeds"].shape[0], kwargs["negative_prompt_embeds"].shape[1], -1)
            except Exception as e:
                shared.log.debug(f"Callback: {e}")
        shared.state.current_latent = kwargs['latents']
        if shared.cmd_opts.profile and shared.profiler is not None:
            shared.profiler.step()
        return kwargs

    def fix_prompts(prompts, negative_prompts, prompts_2, negative_prompts_2):
        if type(prompts) is str:
            prompts = [prompts]
        if type(negative_prompts) is str:
            negative_prompts = [negative_prompts]
        while len(negative_prompts) < len(prompts):
            negative_prompts.append(negative_prompts[-1])
        while len(prompts) < len(negative_prompts):
            prompts.append(prompts[-1])
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
        is_img2img_model = bool('Zero123' in shared.sd_model.__class__.__name__)
        if sd_models.get_diffusers_task(model) == sd_models.DiffusersTaskType.TEXT_2_IMAGE and not is_img2img_model:
            p.ops.append('txt2img')
            if hasattr(p, 'width') and hasattr(p, 'height'):
                task_args = {
                    'width': 8 * math.ceil(p.width / 8),
                    'height': 8 * math.ceil(p.height / 8),
                }
        elif (sd_models.get_diffusers_task(model) == sd_models.DiffusersTaskType.IMAGE_2_IMAGE or is_img2img_model) and len(getattr(p, 'init_images' ,[])) > 0:
            p.ops.append('img2img')
            task_args = {
                'image': p.init_images,
                'strength': p.denoising_strength,
            }
        elif sd_models.get_diffusers_task(model) == sd_models.DiffusersTaskType.INSTRUCT and len(getattr(p, 'init_images' ,[])) > 0:
            p.ops.append('instruct')
            task_args = {
                'width': 8 * math.ceil(p.width / 8) if hasattr(p, 'width') else None,
                'height': 8 * math.ceil(p.height / 8) if hasattr(p, 'height') else None,
                'image': p.init_images,
                'strength': p.denoising_strength,
            }
        elif (sd_models.get_diffusers_task(model) == sd_models.DiffusersTaskType.INPAINTING or is_img2img_model) and len(getattr(p, 'init_images' ,[])) > 0:
            p.ops.append('inpaint')
            if getattr(p, 'mask', None) is None:
                p.mask = TF.to_pil_image(torch.ones_like(TF.to_tensor(p.init_images[0]))).convert("L")
            p.mask = shared.sd_model.mask_processor.blur(p.mask, blur_factor=p.mask_blur)
            width = 8 * math.ceil(p.init_images[0].width / 8)
            height = 8 * math.ceil(p.init_images[0].height / 8)
            task_args = {
                'image': p.init_images,
                'mask_image': p.mask,
                'strength': p.denoising_strength,
                'height': height,
                'width': width,
                # 'padding_mask_crop': p.inpaint_full_res_padding # done back in main processing method
            }
        if model.__class__.__name__ == 'LatentConsistencyModelPipeline' and hasattr(p, 'init_images') and len(p.init_images) > 0:
            p.ops.append('lcm')
            init_latents = [vae_encode(image, model=shared.sd_model, full_quality=p.full_quality).squeeze(dim=0) for image in p.init_images]
            init_latent = torch.stack(init_latents, dim=0).to(shared.device)
            init_noise = p.denoising_strength * create_random_tensors(init_latent.shape[1:], seeds=p.all_seeds, subseeds=p.all_subseeds, subseed_strength=p.subseed_strength, p=p)
            init_latent = (1 - p.denoising_strength) * init_latent + init_noise
            task_args = {
                'latents': init_latent.to(model.dtype),
                'width': p.width if hasattr(p, 'width') else None,
                'height': p.height if hasattr(p, 'height') else None,
            }
        debug(f'Diffusers task specific args: {task_args}')
        return task_args

    def set_pipeline_args(model, prompts: list, negative_prompts: list, prompts_2: typing.Optional[list]=None, negative_prompts_2: typing.Optional[list]=None, desc:str='', **kwargs):
        t0 = time.time()
        if hasattr(model, "set_progress_bar_config"):
            model.set_progress_bar_config(bar_format='Progress {rate_fmt}{postfix} {bar} {percentage:3.0f}% {n_fmt}/{total_fmt} {elapsed} {remaining} ' + '\x1b[38;5;71m' + desc, ncols=80, colour='#327fba')
        args = {}
        signature = inspect.signature(type(model).__call__)
        possible = signature.parameters.keys()
        debug(f'Diffusers pipeline possible: {possible}')
        if shared.opts.diffusers_generator_device == "Unset":
            generator_device = None
            generator = None
        else:
            generator_device = devices.cpu if shared.opts.diffusers_generator_device == "CPU" else shared.device
            generator = [torch.Generator(generator_device).manual_seed(s) for s in p.seeds]
        prompts, negative_prompts, prompts_2, negative_prompts_2 = fix_prompts(prompts, negative_prompts, prompts_2, negative_prompts_2)
        parser = 'Fixed attention'
        if shared.opts.prompt_attention != 'Fixed attention' and 'StableDiffusion' in model.__class__.__name__:
            try:
                prompt_parser_diffusers.encode_prompts(model, p, prompts, negative_prompts, kwargs.get("num_inference_steps", 1), 0, kwargs.pop("clip_skip", None))
                # prompt_embed, pooled, negative_embed, negative_pooled = , , , ,
                parser = shared.opts.prompt_attention
            except Exception as e:
                shared.log.error(f'Prompt parser encode: {e}')
                if os.environ.get('SD_PROMPT_DEBUG', None) is not None:
                    errors.display(e, 'Prompt parser encode')
        if 'prompt' in possible:
            if hasattr(model, 'text_encoder') and 'prompt_embeds' in possible and len(p.prompt_embeds) > 0 and p.prompt_embeds[0] is not None:
                args['prompt_embeds'] = p.prompt_embeds[0]
                if 'XL' in model.__class__.__name__ and len(getattr(p, 'negative_pooleds', [])) > 0:
                    args['pooled_prompt_embeds'] = p.negative_pooleds[0]
            else:
                args['prompt'] = prompts
        if 'negative_prompt' in possible:
            if hasattr(model, 'text_encoder') and 'negative_prompt_embeds' in possible and len(p.negative_embeds) > 0 and p.negative_embeds[0] is not None:
                args['negative_prompt_embeds'] = p.negative_embeds[0]
                if 'XL' in model.__class__.__name__ and len(getattr(p, 'negative_pooleds', [])) > 0:
                    args['negative_pooled_prompt_embeds'] = p.negative_pooleds[0]
            else:
                args['negative_prompt'] = negative_prompts
        if hasattr(model, 'scheduler') and hasattr(model.scheduler, 'noise_sampler_seed') and hasattr(model.scheduler, 'noise_sampler'):
            model.scheduler.noise_sampler = None # noise needs to be reset instead of using cached values
            model.scheduler.noise_sampler_seed = p.seeds[0] # some schedulers have internal noise generator and do not use pipeline generator
        if 'noise_sampler_seed' in possible:
            args['noise_sampler_seed'] = p.seeds[0]
        if 'guidance_scale' in possible:
            args['guidance_scale'] = p.cfg_scale
        if 'generator' in possible and generator is not None:
            args['generator'] = generator
        if 'output_type' in possible:
            args['output_type'] = 'np'
        if 'callback_steps' in possible:
            args['callback_steps'] = 1
        if 'callback' in possible:
            args['callback'] = diffusers_callback_legacy
        elif 'callback_on_step_end_tensor_inputs' in possible:
            args['callback_on_step_end'] = diffusers_callback
            if 'prompt_embeds' in possible and 'negative_prompt_embeds' in possible:
                args['callback_on_step_end_tensor_inputs'] = ['latents', 'prompt_embeds', 'negative_prompt_embeds']
            else:
                args['callback_on_step_end_tensor_inputs'] = ['latents']
        for arg in kwargs:
            if arg in possible: # add kwargs
                args[arg] = kwargs[arg]
            else:
                pass
        task_kwargs = task_specific_kwargs(model)
        for arg in task_kwargs:
            # if arg in possible and arg not in args: # task specific args should not override args
            if arg in possible:
                args[arg] = task_kwargs[arg]
        task_args = getattr(p, 'task_args', {})
        debug(f'Diffusers task args: {task_args}')
        for k, v in task_args.items():
            if k in possible:
                args[k] = v
            else:
                debug(f'Diffusers unknown task args: {k}={v}')

        hypertile_set(p, hr=len(getattr(p, 'init_images', [])) > 0)
        clean = args.copy()
        clean.pop('callback', None)
        clean.pop('callback_steps', None)
        clean.pop('callback_on_step_end', None)
        clean.pop('callback_on_step_end_tensor_inputs', None)
        if 'latents' in clean:
            clean['latents'] = clean['latents'].shape
        if 'image' in clean:
            clean['image'] = type(clean['image'])
        if 'mask_image' in clean:
            clean['mask_image'] = type(clean['mask_image'])
        if 'masked_image_latents' in clean:
            clean['masked_image_latents'] = type(clean['masked_image_latents'])
        if 'ip_adapter_image' in clean:
            clean['ip_adapter_image'] = type(clean['ip_adapter_image'])
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
        if p.hdr_clamp or p.hdr_center or p.hdr_maximize:
            txt = 'HDR:'
            txt += f' Clamp threshold={p.hdr_threshold} boundary={p.hdr_boundary}' if p.hdr_clamp else ' Clamp off'
            txt += f' Center channel-shift={p.hdr_channel_shift} full-shift={p.hdr_full_shift}' if p.hdr_center else ' Center off'
            txt += f' Maximize boundary={p.hdr_max_boundry} center={p.hdr_max_center}' if p.hdr_maximize else ' Maximize off'
            shared.log.debug(txt)
        # components = [{ k: getattr(v, 'device', None) } for k, v in model.components.items()]
        # shared.log.debug(f'Diffuser pipeline components: {components}')
        if shared.cmd_opts.profile:
            t1 = time.time()
            shared.log.debug(f'Profile: pipeline args: {t1-t0:.2f}')
        debug(f'Diffusers pipeline args: {args}')
        return args

    def recompile_model(hires=False):
        if shared.opts.cuda_compile and shared.opts.cuda_compile_backend != 'none':
            if shared.opts.cuda_compile_backend == "openvino_fx":
                compile_height = p.height if not hires and hasattr(p, 'height') else p.hr_upscale_to_y
                compile_width = p.width if not hires and hasattr(p, 'width') else p.hr_upscale_to_x
                if (shared.compiled_model_state is None or
                (not shared.compiled_model_state.first_pass
                and (shared.compiled_model_state.height != compile_height
                or shared.compiled_model_state.width != compile_width
                or shared.compiled_model_state.batch_size != p.batch_size))):
                    shared.log.info("OpenVINO: Parameter change detected")
                    shared.log.info("OpenVINO: Recompiling base model")
                    sd_models.unload_model_weights(op='model')
                    sd_models.reload_model_weights(op='model')
                    if is_refiner_enabled():
                        shared.log.info("OpenVINO: Recompiling refiner")
                        sd_models.unload_model_weights(op='refiner')
                        sd_models.reload_model_weights(op='refiner')
                shared.compiled_model_state.height = compile_height
                shared.compiled_model_state.width = compile_width
                shared.compiled_model_state.batch_size = p.batch_size

    # Downcast UNET after OpenVINO compile
    def downcast_openvino(op="base"):
        if shared.opts.cuda_compile and shared.opts.cuda_compile_backend == "openvino_fx":
            if shared.compiled_model_state.first_pass and op == "base":
                shared.compiled_model_state.first_pass = False
                if hasattr(shared.sd_model, "unet"):
                    shared.sd_model.unet.to(dtype=torch.float8_e4m3fn)
                    devices.torch_gc(force=True)
            if shared.compiled_model_state.first_pass_refiner and op == "refiner":
                shared.compiled_model_state.first_pass_refiner = False
                if hasattr(shared.sd_refiner, "unet"):
                    shared.sd_refiner.unet.to(dtype=torch.float8_e4m3fn)
                    devices.torch_gc(force=True)

    def update_sampler(sd_model, second_pass=False):
        sampler_selection = p.latent_sampler if second_pass else p.sampler_name
        # is_karras_compatible = sd_model.__class__.__init__.__annotations__.get("scheduler", None) == diffusers.schedulers.scheduling_utils.KarrasDiffusionSchedulers
        if sd_model.__class__.__name__ in ['AmusedPipeline']:
            return # models with their own schedulers
        if hasattr(sd_model, 'scheduler') and sampler_selection != 'Default':
            sampler = sd_samplers.all_samplers_map.get(sampler_selection, None)
            if sampler is None:
                sampler = sd_samplers.all_samplers_map.get("UniPC")
            sd_samplers.create_sampler(sampler.name, sd_model)
            # TODO extra_generation_params add sampler options
            # p.extra_generation_params['Sampler options'] = ''

    if len(getattr(p, 'init_images', [])) > 0:
        while len(p.init_images) < len(p.prompts):
            p.init_images.append(p.init_images[-1])

    if shared.state.interrupted or shared.state.skipped:
        return results

    if shared.opts.diffusers_move_base and not getattr(shared.sd_model, 'has_accelerate', False):
        shared.sd_model.to(devices.device)

    # pipeline type is set earlier in processing, but check for sanity
    has_images = len(getattr(p, 'init_images' ,[])) > 0 or getattr(p, 'is_control', False) is True
    if sd_models.get_diffusers_task(shared.sd_model) != sd_models.DiffusersTaskType.TEXT_2_IMAGE and not has_images:
        shared.sd_model = sd_models.set_diffuser_pipe(shared.sd_model, sd_models.DiffusersTaskType.TEXT_2_IMAGE) # reset pipeline
    if hasattr(shared.sd_model, 'unet') and hasattr(shared.sd_model.unet, 'config') and hasattr(shared.sd_model.unet.config, 'in_channels') and shared.sd_model.unet.config.in_channels == 9:
        shared.sd_model = sd_models.set_diffuser_pipe(shared.sd_model, sd_models.DiffusersTaskType.INPAINTING) # force pipeline
        if len(getattr(p, 'init_images' ,[])) == 0:
            p.init_images = [TF.to_pil_image(torch.rand((3, getattr(p, 'height', 512), getattr(p, 'width', 512))))]

    use_refiner_start = is_txt2img() and is_refiner_enabled() and not p.is_hr_pass and p.refiner_start > 0 and p.refiner_start < 1
    use_denoise_start = not is_txt2img() and p.refiner_start > 0 and p.refiner_start < 1

    def calculate_base_steps():
        if not is_txt2img():
            if use_denoise_start and shared.sd_model_type == 'sdxl':
                steps = p.steps // (1 - p.refiner_start)
            elif p.denoising_strength > 0:
                steps = (p.steps // p.denoising_strength) + 1
            else:
                steps = p.steps
        elif use_refiner_start and shared.sd_model_type == 'sdxl':
            steps = (p.steps // p.refiner_start) + 1
        else:
            steps = p.steps
        debug_steps(f'Steps: type=base input={p.steps} output={steps} task={sd_models.get_diffusers_task(shared.sd_model)} refiner={use_refiner_start} denoise={p.denoising_strength} model={shared.sd_model_type}')
        return max(2, int(steps))

    def calculate_hires_steps():
        if p.hr_second_pass_steps > 0:
            steps = (p.hr_second_pass_steps // p.denoising_strength) + 1
        elif p.denoising_strength > 0:
            steps = (p.steps // p.denoising_strength) + 1
        else:
            steps = 0
        debug_steps(f'Steps: type=hires input={p.hr_second_pass_steps} output={steps} denoise={p.denoising_strength} model={shared.sd_model_type}')
        return max(2, int(steps))

    def calculate_refiner_steps():
        if "StableDiffusionXL" in shared.sd_refiner.__class__.__name__:
            if p.refiner_start > 0 and p.refiner_start < 1:
                #steps = p.refiner_steps // (1 - p.refiner_start) # SDXL with denoise strenght
                steps = (p.refiner_steps // (1 - p.refiner_start) // 2) + 1
            elif p.denoising_strength > 0:
                steps = (p.refiner_steps // p.denoising_strength) + 1
            else:
                steps = 0
        else:
            #steps = p.refiner_steps # SD 1.5 with denoise strenght
            steps = (p.refiner_steps * 1.25) + 1
        debug_steps(f'Steps: type=refiner input={p.refiner_steps} output={steps} start={p.refiner_start} denoise={p.denoising_strength}')
        return max(2, int(steps))

    base_args = set_pipeline_args(
        model=shared.sd_model,
        prompts=p.prompts,
        negative_prompts=p.negative_prompts,
        prompts_2=[p.refiner_prompt] if len(p.refiner_prompt) > 0 else p.prompts,
        negative_prompts_2=[p.refiner_negative] if len(p.refiner_negative) > 0 else p.negative_prompts,
        num_inference_steps=calculate_base_steps(),
        eta=shared.opts.scheduler_eta,
        guidance_scale=p.cfg_scale,
        guidance_rescale=p.diffusers_guidance_rescale,
        denoising_start=0 if use_refiner_start else p.refiner_start if use_denoise_start else None,
        denoising_end=p.refiner_start if use_refiner_start else 1 if use_denoise_start else None,
        output_type='latent' if hasattr(shared.sd_model, 'vae') else 'np',
        clip_skip=p.clip_skip,
        desc='Base',
    )
    recompile_model()
    update_sampler(shared.sd_model)
    shared.state.sampling_steps = base_args['num_inference_steps']
    p.extra_generation_params['Pipeline'] = shared.sd_model.__class__.__name__
    if shared.opts.scheduler_eta is not None and shared.opts.scheduler_eta > 0 and shared.opts.scheduler_eta < 1:
        p.extra_generation_params["Sampler Eta"] = shared.opts.scheduler_eta
    try:
        t0 = time.time()
        output = shared.sd_model(**base_args) # pylint: disable=not-callable
        downcast_openvino(op="base")
        if shared.cmd_opts.profile:
            t1 = time.time()
            shared.log.debug(f'Profile: pipeline call: {t1-t0:.2f}')
        if not hasattr(output, 'images') and hasattr(output, 'frames'):
            if hasattr(output.frames[0], 'shape'):
                shared.log.debug(f'Generated: frames={output.frames[0].shape[1]}')
            else:
                shared.log.debug(f'Generated: frames={len(output.frames[0])}')
            output.images = output.frames[0]
    except AssertionError as e:
        shared.log.info(e)
    except ValueError as e:
        shared.state.interrupted = True
        shared.log.error(f'Processing: args={base_args} {e}')
        if shared.cmd_opts.debug:
            errors.display(e, 'Processing')
    except RuntimeError as e:
        shared.state.interrupted = True
        shared.log.error(f'Processing: args={base_args} {e}')
        errors.display(e, 'Processing')

    if hasattr(shared.sd_model, 'embedding_db') and len(shared.sd_model.embedding_db.embeddings_used) > 0:
        p.extra_generation_params['Embeddings'] = ', '.join(shared.sd_model.embedding_db.embeddings_used)

    shared.state.nextjob()
    if shared.state.interrupted or shared.state.skipped:
        return results

    # optional hires pass
    if p.enable_hr and getattr(p, 'hr_upscaler', 'None') != 'None' and len(getattr(p, 'init_images', [])) == 0:
        p.is_hr_pass = True
    latent_scale_mode = shared.latent_upscale_modes.get(p.hr_upscaler, None) if (hasattr(p, "hr_upscaler") and p.hr_upscaler is not None) else shared.latent_upscale_modes.get(shared.latent_upscale_default_mode, "None")
    if p.is_hr_pass:
        p.init_hr()
        prev_job = shared.state.job
        if hasattr(p, 'height') and hasattr(p, 'width') and (p.width != p.hr_upscale_to_x or p.height != p.hr_upscale_to_y):
            p.ops.append('upscale')
            if shared.opts.save and not p.do_not_save_samples and shared.opts.save_images_before_highres_fix and hasattr(shared.sd_model, 'vae'):
                save_intermediate(latents=output.images, suffix="-before-hires")
            shared.state.job = 'upscale'
            output.images = hires_resize(latents=output.images)
            if (latent_scale_mode is not None or p.hr_force) and p.denoising_strength > 0:
                p.ops.append('hires')
                shared.sd_model = sd_models.set_diffuser_pipe(shared.sd_model, sd_models.DiffusersTaskType.IMAGE_2_IMAGE)
                recompile_model(hires=True)
                update_sampler(shared.sd_model, second_pass=True)
                hires_args = set_pipeline_args(
                    model=shared.sd_model,
                    prompts=[p.refiner_prompt] if len(p.refiner_prompt) > 0 else p.prompts,
                    negative_prompts=[p.refiner_negative] if len(p.refiner_negative) > 0 else p.negative_prompts,
                    prompts_2=[p.refiner_prompt] if len(p.refiner_prompt) > 0 else p.prompts,
                    negative_prompts_2=[p.refiner_negative] if len(p.refiner_negative) > 0 else p.negative_prompts,
                    num_inference_steps=calculate_hires_steps(),
                    eta=shared.opts.scheduler_eta,
                    guidance_scale=p.image_cfg_scale if p.image_cfg_scale is not None else p.cfg_scale,
                    guidance_rescale=p.diffusers_guidance_rescale,
                    output_type='latent' if hasattr(shared.sd_model, 'vae') else 'np',
                    clip_skip=p.clip_skip,
                    image=output.images,
                    strength=p.denoising_strength,
                    desc='Hires',
                )
                shared.state.job = 'hires'
                shared.state.sampling_steps = hires_args['num_inference_steps']
                try:
                    output = shared.sd_model(**hires_args) # pylint: disable=not-callable
                    downcast_openvino(op="base")
                except AssertionError as e:
                    shared.log.info(e)
                p.init_images = []
        shared.state.job = prev_job
        shared.state.nextjob()
        p.is_hr_pass = False

    # optional refiner pass or decode
    if is_refiner_enabled():
        prev_job = shared.state.job
        shared.state.job = 'refine'
        shared.state.job_count +=1
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
            refiner_args = set_pipeline_args(
                model=shared.sd_refiner,
                prompts=[p.refiner_prompt] if len(p.refiner_prompt) > 0 else p.prompts[i],
                negative_prompts=[p.refiner_negative] if len(p.refiner_negative) > 0 else p.negative_prompts[i],
                num_inference_steps=calculate_refiner_steps(),
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
            shared.state.sampling_steps = refiner_args['num_inference_steps']
            try:
                shared.sd_refiner.register_to_config(requires_aesthetics_score=shared.opts.diffusers_aesthetics_score)
                refiner_output = shared.sd_refiner(**refiner_args) # pylint: disable=not-callable
                downcast_openvino(op="refiner")
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
        shared.state.job = prev_job
        shared.state.nextjob()
        p.is_refiner_pass = False

    # final decode since there is no refiner
    if not is_refiner_enabled():
        if output is not None:
            if not hasattr(output, 'images') and hasattr(output, 'frames'):
                shared.log.debug(f'Generated: frames={len(output.frames[0])}')
                output.images = output.frames[0]
            if output.images is not None and len(output.images) > 0:
                results = vae_decode(latents=output.images, model=shared.sd_model, full_quality=p.full_quality)
            else:
                shared.log.warning('Processing returned no results')
                results = []
        else:
            shared.log.warning('Processing returned no results')
            results = []

    return results
