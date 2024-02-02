import os
import json
import time
from contextlib import nullcontext
import numpy as np
from PIL import Image
from modules import shared, devices, errors, images, scripts, memstats, lowvram, script_callbacks, extra_networks, face_restoration, sd_hijack_freeu, sd_models, sd_vae, processing_helpers
from modules.sd_hijack_hypertile import context_hypertile_vae, context_hypertile_unet
from modules.processing_class import StableDiffusionProcessing, StableDiffusionProcessingTxt2Img, StableDiffusionProcessingImg2Img # pylint: disable=unused-import
from modules.processing_info import create_infotext


opt_C = 4
opt_f = 8
debug = shared.log.trace if os.environ.get('SD_PROCESS_DEBUG', None) is not None else lambda *args, **kwargs: None
debug('Trace: PROCESS')
create_binary_mask = processing_helpers.create_binary_mask
apply_overlay = processing_helpers.apply_overlay
apply_color_correction = processing_helpers.apply_color_correction
setup_color_correction = processing_helpers.setup_color_correction
txt2img_image_conditioning = processing_helpers.txt2img_image_conditioning
img2img_image_conditioning = processing_helpers.img2img_image_conditioning
fix_seed = processing_helpers.fix_seed
get_fixed_seed = processing_helpers.get_fixed_seed
create_random_tensors = processing_helpers.create_random_tensors
old_hires_fix_first_pass_dimensions = processing_helpers.old_hires_fix_first_pass_dimensions
get_sampler_name = processing_helpers.get_sampler_name
validate_sample = processing_helpers.validate_sample
decode_first_stage = processing_helpers.decode_first_stage
images_tensor_to_samples = processing_helpers.images_tensor_to_samples


class Processed:
    def __init__(self, p: StableDiffusionProcessing, images_list, seed=-1, info="", subseed=None, all_prompts=None, all_negative_prompts=None, all_seeds=None, all_subseeds=None, index_of_first_image=0, infotexts=None, comments=""):
        self.images = images_list
        self.prompt = p.prompt
        self.negative_prompt = p.negative_prompt
        self.seed = seed
        self.subseed = subseed
        self.subseed_strength = p.subseed_strength
        self.info = info
        self.comments = comments
        self.width = p.width if hasattr(p, 'width') else (self.images[0].width if len(self.images) > 0 else 0)
        self.height = p.height if hasattr(p, 'height') else (self.images[0].height if len(self.images) > 0 else 0)
        self.sampler_name = p.sampler_name
        self.cfg_scale = p.cfg_scale
        self.image_cfg_scale = p.image_cfg_scale
        self.steps = p.steps
        self.batch_size = p.batch_size
        self.restore_faces = p.restore_faces
        self.face_restoration_model = shared.opts.face_restoration_model if p.restore_faces else None
        self.sd_model_hash = getattr(shared.sd_model, 'sd_model_hash', '')
        self.seed_resize_from_w = p.seed_resize_from_w
        self.seed_resize_from_h = p.seed_resize_from_h
        self.denoising_strength = p.denoising_strength
        self.extra_generation_params = p.extra_generation_params
        self.index_of_first_image = index_of_first_image
        self.styles = p.styles
        self.job_timestamp = shared.state.job_timestamp
        self.clip_skip = p.clip_skip
        self.eta = p.eta
        self.ddim_discretize = p.ddim_discretize
        self.s_churn = p.s_churn
        self.s_tmin = p.s_tmin
        self.s_tmax = p.s_tmax
        self.s_noise = p.s_noise
        self.s_min_uncond = p.s_min_uncond
        self.prompt = self.prompt if type(self.prompt) != list else self.prompt[0]
        self.negative_prompt = self.negative_prompt if type(self.negative_prompt) != list else self.negative_prompt[0]
        self.seed = int(self.seed if type(self.seed) != list else self.seed[0]) if self.seed is not None else -1
        self.subseed = int(self.subseed if type(self.subseed) != list else self.subseed[0]) if self.subseed is not None else -1
        self.is_using_inpainting_conditioning = p.is_using_inpainting_conditioning
        self.all_prompts = all_prompts or p.all_prompts or [self.prompt]
        self.all_negative_prompts = all_negative_prompts or p.all_negative_prompts or [self.negative_prompt]
        self.all_seeds = all_seeds or p.all_seeds or [self.seed]
        self.all_subseeds = all_subseeds or p.all_subseeds or [self.subseed]
        self.token_merging_ratio = p.token_merging_ratio
        self.token_merging_ratio_hr = p.token_merging_ratio_hr
        self.infotexts = infotexts or [info]

    def js(self):
        obj = {
            "prompt": self.all_prompts[0],
            "all_prompts": self.all_prompts,
            "negative_prompt": self.all_negative_prompts[0],
            "all_negative_prompts": self.all_negative_prompts,
            "seed": self.seed,
            "all_seeds": self.all_seeds,
            "subseed": self.subseed,
            "all_subseeds": self.all_subseeds,
            "subseed_strength": self.subseed_strength,
            "width": self.width,
            "height": self.height,
            "sampler_name": self.sampler_name,
            "cfg_scale": self.cfg_scale,
            "steps": self.steps,
            "batch_size": self.batch_size,
            "restore_faces": self.restore_faces,
            "face_restoration_model": self.face_restoration_model,
            "sd_model_hash": self.sd_model_hash,
            "seed_resize_from_w": self.seed_resize_from_w,
            "seed_resize_from_h": self.seed_resize_from_h,
            "denoising_strength": self.denoising_strength,
            "extra_generation_params": self.extra_generation_params,
            "index_of_first_image": self.index_of_first_image,
            "infotexts": self.infotexts,
            "styles": self.styles,
            "job_timestamp": self.job_timestamp,
            "clip_skip": self.clip_skip,
        }
        return json.dumps(obj)

    def infotext(self, p: StableDiffusionProcessing, index):
        return create_infotext(p, self.all_prompts, self.all_seeds, self.all_subseeds, comments=[], position_in_batch=index % self.batch_size, iteration=index // self.batch_size)

    def get_token_merging_ratio(self, for_hr=False):
        return self.token_merging_ratio_hr if for_hr else self.token_merging_ratio


def process_images(p: StableDiffusionProcessing) -> Processed:
    debug(f'Process images: {vars(p)}')
    if not hasattr(p.sd_model, 'sd_checkpoint_info'):
        return None
    if p.scripts is not None and isinstance(p.scripts, scripts.ScriptRunner):
        p.scripts.before_process(p)
    stored_opts = {}
    for k, v in p.override_settings.copy().items():
        if shared.opts.data.get(k, None) is None and shared.opts.data_labels.get(k, None) is None:
            continue
        orig = shared.opts.data.get(k, None) or shared.opts.data_labels[k].default
        if orig == v or (type(orig) == str and os.path.splitext(orig)[0] == v):
            p.override_settings.pop(k, None)
    for k in p.override_settings.keys():
        stored_opts[k] = shared.opts.data.get(k, None) or shared.opts.data_labels[k].default
    processed = None
    try:
        # if no checkpoint override or the override checkpoint can't be found, remove override entry and load opts checkpoint
        if p.override_settings.get('sd_model_checkpoint', None) is not None and sd_models.checkpoint_aliases.get(p.override_settings.get('sd_model_checkpoint')) is None:
            shared.log.warning(f"Override not found: checkpoint={p.override_settings.get('sd_model_checkpoint', None)}")
            p.override_settings.pop('sd_model_checkpoint', None)
            sd_models.reload_model_weights()
        if p.override_settings.get('sd_model_refiner', None) is not None and sd_models.checkpoint_aliases.get(p.override_settings.get('sd_model_refiner')) is None:
            shared.log.warning(f"Override not found: refiner={p.override_settings.get('sd_model_refiner', None)}")
            p.override_settings.pop('sd_model_refiner', None)
            sd_models.reload_model_weights()
        if p.override_settings.get('sd_vae', None) is not None:
            if p.override_settings.get('sd_vae', None) == 'TAESD':
                p.full_quality = False
                p.override_settings.pop('sd_vae', None)
        if p.override_settings.get('Hires upscaler', None) is not None:
            p.enable_hr = True
        if len(p.override_settings.keys()) > 0:
            shared.log.debug(f'Override: {p.override_settings}')
        for k, v in p.override_settings.items():
            setattr(shared.opts, k, v)
            if k == 'sd_model_checkpoint':
                sd_models.reload_model_weights()
            if k == 'sd_vae':
                sd_vae.reload_vae_weights()

        shared.prompt_styles.apply_styles_to_extra(p)
        if not shared.opts.cuda_compile:
            sd_models.apply_token_merging(p.sd_model, p.get_token_merging_ratio())
            sd_hijack_freeu.apply_freeu(p, shared.backend == shared.Backend.ORIGINAL)

        script_callbacks.before_process_callback(p)

        if shared.cmd_opts.profile:
            import cProfile
            profile_python = cProfile.Profile()
            profile_python.enable()
            with context_hypertile_vae(p), context_hypertile_unet(p):
                import torch.profiler # pylint: disable=redefined-outer-name
                activities=[torch.profiler.ProfilerActivity.CPU]
                if torch.cuda.is_available():
                    activities.append(torch.profiler.ProfilerActivity.CUDA)
                shared.log.debug(f'Torch profile: activities={activities}')
                if shared.profiler is None:
                    shared.profiler = torch.profiler.profile(activities=activities, profile_memory=True, with_modules=True)
                shared.profiler.start()
                shared.profiler.step()
                processed = process_images_inner(p)
                errors.profile_torch(shared.profiler, 'Process')
            errors.profile(profile_python, 'Process')
        else:
            with context_hypertile_vae(p), context_hypertile_unet(p):
                processed = process_images_inner(p)

    finally:
        if not shared.opts.cuda_compile:
            sd_models.apply_token_merging(p.sd_model, 0)

        script_callbacks.after_process_callback(p)

        if p.override_settings_restore_afterwards: # restore opts to original state
            for k, v in stored_opts.items():
                setattr(shared.opts, k, v)
                if k == 'sd_model_checkpoint':
                    sd_models.reload_model_weights()
                if k == 'sd_model_refiner':
                    sd_models.reload_model_weights()
                if k == 'sd_vae':
                    sd_vae.reload_vae_weights()
    return processed


def process_init(p: StableDiffusionProcessing):
    seed = get_fixed_seed(p.seed)
    subseed = get_fixed_seed(p.subseed)
    if type(p.prompt) == list:
        p.all_prompts = [shared.prompt_styles.apply_styles_to_prompt(x, p.styles) for x in p.prompt]
    else:
        p.all_prompts = p.batch_size * p.n_iter * [shared.prompt_styles.apply_styles_to_prompt(p.prompt, p.styles)]
    if type(p.negative_prompt) == list:
        p.all_negative_prompts = [shared.prompt_styles.apply_negative_styles_to_prompt(x, p.styles) for x in p.negative_prompt]
    else:
        p.all_negative_prompts = p.batch_size * p.n_iter * [shared.prompt_styles.apply_negative_styles_to_prompt(p.negative_prompt, p.styles)]
    if type(seed) == list:
        p.all_seeds = seed
    else:
        p.all_seeds = [int(seed) + (x if p.subseed_strength == 0 else 0) for x in range(len(p.all_prompts))]
    if type(subseed) == list:
        p.all_subseeds = subseed
    else:
        p.all_subseeds = [int(subseed) + x for x in range(len(p.all_prompts))]


def process_images_inner(p: StableDiffusionProcessing) -> Processed:
    """this is the main loop that both txt2img and img2img use; it calls func_init once inside all the scopes and func_sample once per batch"""

    if type(p.prompt) == list:
        assert len(p.prompt) > 0
    else:
        assert p.prompt is not None

    if shared.backend == shared.Backend.ORIGINAL:
        import modules.sd_hijack # pylint: disable=redefined-outer-name
        modules.sd_hijack.model_hijack.apply_circular(p.tiling)
        modules.sd_hijack.model_hijack.clear_comments()
    comments = {}
    infotexts = []
    output_images = []

    process_init(p)
    if os.path.exists(shared.opts.embeddings_dir) and not p.do_not_reload_embeddings and shared.backend == shared.Backend.ORIGINAL:
        modules.sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings(force_reload=False)
    if p.scripts is not None and isinstance(p.scripts, scripts.ScriptRunner):
        p.scripts.process(p)

    if shared.backend == shared.Backend.DIFFUSERS:
        from modules import ipadapter
        ipadapter.apply(shared.sd_model, p)

    def infotext(_inxex=0): # dummy function overriden if there are iterations
        return ''

    ema_scope_context = p.sd_model.ema_scope if shared.backend == shared.Backend.ORIGINAL else nullcontext
    shared.state.job_count = p.n_iter
    with devices.inference_context(), ema_scope_context():
        t0 = time.time()
        with devices.autocast():
            p.init(p.all_prompts, p.all_seeds, p.all_subseeds)
        extra_network_data = None
        debug(f'Processing inner: args={vars(p)}')
        for n in range(p.n_iter):
            p.iteration = n
            if shared.state.skipped:
                shared.log.debug(f'Process skipped: {n}/{p.n_iter}')
                shared.state.skipped = False
                continue
            if shared.state.interrupted:
                shared.log.debug(f'Process interrupted: {n}/{p.n_iter}')
                break
            p.prompts = p.all_prompts[n * p.batch_size:(n + 1) * p.batch_size]
            p.negative_prompts = p.all_negative_prompts[n * p.batch_size:(n + 1) * p.batch_size]
            p.seeds = p.all_seeds[n * p.batch_size:(n + 1) * p.batch_size]
            p.subseeds = p.all_subseeds[n * p.batch_size:(n + 1) * p.batch_size]
            if p.scripts is not None and isinstance(p.scripts, scripts.ScriptRunner):
                p.scripts.before_process_batch(p, batch_number=n, prompts=p.prompts, seeds=p.seeds, subseeds=p.subseeds)
            if len(p.prompts) == 0:
                break
            p.prompts, extra_network_data = extra_networks.parse_prompts(p.prompts)
            if not p.disable_extra_networks:
                with devices.autocast():
                    extra_networks.activate(p, extra_network_data)
            if p.scripts is not None and isinstance(p.scripts, scripts.ScriptRunner):
                p.scripts.process_batch(p, batch_number=n, prompts=p.prompts, seeds=p.seeds, subseeds=p.subseeds)

            x_samples_ddim = None
            if p.scripts is not None and isinstance(p.scripts, scripts.ScriptRunner):
                x_samples_ddim = p.scripts.process_images(p)
            if x_samples_ddim is None:
                if shared.backend == shared.Backend.ORIGINAL:
                    from modules.processing_original import process_original
                    x_samples_ddim = process_original(p)
                elif shared.backend == shared.Backend.DIFFUSERS:
                    from modules.processing_diffusers import process_diffusers
                    x_samples_ddim = process_diffusers(p)
                else:
                    raise ValueError(f"Unknown backend {shared.backend}")

            if not shared.opts.keep_incomplete and shared.state.interrupted:
                x_samples_ddim = []

            if shared.cmd_opts.lowvram or shared.cmd_opts.medvram and shared.backend == shared.Backend.ORIGINAL:
                lowvram.send_everything_to_cpu()
                devices.torch_gc()
            if p.scripts is not None and isinstance(p.scripts, scripts.ScriptRunner):
                p.scripts.postprocess_batch(p, x_samples_ddim, batch_number=n)
            if p.scripts is not None and isinstance(p.scripts, scripts.ScriptRunner):
                p.prompts = p.all_prompts[n * p.batch_size:(n + 1) * p.batch_size]
                p.negative_prompts = p.all_negative_prompts[n * p.batch_size:(n + 1) * p.batch_size]
                batch_params = scripts.PostprocessBatchListArgs(list(x_samples_ddim))
                p.scripts.postprocess_batch_list(p, batch_params, batch_number=n)
                x_samples_ddim = batch_params.images

            def infotext(index): # pylint: disable=function-redefined # noqa: F811
                return create_infotext(p, p.prompts, p.seeds, p.subseeds, index=index, all_negative_prompts=p.negative_prompts)

            for i, x_sample in enumerate(x_samples_ddim):
                p.batch_index = i
                if type(x_sample) == Image.Image:
                    image = x_sample
                    x_sample = np.array(x_sample)
                else:
                    x_sample = validate_sample(x_sample)
                    image = Image.fromarray(x_sample)
                if p.restore_faces:
                    if not p.do_not_save_samples and shared.opts.save_images_before_face_restoration:
                        orig = p.restore_faces
                        p.restore_faces = False
                        info = infotext(i)
                        p.restore_faces = orig
                        images.save_image(Image.fromarray(x_sample), path=p.outpath_samples, basename="", seed=p.seeds[i], prompt=p.prompts[i], extension=shared.opts.samples_format, info=info, p=p, suffix="-before-face-restore")
                    p.ops.append('face')
                    x_sample = face_restoration.restore_faces(x_sample)
                    image = Image.fromarray(x_sample)
                if p.scripts is not None and isinstance(p.scripts, scripts.ScriptRunner):
                    pp = scripts.PostprocessImageArgs(image)
                    p.scripts.postprocess_image(p, pp)
                    image = pp.image
                if p.color_corrections is not None and i < len(p.color_corrections):
                    if not p.do_not_save_samples and shared.opts.save_images_before_color_correction:
                        orig = p.color_corrections
                        p.color_corrections = None
                        info = infotext(i)
                        p.color_corrections = orig
                        image_without_cc = apply_overlay(image, p.paste_to, i, p.overlay_images)
                        images.save_image(image_without_cc, path=p.outpath_samples, basename="", seed=p.seeds[i], prompt=p.prompts[i], extension=shared.opts.samples_format, info=info, p=p, suffix="-before-color-correct")
                    p.ops.append('color')
                    image = apply_color_correction(p.color_corrections[i], image)
                image = apply_overlay(image, p.paste_to, i, p.overlay_images)
                text = infotext(i)
                infotexts.append(text)
                image.info["parameters"] = text
                output_images.append(image)
                if shared.opts.samples_save and not p.do_not_save_samples:
                    images.save_image(image, p.outpath_samples, "", p.seeds[i], p.prompts[i], shared.opts.samples_format, info=text, p=p) # main save image
                if hasattr(p, 'mask_for_overlay') and p.mask_for_overlay and any([shared.opts.save_mask, shared.opts.save_mask_composite, shared.opts.return_mask, shared.opts.return_mask_composite]):
                    image_mask = p.mask_for_overlay.convert('RGB')
                    image_mask_composite = Image.composite(image.convert('RGBA').convert('RGBa'), Image.new('RGBa', image.size), images.resize_image(3, p.mask_for_overlay, image.width, image.height).convert('L')).convert('RGBA')
                    if shared.opts.save_mask:
                        images.save_image(image_mask, p.outpath_samples, "", p.seeds[i], p.prompts[i], shared.opts.samples_format, info=text, p=p, suffix="-mask")
                    if shared.opts.save_mask_composite:
                        images.save_image(image_mask_composite, p.outpath_samples, "", p.seeds[i], p.prompts[i], shared.opts.samples_format, info=text, p=p, suffix="-mask-composite")
                    if shared.opts.return_mask:
                        output_images.append(image_mask)
                    if shared.opts.return_mask_composite:
                        output_images.append(image_mask_composite)
            del x_samples_ddim
            devices.torch_gc()

        t1 = time.time()
        shared.log.info(f'Processed: images={len(output_images)} time={t1 - t0:.2f} its={(p.steps * len(output_images)) / (t1 - t0):.2f} memory={memstats.memory_stats()}')

        p.color_corrections = None
        index_of_first_image = 0
        if (shared.opts.return_grid or shared.opts.grid_save) and not p.do_not_save_grid and len(output_images) > 1:
            if images.check_grid_size(output_images):
                grid = images.image_grid(output_images, p.batch_size)
                if shared.opts.return_grid:
                    text = infotext(-1)
                    infotexts.insert(0, text)
                    grid.info["parameters"] = text
                    output_images.insert(0, grid)
                    index_of_first_image = 1
                if shared.opts.grid_save:
                    images.save_image(grid, p.outpath_grids, "", p.all_seeds[0], p.all_prompts[0], shared.opts.grid_format, info=infotext(-1), p=p, grid=True, suffix="-grid") # main save grid

    if shared.backend == shared.Backend.DIFFUSERS:
        ipadapter.unapply(shared.sd_model)

    if not p.disable_extra_networks:
        extra_networks.deactivate(p, extra_network_data)

    processed = Processed(
        p,
        images_list=output_images,
        seed=p.all_seeds[0],
        info=infotext(0),
        comments="\n".join(comments),
        subseed=p.all_subseeds[0],
        index_of_first_image=index_of_first_image,
        infotexts=infotexts,
    )
    if p.scripts is not None and isinstance(p.scripts, scripts.ScriptRunner) and not (shared.state.interrupted or shared.state.skipped):
        p.scripts.postprocess(p, processed)
    return processed
