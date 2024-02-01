import torch
import numpy as np
from PIL import Image
from modules import shared, devices, processing, images, sd_models, sd_vae, sd_samplers, processing_helpers, prompt_parser
from modules.sd_hijack_hypertile import hypertile_set


create_binary_mask = processing_helpers.create_binary_mask
apply_overlay = processing_helpers.apply_overlay
apply_color_correction = processing_helpers.apply_color_correction
setup_color_correction = processing_helpers.setup_color_correction
images_tensor_to_samples = processing_helpers.images_tensor_to_samples
txt2img_image_conditioning = processing_helpers.txt2img_image_conditioning
img2img_image_conditioning = processing_helpers.img2img_image_conditioning
get_fixed_seed = processing_helpers.get_fixed_seed
create_random_tensors = processing_helpers.create_random_tensors
decode_first_stage = processing_helpers.decode_first_stage
old_hires_fix_first_pass_dimensions = processing_helpers.old_hires_fix_first_pass_dimensions
validate_sample = processing_helpers.validate_sample


def get_conds_with_caching(function, required_prompts, steps, cache):
    if cache[0] is not None and (required_prompts, steps) == cache[0]:
        return cache[1]
    with devices.autocast():
        cache[1] = function(shared.sd_model, required_prompts, steps)
    cache[0] = (required_prompts, steps)
    return cache[1]


def process_original(p: processing.StableDiffusionProcessing):
    cached_uc = [None, None]
    cached_c = [None, None]
    sampler_config = sd_samplers.find_sampler_config(p.sampler_name)
    step_multiplier = 2 if sampler_config and sampler_config.options.get("second_order", False) else 1
    uc = get_conds_with_caching(prompt_parser.get_learned_conditioning, p.negative_prompts, p.steps * step_multiplier, cached_uc)
    c = get_conds_with_caching(prompt_parser.get_multicond_learned_conditioning, p.prompts, p.steps * step_multiplier, cached_c)
    with devices.without_autocast() if devices.unet_needs_upcast else devices.autocast():
        samples_ddim = p.sample(conditioning=c, unconditional_conditioning=uc, seeds=p.seeds, subseeds=p.subseeds, subseed_strength=p.subseed_strength, prompts=p.prompts)
    x_samples_ddim = [processing.decode_first_stage(p.sd_model, samples_ddim[i:i+1].to(dtype=devices.dtype_vae), p.full_quality)[0].cpu() for i in range(samples_ddim.size(0))]
    try:
        for x in x_samples_ddim:
            devices.test_for_nans(x, "vae")
    except devices.NansException as e:
        if not shared.opts.no_half and not shared.opts.no_half_vae and shared.cmd_opts.rollback_vae:
            shared.log.warning('Tensor with all NaNs was produced in VAE')
            devices.dtype_vae = torch.bfloat16
            vae_file, vae_source = sd_vae.resolve_vae(p.sd_model.sd_model_checkpoint)
            sd_vae.load_vae(p.sd_model, vae_file, vae_source)
            x_samples_ddim = [processing.decode_first_stage(p.sd_model, samples_ddim[i:i+1].to(dtype=devices.dtype_vae), p.full_quality)[0].cpu() for i in range(samples_ddim.size(0))]
            for x in x_samples_ddim:
                devices.test_for_nans(x, "vae")
        else:
            raise e
    x_samples_ddim = torch.stack(x_samples_ddim).float()
    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
    del samples_ddim
    return x_samples_ddim


def sample_txt2img(p: processing.StableDiffusionProcessingTxt2Img, conditioning, unconditional_conditioning, seeds, subseeds, subseed_strength, prompts):
    latent_scale_mode = shared.latent_upscale_modes.get(p.hr_upscaler, None) if p.hr_upscaler is not None else shared.latent_upscale_modes.get(shared.latent_upscale_default_mode, "None")
    if latent_scale_mode is not None:
        p.hr_force = False # no need to force anything
    if p.enable_hr and (latent_scale_mode is None or p.hr_force):
        if len([x for x in shared.sd_upscalers if x.name == p.hr_upscaler]) == 0:
            shared.log.warning(f"Cannot find upscaler for hires: {p.hr_upscaler}")
            p.enable_hr = False

    p.ops.append('txt2img')
    hypertile_set(p)
    p.sampler = sd_samplers.create_sampler(p.sampler_name, p.sd_model)
    if hasattr(p.sampler, "initialize"):
        p.sampler.initialize(p)
    x = create_random_tensors([4, p.height // 8, p.width // 8], seeds=seeds, subseeds=subseeds, subseed_strength=p.subseed_strength, seed_resize_from_h=p.seed_resize_from_h, seed_resize_from_w=p.seed_resize_from_w, p=p)
    samples = p.sampler.sample(p, x, conditioning, unconditional_conditioning, image_conditioning=txt2img_image_conditioning(p, x))
    shared.state.nextjob()
    if not p.enable_hr or shared.state.interrupted or shared.state.skipped:
        return samples

    p.init_hr()
    if p.is_hr_pass:
        prev_job = shared.state.job
        target_width = p.hr_upscale_to_x
        target_height = p.hr_upscale_to_y
        decoded_samples = None
        if shared.opts.save and shared.opts.save_images_before_highres_fix and not p.do_not_save_samples:
            decoded_samples = decode_first_stage(p.sd_model, samples.to(dtype=devices.dtype_vae), p.full_quality)
            decoded_samples = torch.clamp((decoded_samples + 1.0) / 2.0, min=0.0, max=1.0)
            for i, x_sample in enumerate(decoded_samples):
                x_sample = validate_sample(x_sample)
                image = Image.fromarray(x_sample)
                bak_extra_generation_params, bak_restore_faces = p.extra_generation_params, p.restore_faces
                p.extra_generation_params = {}
                p.restore_faces = False
                info = processing.create_infotext(p, p.all_prompts, p.all_seeds, p.all_subseeds, [], iteration=p.iteration, position_in_batch=i)
                p.extra_generation_params, p.restore_faces = bak_extra_generation_params, bak_restore_faces
                images.save_image(image, p.outpath_samples, "", seeds[i], prompts[i], shared.opts.samples_format, info=info, suffix="-before-hires")
        if latent_scale_mode is None or p.hr_force: # non-latent upscaling
            shared.state.job = 'upscale'
            if decoded_samples is None:
                decoded_samples = decode_first_stage(p.sd_model, samples.to(dtype=devices.dtype_vae), p.full_quality)
                decoded_samples = torch.clamp((decoded_samples + 1.0) / 2.0, min=0.0, max=1.0)
            batch_images = []
            for _i, x_sample in enumerate(decoded_samples):
                x_sample = validate_sample(x_sample)
                image = Image.fromarray(x_sample)
                image = images.resize_image(1, image, target_width, target_height, upscaler_name=p.hr_upscaler)
                image = np.array(image).astype(np.float32) / 255.0
                image = np.moveaxis(image, 2, 0)
                batch_images.append(image)
            resized_samples = torch.from_numpy(np.array(batch_images))
            resized_samples = resized_samples.to(device=shared.device, dtype=devices.dtype_vae)
            resized_samples = 2.0 * resized_samples - 1.0
            if shared.opts.sd_vae_sliced_encode and len(decoded_samples) > 1:
                samples = torch.stack([p.sd_model.get_first_stage_encoding(p.sd_model.encode_first_stage(torch.unsqueeze(resized_sample, 0)))[0] for resized_sample in resized_samples])
            else:
                samples = p.sd_model.get_first_stage_encoding(p.sd_model.encode_first_stage(resized_samples))
            image_conditioning = img2img_image_conditioning(p, resized_samples, samples)
        else:
            samples = torch.nn.functional.interpolate(samples, size=(target_height // 8, target_width // 8), mode=latent_scale_mode["mode"], antialias=latent_scale_mode["antialias"])
            if getattr(p, "inpainting_mask_weight", shared.opts.inpainting_mask_weight) < 1.0:
                image_conditioning = img2img_image_conditioning(p, decode_first_stage(p.sd_model, samples.to(dtype=devices.dtype_vae), p.full_quality), samples)
            else:
                image_conditioning = txt2img_image_conditioning(p, samples.to(dtype=devices.dtype_vae))
            if p.hr_sampler_name == "PLMS":
                p.hr_sampler_name = 'UniPC'
        if p.hr_force or latent_scale_mode is not None:
            shared.state.job = 'hires'
            if p.denoising_strength > 0:
                p.ops.append('hires')
                devices.torch_gc() # GC now before running the next img2img to prevent running out of memory
                p.sampler = sd_samplers.create_sampler(p.hr_sampler_name or p.sampler_name, p.sd_model)
                if hasattr(p.sampler, "initialize"):
                    p.sampler.initialize(p)
                samples = samples[:, :, p.truncate_y//2:samples.shape[2]-(p.truncate_y+1)//2, p.truncate_x//2:samples.shape[3]-(p.truncate_x+1)//2]
                noise = create_random_tensors(samples.shape[1:], seeds=seeds, subseeds=subseeds, subseed_strength=subseed_strength, p=p)
                sd_models.apply_token_merging(p.sd_model, p.get_token_merging_ratio(for_hr=True))
                hypertile_set(p, hr=True)
                samples = p.sampler.sample_img2img(p, samples, noise, conditioning, unconditional_conditioning, steps=p.hr_second_pass_steps or p.steps, image_conditioning=image_conditioning)
                sd_models.apply_token_merging(p.sd_model, p.get_token_merging_ratio())
            else:
                p.ops.append('upscale')
        x = None
        p.is_hr_pass = False
        shared.state.job = prev_job
        shared.state.nextjob()

    return samples


def sample_img2img(p, conditioning, unconditional_conditioning, seeds, subseeds, subseed_strength, prompts):
    hypertile_set(p)
    x = create_random_tensors([4, p.height // 8, p.width // 8], seeds=seeds, subseeds=subseeds, subseed_strength=p.subseed_strength, seed_resize_from_h=p.seed_resize_from_h, seed_resize_from_w=p.seed_resize_from_w, p=p)
    x *= p.initial_noise_multiplier
    samples = p.sampler.sample_img2img(p, p.init_latent, x, conditioning, unconditional_conditioning, image_conditioning=p.image_conditioning)
    if p.mask is not None:
        samples = samples * p.nmask + p.init_latent * p.mask
    del x
    devices.torch_gc()
    shared.state.nextjob()

    return samples
