import os
import math
import random
import warnings
from einops import repeat, rearrange
import torch
import numpy as np
import cv2
from PIL import Image
from skimage import exposure
from blendmodes.blend import blendLayers, BlendType
from modules import shared, devices, images, sd_models, sd_samplers, sd_hijack_hypertile, processing_vae


debug = shared.log.trace if os.environ.get('SD_PROCESS_DEBUG', None) is not None else lambda *args, **kwargs: None
debug_steps = shared.log.trace if os.environ.get('SD_STEPS_DEBUG', None) is not None else lambda *args, **kwargs: None
debug_steps('Trace: STEPS')


def setup_color_correction(image):
    debug("Calibrating color correction")
    correction_target = cv2.cvtColor(np.asarray(image.copy()), cv2.COLOR_RGB2LAB)
    return correction_target


def apply_color_correction(correction, original_image):
    shared.log.debug(f"Applying color correction: correction={correction.shape} image={original_image}")
    np_image = np.asarray(original_image)
    np_recolor = cv2.cvtColor(np_image, cv2.COLOR_RGB2LAB)
    np_match = exposure.match_histograms(np_recolor, correction, channel_axis=2)
    np_output = cv2.cvtColor(np_match, cv2.COLOR_LAB2RGB)
    image = Image.fromarray(np_output.astype("uint8"))
    image = blendLayers(image, original_image, BlendType.LUMINOSITY)
    return image


def apply_overlay(image: Image, paste_loc, index, overlays):
    debug(f'Apply overlay: image={image} loc={paste_loc} index={index} overlays={overlays}')
    if overlays is None or index >= len(overlays):
        return image
    overlay = overlays[index]
    if paste_loc is not None:
        x, y, w, h = paste_loc
        if image.width != w or image.height != h or x != 0 or y != 0:
            base_image = Image.new('RGBA', (overlay.width, overlay.height))
            image = images.resize_image(2, image, w, h)
            base_image.paste(image, (x, y))
            image = base_image
    image = image.convert('RGBA')
    image.alpha_composite(overlay)
    image = image.convert('RGB')
    return image


def create_binary_mask(image):
    if image.mode == 'RGBA' and image.getextrema()[-1] != (255, 255):
        image = image.split()[-1].convert("L").point(lambda x: 255 if x > 128 else 0)
    else:
        image = image.convert('L')
    return image


def images_tensor_to_samples(image, approximation=None, model=None): # pylint: disable=unused-argument
    if model is None:
        model = shared.sd_model
    model.first_stage_model.to(devices.dtype_vae)
    image = image.to(shared.device, dtype=devices.dtype_vae)
    image = image * 2 - 1
    if len(image) > 1:
        x_latent = torch.stack([
            model.get_first_stage_encoding(model.encode_first_stage(torch.unsqueeze(img, 0)))[0]
            for img in image
        ])
    else:
        x_latent = model.get_first_stage_encoding(model.encode_first_stage(image))
    return x_latent


def get_sampler_name(sampler_index: int, img: bool = False) -> str:
    sampler_index = sampler_index or 0
    if len(sd_samplers.samplers) > sampler_index:
        sampler_name = sd_samplers.samplers[sampler_index].name
    else:
        sampler_name = "UniPC"
        shared.log.warning(f'Sampler not found: index={sampler_index} available={[s.name for s in sd_samplers.samplers]} fallback={sampler_name}')
    if img and sampler_name == "PLMS":
        sampler_name = "UniPC"
        shared.log.warning(f'Sampler not compatible: name=PLMS fallback={sampler_name}')
    return sampler_name


def slerp(val, low, high): # from https://discuss.pytorch.org/t/help-regarding-slerp-function-for-generative-model-sampling/32475/3
    low_norm = low/torch.norm(low, dim=1, keepdim=True)
    high_norm = high/torch.norm(high, dim=1, keepdim=True)
    dot = (low_norm*high_norm).sum(1)

    if dot.mean() > 0.9995:
        return low * val + high * (1 - val)

    omega = torch.acos(dot)
    so = torch.sin(omega)
    res = (torch.sin((1.0-val)*omega)/so).unsqueeze(1)*low + (torch.sin(val*omega)/so).unsqueeze(1) * high
    return res


def create_random_tensors(shape, seeds, subseeds=None, subseed_strength=0.0, seed_resize_from_h=0, seed_resize_from_w=0, p=None):
    eta_noise_seed_delta = shared.opts.eta_noise_seed_delta or 0
    xs = []
    # if we have multiple seeds, this means we are working with batch size>1; this then
    # enables the generation of additional tensors with noise that the sampler will use during its processing.
    # Using those pre-generated tensors instead of simple torch.randn allows a batch with seeds [100, 101] to
    # produce the same images as with two batches [100], [101].
    if p is not None and p.sampler is not None and (len(seeds) > 1 and shared.opts.enable_batch_seeds or eta_noise_seed_delta > 0):
        sampler_noises = [[] for _ in range(p.sampler.number_of_needed_noises(p))]
    else:
        sampler_noises = None
    for i, seed in enumerate(seeds):
        noise_shape = shape if seed_resize_from_h <= 0 or seed_resize_from_w <= 0 else (shape[0], seed_resize_from_h//8, seed_resize_from_w//8)
        subnoise = None
        if subseeds is not None:
            subseed = 0 if i >= len(subseeds) else subseeds[i]
            subnoise = devices.randn(subseed, noise_shape)
        # randn results depend on device; gpu and cpu get different results for same seed;
        # the way I see it, it's better to do this on CPU, so that everyone gets same result;
        # but the original script had it like this, so I do not dare change it for now because
        # it will break everyone's seeds.
        noise = devices.randn(seed, noise_shape)
        if subnoise is not None:
            noise = slerp(subseed_strength, noise, subnoise)
        if noise_shape != shape:
            x = devices.randn(seed, shape)
            dx = (shape[2] - noise_shape[2]) // 2
            dy = (shape[1] - noise_shape[1]) // 2
            w = noise_shape[2] if dx >= 0 else noise_shape[2] + 2 * dx
            h = noise_shape[1] if dy >= 0 else noise_shape[1] + 2 * dy
            tx = 0 if dx < 0 else dx
            ty = 0 if dy < 0 else dy
            dx = max(-dx, 0)
            dy = max(-dy, 0)
            x[:, ty:ty+h, tx:tx+w] = noise[:, dy:dy+h, dx:dx+w]
            noise = x
        if sampler_noises is not None:
            cnt = p.sampler.number_of_needed_noises(p)
            if eta_noise_seed_delta > 0:
                torch.manual_seed(seed + eta_noise_seed_delta)
            for j in range(cnt):
                sampler_noises[j].append(devices.randn_without_seed(tuple(noise_shape)))
        xs.append(noise)
    if sampler_noises is not None:
        p.sampler.sampler_noises = [torch.stack(n).to(shared.device) for n in sampler_noises]
    x = torch.stack(xs).to(shared.device)
    return x


def decode_first_stage(model, x, full_quality=True):
    if not shared.opts.keep_incomplete and (shared.state.skipped or shared.state.interrupted):
        shared.log.debug(f'Decode VAE: skipped={shared.state.skipped} interrupted={shared.state.interrupted}')
        x_sample = torch.zeros((len(x), 3, x.shape[2] * 8, x.shape[3] * 8), dtype=devices.dtype_vae, device=devices.device)
        return x_sample
    prev_job = shared.state.job
    shared.state.job = 'vae'
    with devices.autocast(disable = x.dtype==devices.dtype_vae):
        try:
            if full_quality:
                if hasattr(model, 'decode_first_stage'):
                    x_sample = model.decode_first_stage(x)
                elif hasattr(model, 'vae'):
                    x_sample = model.vae(x)
                else:
                    x_sample = x
                    shared.log.error('Decode VAE unknown model')
            else:
                from modules import sd_vae_taesd
                x_sample = torch.zeros((len(x), 3, x.shape[2] * 8, x.shape[3] * 8), dtype=devices.dtype_vae, device=devices.device)
                for i in range(len(x_sample)):
                    x_sample[i] = sd_vae_taesd.decode(x[i])
        except Exception as e:
            x_sample = x
            shared.log.error(f'Decode VAE: {e}')
    shared.state.job = prev_job
    return x_sample


def get_fixed_seed(seed):
    if seed is None or seed == '' or seed == -1:
        return int(random.randrange(4294967294))
    return seed


def fix_seed(p):
    p.seed = get_fixed_seed(p.seed)
    p.subseed = get_fixed_seed(p.subseed)


def old_hires_fix_first_pass_dimensions(width, height):
    """old algorithm for auto-calculating first pass size"""
    desired_pixel_count = 512 * 512
    actual_pixel_count = width * height
    scale = math.sqrt(desired_pixel_count / actual_pixel_count)
    width = math.ceil(scale * width / 64) * 64
    height = math.ceil(scale * height / 64) * 64
    return width, height


def txt2img_image_conditioning(p, x, width=None, height=None):
    width = width or p.width
    height = height or p.height
    if p.sd_model.model.conditioning_key in {'hybrid', 'concat'}: # Inpainting models
        image_conditioning = torch.zeros(x.shape[0], 3, height, width, device=x.device)
        image_conditioning = p.sd_model.get_first_stage_encoding(p.sd_model.encode_first_stage(image_conditioning))
        image_conditioning = torch.nn.functional.pad(image_conditioning, (0, 0, 0, 0, 1, 0), value=1.0) # pylint: disable=not-callable
        image_conditioning = image_conditioning.to(x.dtype)
        return image_conditioning
    elif p.sd_model.model.conditioning_key == "crossattn-adm": # UnCLIP models
        return x.new_zeros(x.shape[0], 2*p.sd_model.noise_augmentor.time_embed.dim, dtype=x.dtype, device=x.device)
    else:
        return x.new_zeros(x.shape[0], 5, 1, 1, dtype=x.dtype, device=x.device)


def img2img_image_conditioning(p, source_image, latent_image, image_mask=None):
    from ldm.models.diffusion.ddpm import LatentDepth2ImageDiffusion
    source_image = devices.cond_cast_float(source_image)

    def depth2img_image_conditioning(source_image):
        # Use the AddMiDaS helper to Format our source image to suit the MiDaS model
        from ldm.data.util import AddMiDaS
        transformer = AddMiDaS(model_type="dpt_hybrid")
        transformed = transformer({"jpg": rearrange(source_image[0], "c h w -> h w c")})
        midas_in = torch.from_numpy(transformed["midas_in"][None, ...]).to(device=shared.device)
        midas_in = repeat(midas_in, "1 ... -> n ...", n=p.batch_size)
        conditioning_image = p.sd_model.get_first_stage_encoding(p.sd_model.encode_first_stage(source_image))
        conditioning = torch.nn.functional.interpolate(
            p.sd_model.depth_model(midas_in),
            size=conditioning_image.shape[2:],
            mode="bicubic",
            align_corners=False,
        )
        (depth_min, depth_max) = torch.aminmax(conditioning)
        conditioning = 2. * (conditioning - depth_min) / (depth_max - depth_min) - 1.
        return conditioning

    def edit_image_conditioning(source_image):
        conditioning_image = p.sd_model.encode_first_stage(source_image).mode()
        return conditioning_image

    def unclip_image_conditioning(source_image):
        c_adm = p.sd_model.embedder(source_image)
        if p.sd_model.noise_augmentor is not None:
            noise_level = 0
            c_adm, noise_level_emb = p.sd_model.noise_augmentor(c_adm, noise_level=repeat(torch.tensor([noise_level]).to(c_adm.device), '1 -> b', b=c_adm.shape[0]))
            c_adm = torch.cat((c_adm, noise_level_emb), 1)
        return c_adm

    def inpainting_image_conditioning(source_image, latent_image, image_mask=None):
        # Handle the different mask inputs
        if image_mask is not None:
            if torch.is_tensor(image_mask):
                conditioning_mask = image_mask
            else:
                conditioning_mask = np.array(image_mask.convert("L"))
                conditioning_mask = conditioning_mask.astype(np.float32) / 255.0
                conditioning_mask = torch.from_numpy(conditioning_mask[None, None])
                # Inpainting model uses a discretized mask as input, so we round to either 1.0 or 0.0
                conditioning_mask = torch.round(conditioning_mask)
        else:
            conditioning_mask = source_image.new_ones(1, 1, *source_image.shape[-2:])
        # Create another latent image, this time with a masked version of the original input.
        # Smoothly interpolate between the masked and unmasked latent conditioning image using a parameter.
        conditioning_mask = conditioning_mask.to(device=source_image.device, dtype=source_image.dtype)
        conditioning_image = torch.lerp(
            source_image,
            source_image * (1.0 - conditioning_mask),
            getattr(p, "inpainting_mask_weight", shared.opts.inpainting_mask_weight)
        )
        # Encode the new masked image using first stage of network.
        conditioning_image = p.sd_model.get_first_stage_encoding(p.sd_model.encode_first_stage(conditioning_image))
        # Create the concatenated conditioning tensor to be fed to `c_concat`
        conditioning_mask = torch.nn.functional.interpolate(conditioning_mask, size=latent_image.shape[-2:])
        conditioning_mask = conditioning_mask.expand(conditioning_image.shape[0], -1, -1, -1)
        image_conditioning = torch.cat([conditioning_mask, conditioning_image], dim=1)
        image_conditioning = image_conditioning.to(device=shared.device, dtype=source_image.dtype)
        return image_conditioning

    def diffusers_image_conditioning(_source_image, latent_image, _image_mask=None):
        # shared.log.warning('Diffusers not implemented: img2img_image_conditioning')
        return latent_image.new_zeros(latent_image.shape[0], 5, 1, 1)

    # HACK: Using introspection as the Depth2Image model doesn't appear to uniquely
    # identify itself with a field common to all models. The conditioning_key is also hybrid.
    if shared.backend == shared.Backend.DIFFUSERS:
        return diffusers_image_conditioning(source_image, latent_image, image_mask)
    if isinstance(p.sd_model, LatentDepth2ImageDiffusion):
        return depth2img_image_conditioning(source_image)
    if hasattr(p.sd_model, 'cond_stage_key') and p.sd_model.cond_stage_key == "edit":
        return edit_image_conditioning(source_image)
    if hasattr(p.sampler, 'conditioning_key') and p.sampler.conditioning_key in {'hybrid', 'concat'}:
        return inpainting_image_conditioning(source_image, latent_image, image_mask=image_mask)
    if hasattr(p.sampler, 'conditioning_key') and p.sampler.conditioning_key == "crossattn-adm":
        return unclip_image_conditioning(source_image)
    # Dummy zero conditioning if we're not using inpainting or depth model.
    return latent_image.new_zeros(latent_image.shape[0], 5, 1, 1)


def validate_sample(tensor):
    if not isinstance(tensor, np.ndarray) and not isinstance(tensor, torch.Tensor):
        return tensor
    if tensor.dtype == torch.bfloat16: # numpy does not support bf16
        tensor = tensor.to(torch.float16)
    if isinstance(tensor, torch.Tensor) and hasattr(tensor, 'detach'):
        sample = tensor.detach().cpu().numpy()
    elif isinstance(tensor, np.ndarray):
        sample = tensor
    else:
        shared.log.warning(f'Unknown sample type: {type(tensor)}')
    sample = 255.0 * np.moveaxis(sample, 0, 2) if shared.backend == shared.Backend.ORIGINAL else 255.0 * sample
    with warnings.catch_warnings(record=True) as w:
        cast = sample.astype(np.uint8)
    if len(w) > 0:
        nans = np.isnan(sample).sum()
        shared.log.error(f'Failed to validate samples: sample={sample.shape} invalid={nans}')
        cast = np.nan_to_num(sample)
        minimum, maximum, mean = np.min(cast), np.max(cast), np.mean(cast)
        cast = cast.astype(np.uint8)
        shared.log.warning(f'Attempted to correct samples: min={minimum:.2f} max={maximum:.2f} mean={mean:.2f}')
    return cast


def resize_init_images(p):
    if getattr(p, 'image', None) is not None and getattr(p, 'init_images', None) is None:
        p.init_images = [p.image]
    if getattr(p, 'init_images', None) is not None and len(p.init_images) > 0:
        tgt_width, tgt_height = 8 * math.ceil(p.init_images[0].width / 8), 8 * math.ceil(p.init_images[0].height / 8)
        if p.init_images[0].size != (tgt_width, tgt_height):
            shared.log.debug(f'Resizing init images: original={p.init_images[0].width}x{p.init_images[0].height} target={tgt_width}x{tgt_height}')
            p.init_images = [images.resize_image(1, image, tgt_width, tgt_height, upscaler_name=None) for image in p.init_images]
            p.height = tgt_height
            p.width = tgt_width
            sd_hijack_hypertile.hypertile_set(p)
        if getattr(p, 'mask', None) is not None and p.mask.size != (tgt_width, tgt_height):
            p.mask = images.resize_image(1, p.mask, tgt_width, tgt_height, upscaler_name=None)
        if getattr(p, 'init_mask', None) is not None and p.init_mask.size != (tgt_width, tgt_height):
            p.init_mask = images.resize_image(1, p.init_mask, tgt_width, tgt_height, upscaler_name=None)
        if getattr(p, 'mask_for_overlay', None) is not None and p.mask_for_overlay.size != (tgt_width, tgt_height):
            p.mask_for_overlay = images.resize_image(1, p.mask_for_overlay, tgt_width, tgt_height, upscaler_name=None)
        return tgt_width, tgt_height
    return p.width, p.height


def resize_hires(p, latents): # input=latents output=pil
    if not torch.is_tensor(latents):
        shared.log.warning('Hires: input is not tensor')
        first_pass_images = processing_vae.vae_decode(latents=latents, model=shared.sd_model, full_quality=p.full_quality, output_type='pil')
        return first_pass_images
    latent_upscaler = shared.latent_upscale_modes.get(p.hr_upscaler, None)
    shared.log.info(f'Hires: upscaler={p.hr_upscaler} width={p.hr_upscale_to_x} height={p.hr_upscale_to_y} images={latents.shape[0]}')
    if latent_upscaler is not None:
        latents = torch.nn.functional.interpolate(latents, size=(p.hr_upscale_to_y // 8, p.hr_upscale_to_x // 8), mode=latent_upscaler["mode"], antialias=latent_upscaler["antialias"])
    first_pass_images = processing_vae.vae_decode(latents=latents, model=shared.sd_model, full_quality=p.full_quality, output_type='pil')
    resized_images = []
    for img in first_pass_images:
        if latent_upscaler is None:
            resized_image = images.resize_image(1, img, p.hr_upscale_to_x, p.hr_upscale_to_y, upscaler_name=p.hr_upscaler)
        else:
            resized_image = img
        resized_images.append(resized_image)
    return resized_images

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

def calculate_base_steps(p, use_denoise_start, use_refiner_start):
    is_txt2img = sd_models.get_diffusers_task(shared.sd_model) == sd_models.DiffusersTaskType.TEXT_2_IMAGE
    if not is_txt2img:
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
    return max(1, int(steps))

def calculate_hires_steps(p):
    if p.hr_second_pass_steps > 0:
        steps = (p.hr_second_pass_steps // p.denoising_strength) + 1
    elif p.denoising_strength > 0:
        steps = (p.steps // p.denoising_strength) + 1
    else:
        steps = 0
    debug_steps(f'Steps: type=hires input={p.hr_second_pass_steps} output={steps} denoise={p.denoising_strength} model={shared.sd_model_type}')
    return max(1, int(steps))

def calculate_refiner_steps(p):
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
    return max(1, int(steps))
