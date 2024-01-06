import os
import time
import torch
import torchvision.transforms.functional as TF
from modules import shared, devices, sd_vae, sd_models
import modules.taesd.sd_vae_taesd as sd_vae_taesd


debug = shared.log.trace if os.environ.get('SD_VAE_DEBUG', None) is not None else lambda *args, **kwargs: None
debug('Trace: VAE')


def create_latents(image, p, dtype=None, device=None):
    from modules.processing import create_random_tensors
    from PIL import Image
    if image is None:
        return image
    elif isinstance(image, Image.Image):
        latents = vae_encode(image, model=shared.sd_model, full_quality=p.full_quality)
    elif isinstance(image, list):
        latents = [vae_encode(i, model=shared.sd_model, full_quality=p.full_quality).squeeze(dim=0) for i in image]
        latents = torch.stack(latents, dim=0).to(shared.device)
    else:
        shared.log.warning(f'Latents: input type: {type(image)} {image}')
        return image
    noise = p.denoising_strength * create_random_tensors(latents.shape[1:], seeds=p.all_seeds, subseeds=p.all_subseeds, subseed_strength=p.subseed_strength, p=p)
    latents = (1 - p.denoising_strength) * latents + noise
    if dtype is not None:
        latents = latents.to(dtype=dtype)
    if device is not None:
        latents = latents.to(device=device)
    return latents


def full_vae_decode(latents, model):
    t0 = time.time()
    if shared.opts.diffusers_move_unet and not getattr(model, 'has_accelerate', False) and hasattr(model, 'unet'):
        shared.log.debug('Moving to CPU: model=UNet')
        unet_device = model.unet.device
        model.unet.to(devices.cpu)
        devices.torch_gc()
    if not shared.cmd_opts.lowvram and not shared.opts.diffusers_seq_cpu_offload and hasattr(model, 'vae'):
        model.vae.to(devices.device)
    latents.to(model.vae.device)

    upcast = (model.vae.dtype == torch.float16) and getattr(model.vae.config, 'force_upcast', False) and hasattr(model, 'upcast_vae')
    if upcast: # this is done by diffusers automatically if output_type != 'latent'
        model.upcast_vae()
        latents = latents.to(next(iter(model.vae.post_quant_conv.parameters())).dtype)

    decoded = model.vae.decode(latents / model.vae.config.scaling_factor, return_dict=False)[0]

    # Delete PyTorch VAE after OpenVINO compile
    if shared.opts.cuda_compile and shared.opts.cuda_compile_backend == "openvino_fx" and shared.compiled_model_state.first_pass_vae:
        shared.compiled_model_state.first_pass_vae = False
        if hasattr(shared.sd_model, "vae"):
            model.vae.apply(sd_models.convert_to_faketensors)
            devices.torch_gc(force=True)

    if shared.opts.diffusers_move_unet and not getattr(model, 'has_accelerate', False) and hasattr(model, 'unet'):
        model.unet.to(unet_device)
    t1 = time.time()
    debug(f'VAE decode: name={sd_vae.loaded_vae_file if sd_vae.loaded_vae_file is not None else "baked"} dtype={model.vae.dtype} upcast={upcast} images={latents.shape[0]} latents={latents.shape} time={round(t1-t0, 3)}')
    return decoded


def full_vae_encode(image, model):
    debug(f'VAE encode: name={sd_vae.loaded_vae_file if sd_vae.loaded_vae_file is not None else "baked"} dtype={model.vae.dtype} upcast={model.vae.config.get("force_upcast", None)}')
    if shared.opts.diffusers_move_unet and not getattr(model, 'has_accelerate', False) and hasattr(model, 'unet'):
        debug('Moving to CPU: model=UNet')
        unet_device = model.unet.device
        model.unet.to(devices.cpu)
        devices.torch_gc()
    if not shared.cmd_opts.lowvram and not shared.opts.diffusers_seq_cpu_offload and hasattr(model, 'vae'):
        model.vae.to(devices.device)
    encoded = model.vae.encode(image.to(model.vae.device, model.vae.dtype)).latent_dist.sample()
    if shared.opts.diffusers_move_unet and not getattr(model, 'has_accelerate', False) and hasattr(model, 'unet'):
        model.unet.to(unet_device)
    return encoded


def taesd_vae_decode(latents):
    debug(f'VAE decode: name=TAESD images={len(latents)} latents={latents.shape} slicing={shared.opts.diffusers_vae_slicing}')
    if len(latents) == 0:
        return []
    if shared.opts.diffusers_vae_slicing:
        decoded = torch.zeros((len(latents), 3, latents.shape[2] * 8, latents.shape[3] * 8), dtype=devices.dtype_vae, device=devices.device)
        for i in range(latents.shape[0]):
            decoded[i] = sd_vae_taesd.decode(latents[i])
    else:
        decoded = sd_vae_taesd.decode(latents)
    return decoded


def taesd_vae_encode(image):
    debug(f'VAE encode: name=TAESD image={image.shape}')
    encoded = sd_vae_taesd.encode(image)
    return encoded


def vae_decode(latents, model, output_type='np', full_quality=True):
    t0 = time.time()
    prev_job = shared.state.job
    shared.state.job = 'vae'
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
    if latents.shape[0] == 4 and latents.shape[1] != 4: # likely animatediff latent
        latents = latents.permute(1, 0, 2, 3)
    if len(latents.shape) == 3: # lost a batch dim in hires
        latents = latents.unsqueeze(0)
    if full_quality:
        decoded = full_vae_decode(latents=latents, model=shared.sd_model)
    else:
        decoded = taesd_vae_decode(latents=latents)
    # TODO validate decoded sample diffusers
    # decoded = validate_sample(decoded)
    if hasattr(model, 'image_processor'):
        imgs = model.image_processor.postprocess(decoded, output_type=output_type)
    else:
        import diffusers
        image_processor = diffusers.image_processor.VaeImageProcessor()
        imgs = image_processor.postprocess(decoded, output_type=output_type)
    shared.state.job = prev_job
    if shared.cmd_opts.profile:
        t1 = time.time()
        shared.log.debug(f'Profile: VAE decode: {t1-t0:.2f}')
    return imgs


def vae_encode(image, model, full_quality=True): # pylint: disable=unused-variable
    if shared.state.interrupted or shared.state.skipped:
        return []
    if not hasattr(model, 'vae'):
        shared.log.error('VAE not found in model')
        return []
    tensor = TF.to_tensor(image.convert("RGB")).unsqueeze(0).to(devices.device, devices.dtype_vae)
    if full_quality:
        tensor = tensor * 2 - 1
        latents = full_vae_encode(image=tensor, model=shared.sd_model)
    else:
        latents = taesd_vae_encode(image=tensor)
    return latents
