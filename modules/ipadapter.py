"""
Lightweight IP-Adapter applied to existing pipeline in Diffusers
- Downloads image_encoder or first usage (2.5GB)
- Introduced via: https://github.com/huggingface/diffusers/pull/5713
- IP adapters: https://huggingface.co/h94/IP-Adapter
TODO ipadapter items:
- SD/SDXL autodetect
"""

import time
from modules import processing, shared, devices


image_encoder = None
feature_extractor = None
image_encoder_type = None
image_encoder_name = None
base_repo = "h94/IP-Adapter"
ADAPTERS = {
    'None': 'none',
    'Base': 'ip-adapter_sd15.safetensors',
    'Base ViT-G': 'ip-adapter_sd15_vit-G.safetensors',
    'Light': 'ip-adapter_sd15_light.safetensors',
    'Plus': 'ip-adapter-plus_sd15.safetensors',
    'Plus Face': 'ip-adapter-plus-face_sd15.safetensors',
    'Full Face': 'ip-adapter-full-face_sd15.safetensors',
    'Base SXDL': 'ip-adapter_sdxl.safetensors',
    'Base ViT-H SXDL': 'ip-adapter_sdxl_vit-h.safetensors',
    'Plus ViT-H SXDL': 'ip-adapter-plus_sdxl_vit-h.safetensors',
    'Plus Face ViT-H SXDL': 'ip-adapter-plus-face_sdxl_vit-h.safetensors',
}

def unapply(pipe): # pylint: disable=arguments-differ
    try:
        if hasattr(pipe, 'set_ip_adapter_scale'):
            pipe.set_ip_adapter_scale(0)
        if hasattr(pipe, 'unet') and hasattr(pipe.unet, 'config')and pipe.unet.config.encoder_hid_dim_type == 'ip_image_proj':
            pipe.unet.encoder_hid_proj = None
            pipe.config.encoder_hid_dim_type = None
            pipe.unet.set_default_attn_processor()
    except Exception:
        pass


def apply(pipe, p: processing.StableDiffusionProcessing, adapter_name='None', scale=1.0, image=None):
    # overrides
    if hasattr(p, 'ip_adapter_name'):
        adapter = ADAPTERS.get(p.ip_adapter_name, None)
        adapter_name = p.ip_adapter_name
    else:
        adapter = ADAPTERS.get(adapter_name, None)
    if hasattr(p, 'ip_adapter_scale'):
        scale = p.ip_adapter_scale
    if hasattr(p, 'ip_adapter_image'):
        image = p.ip_adapter_image
    if adapter is None:
        unapply(pipe)
        return False
    # init code
    global image_encoder, image_encoder_type, image_encoder_name, feature_extractor # pylint: disable=global-statement
    if pipe is None:
        return False
    if shared.backend != shared.Backend.DIFFUSERS:
        shared.log.warning('IP adapter: not in diffusers mode')
        return False
    if image is None and adapter != 'none':
        shared.log.error('IP adapter: no image provided')
        adapter = 'none' # unload adapter if previously loaded as it will cause runtime errors
    if adapter == 'none':
        unapply(pipe)
    if not hasattr(pipe, 'load_ip_adapter'):
        shared.log.error(f'IP adapter: pipeline not supported: {pipe.__class__.__name__}')
        return False
    if shared.sd_model_type != 'sd' and shared.sd_model_type != 'sdxl':
        shared.log.error(f'IP adapter: unsupported model type: {shared.sd_model_type}')
        return False

    # which clip to use
    if 'ViT' not in adapter_name:
        clip_repo = base_repo
        subfolder = 'models/image_encoder' if shared.sd_model_type == 'sd' else 'sdxl_models/image_encoder' # defaults per model
    elif 'ViT-H' in adapter_name:
        clip_repo = base_repo
        subfolder = 'models/image_encoder' # this is vit-h
    elif 'ViT-G' in adapter_name:
        clip_repo = base_repo
        subfolder = 'sdxl_models/image_encoder' # this is vit-g
    else:
        shared.log.error(f'IP adapter: unknown model type: {adapter_name}')
        return False

    # load feature extractor used by ip adapter
    if feature_extractor is None:
        from transformers import CLIPImageProcessor
        shared.log.debug('IP adapter load: feature extractor')
        feature_extractor = CLIPImageProcessor()
    # load image encoder used by ip adapter
    if image_encoder is None or image_encoder_name != clip_repo + '/' + subfolder or image_encoder_type != shared.sd_model_type:
        try:
            from transformers import CLIPVisionModelWithProjection
            shared.log.debug(f'IP adapter load: image encoder="{clip_repo}/{subfolder}"')
            image_encoder = CLIPVisionModelWithProjection.from_pretrained(clip_repo, subfolder=subfolder, torch_dtype=devices.dtype, cache_dir=shared.opts.diffusers_dir, use_safetensors=True).to(devices.device)
            image_encoder_type = shared.sd_model_type
            image_encoder_name = clip_repo + '/' + subfolder
        except Exception as e:
            shared.log.error(f'IP adapter: failed to load image encoder: {e}')
            return

    # main code
    # subfolder = 'models' if 'sd15' in adapter else 'sdxl_models'
    t0 = time.time()
    subfolder = 'models' if shared.sd_model_type == 'sd' else 'sdxl_models'
    pipe.image_encoder = image_encoder
    pipe.feature_extractor = feature_extractor
    pipe.load_ip_adapter(base_repo, subfolder=subfolder, weight_name=adapter)
    pipe.set_ip_adapter_scale(scale)
    t1 = time.time()
    shared.log.info(f'IP adapter: adapter="{adapter}" scale={scale} image={image} time={t1-t0:.2f}')

    if isinstance(image, str):
        from modules.api.api import decode_base64_to_image
        image = decode_base64_to_image(image).convert("RGB")

    p.task_args['ip_adapter_image'] = p.batch_size * [image]
    p.extra_generation_params["IP Adapter"] = f'{adapter}:{scale}'
    return True
