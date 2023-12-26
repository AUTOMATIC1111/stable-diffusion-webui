import time
from PIL import Image
from modules import shared, processing, devices


image_encoder = None
image_encoder_type = None
loaded = None
ADAPTERS = [
    'none',
    'ip-adapter_sd15',
    'ip-adapter_sd15_light',
    'ip-adapter-plus_sd15',
    'ip-adapter-plus-face_sd15',
    'ip-adapter-full-face_sd15',
    # 'models/ip-adapter_sd15_vit-G', # RuntimeError: mat1 and mat2 shapes cannot be multiplied (2x1024 and 1280x3072)
    'ip-adapter_sdxl',
    # 'sdxl_models/ip-adapter_sdxl_vit-h',
    # 'sdxl_models/ip-adapter-plus_sdxl_vit-h',
    # 'sdxl_models/ip-adapter-plus-face_sdxl_vit-h',
]


def apply_ip_adapter(pipe, p: processing.StableDiffusionProcessing, adapter, scale, image, reset=False): # pylint: disable=arguments-differ
    from transformers import CLIPVisionModelWithProjection
    # overrides
    if hasattr(p, 'ip_adapter_name'):
        adapter = p.ip_adapter_name
    if hasattr(p, 'ip_adapter_scale'):
        scale = p.ip_adapter_scale
    if hasattr(p, 'ip_adapter_image'):
        image = p.ip_adapter_image
    # init code
    global loaded, image_encoder, image_encoder_type # pylint: disable=global-statement
    if pipe is None:
        return
    if shared.backend != shared.Backend.DIFFUSERS:
        shared.log.warning('IP adapter: not in diffusers mode')
        return False
    if adapter == 'none':
        if hasattr(pipe, 'set_ip_adapter_scale'):
            pipe.set_ip_adapter_scale(0)
        if loaded is not None:
            shared.log.debug('IP adapter: unload attention processor')
            pipe.unet.set_default_attn_processor()
            pipe.unet.config.encoder_hid_dim_type = None
            loaded = None
        return False
    if image is None:
        image = Image.new('RGB', (512, 512), (0, 0, 0))
    if not hasattr(pipe, 'load_ip_adapter'):
        shared.log.error(f'IP adapter: pipeline not supported: {pipe.__class__.__name__}')
        return False
    if getattr(pipe, 'image_encoder', None) is None or getattr(pipe, 'image_encoder', None) == (None, None):
        if shared.sd_model_type == 'sd':
            subfolder = 'models/image_encoder'
        elif shared.sd_model_type == 'sdxl':
            subfolder = 'sdxl_models/image_encoder'
        else:
            shared.log.error(f'IP adapter: unsupported model type: {shared.sd_model_type}')
            return False
        if image_encoder is None or image_encoder_type != shared.sd_model_type:
            try:
                image_encoder = CLIPVisionModelWithProjection.from_pretrained("h94/IP-Adapter", subfolder=subfolder, torch_dtype=devices.dtype, cache_dir=shared.opts.diffusers_dir, use_safetensors=True).to(devices.device)
                image_encoder_type = shared.sd_model_type
            except Exception as e:
                shared.log.error(f'IP adapter: failed to load image encoder: {e}')
                return False
        pipe.image_encoder = image_encoder

    # main code
    subfolder = 'models' if 'sd15' in adapter else 'sdxl_models'
    if adapter != loaded or getattr(pipe.unet.config, 'encoder_hid_dim_type', None) is None or reset:
        t0 = time.time()
        if loaded is not None:
            # shared.log.debug('IP adapter: reset attention processor')
            pipe.unet.set_default_attn_processor()
            loaded = None
        else:
            shared.log.debug('IP adapter: load attention processor')
        pipe.load_ip_adapter("h94/IP-Adapter", subfolder=subfolder, weight_name=f'{adapter}.safetensors')
        t1 = time.time()
        shared.log.info(f'IP adapter load: adapter="{adapter}" scale={scale} image={image} time={t1-t0:.2f}')
        loaded = adapter
    else:
        shared.log.debug(f'IP adapter cache: adapter="{adapter}" scale={scale} image={image}')
    pipe.set_ip_adapter_scale(scale)
    p.task_args['ip_adapter_image'] = p.batch_size * [image]
    p.extra_generation_params["IP Adapter"] = f'{adapter}:{scale}'
    return True
