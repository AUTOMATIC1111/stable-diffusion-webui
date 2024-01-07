"""
Lightweight IP-Adapter applied to existing pipeline in Diffusers
- Downloads image_encoder or first usage (2.5GB)
- Introduced via: https://github.com/huggingface/diffusers/pull/5713
- IP adapters: https://huggingface.co/h94/IP-Adapter
TODO:
- Additional IP addapters
- SD/SDXL autodetect
"""

import time
import gradio as gr
from modules import scripts, processing, shared, devices


image_encoder = None
image_encoder_type = None
image_encoder_name = None
loaded = None
checkpoint = None
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


def apply(pipe, p: processing.StableDiffusionProcessing, adapter_name, scale, image): # pylint: disable=arguments-differ
    # overrides
    adapter = ADAPTERS.get(adapter_name, None)
    if hasattr(p, 'ip_adapter_name'):
        adapter = p.ip_adapter_name
    if hasattr(p, 'ip_adapter_scale'):
        scale = p.ip_adapter_scale
    if hasattr(p, 'ip_adapter_image'):
        image = p.ip_adapter_image
    if adapter is None:
        return False
    # init code
    global loaded, checkpoint, image_encoder, image_encoder_type, image_encoder_name # pylint: disable=global-statement
    if pipe is None:
        return False
    if shared.backend != shared.Backend.DIFFUSERS:
        shared.log.warning('IP adapter: not in diffusers mode')
        return
    if image is None and adapter != 'none':
        shared.log.error('IP adapter: no image provided')
        adapter = 'none' # unload adapter if previously loaded as it will cause runtime errors
    if adapter == 'none':
        if hasattr(pipe, 'set_ip_adapter_scale'):
            pipe.set_ip_adapter_scale(0)
        if loaded is not None:
            shared.log.debug('IP adapter: unload attention processor')
            pipe.unet.config.encoder_hid_dim_type = None
            loaded = None
        return False
    if not hasattr(pipe, 'load_ip_adapter'):
        import diffusers
        diffusers.StableDiffusionPipeline.load_ip_adapter()
        shared.log.error(f'IP adapter: pipeline not supported: {pipe.__class__.__name__}')
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

    # load image encoder used by ip adapter
    if getattr(pipe, 'image_encoder', None) is None or image_encoder_name != clip_repo + '/' + subfolder:
        if image_encoder is None or image_encoder_type != shared.sd_model_type or checkpoint != shared.opts.sd_model_checkpoint or image_encoder_name != clip_repo + '/' + subfolder:
            if shared.sd_model_type != 'sd' and shared.sd_model_type != 'sdxl':
                shared.log.error(f'IP adapter: unsupported model type: {shared.sd_model_type}')
                return False
            try:
                from transformers import CLIPVisionModelWithProjection
                shared.log.debug(f'IP adapter: load image encoder: {clip_repo}/{subfolder}')
                image_encoder = CLIPVisionModelWithProjection.from_pretrained(clip_repo, subfolder=subfolder, torch_dtype=devices.dtype, cache_dir=shared.opts.diffusers_dir, use_safetensors=True).to(devices.device)
                image_encoder_type = shared.sd_model_type
                image_encoder_name = clip_repo + '/' + subfolder
            except Exception as e:
                shared.log.error(f'IP adapter: failed to load image encoder: {e}')
                return
    if getattr(pipe, 'feature_extractor', None) is None:
        from transformers import CLIPImageProcessor
        shared.log.debug('IP adapter: load feature extractor')
        pipe.feature_extractor = CLIPImageProcessor()

    # main code
    # subfolder = 'models' if 'sd15' in adapter else 'sdxl_models'
    if adapter != loaded or getattr(pipe.unet.config, 'encoder_hid_dim_type', None) is None or checkpoint != shared.opts.sd_model_checkpoint:
        t0 = time.time()
        if loaded is not None:
            shared.log.debug('IP adapter: reset attention processor')
            loaded = None
        else:
            shared.log.debug('IP adapter: load attention processor')
        pipe.image_encoder = image_encoder
        subfolder = 'models' if shared.sd_model_type == 'sd' else 'sdxl_models'
        pipe.load_ip_adapter(base_repo, subfolder=subfolder, weight_name=adapter)
        t1 = time.time()
        shared.log.info(f'IP adapter load: adapter="{adapter}" scale={scale} image={image} time={t1-t0:.2f}')
        loaded = adapter
        checkpoint = shared.opts.sd_model_checkpoint
    else:
        shared.log.debug(f'IP adapter cache: adapter="{adapter}" scale={scale} image={image}')
    pipe.set_ip_adapter_scale(scale)

    if isinstance(image, str):
        from modules.api.api import decode_base64_to_image
        image = decode_base64_to_image(image)

    p.task_args['ip_adapter_image'] = p.batch_size * [image]
    p.extra_generation_params["IP Adapter"] = f'{adapter}:{scale}'
    return True


class Script(scripts.Script):
    def title(self):
        return 'IP Adapter'

    def show(self, is_img2img):
        return scripts.AlwaysVisible if shared.backend == shared.Backend.DIFFUSERS else False

    def ui(self, _is_img2img):
        with gr.Accordion('IP Adapter', open=False, elem_id='ipadapter'):
            with gr.Row():
                adapter = gr.Dropdown(label='Adapter', choices=list(ADAPTERS), value='none')
                scale = gr.Slider(label='Scale', minimum=0.0, maximum=1.0, step=0.01, value=0.5)
            with gr.Row():
                image = gr.Image(image_mode='RGB', label='Image', source='upload', type='pil', width=512)
        return [adapter, scale, image]

    def process(self, p: processing.StableDiffusionProcessing, adapter_name, scale, image): # pylint: disable=arguments-differ
        apply(shared.sd_model, p, adapter_name, scale, image)
