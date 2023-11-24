"""
lightweight ip-adapter applied to existing pipeline
- downloads image_encoder or first usage (2.5GB)
- introduced via: https://github.com/huggingface/diffusers/pull/5713
- ip adapters: https://huggingface.co/h94/IP-Adapter
"""

import gradio as gr
from modules import scripts, processing


image_encoder = None
ADAPTERS = [
    'none',
    'models/ip-adapter_sd15',
    'models/ip-adapter_sd15_light',
    # 'models/ip-adapter_sd15_vit-G', # RuntimeError: mat1 and mat2 shapes cannot be multiplied (2x1024 and 1280x3072)
    # 'models/ip-adapter-plus_sd15', # KeyError: 'proj.weight'
    # 'models/ip-adapter-plus-face_sd15', # KeyError: 'proj.weight'
    # 'models/ip-adapter-full-face_sd15', # KeyError: 'proj.weight'
    'sdxl_models/ip-adapter_sdxl',
    # 'sdxl_models/ip-adapter_sdxl_vit-h',
    # 'sdxl_models/ip-adapter-plus_sdxl_vit-h',
    # 'sdxl_models/ip-adapter-plus-face_sdxl_vit-h',
]


# main processing used in both modes
def before_process(p: processing.StableDiffusionProcessing, adapter, scale, image):
    import torch
    import transformers
    from modules import shared, devices

    # init code
    if shared.sd_model is None:
        return
    if adapter == 'none' or image is None:
        if hasattr(shared.sd_model, 'set_ip_adapter_scale'):
            shared.sd_model.set_ip_adapter_scale(0)
        return
    if shared.backend != shared.Backend.DIFFUSERS:
        shared.log.warning('IP adapter: not in diffusers mode')
        return
    if not hasattr(shared.sd_model, 'load_ip_adapter'):
        shared.log.error(f'IP adapter: pipeline not supported: {shared.sd_model.__class__.__name__}')
        return
    if getattr(shared.sd_model, 'image_encoder', None) is None:
        if shared.sd_model_type == 'sd':
            subfolder = 'models/image_encoder'
        elif shared.sd_model_type == 'sdxl':
            subfolder = 'sdxl_models/image_encoder'
        else:
            shared.log.error(f'IP adapter: unsupported model type: {shared.sd_model_type}')
            return
        global image_encoder # pylint: disable=global-statement
        if image_encoder is None:
            try:
                image_encoder = transformers.CLIPVisionModelWithProjection.from_pretrained("h94/IP-Adapter", subfolder=subfolder, torch_dtype=torch.float16, cache_dir=shared.opts.diffusers_dir, use_safetensors=True).to(devices.device)
            except Exception as e:
                shared.log.error(f'IP adapter: failed to load image encoder: {e}')
                return

    # main code
    subfolder, model = adapter.split('/')
    shared.log.info(f'IP adapter: scale={scale} adapter="{model}" image={image}')
    shared.sd_model.image_encoder = image_encoder
    shared.sd_model.load_ip_adapter("h94/IP-Adapter", subfolder=subfolder, weight_name=f'{model}.safetensors')
    shared.sd_model.set_ip_adapter_scale(scale)
    p.task_args = { 'ip_adapter_image': image }
    p.extra_generation_params["IP Adapter"] = f'{adapter}:{scale}'


# defines script for dual-mode usage
class Script(scripts.Script):
    # see below for all available options and callbacks
    # <https://github.com/vladmandic/automatic/blob/master/modules/scripts.py#L26>

    def title(self):
        return 'IP Adapter'

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    # return signature is array of gradio components
    def ui(self, _is_img2img):
        with gr.Accordion('IP Adapter', open=False, elem_id='ipadapter'):
            with gr.Row():
                adapter = gr.Dropdown(label='Adapter', choices=ADAPTERS, value='none')
                scale = gr.Slider(label='Scale', minimum=0.0, maximum=1.0, step=0.01, value=0.5)
            with gr.Row():
                image = gr.Image(image_mode='RGB', label='Image', source='upload', type='pil', width=512)
        return [adapter, scale, image]

    # triggered by callback
    def before_process(self, p: processing.StableDiffusionProcessing, *args): # pylint: disable=arguments-differ
        before_process(p, *args)
