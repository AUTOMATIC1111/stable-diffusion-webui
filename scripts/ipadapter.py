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
loaded = None
ADAPTERS = {
    'None': 'none',
    'Base': 'ip-adapter_sd15',
    'Light': 'ip-adapter_sd15_light',
    'Plus': 'ip-adapter-plus_sd15',
    'Plus Face': 'ip-adapter-plus-face_sd15',
    'Full face': 'ip-adapter-full-face_sd15',
    'Base SXDL': 'ip-adapter_sdxl',
    # 'models/ip-adapter_sd15_vit-G', # RuntimeError: mat1 and mat2 shapes cannot be multiplied (2x1024 and 1280x3072)
    # 'sdxl_models/ip-adapter_sdxl_vit-h',
    # 'sdxl_models/ip-adapter-plus_sdxl_vit-h',
    # 'sdxl_models/ip-adapter-plus-face_sdxl_vit-h',
}


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

    def process(self, p: processing.StableDiffusionProcessing, adapter, scale, image): # pylint: disable=arguments-differ
        # overrides
        adapter = ADAPTERS.get(adapter, None)
        if hasattr(p, 'ip_adapter_name'):
            adapter = p.ip_adapter_name
        if hasattr(p, 'ip_adapter_scale'):
            scale = p.ip_adapter_scale
        if hasattr(p, 'ip_adapter_image'):
            image = p.ip_adapter_image
        if adapter is None:
            return
        # init code
        global loaded, image_encoder, image_encoder_type # pylint: disable=global-statement
        if shared.sd_model is None:
            return
        if shared.backend != shared.Backend.DIFFUSERS:
            shared.log.warning('IP adapter: not in diffusers mode')
            return
        if adapter == 'none':
            if hasattr(shared.sd_model, 'set_ip_adapter_scale'):
                shared.sd_model.set_ip_adapter_scale(0)
            if loaded is not None:
                shared.log.debug('IP adapter: unload attention processor')
                shared.sd_model.unet.set_default_attn_processor()
                shared.sd_model.unet.config.encoder_hid_dim_type = None
                loaded = None
            return
        if image is None:
            shared.log.error('IP adapter: no image')
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
            if image_encoder is None or image_encoder_type != shared.sd_model_type:
                try:
                    from transformers import CLIPVisionModelWithProjection
                    image_encoder = CLIPVisionModelWithProjection.from_pretrained("h94/IP-Adapter", subfolder=subfolder, torch_dtype=devices.dtype, cache_dir=shared.opts.diffusers_dir, use_safetensors=True).to(devices.device)
                    image_encoder_type = shared.sd_model_type
                except Exception as e:
                    shared.log.error(f'IP adapter: failed to load image encoder: {e}')
                    return

        # main code
        subfolder = 'models' if 'sd15' in adapter else 'sdxl_models'
        if adapter != loaded or getattr(shared.sd_model.unet.config, 'encoder_hid_dim_type', None) is None:
            t0 = time.time()
            if loaded is not None:
                shared.log.debug('IP adapter: reset attention processor')
                shared.sd_model.unet.set_default_attn_processor()
                loaded = None
            else:
                shared.log.debug('IP adapter: load attention processor')
            shared.sd_model.image_encoder = image_encoder
            shared.sd_model.load_ip_adapter("h94/IP-Adapter", subfolder=subfolder, weight_name=f'{adapter}.safetensors')
            t1 = time.time()
            shared.log.info(f'IP adapter load: adapter="{adapter}" scale={scale} image={image} time={t1-t0:.2f}')
            loaded = adapter
        else:
            shared.log.debug(f'IP adapter cache: adapter="{adapter}" scale={scale} image={image}')
        shared.sd_model.set_ip_adapter_scale(scale)
        p.task_args['ip_adapter_image'] = p.batch_size * [image]
        p.extra_generation_params["IP Adapter"] = f'{adapter}:{scale}'
