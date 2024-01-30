import gradio as gr
from modules import scripts, processing, shared, ipadapter


class Script(scripts.Script):
    def title(self):
        return 'IP Adapter'

    def show(self, is_img2img):
        return scripts.AlwaysVisible if shared.backend == shared.Backend.DIFFUSERS else False

    def ui(self, _is_img2img):
        with gr.Accordion('IP Adapter', open=False, elem_id='ipadapter'):
            with gr.Row():
                enabled = gr.Checkbox(label='Enabled', value=False)
            with gr.Row():
                adapter = gr.Dropdown(label='Adapter', choices=list(ipadapter.ADAPTERS), value='None')
                scale = gr.Slider(label='Scale', minimum=0.0, maximum=1.0, step=0.01, value=0.5)
            with gr.Row():
                image = gr.Image(image_mode='RGB', label='Image', source='upload', type='pil', width=512)
        return [enabled, adapter, scale, image]

    def process(self, p: processing.StableDiffusionProcessing, enabled, adapter_name, scale, image): # pylint: disable=arguments-differ
        if shared.backend != shared.Backend.DIFFUSERS:
            return
        if enabled:
            p.ip_adapter_name = adapter_name
            p.ip_adapter_scale = scale
            p.ip_adapter_image = image
            # ipadapter.apply(shared.sd_model, p, adapter_name, scale, image) # called directly from processing.process_images_inner
