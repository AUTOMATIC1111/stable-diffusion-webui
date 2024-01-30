import gradio as gr
import torch
from modules import shared, devices, scripts, processing, sd_models


checked_ok = False


def check_dependencies():
    global checked_ok # pylint: disable=global-statement
    from installer import installed, install
    packages = [
        ('ligo-segments', 'ligo-segments'),
    ]
    for pkg in packages:
        if not installed(pkg[1], reload=True, quiet=True):
            install(pkg[0], pkg[1], ignore=False)
    try:
        from ligo.segments import segment # pylint: disable=unused-import
        checked_ok = True
        return True
    except Exception as e:
        shared.log.error(f'Mixture tiling: {e}')
        return False


class Script(scripts.Script):
    def title(self):
        return 'Mixture tiling'

    def show(self, is_img2img):
        return not is_img2img if shared.backend == shared.Backend.DIFFUSERS else False

    def ui(self, _is_img2img):
        with gr.Row():
            gr.HTML('<a href="https://arxiv.org/abs/2302.02412">&nbsp Mixture tiling</a>')
        with gr.Row():
            gr.HTML('<span>&nbsp Separated prompts using new lines<br>&nbsp Number of prompts must matcxh X*Y</span>')
        with gr.Row():
            x_size = gr.Slider(label='X components', minimum=1, maximum=5, step=1, value=1)
            y_size = gr.Slider(label='Y components', minimum=1, maximum=5, step=1, value=1)
        with gr.Row():
            x_overlap = gr.Slider(label='X overlap', minimum=0, maximum=1, step=0.01, value=0.5)
            y_overlap = gr.Slider(label='Y overlap', minimum=0, maximum=1, step=0.01, value=0.5)
        return x_size, y_size, x_overlap, y_overlap

    def run(self, p: processing.StableDiffusionProcessing, x_size, y_size, x_overlap, y_overlap): # pylint: disable=arguments-differ
        if not checked_ok:
            if not check_dependencies():
                return
        prompts = p.prompt.splitlines()
        if len(prompts) != x_size * y_size:
            shared.log.error(f'Mixture tiling prompt count mismatch: prompts={len(prompts)} required={x_size * y_size}')
            return
        # backup pipeline and params
        orig_pipeline = shared.sd_model
        orig_dtype = devices.dtype
        orig_prompt_attention = shared.opts.prompt_attention
        # create pipeline
        if shared.sd_model_type != 'sd':
            shared.log.error(f'Mixture tiling: incorrect base model: {shared.sd_model.__class__.__name__}')
            return
        shared.sd_model = sd_models.switch_pipe('mixture_tiling', shared.sd_model)
        if shared.sd_model.__class__.__name__ != 'StableDiffusionTilingPipeline': # switch failed
            shared.log.error(f'Mixture tiling: not a tiling pipeline: {shared.sd_model.__class__.__name__}')
            shared.sd_model = orig_pipeline
            return
        sd_models.set_diffuser_options(shared.sd_model)
        shared.opts.data['prompt_attention'] = 'Fixed attention' # this pipeline is not compatible with embeds
        shared.sd_model.to(torch.float32) # this pipeline unet is not compatible with fp16
        processing.fix_seed(p)
        # set pipeline specific params, note that standard params are applied when applicable
        y_prompts = []
        for y in range(y_size):
            x_prompts = []
            for x in range(x_size):
                x_prompts.append(prompts[y * x_size + x])
            y_prompts.append(x_prompts)
        p.task_args['prompt'] = y_prompts
        p.task_args['seed'] = p.seed
        p.task_args['tile_width'] = p.height
        p.task_args['tile_height'] = p.width
        p.task_args['tile_col_overlap'] = int(p.height * x_overlap)
        p.task_args['tile_row_overlap'] = int(p.width * y_overlap)
        p.task_args['output_type'] = 'np'
        # run pipeline
        shared.log.debug(f'Tiling: args={p.task_args}')
        processed: processing.Processed = processing.process_images(p) # runs processing using main loop
        # restore pipeline and params
        shared.opts.data['prompt_attention'] = orig_prompt_attention
        shared.sd_model = orig_pipeline
        shared.sd_model.to(orig_dtype)
        return processed
