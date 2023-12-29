import gradio as gr
from modules import scripts, processing, shared, sd_models


title = 'BLIP Diffusion'


class Script(scripts.Script):
    def title(self):
        return title

    def show(self, is_img2img):
        return is_img2img if shared.backend == shared.Backend.DIFFUSERS else False

    def ui(self, _is_img2img):
        with gr.Row():
            source_subject = gr.Textbox(value='', label='Source subject')
        with gr.Row():
            target_subject = gr.Textbox(value='', label='Target subject')
        with gr.Row():
            prompt_strength = gr.Slider(label='Prompt strength', minimum=0.0, maximum=1.0, step=0.01, value=0.5)
        return [source_subject, target_subject, prompt_strength]

    def run(self, p: processing.StableDiffusionProcessing, source_subject, target_subject, prompt_strength): # pylint: disable=arguments-differ, unused-argument
        c = shared.sd_model.__class__.__name__ if shared.sd_model is not None else ''
        if c != 'BlipDiffusionPipeline':
            shared.log.error(f'{title}: model selected={c} required=BLIPDiffusion')
            return None
        if hasattr(p, 'init_images') and len(p.init_images) > 0:
            p.task_args['reference_image'] = p.init_images[0]
            p.task_args['prompt'] = [p.prompt]
            p.task_args['neg_prompt'] = p.negative_prompt
            p.task_args['prompt_strength'] = prompt_strength
            p.task_args['source_subject_category'] = [source_subject]
            p.task_args['target_subject_category'] = [target_subject]
            p.task_args['output_type'] = 'pil'
            shared.log.debug(f'BLIP Diffusion: args={p.task_args}')
            shared.sd_model = sd_models.set_diffuser_pipe(shared.sd_model, sd_models.DiffusersTaskType.IMAGE_2_IMAGE)
            processed = processing.process_images(p)
            return processed
        else:
            shared.log.error(f'{title}: no init_images')
            return None
