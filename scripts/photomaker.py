import os
import gradio as gr
import huggingface_hub as hf
from PIL import Image
from modules import shared, processing, sd_models, scripts


class Script(scripts.Script):
    def title(self):
        return 'PhotoMaker'

    def show(self, is_img2img):
        return True if shared.backend == shared.Backend.DIFFUSERS else False

    def load_images(self, files):
        init_images = []
        for file in files or []:
            try:
                if isinstance(file, str):
                    from modules.api.api import decode_base64_to_image
                    image = decode_base64_to_image(file)
                elif isinstance(file, Image.Image):
                    image = file
                elif isinstance(file, dict) and 'name' in file:
                    image = Image.open(file['name']) # _TemporaryFileWrapper from gr.Files
                elif hasattr(file, 'name'):
                    image = Image.open(file.name) # _TemporaryFileWrapper from gr.Files
                else:
                    raise ValueError(f'PhotoMaker unknown input: {file}')
                init_images.append(image)
            except Exception as e:
                shared.log.warning(f'PhotoMaker failed to load image: {e}')
        return init_images

    def ui(self, _is_img2img):
        with gr.Row():
            trigger = gr.Text(label='Trigger word', value="person")
            strength = gr.Slider(label='Strength', minimum=0.0, maximum=2.0, step=0.01, value=1.0)
            start = gr.Slider(label='Start', minimum=0.0, maximum=1.0, step=0.01, value=0.5)
        with gr.Row():
            files = gr.File(label='Input images', file_count='multiple', file_types=['image'], type='file', interactive=True, height=100)
        with gr.Row():
            gallery = gr.Gallery(show_label=False, value=[])
        with gr.Row():
            gr.HTML('<a href="https://github.com/TencentARC/PhotoMaker>PhotoMaker</a>"')
        files.change(fn=self.load_images, inputs=[files], outputs=[gallery])
        return [trigger, strength, start, gallery]

    # Run pipeline
    def run(self, p: processing.StableDiffusionProcessing, trigger, strength, start, images): # pylint: disable=arguments-differ
        from scripts.photomaker_model import PhotoMakerStableDiffusionXLPipeline # pylint: disable=no-name-in-module

        # prepare pipeline
        input_images = self.load_images(images)
        if len(input_images) == 0:
            shared.log.warning('PhotoMaker: no input images')
            return None

        c = shared.sd_model.__class__.__name__ if shared.sd_model is not None else ''
        if c != 'StableDiffusionXLPipeline':
            shared.log.warning(f'PhotoMaker invalid base model: current={c} required=StableDiffusionXLPipeline')
            return None

        # validate prompt
        trigger_ids = shared.sd_model.tokenizer.encode(trigger) + shared.sd_model.tokenizer_2.encode(trigger)
        prompt_ids1 = shared.sd_model.tokenizer.encode(p.prompt)
        prompt_ids2 = shared.sd_model.tokenizer_2.encode(p.prompt)
        for t in trigger_ids:
            if prompt_ids1.count(t) != 1:
                shared.log.error(f'PhotoMaker: trigger word not matched in prompt: {trigger} ids={trigger_ids} prompt={p.prompt} ids={prompt_ids1}')
                return None
            if prompt_ids2.count(t) != 1:
                shared.log.error(f'PhotoMaker: trigger word not matched in prompt: {trigger} ids={trigger_ids} prompt={p.prompt} ids={prompt_ids1}')
                return None

        # create new pipeline
        orig_pipeline = shared.sd_model # backup current pipeline definition
        shared.sd_model = PhotoMakerStableDiffusionXLPipeline(
            vae = shared.sd_model.vae,
            text_encoder=shared.sd_model.text_encoder,
            text_encoder_2=shared.sd_model.text_encoder_2,
            tokenizer=shared.sd_model.tokenizer,
            tokenizer_2=shared.sd_model.tokenizer_2,
            unet=shared.sd_model.unet,
            scheduler=shared.sd_model.scheduler,
            force_zeros_for_empty_prompt=shared.opts.diffusers_force_zeros,
        )
        sd_models.copy_diffuser_options(shared.sd_model, orig_pipeline) # copy options from original pipeline
        sd_models.set_diffuser_options(shared.sd_model) # set all model options such as fp16, offload, etc.
        if not ((shared.opts.diffusers_model_cpu_offload or shared.cmd_opts.medvram) or (shared.opts.diffusers_seq_cpu_offload or shared.cmd_opts.lowvram)):
            shared.sd_model.to(shared.device) # move pipeline if needed, but don't touch if its under automatic managment

        orig_prompt_attention = shared.opts.prompt_attention
        shared.opts.data['prompt_attention'] = 'Fixed attention' # otherwise need to deal with class_tokens_mask
        p.task_args['input_id_images'] = input_images
        p.task_args['start_merge_step'] = int(start * p.steps)
        p.task_args['prompt'] = p.prompt # override all logic

        photomaker_path = hf.hf_hub_download(repo_id="TencentARC/PhotoMaker", filename="photomaker-v1.bin", repo_type="model", cache_dir=shared.opts.diffusers_dir)
        shared.log.debug(f'PhotoMaker: model={photomaker_path} images={len(input_images)} trigger={trigger} args={p.task_args}')

        # load photomaker adapter
        shared.sd_model.load_photomaker_adapter(
            os.path.dirname(photomaker_path),
            subfolder="",
            weight_name=os.path.basename(photomaker_path),
            trigger_word=trigger
        )
        shared.sd_model.set_adapters(["photomaker"], adapter_weights=[strength])

        # run processing
        processed: processing.Processed = processing.process_images(p)
        p.extra_generation_params['PhotoMaker'] = f'{strength}'

        # restore original pipeline
        shared.opts.data['prompt_attention'] = orig_prompt_attention
        shared.sd_model = orig_pipeline
        return processed
