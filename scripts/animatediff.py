"""
Lightweight AnimateDiff implementation in Diffusers
Docs: <https://huggingface.co/docs/diffusers/api/pipelines/animatediff>
TODO:
- SDXL
- Custom models
- Custom LORAs
- Enable second pass
- TemporalDiff: https://huggingface.co/CiaraRowles/TemporalDiff/tree/main
- AnimateFace: https://huggingface.co/nlper2022/animatediff_face_512/tree/main
"""

import os
import gradio as gr
import diffusers
from modules import scripts, processing, shared, devices, sd_models


# config
ADAPTERS = {
    'None': None,
    'Motion 1.5 v3' :'vladmandic/animatediff-v3',
    'Motion 1.5 v2' :'guoyww/animatediff-motion-adapter-v1-5-2',
    'Motion 1.5 v1': 'guoyww/animatediff-motion-adapter-v1-5',
    'Motion 1.4': 'guoyww/animatediff-motion-adapter-v1-4',
    'TemporalDiff': 'vladmandic/temporaldiff',
    'AnimateFace': 'vladmandic/animateface',
    # 'LongAnimateDiff 32': 'vladmandic/longanimatediff-32',
    # 'LongAnimateDiff 64': 'vladmandic/longanimatediff-64',
    # 'Motion SD-XL Beta v1' :'vladmandic/animatediff-sdxl',
}
LORAS = {
    'None': None,
    'Zoom-in': 'guoyww/animatediff-motion-lora-zoom-in',
    'Zoom-out': 'guoyww/animatediff-motion-lora-zoom-out',
    'Pan-left': 'guoyww/animatediff-motion-lora-pan-left',
    'Pan-right': 'guoyww/animatediff-motion-lora-pan-right',
    'Tilt-up': 'guoyww/animatediff-motion-lora-tilt-up',
    'Tilt-down': 'guoyww/animatediff-motion-lora-tilt-down',
    'Roll-left': 'guoyww/animatediff-motion-lora-rolling-anticlockwise',
    'Roll-right': 'guoyww/animatediff-motion-lora-rolling-clockwise',
}

# state
motion_adapter = None # instance of diffusers.MotionAdapter
loaded_adapter = None # name of loaded adapter
orig_pipe = None # original sd_model pipeline


def set_adapter(adapter_name: str = 'None'):
    if shared.sd_model is None:
        return
    if shared.backend != shared.Backend.DIFFUSERS:
        shared.log.warning('AnimateDiff: not in diffusers mode')
        return
    global motion_adapter, loaded_adapter, orig_pipe # pylint: disable=global-statement
    # adapter_name = name if name is not None and isinstance(name, str) else loaded_adapter
    if adapter_name is None or adapter_name == 'None' or shared.sd_model is None:
        motion_adapter = None
        loaded_adapter = None
        if orig_pipe is not None:
            shared.log.debug(f'AnimateDiff restore pipeline: adapter="{loaded_adapter}"')
            shared.sd_model = orig_pipe
            orig_pipe = None
        return
    if shared.sd_model_type != 'sd' and shared.sd_model_type != 'sdxl':
        shared.log.warning(f'AnimateDiff: unsupported model type: {shared.sd_model.__class__.__name__}')
        return
    if motion_adapter is not None and loaded_adapter == adapter_name and shared.sd_model.__class__.__name__ == 'AnimateDiffPipeline':
        shared.log.debug(f'AnimateDiff cache: adapter="{adapter_name}"')
        return
    if getattr(shared.sd_model, 'image_encoder', None) is not None:
        shared.log.debug('AnimateDiff: unloading IP adapter')
        # shared.sd_model.image_encoder = None
        # shared.sd_model.unet.set_default_attn_processor()
        shared.sd_model.unet.config.encoder_hid_dim_type = None
    if adapter_name.endswith('.ckpt') or adapter_name.endswith('.safetensors'):
        import huggingface_hub as hf
        folder, filename = os.path.split(adapter_name)
        adapter_name = hf.hf_hub_download(repo_id=folder, filename=filename, cache_dir=shared.opts.diffusers_dir)
    try:
        shared.log.info(f'AnimateDiff load: adapter="{adapter_name}"')
        motion_adapter = None
        motion_adapter = diffusers.MotionAdapter.from_pretrained(adapter_name, cache_dir=shared.opts.diffusers_dir, torch_dtype=devices.dtype, low_cpu_mem_usage=False, device_map=None)
        motion_adapter.to(shared.device)
        sd_models.set_diffuser_options(motion_adapter, vae=None, op='adapter')
        loaded_adapter = adapter_name
        new_pipe = diffusers.AnimateDiffPipeline(
            vae=shared.sd_model.vae,
            text_encoder=shared.sd_model.text_encoder,
            tokenizer=shared.sd_model.tokenizer,
            unet=shared.sd_model.unet,
            scheduler=shared.sd_model.scheduler,
            feature_extractor=getattr(shared.sd_model, 'feature_extractor', None),
            image_encoder=getattr(shared.sd_model, 'image_encoder', None),
            motion_adapter=motion_adapter,
        )
        orig_pipe = shared.sd_model
        shared.sd_model = new_pipe
        if not ((shared.opts.diffusers_model_cpu_offload or shared.cmd_opts.medvram) or (shared.opts.diffusers_seq_cpu_offload or shared.cmd_opts.lowvram)):
            shared.sd_model.to(shared.device)
        sd_models.copy_diffuser_options(new_pipe, orig_pipe)
        sd_models.set_diffuser_options(shared.sd_model, vae=None, op='model')
        shared.log.debug(f'AnimateDiff create pipeline: adapter="{loaded_adapter}"')
    except Exception as e:
        motion_adapter = None
        loaded_adapter = None
        shared.log.error(f'AnimateDiff load error: adapter="{adapter_name}" {e}')


class Script(scripts.Script):
    def title(self):
        return 'AnimateDiff'

    def show(self, _is_img2img):
        return scripts.AlwaysVisible if shared.backend == shared.Backend.DIFFUSERS else False


    def ui(self, _is_img2img):
        def video_type_change(video_type):
            return [
                gr.update(visible=video_type != 'None'),
                gr.update(visible=video_type == 'GIF' or video_type == 'PNG'),
                gr.update(visible=video_type == 'MP4'),
                gr.update(visible=video_type == 'MP4'),
            ]

        with gr.Accordion('AnimateDiff', open=False, elem_id='animatediff'):
            with gr.Row():
                adapter_index = gr.Dropdown(label='Adapter', choices=list(ADAPTERS), value='None')
                frames = gr.Slider(label='Frames', minimum=1, maximum=64, step=1, value=16)
            with gr.Row():
                override_scheduler = gr.Checkbox(label='Override sampler', value=True)
            with gr.Row():
                lora_index = gr.Dropdown(label='Lora', choices=list(LORAS), value='None')
                strength = gr.Slider(label='Strength', minimum=0.0, maximum=2.0, step=0.05, value=1.0)
            with gr.Row():
                latent_mode = gr.Checkbox(label='Latent mode', value=True, visible=False)
            with gr.Row():
                video_type = gr.Dropdown(label='Video file', choices=['None', 'GIF', 'PNG', 'MP4'], value='None')
                duration = gr.Slider(label='Duration', minimum=0.25, maximum=10, step=0.25, value=2, visible=False)
            with gr.Row():
                gif_loop = gr.Checkbox(label='Loop', value=True, visible=False)
                mp4_pad = gr.Slider(label='Pad frames', minimum=0, maximum=24, step=1, value=1, visible=False)
                mp4_interpolate = gr.Slider(label='Interpolate frames', minimum=0, maximum=24, step=1, value=0, visible=False)
            video_type.change(fn=video_type_change, inputs=[video_type], outputs=[duration, gif_loop, mp4_pad, mp4_interpolate])
        return [adapter_index, frames, lora_index, strength, latent_mode, video_type, duration, gif_loop, mp4_pad, mp4_interpolate, override_scheduler]

    def process(self, p: processing.StableDiffusionProcessing, adapter_index, frames, lora_index, strength, latent_mode, video_type, duration, gif_loop, mp4_pad, mp4_interpolate, override_scheduler): # pylint: disable=arguments-differ, unused-argument
        adapter = ADAPTERS[adapter_index]
        lora = LORAS[lora_index]
        set_adapter(adapter)
        if motion_adapter is None:
            return
        if override_scheduler:
            p.sampler_name = 'Default'
            shared.sd_model.scheduler = diffusers.DDIMScheduler(
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="linear",
                clip_sample=False,
                num_train_timesteps=1000,
                rescale_betas_zero_snr=False,
                set_alpha_to_one=True,
                steps_offset=0,
                timestep_spacing="linspace",
                trained_betas=None,
            )
        shared.log.debug(f'AnimateDiff: adapter="{adapter}" lora="{lora}" strength={strength} video={video_type} scheduler={shared.sd_model.scheduler.__class__.__name__ if override_scheduler else p.sampler_name}')
        if lora is not None and lora != 'None':
            shared.sd_model.load_lora_weights(lora, adapter_name=lora)
            shared.sd_model.set_adapters([lora], adapter_weights=[strength])
            p.extra_generation_params['AnimateDiff Lora'] = f'{lora}:{strength}'
        p.extra_generation_params['AnimateDiff'] = loaded_adapter
        p.do_not_save_grid = True
        if 'animatediff' not in p.ops:
            p.ops.append('animatediff')
        p.task_args['num_frames'] = frames
        p.task_args['num_inference_steps'] = p.steps
        if not latent_mode:
            p.task_args['output_type'] = 'np'

    def postprocess(self, p: processing.StableDiffusionProcessing, processed: processing.Processed, adapter_index, frames, lora_index, strength, latent_mode, video_type, duration, gif_loop, mp4_pad, mp4_interpolate, override_scheduler): # pylint: disable=arguments-differ, unused-argument
        from modules.images import save_video
        if video_type != 'None':
            save_video(p, filename=None, images=processed.images, video_type=video_type, duration=duration, loop=gif_loop, pad=mp4_pad, interpolate=mp4_interpolate)
