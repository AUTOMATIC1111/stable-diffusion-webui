import torch
import gradio as gr
import diffusers
from modules import scripts, processing, shared, images, sd_models, devices


MODELS = [
    { 'name': 'None', 'info': '' },
    { 'name': 'PIA', 'url': 'openmmlab/PIA-condition-adapter', 'info': '<a href="https://huggingface.co/docs/diffusers/main/en/api/pipelines/pia" target="_blank">Open MMLab Personalized Image Animator</a>' },
    { 'name': 'VGen', 'url': 'ali-vilab/i2vgen-xl', 'info': '<a href="https://huggingface.co/ali-vilab/i2vgen-xl" target="_blank">Alibaba VGen</a>' },
]


class Script(scripts.Script):
    def title(self):
        return 'Image-to-Video'

    def show(self, is_img2img):
        # return is_img2img if shared.backend == shared.Backend.DIFFUSERS else False
        return False

    # return signature is array of gradio components
    def ui(self, _is_img2img):

        def video_change(video_type):
            return [
                gr.update(visible=video_type != 'None'),
                gr.update(visible=video_type == 'GIF' or video_type == 'PNG'),
                gr.update(visible=video_type == 'MP4'),
                gr.update(visible=video_type == 'MP4'),
            ]

        def model_change(model_name):
            model = next(m for m in MODELS if m['name'] == model_name)
            return gr.update(value=model['info']), gr.update(visible=model_name == 'PIA'), gr.update(visible=model_name == 'VGen')

        with gr.Row():
            model_name = gr.Dropdown(label='Model', value='None', choices=[m['name'] for m in MODELS])
        with gr.Row():
            model_info = gr.HTML()
        with gr.Row():
            num_frames = gr.Slider(label='Frames', minimum=0, maximum=50, step=1, value=16)
        with gr.Row():
            video_type = gr.Dropdown(label='Video file', choices=['None', 'GIF', 'PNG', 'MP4'], value='None')
            duration = gr.Slider(label='Duration', minimum=0.25, maximum=10, step=0.25, value=2, visible=False)
        with gr.Accordion('FreeInit', open=False, visible=False) as fi_accordion:
            with gr.Row():
                fi_method = gr.Dropdown(label='Method', choices=['none', 'butterworth', 'ideal', 'gaussian'], value='none')
            with gr.Row():
                # fi_fast = gr.Checkbox(label='Fast sampling', value=False)
                fi_iters = gr.Slider(label='Iterations', minimum=1, maximum=10, step=1, value=3)
                fi_order = gr.Slider(label='Order', minimum=1, maximum=10, step=1, value=4)
            with gr.Row():
                fi_spatial = gr.Slider(label='Spatial frequency', minimum=0.0, maximum=1.0, step=0.05, value=0.25)
                fi_temporal = gr.Slider(label='Temporal frequency', minimum=0.0, maximum=1.0, step=0.05, value=0.25)
        with gr.Accordion('VGen params', open=True, visible=False) as vgen_accordion:
            with gr.Row():
                vg_chunks = gr.Slider(label='Decode chunks', minimum=0.1, maximum=1.0, step=0.1, value=0.5)
                vg_fps = gr.Slider(label='Change rate', minimum=0.1, maximum=1.0, step=0.1, value=0.5)
        with gr.Row():
            gif_loop = gr.Checkbox(label='Loop', value=True, visible=False)
            mp4_pad = gr.Slider(label='Pad frames', minimum=0, maximum=24, step=1, value=1, visible=False)
            mp4_interpolate = gr.Slider(label='Interpolate frames', minimum=0, maximum=24, step=1, value=0, visible=False)
        model_name.change(fn=model_change, inputs=[model_name], outputs=[model_info, fi_accordion, vgen_accordion])
        video_type.change(fn=video_change, inputs=[video_type], outputs=[duration, gif_loop, mp4_pad, mp4_interpolate])
        return [model_name, num_frames, video_type, duration, gif_loop, mp4_pad, mp4_interpolate, fi_method, fi_iters, fi_order, fi_spatial, fi_temporal, vg_chunks, vg_fps]

    def run(self, p: processing.StableDiffusionProcessing, model_name, num_frames, video_type, duration, gif_loop, mp4_pad, mp4_interpolate, fi_method, fi_iters, fi_order, fi_spatial, fi_temporal, vg_chunks, vg_fps): # pylint: disable=arguments-differ, unused-argument
        if model_name == 'None':
            return
        if p.init_images is None or len(p.init_images) == 0:
            return
        model = [m for m in MODELS if m['name'] == model_name][0]
        repo_id = model['url']
        shared.log.debug(f'Image2Video: model={model_name} frames={num_frames}, video={video_type} duration={duration} loop={gif_loop} pad={mp4_pad} interpolate={mp4_interpolate}')
        p.ops.append('image2video')
        p.do_not_save_grid = True

        if model_name == 'PIA':
            if shared.sd_model_type != 'sd':
                shared.log.error('Image2Video PIA: base model must be SD15')
                return
            orig_pipeline = shared.sd_model
            shared.log.info(f'Image2Video PIA load: model={repo_id}')
            motion_adapter = diffusers.MotionAdapter.from_pretrained(repo_id)
            motion_adapter.to(devices.device, devices.dtype)
            shared.sd_model = sd_models.switch_pipe(diffusers.PIAPipeline, shared.sd_model, { 'motion_adapter': motion_adapter })
            if num_frames > 0:
                p.task_args['num_frames'] = num_frames
                p.task_args['image'] = p.init_images[0]
            if hasattr(shared.sd_model, 'enable_free_init') and fi_method != 'none':
                shared.sd_model.enable_free_init(
                    num_iters=fi_iters,
                    use_fast_sampling=False,
                    method=fi_method,
                    order=fi_order,
                    spatial_stop_frequency=fi_spatial,
                    temporal_stop_frequency=fi_temporal,
                )
            shared.log.debug(f'Image2Video PIA: args={p.task_args}')
            processed = processing.process_images(p)
            shared.sd_model.motion_adapter = None
            shared.sd_model = orig_pipeline

        if model_name == 'VGen':
            if not isinstance(shared.sd_model, diffusers.I2VGenXLPipeline):
                shared.log.info(f'Image2Video VGen load: model={repo_id}')
                pipe = diffusers.I2VGenXLPipeline.from_pretrained(repo_id, torch_dtype=devices.dtype, cache_dir=shared.opts.diffusers_dir)
                sd_models.copy_diffuser_options(pipe, shared.sd_model)
                sd_models.set_diffuser_options(pipe)
                shared.sd_model = pipe
                shared.sd_model.to(devices.device, torch.float32)
                devices.torch_gc()
            if num_frames > 0:
                p.task_args['image'] = p.init_images[0]
                p.task_args['num_frames'] = num_frames
                p.task_args['target_fps'] = max(1, int(num_frames * vg_fps))
                p.task_args['decode_chunk_size'] = max(1, int(num_frames * vg_chunks))
                p.task_args['output_type'] = 'pil'
            shared.log.debug(f'Image2Video VGen: args={p.task_args}')
            processed = processing.process_images(p)

        if video_type != 'None' and processed is not None:
            images.save_video(p, filename=None, images=processed.images, video_type=video_type, duration=duration, loop=gif_loop, pad=mp4_pad, interpolate=mp4_interpolate)
        return processed
