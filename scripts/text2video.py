"""
Additional params for Text-to-Video
<https://huggingface.co/docs/diffusers/api/pipelines/text_to_video>

TODO text2video items:
- Video-to-Video upscaling: <https://huggingface.co/cerspense/zeroscope_v2_XL>, <https://huggingface.co/damo-vilab/MS-Vid2Vid-XL>
"""

import gradio as gr
from modules import scripts, processing, shared, images, sd_models, modelloader


MODELS = [
    {'name': 'None'},
    {'name': 'ModelScope v1.7b', 'path': 'damo-vilab/text-to-video-ms-1.7b', 'params': [16,320,320]},
    {'name': 'ZeroScope v1', 'path': 'cerspense/zeroscope_v1_320s', 'params': [16,320,320]},
    {'name': 'ZeroScope v1.1', 'path': 'cerspense/zeroscope_v1-1_320s', 'params': [16,320,320]},
    {'name': 'ZeroScope v2', 'path': 'cerspense/zeroscope_v2_576w', 'params': [24,576,320]},
    {'name': 'ZeroScope v2 Dark', 'path': 'cerspense/zeroscope_v2_dark_30x448x256', 'params': [24,448,256]},
    {'name': 'Potat v1', 'path': 'camenduru/potat1', 'params': [24,1024,576]},
]


class Script(scripts.Script):
    def title(self):
        return 'Text-to-Video'

    def show(self, is_img2img):
        return not is_img2img if shared.backend == shared.Backend.DIFFUSERS else False

    # return signature is array of gradio components
    def ui(self, _is_img2img):

        def video_type_change(video_type):
            return [
                gr.update(visible=video_type != 'None'),
                gr.update(visible=video_type == 'GIF' or video_type == 'PNG'),
                gr.update(visible=video_type == 'MP4'),
                gr.update(visible=video_type == 'MP4'),
            ]

        def model_info_change(model_name):
            if model_name == 'None':
                return gr.update(value='')
            else:
                model = next(m for m in MODELS if m['name'] == model_name)
                return gr.update(value=f'<span> &nbsp frames: {model["params"][0]} size: {model["params"][1]}x{model["params"][2]}</span> <a href="https://huggingface.co/{model["path"]}" target="_blank">link</a>')

        with gr.Row():
            model_name = gr.Dropdown(label='Model', value='None', choices=[m['name'] for m in MODELS])
        with gr.Row():
            model_info = gr.HTML()
            model_name.change(fn=model_info_change, inputs=[model_name], outputs=[model_info])
        with gr.Row():
            use_default = gr.Checkbox(label='Use defaults', value=True)
            num_frames = gr.Slider(label='Frames', minimum=1, maximum=50, step=1, value=0)
        with gr.Row():
            video_type = gr.Dropdown(label='Video file', choices=['None', 'GIF', 'PNG', 'MP4'], value='None')
            duration = gr.Slider(label='Duration', minimum=0.25, maximum=10, step=0.25, value=2, visible=False)
        with gr.Row():
            gif_loop = gr.Checkbox(label='Loop', value=True, visible=False)
            mp4_pad = gr.Slider(label='Pad frames', minimum=0, maximum=24, step=1, value=1, visible=False)
            mp4_interpolate = gr.Slider(label='Interpolate frames', minimum=0, maximum=24, step=1, value=0, visible=False)
        video_type.change(fn=video_type_change, inputs=[video_type], outputs=[duration, gif_loop, mp4_pad, mp4_interpolate])
        return [model_name, use_default, num_frames, video_type, duration, gif_loop, mp4_pad, mp4_interpolate]

    def run(self, p: processing.StableDiffusionProcessing, model_name, use_default, num_frames, video_type, duration, gif_loop, mp4_pad, mp4_interpolate): # pylint: disable=arguments-differ, unused-argument
        if model_name == 'None':
            return
        model = [m for m in MODELS if m['name'] == model_name][0]
        shared.log.debug(f'Text2Video: model={model} defaults={use_default} frames={num_frames}, video={video_type} duration={duration} loop={gif_loop} pad={mp4_pad} interpolate={mp4_interpolate}')

        if model['path'] in shared.opts.sd_model_checkpoint:
            shared.log.debug(f'Text2Video cached: model={shared.opts.sd_model_checkpoint}')
        else:
            checkpoint = sd_models.get_closet_checkpoint_match(model['path'])
            if checkpoint is None:
                shared.log.debug(f'Text2Video downloading: model={model["path"]}')
                checkpoint = modelloader.download_diffusers_model(hub_id=model['path'])
                sd_models.list_models()
            if checkpoint is None:
                shared.log.error(f'Text2Video: failed to find model={model["path"]}')
                return
            shared.log.debug(f'Text2Video loading: model={checkpoint}')
            shared.opts.sd_model_checkpoint = checkpoint
            sd_models.reload_model_weights(op='model')

        p.ops.append('text2video')
        p.do_not_save_grid = True
        if use_default:
            p.task_args['num_frames'] = model['params'][0]
            p.width = model['params'][1]
            p.height = model['params'][2]
        elif num_frames > 0:
            p.task_args['num_frames'] = num_frames
        else:
            shared.log.error('Text2Video: invalid number of frames')
            return

        shared.sd_model = sd_models.set_diffuser_pipe(shared.sd_model, sd_models.DiffusersTaskType.TEXT_2_IMAGE)
        shared.log.debug(f'Text2Video: args={p.task_args}')
        processed = processing.process_images(p)

        if video_type != 'None':
            images.save_video(p, filename=None, images=processed.images, video_type=video_type, duration=duration, loop=gif_loop, pad=mp4_pad, interpolate=mp4_interpolate)
        return processed
