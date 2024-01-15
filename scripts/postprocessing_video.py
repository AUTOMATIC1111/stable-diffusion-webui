import gradio as gr
import modules.images
from modules import scripts_postprocessing


class ScriptPostprocessingUpscale(scripts_postprocessing.ScriptPostprocessing):
    name = "Video"

    def ui(self):
        def video_type_change(video_type):
            return [
                gr.update(visible=video_type != 'None'),
                gr.update(visible=video_type == 'GIF' or video_type == 'PNG'),
                gr.update(visible=video_type == 'MP4'),
                gr.update(visible=video_type == 'MP4'),
                gr.update(visible=video_type == 'MP4'),
                gr.update(visible=video_type == 'MP4'),
            ]

        with gr.Row():
            video_type = gr.Dropdown(label='Video file', choices=['None', 'GIF', 'PNG', 'MP4'], value='None')
            duration = gr.Slider(label='Duration', minimum=0.25, maximum=10, step=0.25, value=2, visible=False)
        with gr.Row():
            loop = gr.Checkbox(label='Loop', value=True, visible=False)
            pad = gr.Slider(label='Pad frames', minimum=0, maximum=24, step=1, value=1, visible=False)
            interpolate = gr.Slider(label='Interpolate frames', minimum=0, maximum=24, step=1, value=0, visible=False)
            scale = gr.Slider(label='Rescale', minimum=0.5, maximum=2, step=0.05, value=1, visible=False)
            change = gr.Slider(label='Frame change sensitivity', minimum=0, maximum=1, step=0.05, value=0.3, visible=False)
        with gr.Row():
            filename = gr.Textbox(label='Filename', placeholder='enter filename', lines=1)
        video_type.change(fn=video_type_change, inputs=[video_type], outputs=[duration, loop, pad, interpolate, scale, change])
        return {
            "filename": filename,
            "video_type": video_type,
            "duration": duration,
            "loop": loop,
            "pad": pad,
            "interpolate": interpolate,
            "scale": scale,
            "change": change,
        }

    def postprocess(self, images, filename, video_type, duration, loop, pad, interpolate, scale, change): # pylint: disable=arguments-differ
        filename = filename.strip()
        if video_type == 'None' or len(filename) == 0 or images is None or len(images) < 2:
            return
        modules.images.save_video(p=None, filename=filename, images=images, video_type=video_type, duration=duration, loop=loop, pad=pad, interpolate=interpolate, scale=scale, change=change)
