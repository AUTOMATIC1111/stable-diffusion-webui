"""
Additional params for StableVideoDiffusion
"""
import torch
import gradio as gr
from modules import scripts, processing, shared, sd_models, images


class Script(scripts.Script):
    def title(self):
        return 'Stable Video Diffusion'

    def show(self, is_img2img):
        return is_img2img if shared.backend == shared.Backend.DIFFUSERS else False

    # return signature is array of gradio components
    def ui(self, _is_img2img):
        with gr.Row():
            num_frames = gr.Slider(label='Frames', minimum=1, maximum=50, step=1, value=14)
            min_guidance_scale = gr.Slider(label='Min guidance', minimum=0.0, maximum=10.0, step=0.1, value=1.0)
            max_guidance_scale = gr.Slider(label='Max guidance', minimum=0.0, maximum=10.0, step=0.1, value=3.0)
        with gr.Row():
            decode_chunk_size = gr.Slider(label='Decode chunks', minimum=1, maximum=25, step=1, value=6)
            motion_bucket_id = gr.Slider(label='Motion level', minimum=0, maximum=1, step=0.05, value=0.5)
            noise_aug_strength = gr.Slider(label='Noise strength', minimum=0.0, maximum=1.0, step=0.01, value=0.1)
        with gr.Row():
            override_resolution = gr.Checkbox(label='Override resolution', value=True)
            create_gif = gr.Checkbox(label='GIF', value=False)
            create_mp4 = gr.Checkbox(label='MP4', value=False)
            loop = gr.Checkbox(label='Loop', value=True)
            duration = gr.Slider(label='Duration', minimum=0.25, maximum=10, step=0.25, value=2)
        return [num_frames, override_resolution, min_guidance_scale, max_guidance_scale, decode_chunk_size, motion_bucket_id, noise_aug_strength, create_gif, create_mp4, duration, loop]

    def run(self, p: processing.StableDiffusionProcessing, num_frames, override_resolution, min_guidance_scale, max_guidance_scale, decode_chunk_size, motion_bucket_id, noise_aug_strength, create_gif, create_mp4, duration, loop): # pylint: disable=arguments-differ, unused-argument
        if shared.sd_model is None or shared.sd_model.__class__.__name__ != 'StableVideoDiffusionPipeline':
            return None
        if hasattr(p, 'init_images') and len(p.init_images) > 0:
            if override_resolution:
                p.width = 1024
                p.height = 576
                p.task_args['image'] = images.resize_image(resize_mode=2, im=p.init_images[0], width=p.width, height=p.height, upscaler_name=None, output_type='pil')
            else:
                p.task_args['image'] = p.init_images[0]
            p.ops.append('svd')
            p.do_not_save_grid = True
            p.sampler_name = 'Default' # svd does not support non-default sampler
            p.task_args['generator'] = torch.manual_seed(p.seed) # svd does not support gpu based generator
            p.task_args['width'] = p.width
            p.task_args['height'] = p.height
            p.task_args['num_frames'] = num_frames
            p.task_args['decode_chunk_size'] = decode_chunk_size
            p.task_args['motion_bucket_id'] = round(255 * motion_bucket_id)
            p.task_args['noise_aug_strength'] = noise_aug_strength
            p.task_args['num_inference_steps'] = p.steps
            p.task_args['min_guidance_scale'] = min_guidance_scale
            p.task_args['max_guidance_scale'] = max_guidance_scale
            p.task_args['output_type'] = 'np'
            shared.log.debug(f'StableVideo: args={p.task_args}')
            shared.sd_model = sd_models.set_diffuser_pipe(shared.sd_model, sd_models.DiffusersTaskType.IMAGE_2_IMAGE)
            processed = processing.process_images(p)
            if create_gif:
                images.save_video(p, images=processed.images, video_type='gif', duration=duration, loop=loop)
            if create_mp4:
                images.save_video(p, images=processed.images, video_type='mp4', duration=duration, loop=loop)
            return processed
        else:
            shared.log.error('StableVideo: no init_images')
            return None
