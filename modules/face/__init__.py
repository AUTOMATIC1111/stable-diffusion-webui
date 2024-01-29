import os
import gradio as gr
from PIL import Image
from modules import scripts, processing, shared, images


debug = shared.log.trace if os.environ.get('SD_FACE_DEBUG', None) is not None else lambda *args, **kwargs: None


class Script(scripts.Script):
    def title(self):
        return 'Face'

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

    def mode_change(self, mode):
        return [
            gr.update(visible=mode=='FaceID'),
            gr.update(visible=mode=='FaceSwap'),
            gr.update(visible=mode=='InstantID'),
            gr.update(visible=mode=='PhotoMaker'),
        ]

    # return signature is array of gradio components
    def ui(self, _is_img2img):
        with gr.Row():
            mode = gr.Dropdown(label='Mode', choices=['None', 'FaceID', 'FaceSwap', 'InstantID', 'PhotoMaker'], value='None')
        with gr.Group(visible=False) as cfg_faceid:
            with gr.Row():
                gr.HTML('<a href="https://huggingface.co/h94/IP-Adapter-FaceID" target="_blank">&nbsp Tencent AI Lab IP-Adapter FaceID</a><br>')
            with gr.Row():
                from modules.face.faceid import FACEID_MODELS
                ip_model = gr.Dropdown(choices=list(FACEID_MODELS), label='FaceID Model', value='FaceID Base')
            with gr.Row(visible=True):
                ip_override = gr.Checkbox(label='Override sampler', value=True)
                ip_cache = gr.Checkbox(label='Cache model', value=True)
            with gr.Row(visible=True):
                ip_strength = gr.Slider(label='Strength', minimum=0.0, maximum=2.0, step=0.01, value=1.0)
                ip_structure = gr.Slider(label='Structure', minimum=0.0, maximum=1.0, step=0.01, value=1.0)
        with gr.Group(visible=False) as cfg_faceswap:
            with gr.Row():
                gr.HTML('<a href="https://github.com/deepinsight/insightface/blob/master/examples/in_swapper/README.md" target="_blank">&nbsp InsightFace InSwapper</a><br>')
            with gr.Row(visible=True):
                fs_cache = gr.Checkbox(label='Cache model', value=True)
        with gr.Group(visible=False) as cfg_instantid:
            with gr.Row():
                gr.HTML('<a href="https://github.com/InstantID/InstantID" target="_blank">&nbsp InstantX InstantID</a><br>')
            with gr.Row():
                id_strength = gr.Slider(label='Strength', minimum=0.0, maximum=2.0, step=0.01, value=1.0)
                id_conditioning = gr.Slider(label='Control', minimum=0.0, maximum=2.0, step=0.01, value=0.5)
            with gr.Row(visible=True):
                id_cache = gr.Checkbox(label='Cache model', value=True)
        with gr.Group(visible=False) as cfg_photomaker:
            with gr.Row():
                gr.HTML('<a href="https://photo-maker.github.io/" target="_blank">&nbsp Tenecent ARC Lab PhotoMaker</a><br>')
            with gr.Row():
                pm_trigger = gr.Text(label='Trigger word', value="person")
                pm_strength = gr.Slider(label='Strength', minimum=0.0, maximum=2.0, step=0.01, value=1.0)
                pm_start = gr.Slider(label='Start', minimum=0.0, maximum=1.0, step=0.01, value=0.5)
        with gr.Row():
            files = gr.File(label='Input images', file_count='multiple', file_types=['image'], type='file', interactive=True, height=100)
        with gr.Row():
            gallery = gr.Gallery(show_label=False, value=[])
        files.change(fn=self.load_images, inputs=[files], outputs=[gallery])
        mode.change(fn=self.mode_change, inputs=[mode], outputs=[cfg_faceid, cfg_faceswap, cfg_instantid, cfg_photomaker])

        return [mode, gallery, ip_model, ip_override, ip_cache, ip_strength, ip_structure, id_strength, id_conditioning, id_cache, pm_trigger, pm_strength, pm_start, fs_cache]

    def run(self, p: processing.StableDiffusionProcessing, mode, input_images, ip_model, ip_override, ip_cache, ip_strength, ip_structure, id_strength, id_conditioning, id_cache, pm_trigger, pm_strength, pm_start, fs_cache): # pylint: disable=arguments-differ, unused-argument
        if input_images is None or len(input_images) == 0:
            shared.log.error('Face: no init images')
            return None
        if shared.sd_model_type != 'sd' and shared.sd_model_type != 'sdxl':
            shared.log.error('Face: base model not supported')
            return None

        for i, image in enumerate(input_images):
            if isinstance(image, str):
                from modules.api.api import decode_base64_to_image
                input_images[i] = decode_base64_to_image(image).convert("RGB")

        processed = None
        for i, image in enumerate(input_images):
            if not isinstance(image, Image.Image):
                input_images[i] = Image.open(image['name'])
        source_image = input_images[0]

        if mode == 'FaceID': # faceid runs as ipadapter in its own pipeline
            from modules.face.faceid import face_id
            from modules.face.insightface import get_app
            processed_images = face_id(p, app=get_app('buffalo_l'), source_image=source_image, model=ip_model, override=ip_override, cache=ip_cache, scale=ip_strength, structure=ip_structure) # run faceid pipeline
            processed = processing.Processed(p, images_list=processed_images, seed=p.seed, subseed=p.subseed, index_of_first_image=0) # manually created processed object
        elif mode == 'PhotoMaker': # photomaker creates pipeline and triggers original process_images
            from modules.face.photomaker import photo_maker
            processed = photo_maker(p, input_images=input_images, trigger=pm_trigger, strength=pm_strength, start=pm_start)
        elif mode == 'InstantID':
            from modules.face.instantid import instant_id # instantid creates pipeline and triggers original process_images
            from modules.face.insightface import get_app
            processed = instant_id(p, app=get_app('antelopev2'), source_image=source_image, strength=id_strength, conditioning=id_conditioning, cache=id_cache)

        if processed is None: # run normal pipeline
            processed = processing.process_images(p)

        if mode == 'FaceSwap': # faceswap runs as postprocessing
            from modules.face.faceswap import face_swap
            from modules.face.insightface import get_app
            if shared.opts.save_images_before_face_restoration and not p.do_not_save_samples:
                for i, image in enumerate(processed.images):
                    info = processing.create_infotext(p, index=i)
                    images.save_image(image, path=p.outpath_samples, seed=p.all_seeds[i], prompt=p.all_prompts[i], info=info, p=p, suffix="-before-faceswap")
            processed.images = face_swap(p, app=get_app('buffalo_l'), input_images=processed.images, source_image=source_image, cache=fs_cache)

        processed.info = processed.infotext(p, 0)
        processed.infotexts = [processed.info]
        if shared.opts.samples_save and not p.do_not_save_samples:
            for i, image in enumerate(processed.images):
                info = processing.create_infotext(p, index=i)
                images.save_image(image, path=p.outpath_samples, seed=p.all_seeds[i], prompt=p.all_prompts[i], info=info, p=p)

        return processed
