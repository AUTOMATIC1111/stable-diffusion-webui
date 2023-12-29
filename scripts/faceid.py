import os
import cv2
import torch
import numpy as np
import gradio as gr
import diffusers
import huggingface_hub as hf
from modules import scripts, processing, shared, devices


app = None
try:
    import onnxruntime
    from insightface.app import FaceAnalysis
    from ip_adapter.ip_adapter_faceid import IPAdapterFaceID
    ok = True
except Exception as e:
    shared.log.error(f'FaceID: {e}')
    ok = False


class Script(scripts.Script):
    def title(self):
        return 'FaceID'

    def show(self, is_img2img):
        return ok if shared.backend == shared.Backend.DIFFUSERS else False

    # return signature is array of gradio components
    def ui(self, _is_img2img):
        with gr.Row():
            scale = gr.Slider(label='Scale', minimum=0.0, maximum=1.0, step=0.01, value=1.0)
        with gr.Row():
            image = gr.Image(image_mode='RGB', label='Image', source='upload', type='pil', width=512)
        return [scale, image]

    def run(self, p: processing.StableDiffusionProcessing, scale, image): # pylint: disable=arguments-differ, unused-argument
        if not ok:
            shared.log.error('FaceID: missing dependencies')
            return None
        if image is None:
            shared.log.error('FaceID: no init_images')
            return None
        if shared.sd_model_type != 'sd':
            shared.log.error('FaceID: base model not supported')
            return None

        global app # pylint: disable=global-statement
        if app is None:
            shared.log.debug(f"ONNX: device={onnxruntime.get_device()} providers={onnxruntime.get_available_providers()}")
            app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            onnxruntime.set_default_logger_severity(3)
            app.prepare(ctx_id=0, det_thresh=0.5, det_size=(640, 640))

        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        faces = app.get(image)
        if len(faces) == 0:
            shared.log.error('FaceID: no faces found')
            return None
        for face in faces:
            shared.log.debug(f'FaceID face: score={face.det_score:.2f} gender={"female" if face.gender==0 else "male"} age={face.age} bbox={face.bbox}')
        embeds = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)

        ip_ckpt = "h94/IP-Adapter-FaceID/ip-adapter-faceid_sd15.bin"
        shared.log.debug(f'FaceID model load: {ip_ckpt}')
        folder, filename = os.path.split(ip_ckpt)
        basename, _ext = os.path.splitext(filename)
        model_path = hf.hf_hub_download(repo_id=folder, filename=filename, cache_dir=shared.opts.diffusers_dir)
        if model_path is None:
            shared.log.error(f'FaceID: model download failed: {ip_ckpt}')
            return None

        processing.process_init(p)
        shared.sd_model.scheduler = diffusers.DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
        )
        ip_model = IPAdapterFaceID(shared.sd_model, model_path, devices.device)
        ip_model_dict = {
            'prompt': p.all_prompts[0],
            'negative_prompt': p.all_negative_prompts[0],
            'num_samples': p.batch_size,
            'width': p.width,
            'height': p.height,
            'num_inference_steps': p.steps,
            'scale': scale,
            'guidance_scale': p.cfg_scale,
            'seed': int(p.all_seeds[0]),
            'faceid_embeds': None,
        }
        shared.log.debug(f'FaceID args: {ip_model_dict}')
        ip_model_dict['faceid_embeds'] = embeds
        images = ip_model.generate(**ip_model_dict)

        ip_model = None
        p.extra_generation_params["IP Adapter"] = f'{basename}:{scale}'
        for i, face in enumerate(faces):
            p.extra_generation_params[f"FaceID {i} score"] = f'{face.det_score:.2f}'
            p.extra_generation_params[f"FaceID {i} gender"] = "female" if face.gender==0 else "male"
            p.extra_generation_params[f"FaceID {i} age"] = face.age
        processed = processing.Processed(
            p,
            images_list=images,
            seed=p.seed,
            subseed=p.subseed,
            index_of_first_image=0,
        )
        processed.info = processed.infotext(p, 0)
        processed.infotexts = [processed.info]
        devices.torch_gc()
        return processed
