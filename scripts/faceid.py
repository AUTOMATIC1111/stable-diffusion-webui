import os
import cv2
import torch
import numpy as np
import gradio as gr
import diffusers
import huggingface_hub as hf
from modules import scripts, processing, shared, devices


MODELS = {
    'FaceID Base': 'h94/IP-Adapter-FaceID/ip-adapter-faceid_sd15.bin',
    'FaceID Plus': 'h94/IP-Adapter-FaceID/ip-adapter-faceid-plus_sd15.bin',
    'FaceID Plus v2': 'h94/IP-Adapter-FaceID/ip-adapter-faceid-plusv2_sd15.bin',
    'FaceID XL': 'h94/IP-Adapter-FaceID/ip-adapter-faceid_sdxl.bin'
}
app = None
ip_model = None
ip_model_name = None
ip_model_tokens = None
ip_model_rank = None


def dependencies():
    from installer import installed, install
    packages = [
        ('insightface', 'insightface'),
        ('git+https://github.com/tencent-ailab/IP-Adapter.git', 'ip_adapter'),
    ]
    for pkg in packages:
        if not installed(pkg[1], reload=False, quiet=True):
            install(pkg[0], pkg[1], ignore=True)


class Script(scripts.Script):
    def title(self):
        return 'FaceID'

    def show(self, is_img2img):
        return True if shared.backend == shared.Backend.DIFFUSERS else False

    # return signature is array of gradio components
    def ui(self, _is_img2img):
        with gr.Row():
            model = gr.Dropdown(choices=list(MODELS), label='Model', value='FaceID Base')
        with gr.Row(visible=True):
            override = gr.Checkbox(label='Override sampler', value=True)
            cache = gr.Checkbox(label='Cache model', value=True)
        with gr.Row(visible=True):
            scale = gr.Slider(label='Strength', minimum=0.0, maximum=1.0, step=0.01, value=1.0)
            structure = gr.Slider(label='Structure', minimum=0.0, maximum=1.0, step=0.01, value=1.0)
        with gr.Row(visible=False):
            rank = gr.Slider(label='Rank', minimum=4, maximum=256, step=4, value=128)
            tokens = gr.Slider(label='Tokens', minimum=1, maximum=16, step=1, value=4)
        with gr.Row():
            image = gr.Image(image_mode='RGB', label='Image', source='upload', type='pil', width=512)
        return [model, scale, image, override, rank, tokens, structure, cache]

    def run(self, p: processing.StableDiffusionProcessing, model, scale, image, override, rank, tokens, structure, cache): # pylint: disable=arguments-differ, unused-argument
        dependencies()
        try:
            import onnxruntime
            from insightface.app import FaceAnalysis
            from insightface.utils import face_align
            from ip_adapter.ip_adapter_faceid import IPAdapterFaceID, IPAdapterFaceIDPlus, IPAdapterFaceIDXL
        except Exception as e:
            shared.log.error(f'FaceID: {e}')
            return None
        if image is None:
            shared.log.error('FaceID: no init_images')
            return None
        if shared.sd_model_type != 'sd' and shared.sd_model_type != 'sdxl':
            shared.log.error('FaceID: base model not supported')
            return None

        global app, ip_model, ip_model_name, ip_model_tokens, ip_model_rank # pylint: disable=global-statement
        if app is None:
            shared.log.debug(f"ONNX: device={onnxruntime.get_device()} providers={onnxruntime.get_available_providers()}")
            app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            onnxruntime.set_default_logger_severity(3)
            app.prepare(ctx_id=0, det_thresh=0.5, det_size=(640, 640))

        if isinstance(image, str):
            from modules.api.api import decode_base64_to_image
            image = decode_base64_to_image(image)

        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        faces = app.get(image)
        if len(faces) == 0:
            shared.log.error('FaceID: no faces found')
            return None
        for face in faces:
            shared.log.debug(f'FaceID face: score={face.det_score:.2f} gender={"female" if face.gender==0 else "male"} age={face.age} bbox={face.bbox}')
        face_embeds = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)
        face_image = face_align.norm_crop(image, landmark=faces[0].kps, image_size=224) # you can also segment the face

        ip_ckpt = MODELS[model]
        folder, filename = os.path.split(ip_ckpt)
        basename, _ext = os.path.splitext(filename)
        model_path = hf.hf_hub_download(repo_id=folder, filename=filename, cache_dir=shared.opts.diffusers_dir)
        if model_path is None:
            shared.log.error(f'FaceID download failed: model={model} file={ip_ckpt}')
            return None

        processing.process_init(p)
        if override:
            shared.sd_model.scheduler = diffusers.DDIMScheduler(
                num_train_timesteps=1000,
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                clip_sample=False,
                set_alpha_to_one=False,
                steps_offset=1,
            )
        shortcut = None
        if ip_model is None or ip_model_name != model or ip_model_tokens != tokens or ip_model_rank != rank or not cache:
            shared.log.debug(f'FaceID load: model={model} file={ip_ckpt} tokens={tokens} rank={rank}')
            if 'Plus' in model:
                image_encoder_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
                ip_model = IPAdapterFaceIDPlus(
                    sd_pipe=shared.sd_model,
                    image_encoder_path=image_encoder_path,
                    ip_ckpt=model_path,
                    lora_rank=rank,
                    num_tokens=tokens,
                    device=devices.device,
                    torch_dtype=devices.dtype,
                )
                shortcut = 'v2' in model
            elif 'XL' in model:
                ip_model = IPAdapterFaceIDXL(
                    sd_pipe=shared.sd_model,
                    ip_ckpt=model_path,
                    lora_rank=rank,
                    num_tokens=tokens,
                    device=devices.device,
                    torch_dtype=devices.dtype,
                )
            else:
                ip_model = IPAdapterFaceID(
                    sd_pipe=shared.sd_model,
                    ip_ckpt=model_path,
                    lora_rank=rank,
                    num_tokens=tokens,
                    device=devices.device,
                    torch_dtype=devices.dtype,
                )
            ip_model_name = model
            ip_model_tokens = tokens
            ip_model_rank = rank
        else:
            shared.log.debug(f'FaceID cached: model={model} file={ip_ckpt} tokens={tokens} rank={rank}')

        # main generate dict
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
            'faceid_embeds': face_embeds.shape,
        }

        # optional generate dict
        if shortcut is not None:
            ip_model_dict['shortcut'] = shortcut
        if 'Plus' in model:
            ip_model_dict['s_scale'] = structure
            ip_model_dict['face_image'] = face_image.shape
        shared.log.debug(f'FaceID args: {ip_model_dict}')
        if 'Plus' in model:
            ip_model_dict['face_image'] = face_image
        ip_model_dict['faceid_embeds'] = face_embeds

        # run generate
        images = ip_model.generate(**ip_model_dict)

        if not cache:
            ip_model = None
            ip_model_name = None
        devices.torch_gc()

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
        return processed
