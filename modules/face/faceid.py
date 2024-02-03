import os
import cv2
import torch
import numpy as np
import diffusers
import huggingface_hub as hf
from PIL import Image
from modules import processing, shared, devices


FACEID_MODELS = {
    'FaceID Base': 'h94/IP-Adapter-FaceID/ip-adapter-faceid_sd15.bin',
    'FaceID Plus v1': 'h94/IP-Adapter-FaceID/ip-adapter-faceid-plus_sd15.bin',
    'FaceID Plus v2': 'h94/IP-Adapter-FaceID/ip-adapter-faceid-plusv2_sd15.bin',
    'FaceID XL': 'h94/IP-Adapter-FaceID/ip-adapter-faceid_sdxl.bin'
}
faceid_model = None
faceid_model_name = None
debug = shared.log.trace if os.environ.get('SD_FACE_DEBUG', None) is not None else lambda *args, **kwargs: None


def face_id(p: processing.StableDiffusionProcessing, app, source_image: Image.Image, model: str, override: bool, cache: bool, scale: float, structure: float):
    global faceid_model, faceid_model_name # pylint: disable=global-statement
    from insightface.utils import face_align
    from ip_adapter.ip_adapter_faceid import IPAdapterFaceID, IPAdapterFaceIDPlus, IPAdapterFaceIDXL

    ip_ckpt = FACEID_MODELS[model]
    folder, filename = os.path.split(ip_ckpt)
    basename, _ext = os.path.splitext(filename)
    model_path = hf.hf_hub_download(repo_id=folder, filename=filename, cache_dir=shared.opts.diffusers_dir)
    if model_path is None:
        shared.log.error(f'FaceID download failed: model={model} file={ip_ckpt}')
        return None

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
    if faceid_model is None or faceid_model_name != model or not cache:
        shared.log.debug(f'FaceID load: model={model} file={ip_ckpt}')
        if 'Plus' in model:
            image_encoder_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
            faceid_model = IPAdapterFaceIDPlus(
                sd_pipe=shared.sd_model,
                image_encoder_path=image_encoder_path,
                ip_ckpt=model_path,
                lora_rank=128, num_tokens=4, device=devices.device, torch_dtype=devices.dtype,
            )
            shortcut = 'v2' in model
        elif 'XL' in model:
            faceid_model = IPAdapterFaceIDXL(
                sd_pipe=shared.sd_model,
                ip_ckpt=model_path,
                lora_rank=128, num_tokens=4, device=devices.device, torch_dtype=devices.dtype,
            )
        else:
            faceid_model = IPAdapterFaceID(
                sd_pipe=shared.sd_model,
                ip_ckpt=model_path,
                lora_rank=128, num_tokens=4, device=devices.device, torch_dtype=devices.dtype,
            )
        faceid_model_name = model
    else:
        shared.log.debug(f'FaceID cached: model={model} file={ip_ckpt}')

    processed_images = []
    np_image = cv2.cvtColor(np.array(source_image), cv2.COLOR_RGB2BGR)
    faces = app.get(np_image)
    if len(faces) == 0:
        shared.log.error('FaceID: no faces found')
        return None
    face_embeds = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)
    face_image = face_align.norm_crop(np_image, landmark=faces[0].kps, image_size=224) # you can also segment the face

    for i, face in enumerate(faces):
        shared.log.debug(f'FaceID face: i={i+1} score={face.det_score:.2f} gender={"female" if face.gender==0 else "male"} age={face.age} bbox={face.bbox}')
        p.extra_generation_params[f"FaceID {i+1}"] = f'{face.det_score:.2f} {"female" if face.gender==0 else "male"} {face.age}y'
    ip_model_dict = { # main generate dict
        'num_samples': p.batch_size,
        'width': p.width,
        'height': p.height,
        'num_inference_steps': p.steps,
        'scale': scale,
        'guidance_scale': p.cfg_scale,
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
    faceid_model.set_scale(scale)
    for i in range(p.n_iter):
        ip_model_dict.update({
                'prompt': p.all_prompts[i],
                'negative_prompt': p.all_negative_prompts[i],
                'seed': int(p.all_seeds[i]),
            })
        debug(f'FaceID: {ip_model_dict}')
        res = faceid_model.generate(**ip_model_dict)
        if isinstance(res, list):
            processed_images += res
    faceid_model.set_scale(0)

    if not cache:
        faceid_model = None
        faceid_model_name = None
    devices.torch_gc()

    p.extra_generation_params["IP Adapter"] = f'{basename}:{scale}'
    return processed_images
