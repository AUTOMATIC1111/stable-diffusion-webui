import os
import cv2
import numpy as np
import huggingface_hub as hf
from modules import shared, processing, sd_models, devices


REPO_ID = "InstantX/InstantID"
controlnet_model = None
debug = shared.log.trace if os.environ.get('SD_FACE_DEBUG', None) is not None else lambda *args, **kwargs: None


def instant_id(p: processing.StableDiffusionProcessing, app, source_image, strength=1.0, conditioning=0.5, cache=True): # pylint: disable=arguments-differ
    from modules.face.instantid_model import StableDiffusionXLInstantIDPipeline, draw_kps
    from diffusers.models import ControlNetModel
    global controlnet_model # pylint: disable=global-statement

    # prepare pipeline
    if source_image is None:
        shared.log.warning('InstantID: no input images')
        return None

    c = shared.sd_model.__class__.__name__ if shared.sd_model is not None else ''
    if c != 'StableDiffusionXLPipeline':
        shared.log.warning(f'InstantID invalid base model: current={c} required=StableDiffusionXLPipeline')
        return None

    # prepare face emb
    faces = app.get(cv2.cvtColor(np.array(source_image), cv2.COLOR_RGB2BGR))
    face = sorted(faces, key=lambda x:(x['bbox'][2]-x['bbox'][0])*x['bbox'][3]-x['bbox'][1])[-1]  # only use the maximum face
    face_emb = face['embedding']
    face_kps = draw_kps(source_image, face['kps'])

    shared.log.debug(f'InstantID face: score={face.det_score:.2f} gender={"female" if face.gender==0 else "male"} age={face.age} bbox={face.bbox}')
    shared.log.debug(f'InstantID loading: model={REPO_ID}')
    face_adapter = hf.hf_hub_download(repo_id=REPO_ID, filename="ip-adapter.bin")
    if controlnet_model is None or not cache:
        controlnet_model = ControlNetModel.from_pretrained(REPO_ID, subfolder="ControlNetModel", torch_dtype=devices.dtype, cache_dir=shared.opts.diffusers_dir)
        controlnet_model.to(devices.device, devices.dtype)

    processing.process_init(p)

    # create new pipeline
    orig_pipeline = shared.sd_model # backup current pipeline definition
    shared.sd_model = StableDiffusionXLInstantIDPipeline(
        vae = shared.sd_model.vae,
        text_encoder=shared.sd_model.text_encoder,
        text_encoder_2=shared.sd_model.text_encoder_2,
        tokenizer=shared.sd_model.tokenizer,
        tokenizer_2=shared.sd_model.tokenizer_2,
        unet=shared.sd_model.unet,
        scheduler=shared.sd_model.scheduler,
        controlnet=controlnet_model,
        force_zeros_for_empty_prompt=shared.opts.diffusers_force_zeros,
    )
    sd_models.copy_diffuser_options(shared.sd_model, orig_pipeline) # copy options from original pipeline
    sd_models.set_diffuser_options(shared.sd_model) # set all model options such as fp16, offload, etc.
    shared.sd_model.load_ip_adapter_instantid(face_adapter, scale=strength)
    shared.sd_model.set_ip_adapter_scale(strength)
    sd_models.move_model(shared.sd_model, devices.device) # move pipeline to device
    shared.sd_model.to(dtype=devices.dtype)

    # pipeline specific args
    orig_prompt_attention = shared.opts.prompt_attention
    shared.opts.data['prompt_attention'] = 'Fixed attention' # otherwise need to deal with class_tokens_mask
    p.task_args['prompt'] = p.all_prompts[0] # override all logic
    p.task_args['negative_prompt'] = p.all_negative_prompts[0]
    p.task_args['image_embeds'] = face_emb
    p.task_args['image'] = face_kps
    p.task_args['controlnet_conditioning_scale'] = float(conditioning)
    p.task_args['ip_adapter_scale'] = float(strength)
    debug(f'InstantID: args={p.task_args}')

    # run processing
    shared.log.debug(f'InstantID: strength={strength} conditioning={conditioning} image={source_image}')
    processed: processing.Processed = processing.process_images(p)
    shared.sd_model.set_ip_adapter_scale(0)
    p.extra_generation_params['InstantID'] = f'{strength}/{conditioning}'
    p.extra_generation_params["Face"] = f'{face.det_score:.2f} {"female" if face.gender==0 else "male"} {face.age}y'

    if not cache:
        controlnet_model = None
        devices.torch_gc()

    # restore original pipeline
    shared.opts.data['prompt_attention'] = orig_prompt_attention
    shared.sd_model = orig_pipeline
    return processed
