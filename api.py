import io
import json

from fastapi import Body, FastAPI
from pydantic import BaseModel, Field
import uvicorn
import os
import threading
import base64

from modules.paths import script_path

import signal

from modules.shared import opts, cmd_opts, state
import modules.shared as shared
import modules.ui
import modules.scripts
import modules.sd_hijack
import modules.codeformer_model
import modules.gfpgan_model
import modules.face_restoration
import modules.realesrgan_model as realesrgan
import modules.esrgan_model as esrgan
import modules.extras
import modules.lowvram
import modules.txt2img
import modules.img2img
import modules.sd_models

app = FastAPI()

modules.codeformer_model.setup_codeformer()
modules.gfpgan_model.setup_gfpgan()
shared.face_restorers.append(modules.face_restoration.FaceRestoration())

esrgan.load_models(cmd_opts.esrgan_models_path)
realesrgan.setup_realesrgan()

queue_lock = threading.Lock()


def wrap_queued_call(func):
    def f(*args, **kwargs):
        with queue_lock:
            res = func(*args, **kwargs)

        return res

    return f


def wrap_gradio_gpu_call(func):
    def f(*args, **kwargs):
        shared.state.sampling_step = 0
        shared.state.job_count = -1
        shared.state.job_no = 0
        shared.state.current_latent = None
        shared.state.current_image = None
        shared.state.current_image_sampling_step = 0

        with queue_lock:
            res = func(*args, **kwargs)

        shared.state.job = ""
        shared.state.job_count = 0

        return res

    return modules.ui.wrap_gradio_call(f)


modules.scripts.load_scripts(os.path.join(script_path, "scripts"))

shared.sd_model = modules.sd_models.load_model()
shared.opts.onchange("sd_model_checkpoint",
                     wrap_queued_call(lambda: modules.sd_models.reload_model_weights(shared.sd_model)))


class TextToImage(BaseModel):
    prompt: str = Field(..., title="Prompt Text", description="The text to generate an image from.")
    negative_prompt: str = Field(default="", title="Negative Prompt Text")
    prompt_style: str = Field(default="None", title="Prompt Style")
    prompt_style2: str = Field(default="None", title="Prompt Style 2")
    steps: int = Field(default=20, title="Steps")
    sampler_index: int = Field(0, title="Sampler Index")
    restore_faces: bool = Field(default=False, title="Restore Faces")
    tiling: bool = Field(default=False, title="Tiling")
    n_iter: int = Field(default=1, title="N Iter")
    batch_size: int = Field(default=1, title="Batch Size")
    cfg_scale: float = Field(default=7, title="Config Scale")
    seed: int = Field(default=-1.0, title="Seed")
    subseed: int = Field(default=-1.0, title="Subseed")
    subseed_strength: float = Field(default=0, title="Subseed Strength")
    seed_resize_from_h: int = Field(default=0, title="Seed Resize From Height")
    seed_resize_from_w: int = Field(default=0, title="Seed Resize From Width")
    height: int = Field(default=512, title="Height")
    width: int = Field(default=512, title="Width")
    enable_hr: bool = Field(default=False, title="Enable HR")
    scale_latent: bool = Field(default=True, title="Scale Latent")
    denoising_strength: float = Field(default=0.7, title="Denoising Strength")


class TextToImageResponse(BaseModel):
    images: list[str] = Field(..., title="Image", description="The generated image in base64 format.")
    all_prompts: list[str] = Field(..., title="All Prompts", description="The prompt text.")
    negative_prompt: str = Field(..., title="Negative Prompt Text")
    seed: int = Field(..., title="Seed")
    all_seeds: list[int] = Field(..., title="All Seeds")
    subseed: int = Field(..., title="Subseed")
    all_subseeds: list[int] = Field(..., title="All Subseeds")
    subseed_strength: float = Field(..., title="Subseed Strength")
    width: int = Field(..., title="Width")
    height: int = Field(..., title="Height")
    sampler_index: int = Field(..., title="Sampler Index")
    sampler: str = Field(..., title="Sampler")
    cfg_scale: float = Field(..., title="Config Scale")
    steps: int = Field(..., title="Steps")
    batch_size: int = Field(..., title="Batch Size")
    restore_faces: bool = Field(..., title="Restore Faces")
    face_restoration_model: str = Field(default=None, title="Face Restoration Model")
    sd_model_hash: str = Field(..., title="SD Model Hash")
    seed_resize_from_w: int = Field(..., title="Seed Resize From Width")
    seed_resize_from_h: int = Field(..., title="Seed Resize From Height")
    denoising_strength: float = Field(default=None, title="Denoising Strength")
    extra_generation_params: str | None = Field(default="", title="Extra Generation Params")
    index_of_first_image: int = Field(..., title="Index of First Image")
    html: str = Field(..., title="HTML")


@app.get("/txt2img", response_model=TextToImageResponse)
def text2img(txt2img: TextToImage = Body(embed=True)):
    resp = modules.txt2img.txt2img(txt2img.prompt, txt2img.negative_prompt, txt2img.prompt_style,
                                   txt2img.prompt_style2, txt2img.steps, txt2img.sampler_index,
                                   txt2img.restore_faces, txt2img.tiling, txt2img.n_iter,
                                   txt2img.batch_size, txt2img.cfg_scale, txt2img.seed,
                                   txt2img.subseed, txt2img.subseed_strength,
                                   txt2img.seed_resize_from_h, txt2img.seed_resize_from_w,
                                   txt2img.height, txt2img.width, txt2img.enable_hr,
                                   txt2img.scale_latent, txt2img.denoising_strength,
                                   0, False, None, '', False, 1, '', 4, '', True)

    images = []
    for i in resp[0]:
        buffer = io.BytesIO()
        i.save(buffer, format="png")
        images.append(base64.b64encode(buffer.getvalue()))
    j = json.loads(resp[1])

    return TextToImageResponse(images=images, all_prompts=j['all_prompts'],
                               negative_prompt=j['negative_prompt'],
                               seed=j['seed'], all_seeds=j['all_seeds'],
                               subseed=j['subseed'], all_subseeds=j['all_subseeds'],
                               subseed_strength=j['subseed_strength'], width=j['width'],
                               height=j['height'], sampler_index=j['sampler_index'],
                               sampler=j['sampler'], cfg_scale=j['cfg_scale'],
                               steps=j['steps'], batch_size=j['batch_size'],
                               restore_faces=j['restore_faces'],
                               face_restoration_model=j['face_restoration_model'],
                               sd_model_hash=j['sd_model_hash'],
                               seed_resize_from_w=j['seed_resize_from_w'],
                               seed_resize_from_h=j['seed_resize_from_h'],
                               denoising_strength=j['denoising_strength'],
                               extra_generation_params=j['extra_generation_params'],
                               index_of_first_image=j['index_of_first_image'],
                               html=resp[2])


if __name__ == '__main__':
    uvicorn.run(app, port=8080, host='0.0.0.0')
