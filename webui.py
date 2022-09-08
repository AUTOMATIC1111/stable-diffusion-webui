import os
import threading

from modules.paths import script_path

import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image

import signal

from ldm.util import instantiate_from_config

from modules.shared import opts, cmd_opts, state
import modules.shared as shared
import modules.ui
from modules.ui import plaintext_to_html
import modules.scripts
import modules.processing as processing
import modules.sd_hijack
import modules.codeformer_model
import modules.gfpgan_model
import modules.face_restoration
import modules.realesrgan_model as realesrgan
import modules.esrgan_model as esrgan
import modules.images as images
import modules.lowvram
import modules.txt2img
import modules.img2img


modules.codeformer_model.setup_codeformer()
modules.gfpgan_model.setup_gfpgan()
shared.face_restorers.append(modules.face_restoration.FaceRestoration())

esrgan.load_models(cmd_opts.esrgan_models_path)
realesrgan.setup_realesrgan()

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.eval()
    return model

cached_images = {}


def run_extras(image, gfpgan_visibility, codeformer_visibility, codeformer_weight, upscaling_resize, extras_upscaler_1, extras_upscaler_2, extras_upscaler_2_visibility):
    processing.torch_gc()

    image = image.convert("RGB")

    outpath = opts.outdir_samples or opts.outdir_extras_samples

    if gfpgan_visibility > 0:
        restored_img = modules.gfpgan_model.gfpgan_fix_faces(np.array(image, dtype=np.uint8))
        res = Image.fromarray(restored_img)

        if gfpgan_visibility < 1.0:
            res = Image.blend(image, res, gfpgan_visibility)

        image = res

    if codeformer_visibility > 0:
        restored_img = modules.codeformer_model.codeformer.restore(np.array(image, dtype=np.uint8), w=codeformer_weight)
        res = Image.fromarray(restored_img)

        if codeformer_visibility < 1.0:
            res = Image.blend(image, res, codeformer_visibility)

        image = res

    if upscaling_resize != 1.0:
        def upscale(image, scaler_index, resize):
            small = image.crop((image.width // 2, image.height // 2, image.width // 2 + 10, image.height // 2 + 10))
            pixels = tuple(np.array(small).flatten().tolist())
            key = (resize, scaler_index, image.width, image.height) + pixels

            c = cached_images.get(key)
            if c is None:
                upscaler = shared.sd_upscalers[scaler_index]
                c = upscaler.upscale(image, image.width * resize, image.height * resize)
                cached_images[key] = c

            return c

        res = upscale(image, extras_upscaler_1, upscaling_resize)

        if extras_upscaler_2 != 0 and extras_upscaler_2_visibility>0:
            res2 = upscale(image, extras_upscaler_2, upscaling_resize)
            res = Image.blend(res, res2, extras_upscaler_2_visibility)

        image = res

    while len(cached_images) > 2:
        del cached_images[next(iter(cached_images.keys()))]

    images.save_image(image, outpath, "", None, '', opts.samples_format, short_filename=True, no_prompt=True)

    return image, '', ''


def run_pnginfo(image):
    info = ''
    for key, text in image.info.items():
        info += f"""
<div>
<p><b>{plaintext_to_html(str(key))}</b></p>
<p>{plaintext_to_html(str(text))}</p>
</div>
""".strip()+"\n"

    if len(info) == 0:
        message = "Nothing found in the image."
        info = f"<div><p>{message}<p></div>"

    return '', '', info


queue_lock = threading.Lock()


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


try:
    # this silences the annoying "Some weights of the model checkpoint were not used when initializing..." message at start.

    from transformers import logging

    logging.set_verbosity_error()
except Exception:
    pass

sd_config = OmegaConf.load(cmd_opts.config)
shared.sd_model = load_model_from_config(sd_config, cmd_opts.ckpt)
shared.sd_model = (shared.sd_model if cmd_opts.no_half else shared.sd_model.half())

if cmd_opts.lowvram or cmd_opts.medvram:
    modules.lowvram.setup_for_low_vram(shared.sd_model, cmd_opts.medvram)
else:
    shared.sd_model = shared.sd_model.to(shared.device)

modules.sd_hijack.model_hijack.hijack(shared.sd_model)

modules.scripts.load_scripts(os.path.join(script_path, "scripts"))

if __name__ == "__main__":
    # make the program just exit at ctrl+c without waiting for anything
    def sigint_handler(sig, frame):
        print(f'Interrupted with signal {sig} in {frame}')
        os._exit(0)


    signal.signal(signal.SIGINT, sigint_handler)

    demo = modules.ui.create_ui(
        txt2img=wrap_gradio_gpu_call(modules.txt2img.txt2img),
        img2img=wrap_gradio_gpu_call(modules.img2img.img2img),
        run_extras=wrap_gradio_gpu_call(run_extras),
        run_pnginfo=run_pnginfo
    )

    demo.launch(share=cmd_opts.share, server_name="0.0.0.0" if cmd_opts.listen else None, server_port=cmd_opts.port if cmd_opts.port else None)
