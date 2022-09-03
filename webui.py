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
import modules.gfpgan_model as gfpgan
import modules.realesrgan_model as realesrgan
import modules.images as images
import modules.lowvram
import modules.txt2img
import modules.img2img


shared.sd_upscalers = {
    "RealESRGAN": lambda img: realesrgan.upscale_with_realesrgan(img, 2, 0),
    "Lanczos": lambda img: img.resize((img.width*2, img.height*2), resample=images.LANCZOS),
    "None": lambda img: img
}
realesrgan.setup_realesrgan()
gfpgan.setup_gfpgan()


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


def run_extras(image, GFPGAN_strength, RealESRGAN_upscaling, RealESRGAN_model_index):
    processing.torch_gc()

    image = image.convert("RGB")

    outpath = opts.outdir_samples or opts.outdir_extras_samples

    if gfpgan.have_gfpgan is not None and GFPGAN_strength > 0:

        restored_img = gfpgan.gfpgan_fix_faces(np.array(image, dtype=np.uint8))
        res = Image.fromarray(restored_img)

        if GFPGAN_strength < 1.0:
            res = Image.blend(image, res, GFPGAN_strength)

        image = res

    if realesrgan.have_realesrgan and RealESRGAN_upscaling != 1.0:
        image = realesrgan.upscale_with_realesrgan(image, RealESRGAN_upscaling, RealESRGAN_model_index)

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
        with queue_lock:
            res = func(*args, **kwargs)

        shared.state.job = ""

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


# make the program just exit at ctrl+c without waiting for anything
def sigint_handler(sig, frame):
    print(f'Interrupted with singal {sig} in {frame}')
    os._exit(0)


signal.signal(signal.SIGINT, sigint_handler)

demo = modules.ui.create_ui(
    txt2img=wrap_gradio_gpu_call(modules.txt2img.txt2img),
    img2img=wrap_gradio_gpu_call(modules.img2img.img2img),
    run_extras=wrap_gradio_gpu_call(run_extras),
    run_pnginfo=run_pnginfo
)

demo.launch(share=cmd_opts.share)
