import os
import threading

from modules.paths import script_path

import torch
from omegaconf import OmegaConf

import signal

from ldm.util import instantiate_from_config

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


modules.codeformer_model.setup_codeformer()
modules.gfpgan_model.setup_gfpgan()
shared.face_restorers.append(modules.face_restoration.FaceRestoration())

esrgan.load_models(cmd_opts.esrgan_models_path)
realesrgan.setup_realesrgan()


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model [{shared.sd_model_hash}] from {ckpt}")
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
    if cmd_opts.opt_channelslast:
        model = model.to(memory_format=torch.channels_last)
    model.eval()
    return model


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


modules.scripts.load_scripts(os.path.join(script_path, "scripts"))

try:
    # this silences the annoying "Some weights of the model checkpoint were not used when initializing..." message at start.

    from transformers import logging

    logging.set_verbosity_error()
except Exception:
    pass

with open(cmd_opts.ckpt, "rb") as file:
    import hashlib
    m = hashlib.sha256()

    file.seek(0x100000)
    m.update(file.read(0x10000))
    shared.sd_model_hash = m.hexdigest()[0:8]

sd_config = OmegaConf.load(cmd_opts.config)
shared.sd_model = load_model_from_config(sd_config, cmd_opts.ckpt)
shared.sd_model = (shared.sd_model if cmd_opts.no_half else shared.sd_model.half())

if cmd_opts.lowvram or cmd_opts.medvram:
    modules.lowvram.setup_for_low_vram(shared.sd_model, cmd_opts.medvram)
else:
    shared.sd_model = shared.sd_model.to(shared.device)

modules.sd_hijack.model_hijack.hijack(shared.sd_model)


def webui():
    # make the program just exit at ctrl+c without waiting for anything
    def sigint_handler(sig, frame):
        print(f'Interrupted with signal {sig} in {frame}')
        os._exit(0)

    signal.signal(signal.SIGINT, sigint_handler)

    demo = modules.ui.create_ui(
        txt2img=wrap_gradio_gpu_call(modules.txt2img.txt2img),
        img2img=wrap_gradio_gpu_call(modules.img2img.img2img),
        run_extras=wrap_gradio_gpu_call(modules.extras.run_extras),
        run_pnginfo=modules.extras.run_pnginfo
    )

    demo.launch(
        share=cmd_opts.share,
        server_name="0.0.0.0" if cmd_opts.listen else None,
        server_port=cmd_opts.port,
        debug=cmd_opts.gradio_debug,
        auth=[tuple(cred.split(':')) for cred in cmd_opts.gradio_auth.strip('"').split(',')] if cmd_opts.gradio_auth else None,
    )


if __name__ == "__main__":
    webui()
