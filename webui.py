import os
import threading

from modules.paths import script_path

import signal

from modules.shared import opts, cmd_opts, state
import modules.shared as shared
import modules.ui
import modules.scripts
import modules.sd_hijack
import modules.model_codeformer
import modules.model_gfpgan
import modules.face_restoration
import modules.model_realesrgan as realesrgan
import modules.model_esrgan as esrgan
import modules.model_ldsr as ldsr
import modules.model_swinir as swinir


import modules.extras
import modules.lowvram
import modules.txt2img
import modules.img2img
import modules.sd_models


modules.model_codeformer.setup_model(cmd_opts.codeformer_models_path)
modules.model_gfpgan.setup_model(cmd_opts.gfpgan_models_path)
shared.face_restorers.append(modules.face_restoration.FaceRestoration())

esrgan.setup_model(cmd_opts.esrgan_models_path)
swinir.setup_model(cmd_opts.swinir_models_path)
realesrgan.setup_model(cmd_opts.realesrgan_models_path)
ldsr.setup_model(cmd_opts.ldsr_models_path)
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
        shared.state.interrupted = False

        with queue_lock:
            res = func(*args, **kwargs)

        shared.state.job = ""
        shared.state.job_count = 0

        return res

    return modules.ui.wrap_gradio_call(f)


modules.scripts.load_scripts(os.path.join(script_path, "scripts"))

shared.sd_model = modules.sd_models.load_model()
shared.opts.onchange("sd_model_checkpoint", wrap_queued_call(lambda: modules.sd_models.reload_model_weights(shared.sd_model)))


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
        inbrowser=cmd_opts.autolaunch,
    )


if __name__ == "__main__":
    webui()
