import os
import re
import time
import signal
import warnings
import logging
from rich import print # pylint: disable=W0622
from modules import timer, errors

startup_timer = timer.Timer()

import torch # pylint: disable=C0411
import torchvision # pylint: disable=W0611,C0411
import pytorch_lightning # pytorch_lightning should be imported after torch, but it re-enables warnings on import so import once to disable them # pylint: disable=W0611,C0411
logging.getLogger("xformers").addFilter(lambda record: 'A matching Triton is not available' not in record.getMessage())
logging.getLogger("pytorch_lightning").disabled = True
warnings.filterwarnings(action="ignore", category=DeprecationWarning, module="pytorch_lightning")
warnings.filterwarnings(action="ignore", category=UserWarning, module="torchvision")
startup_timer.record("torch")

from modules import import_hook # pylint: disable=W0611,C0411,C0412
import gradio # pylint: disable=W0611,C0411
startup_timer.record("gradio")
errors.install([gradio])

import ldm.modules.encoders.modules # pylint: disable=W0611,C0411
from modules import extra_networks, ui_extra_networks_checkpoints # pylint: disable=C0411,C0412
from modules import extra_networks_hypernet, ui_extra_networks_hypernets, ui_extra_networks_textual_inversion
from modules.call_queue import wrap_queued_call, queue_lock, wrap_gradio_gpu_call # pylint: disable=W0611,C0411
from modules.paths import create_paths

# Truncate version number of nightly/local build of PyTorch to not cause exceptions with CodeFormer or Safetensors
if ".dev" in torch.__version__ or "+git" in torch.__version__:
    torch.__long_version__ = torch.__version__
    torch.__version__ = re.search(r'[\d.]+[\d]', torch.__version__).group(0)

from modules import shared, extensions, ui_tempdir, ui_extra_networks
import modules.devices
import modules.sd_samplers
import modules.upscaler
import modules.codeformer_model as codeformer
import modules.face_restoration
import modules.gfpgan_model as gfpgan
import modules.img2img
import modules.lowvram
import modules.scripts
import modules.sd_hijack
import modules.sd_models
import modules.sd_vae
import modules.txt2img
import modules.script_callbacks
import modules.textual_inversion.textual_inversion
import modules.progress
import modules.ui
from modules import modelloader
from modules.shared import cmd_opts, opts
import modules.hypernetworks.hypernetwork
from modules.middleware import setup_middleware
startup_timer.record("libraries")

if cmd_opts.server_name:
    server_name = cmd_opts.server_name
else:
    server_name = "0.0.0.0" if cmd_opts.listen else None


def check_rollback_vae():
    if shared.cmd_opts.rollback_vae:
        if not torch.__version__.startswith('2.1'):
            print("Rollback VAE functionality requires Torch 2.1 or higher")
            shared.cmd_opts.rollback_vae = False
        if 0 < torch.cuda.get_device_capability()[0] < 8:
            print('Rollback VAE functionality device capabilities not met')
            shared.cmd_opts.rollback_vae = False


def initialize():
    check_rollback_vae()

    extensions.list_extensions()
    startup_timer.record("extensions")

    modelloader.cleanup_models()
    modules.sd_models.setup_model()
    startup_timer.record("models")

    codeformer.setup_model(opts.codeformer_models_path)
    startup_timer.record("codeformer")

    gfpgan.setup_model(opts.gfpgan_models_path)
    startup_timer.record("gfpgan")

    modelloader.list_builtin_upscalers()
    startup_timer.record("upscalers")

    modules.scripts.load_scripts()
    startup_timer.record("scripts")

    modelloader.load_upscalers()
    startup_timer.record("upscalers")

    modules.sd_vae.refresh_vae_list()
    startup_timer.record("vae")

    shared.opts.onchange("sd_vae", wrap_queued_call(lambda: modules.sd_vae.reload_vae_weights()), call=False)
    shared.opts.onchange("sd_vae_as_default", wrap_queued_call(lambda: modules.sd_vae.reload_vae_weights()), call=False)
    shared.opts.onchange("temp_dir", ui_tempdir.on_tmpdir_changed)
    shared.opts.onchange("gradio_theme", shared.reload_gradio_theme)
    startup_timer.record("opts onchange")

    modules.textual_inversion.textual_inversion.list_textual_inversion_templates()
    shared.reload_hypernetworks()
    ui_extra_networks.intialize()
    ui_extra_networks.register_page(ui_extra_networks_hypernets.ExtraNetworksPageHypernetworks())
    ui_extra_networks.register_page(ui_extra_networks_checkpoints.ExtraNetworksPageCheckpoints())
    ui_extra_networks.register_page(ui_extra_networks_textual_inversion.ExtraNetworksPageTextualInversion())
    extra_networks.initialize()
    extra_networks.register_extra_network(extra_networks_hypernet.ExtraNetworkHypernet())
    startup_timer.record("extra networks")

    if cmd_opts.tls_keyfile is not None and cmd_opts.tls_keyfile is not None:
        try:
            if not os.path.exists(cmd_opts.tls_keyfile):
                print("Invalid path to TLS keyfile given")
            if not os.path.exists(cmd_opts.tls_certfile):
                print(f"Invalid path to TLS certfile: '{cmd_opts.tls_certfile}'")
        except TypeError:
            cmd_opts.tls_keyfile = cmd_opts.tls_certfile = None
            print("TLS setup invalid, running webui without TLS")
        else:
            print("Running with TLS")
        startup_timer.record("TLS")

    # make the program just exit at ctrl+c without waiting for anything
    def sigint_handler(_sig, _frame):
        print('Exiting')
        os._exit(0)

    signal.signal(signal.SIGINT, sigint_handler)


def load_model():
    shared.state.begin()
    shared.state.job = 'load model'
    try:
        modules.sd_models.load_model()
    except Exception as e:
        errors.display(e, "loading stable diffusion model")
        print("Stable diffusion model failed to load")
        exit(1)
    if shared.sd_model is None:
        print("No stable diffusion model loaded")
        exit(1)
    shared.opts.data["sd_model_checkpoint"] = shared.sd_model.sd_checkpoint_info.title
    shared.opts.onchange("sd_model_checkpoint", wrap_queued_call(lambda: modules.sd_models.reload_model_weights()))
    shared.state.end()
    startup_timer.record("checkpoint")


def create_api(app):
    from modules.api.api import Api
    api = Api(app, queue_lock)
    return api


def start_ui():
    logging.disable(logging.INFO)
    create_paths(opts)
    initialize()
    if shared.opts.clean_temp_dir_at_start:
        ui_tempdir.cleanup_tmpdr()
        startup_timer.record("cleanup")
    modules.script_callbacks.before_ui_callback()
    startup_timer.record("scripts before_ui_callback")
    shared.demo = modules.ui.create_ui()
    startup_timer.record("ui")
    if cmd_opts.disable_queue:
        print('Server queues disabled')
    else:
        shared.demo.queue(16)

    gradio_auth_creds = []
    if cmd_opts.gradio_auth:
        gradio_auth_creds += [x.strip() for x in cmd_opts.gradio_auth.strip('"').replace('\n', '').split(',') if x.strip()]
    if cmd_opts.gradio_auth_path:
        with open(cmd_opts.gradio_auth_path, 'r', encoding="utf8") as file:
            for line in file.readlines():
                gradio_auth_creds += [x.strip() for x in line.split(',') if x.strip()]

    app, _local_url, _share_url = shared.demo.launch(
        share=cmd_opts.share,
        server_name=server_name,
        server_port=cmd_opts.port,
        ssl_keyfile=cmd_opts.tls_keyfile,
        ssl_certfile=cmd_opts.tls_certfile,
        debug=False,
        auth=[tuple(cred.split(':')) for cred in gradio_auth_creds] if gradio_auth_creds else None,
        inbrowser=cmd_opts.autolaunch,
        prevent_thread_lock=True,
        favicon_path='automatic.ico',
    )
    setup_middleware(app, cmd_opts)

    cmd_opts.autolaunch = False
    startup_timer.record("start")

    modules.progress.setup_progress_api(app)
    create_api(app)
    ui_extra_networks.add_pages_to_demo(app)

    modules.script_callbacks.app_started_callback(shared.demo, app)
    startup_timer.record("scripts app_started_callback")


def webui():
    start_ui()
    load_model()
    print(f"Startup time: {startup_timer.summary()}")
    logging.disable(logging.DEBUG)

    while True:
        try:
            alive = shared.demo.server.thread.is_alive()
        except:
            alive = False
        if not alive:
            print('Server restart')
            startup_timer.reset()
            start_ui()
            print(f"Startup time: {startup_timer.summary()}")
        time.sleep(1)

    """
    import sys
    import types
    from modules.paths_internal import script_path
    libs = [name for name, m in sys.modules.items() if isinstance(m, types.ModuleType) and (getattr(m, '__file__', '') or '').startswith(script_path)]
    print(libs)
    """


if __name__ == "__main__":
    webui()
