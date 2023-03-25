import os
import sys
import time
import importlib
import signal
import re
import warnings
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from packaging import version
from rich import print

import logging
logging.getLogger("xformers").addFilter(lambda record: 'A matching Triton is not available' not in record.getMessage())

from modules import paths, timer, import_hook, errors

startup_timer = timer.Timer()

import torch
import pytorch_lightning # pytorch_lightning should be imported after torch, but it re-enables warnings on import so import once to disable them
warnings.filterwarnings(action="ignore", category=DeprecationWarning, module="pytorch_lightning")
startup_timer.record("import torch")

import gradio
import ldm.modules.encoders.modules

from modules import extra_networks, ui_extra_networks_checkpoints
from modules import extra_networks_hypernet, ui_extra_networks_hypernets, ui_extra_networks_textual_inversion
from modules.call_queue import wrap_queued_call, queue_lock, wrap_gradio_gpu_call

# Truncate version number of nightly/local build of PyTorch to not cause exceptions with CodeFormer or Safetensors
if ".dev" in torch.__version__ or "+git" in torch.__version__:
    torch.__long_version__ = torch.__version__
    torch.__version__ = re.search(r'[\d.]+[\d]', torch.__version__).group(0)

from modules import shared, devices, sd_samplers, upscaler, extensions, ui_tempdir, ui_extra_networks
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
from modules.shared import cmd_opts
import modules.hypernetworks.hypernetwork

startup_timer.record("import libraries")


if cmd_opts.server_name:
    server_name = cmd_opts.server_name
else:
    server_name = "0.0.0.0" if cmd_opts.listen else None


def initialize():
    if torch.cuda.is_available():
        if torch.version.cuda: cuda_version = f'CUDA {torch.version.cuda} cuDNN {torch.backends.cudnn.version()}'
        elif torch.version.hip: cuda_version = f'HIP {torch.version.hip}'
        else: cuda_version = ''
        print(f'Torch {getattr(torch, "__long_version__", torch.__version__)} {cuda_version}')
        for device in [torch.cuda.device(i) for i in range(torch.cuda.device_count())]:
            print(f'GPU {torch.cuda.get_device_name(device)} VRAM {round(torch.cuda.get_device_properties(device).total_memory / 1024 / 1024)} Arch {torch.cuda.get_device_capability(device)} Cores {torch.cuda.get_device_properties(device).multi_processor_count}')
    else:
        print(f'Torch {getattr(torch, "__long_version__", torch.__version__)} running on CPU')

    extensions.list_extensions()
    startup_timer.record("list extensions")

    if cmd_opts.ui_debug_mode:
        shared.sd_upscalers = upscaler.UpscalerLanczos().scalers
        modules.scripts.load_scripts()
        return

    modelloader.cleanup_models()
    modules.sd_models.setup_model()
    startup_timer.record("list models")

    codeformer.setup_model(cmd_opts.codeformer_models_path)
    startup_timer.record("setup codeformer")

    gfpgan.setup_model(cmd_opts.gfpgan_models_path)
    startup_timer.record("setup gfpgan")

    modelloader.list_builtin_upscalers()
    startup_timer.record("list builtin upscalers")

    modules.scripts.load_scripts()
    startup_timer.record("load scripts")

    modelloader.load_upscalers()
    startup_timer.record("load upscalers")

    modules.sd_vae.refresh_vae_list()
    startup_timer.record("refresh VAE")

    modules.textual_inversion.textual_inversion.list_textual_inversion_templates()
    startup_timer.record("refresh textual inversion templates")

    shared.opts.onchange("sd_vae", wrap_queued_call(lambda: modules.sd_vae.reload_vae_weights()), call=False)
    shared.opts.onchange("sd_vae_as_default", wrap_queued_call(lambda: modules.sd_vae.reload_vae_weights()), call=False)
    shared.opts.onchange("temp_dir", ui_tempdir.on_tmpdir_changed)
    startup_timer.record("opts onchange")

    shared.reload_hypernetworks()
    startup_timer.record("reload hypernets")

    ui_extra_networks.intialize()
    ui_extra_networks.register_page(ui_extra_networks_textual_inversion.ExtraNetworksPageTextualInversion())
    ui_extra_networks.register_page(ui_extra_networks_hypernets.ExtraNetworksPageHypernetworks())
    ui_extra_networks.register_page(ui_extra_networks_checkpoints.ExtraNetworksPageCheckpoints())

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
        print("", file=sys.stderr)
        print("Stable diffusion model failed to load, exiting", file=sys.stderr)
        exit(1)
    shared.opts.data["sd_model_checkpoint"] = shared.sd_model.sd_checkpoint_info.title
    shared.opts.onchange("sd_model_checkpoint", wrap_queued_call(lambda: modules.sd_models.reload_model_weights()))
    shared.state.end()
    startup_timer.record("load checkpoint")


def setup_middleware(app):
    app.middleware_stack = None # reset current middleware to allow modifying user provided list
    app.add_middleware(GZipMiddleware, minimum_size=1024)
    if cmd_opts.cors_allow_origins and cmd_opts.cors_allow_origins_regex:
        app.add_middleware(CORSMiddleware, allow_origins=cmd_opts.cors_allow_origins.split(','), allow_origin_regex=cmd_opts.cors_allow_origins_regex, allow_methods=['*'], allow_credentials=True, allow_headers=['*'])
    elif cmd_opts.cors_allow_origins:
        app.add_middleware(CORSMiddleware, allow_origins=cmd_opts.cors_allow_origins.split(','), allow_methods=['*'], allow_credentials=True, allow_headers=['*'])
    elif cmd_opts.cors_allow_origins_regex:
        app.add_middleware(CORSMiddleware, allow_origin_regex=cmd_opts.cors_allow_origins_regex, allow_methods=['*'], allow_credentials=True, allow_headers=['*'])
    app.build_middleware_stack() # rebuild middleware stack on-the-fly


def create_api(app):
    from modules.api.api import Api
    api = Api(app, queue_lock)
    return api


def api_only():
    initialize()
    load_model()

    app = FastAPI()
    setup_middleware(app)
    api = create_api(app)

    modules.script_callbacks.app_started_callback(None, app)

    print(f"Startup time: {startup_timer.summary()}.")
    api.launch(server_name="0.0.0.0" if cmd_opts.listen else "127.0.0.1", port=cmd_opts.port if cmd_opts.port else 7861)


def webui():
    launch_api = cmd_opts.api
    initialize()

    if shared.opts.clean_temp_dir_at_start:
        ui_tempdir.cleanup_tmpdr()
        startup_timer.record("cleanup temp dir")

    modules.script_callbacks.before_ui_callback()
    startup_timer.record("scripts before_ui_callback")

    shared.demo = modules.ui.create_ui()
    startup_timer.record("create ui")

    if not cmd_opts.no_gradio_queue:
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
        debug=cmd_opts.gradio_debug,
        auth=[tuple(cred.split(':')) for cred in gradio_auth_creds] if gradio_auth_creds else None,
        inbrowser=cmd_opts.autolaunch,
        prevent_thread_lock=True,
        favicon_path='automatic.ico',
    )
    for dep in shared.demo.dependencies:
        dep['show_progress'] = False  # disable gradio css animation on component update

    # app is instance of FastAPI server
    # shared.demo.server is instance of gradio class which inherits from uvicorn.Server
    # shared.demo.config is instance of uvicorn.Config
    # shared.demo.app is instance of ASGIApp

    cmd_opts.autolaunch = False

    startup_timer.record("gradio launch")

    app.user_middleware = [x for x in app.user_middleware if x.cls.__name__ != 'CORSMiddleware']

    setup_middleware(app)

    modules.progress.setup_progress_api(app)

    if launch_api:
        create_api(app)

    ui_extra_networks.add_pages_to_demo(app)

    modules.script_callbacks.app_started_callback(shared.demo, app)
    startup_timer.record("scripts app_started_callback")

    load_model()

    print(f"Startup time: {startup_timer.summary()}.")

    while True:
        time.sleep(0.1)


if __name__ == "__main__":
    if cmd_opts.nowebui:
        api_only()
    else:
        webui()
