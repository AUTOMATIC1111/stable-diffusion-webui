from __future__ import annotations
import os
import re
import sys
import signal
import asyncio
import logging
import warnings
import importlib
from threading import Thread
from modules import timer, errors, paths # pylint: disable=unused-import

startup_timer = timer.Timer()
local_url = None

errors.log.debug('Loading Torch')
import torch # pylint: disable=C0411
try:
    import intel_extension_for_pytorch as ipex # pylint: disable=import-error, unused-import
except Exception:
    pass
import torchvision # pylint: disable=W0611,C0411
import pytorch_lightning # pytorch_lightning should be imported after torch, but it re-enables warnings on import so import once to disable them # pylint: disable=W0611,C0411
if ".dev" in torch.__version__ or "+git" in torch.__version__:
    torch.__long_version__ = torch.__version__
    torch.__version__ = re.search(r'[\d.]+[\d]', torch.__version__).group(0)
logging.getLogger("xformers").addFilter(lambda record: 'A matching Triton is not available' not in record.getMessage())
logging.getLogger("pytorch_lightning").disabled = True
warnings.filterwarnings(action="ignore", category=DeprecationWarning)
warnings.filterwarnings(action="ignore", category=FutureWarning)
warnings.filterwarnings(action="ignore", category=UserWarning, module="torchvision")
startup_timer.record("torch")

errors.log.debug('Loading Gradio')
from fastapi import FastAPI # pylint: disable=W0611,C0411
import gradio # pylint: disable=W0611,C0411
startup_timer.record("gradio")
errors.install([gradio])

errors.log.debug('Loading Modules')
from installer import log, setup_logging, git_commit
import ldm.modules.encoders.modules # pylint: disable=W0611,C0411,E0401
from modules.call_queue import queue_lock, wrap_queued_call, wrap_gradio_gpu_call # pylint: disable=W0611,C0411,C0412
from modules.paths import create_paths
from modules import shared, extensions, extra_networks, ui_tempdir, ui_extra_networks, modelloader
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
from modules.shared import cmd_opts, opts
import modules.hypernetworks.hypernetwork
from modules.middleware import setup_middleware
startup_timer.record("libraries")
log.info('Libraries loaded')
log.setLevel(logging.DEBUG if cmd_opts.debug else logging.INFO)
logging.disable(logging.NOTSET if cmd_opts.debug else logging.DEBUG)
if cmd_opts.server_name:
    server_name = cmd_opts.server_name
else:
    server_name = "0.0.0.0" if cmd_opts.listen else None

fastapi_args = {
    "version": f'0.0.{git_commit}',
    "title": "SD.Next",
    "description": "SD.Next",
    "docs_url": "/docs" if cmd_opts.docs else None,
    "redocs_url": "/redocs" if cmd_opts.docs else None,
    "swagger_ui_parameters": {
        "displayOperationId": True,
        "showCommonExtensions": True,
        "deepLinking": False,
    }
}


def check_rollback_vae():
    if shared.cmd_opts.rollback_vae:
        if not torch.cuda.is_available():
            log.error("Rollback VAE functionality requires compatible GPU")
            shared.cmd_opts.rollback_vae = False
        elif not torch.__version__.startswith('2.1'):
            log.error("Rollback VAE functionality requires Torch 2.1 or higher")
            shared.cmd_opts.rollback_vae = False
        elif 0 < torch.cuda.get_device_capability()[0] < 8:
            log.error('Rollback VAE functionality device capabilities not met')
            shared.cmd_opts.rollback_vae = False


def initialize():
    log.debug('Entering initialize')
    check_rollback_vae()

    modules.sd_vae.refresh_vae_list()
    startup_timer.record("vae")

    extensions.list_extensions()
    startup_timer.record("extensions")

    modelloader.cleanup_models()
    modules.sd_models.setup_model()
    startup_timer.record("models")

    codeformer.setup_model(opts.codeformer_models_path)
    startup_timer.record("codeformer")

    gfpgan.setup_model(opts.gfpgan_models_path)
    startup_timer.record("gfpgan")

    log.debug('Loading scripts')
    modules.scripts.load_scripts()
    startup_timer.record("scripts")

    modelloader.load_upscalers()
    startup_timer.record("upscalers")

    setup_logging() # needs a reset since scripts can hijaack logging

    shared.opts.onchange("sd_vae", wrap_queued_call(lambda: modules.sd_vae.reload_vae_weights()), call=False)
    shared.opts.onchange("temp_dir", ui_tempdir.on_tmpdir_changed)
    shared.opts.onchange("gradio_theme", shared.reload_gradio_theme)
    startup_timer.record("onchange")

    modules.textual_inversion.textual_inversion.list_textual_inversion_templates()
    shared.reload_hypernetworks()

    ui_extra_networks.initialize()
    ui_extra_networks.register_default_pages()
    extra_networks.initialize()
    extra_networks.register_default_extra_networks()
    startup_timer.record("extra-networks")

    if cmd_opts.tls_keyfile is not None and cmd_opts.tls_certfile is not None:
        try:
            if not os.path.exists(cmd_opts.tls_keyfile):
                log.error("Invalid path to TLS keyfile given")
            if not os.path.exists(cmd_opts.tls_certfile):
                log.error(f"Invalid path to TLS certfile: '{cmd_opts.tls_certfile}'")
        except TypeError:
            cmd_opts.tls_keyfile = cmd_opts.tls_certfile = None
            log.error("TLS setup invalid, running webui without TLS")
        else:
            log.info("Running with TLS")
        startup_timer.record("tls")

    # make the program just exit at ctrl+c without waiting for anything
    def sigint_handler(_sig, _frame):
        log.info('Exiting')
        os._exit(0)

    signal.signal(signal.SIGINT, sigint_handler)


def load_model():
    shared.state.begin()
    shared.state.job = 'load model'
    thread = Thread(target=lambda: shared.sd_model)
    thread.start()
    shared.opts.onchange("sd_model_checkpoint", wrap_queued_call(lambda: modules.sd_models.reload_model_weights()), call=False)
    shared.opts.onchange("sd_model_dict", wrap_queued_call(lambda: modules.sd_models.reload_model_weights()), call=False)
    shared.state.end()
    thread.join()
    startup_timer.record("checkpoint")


def create_api(app):
    log.debug('Creating API')
    from modules.api.api import Api
    api = Api(app, queue_lock)
    return api


def async_policy():
    _BasePolicy = asyncio.WindowsSelectorEventLoopPolicy if sys.platform == "win32" and hasattr(asyncio, "WindowsSelectorEventLoopPolicy") else asyncio.DefaultEventLoopPolicy

    class AnyThreadEventLoopPolicy(_BasePolicy):
        def handle_exception(self, context):
            msg = context.get("exception", context["message"])
            log.error(f"AsyncIO loop: {msg}")

        def get_event_loop(self) -> asyncio.AbstractEventLoop:
            try:
                self.loop = super().get_event_loop()
            except (RuntimeError, AssertionError):
                self.loop = self.new_event_loop()
                self.set_event_loop(self.loop)
            return self.loop

        def __init__(self):
            super().__init__()
            self.loop = self.get_event_loop()
            self.loop.set_exception_handler(self.handle_exception)
            log.debug(f"Event loop: {self.loop}")

    asyncio.set_event_loop_policy(AnyThreadEventLoopPolicy())


def start_common():
    log.debug('Entering start sequence')
    if cmd_opts.debug and hasattr(shared, 'get_version'):
        log.debug(f'Version: {shared.get_version()}')
    logging.disable(logging.NOTSET if cmd_opts.debug else logging.DEBUG)
    if shared.cmd_opts.data_dir is not None or len(shared.cmd_opts.data_dir) > 0:
        log.info(f'Using data path: {shared.cmd_opts.data_dir}')
    create_paths(opts)
    async_policy()
    initialize()
    if shared.opts.clean_temp_dir_at_start:
        ui_tempdir.cleanup_tmpdr()
        startup_timer.record("cleanup")


def start_ui():
    log.debug('Creating UI')
    modules.script_callbacks.before_ui_callback()
    startup_timer.record("before-ui")
    shared.demo = modules.ui.create_ui()
    startup_timer.record("ui")
    if cmd_opts.disable_queue:
        log.info('Server queues disabled')
        shared.demo.progress_tracking = False
    else:
        shared.demo.queue(concurrency_count=64)

    gradio_auth_creds = []
    if cmd_opts.auth:
        gradio_auth_creds += [x.strip() for x in cmd_opts.auth.strip('"').replace('\n', '').split(',') if x.strip()]
    if cmd_opts.auth_file:
        if not os.path.exists(cmd_opts.auth_file):
            log.error(f"Invalid path to auth file: '{cmd_opts.auth_file}'")
        else:
            with open(cmd_opts.auth_file, 'r', encoding="utf8") as file:
                for line in file.readlines():
                    gradio_auth_creds += [x.strip() for x in line.split(',') if x.strip()]

    global local_url # pylint: disable=global-statement
    app, local_url, share_url = shared.demo.launch( # app is FastAPI(Starlette) instance
        share=cmd_opts.share,
        server_name=server_name,
        server_port=cmd_opts.port if cmd_opts.port != 7860 else None,
        ssl_keyfile=cmd_opts.tls_keyfile,
        ssl_certfile=cmd_opts.tls_certfile,
        ssl_verify=not cmd_opts.tls_selfsign,
        debug=False,
        auth=[tuple(cred.split(':')) for cred in gradio_auth_creds] if gradio_auth_creds else None,
        prevent_thread_lock=True,
        max_threads=64,
        show_api=True,
        quiet=True,
        favicon_path='html/logo.ico',
        allowed_paths=[os.path.dirname(__file__), cmd_opts.data_dir],
        app_kwargs=fastapi_args,
    )
    if cmd_opts.data_dir is not None:
        ui_tempdir.register_tmp_file(shared.demo, os.path.join(cmd_opts.data_dir, 'x'))
    shared.log.info(f'Local URL: {local_url}')
    if cmd_opts.docs:
        shared.log.info(f'API Docs: {local_url[:-1]}/docs') # {local_url[:-1]}?view=api
    if share_url is not None:
        shared.log.info(f'Share URL: {share_url}')
    shared.log.debug(f'Gradio registered functions: {len(shared.demo.fns)}')
    shared.demo.server.wants_restart = False
    setup_middleware(app, cmd_opts)

    if cmd_opts.subpath:
        gradio.mount_gradio_app(app, shared.demo, path=f"/{cmd_opts.subpath}")
        shared.log.info(f'Redirector mounted: /{cmd_opts.subpath}')

    startup_timer.record("launch")

    modules.progress.setup_progress_api(app)
    create_api(app)
    ui_extra_networks.add_pages_to_demo(app)

    modules.script_callbacks.app_started_callback(shared.demo, app)
    startup_timer.record("app-started")

    time_setup = [f'{k}:{round(v,3)}s' for (k,v) in modules.scripts.time_setup.items() if v > 0.005]
    shared.log.debug(f'Scripts setup: {time_setup}')
    time_component = [f'{k}:{round(v,3)}s' for (k,v) in modules.scripts.time_component.items() if v > 0.005]
    shared.log.debug(f'Scripts components: {time_component}')


def webui(restart=False):
    if restart:
        modules.script_callbacks.app_reload_callback()
        modules.script_callbacks.script_unloaded_callback()

    start_common()
    start_ui()
    modules.sd_models.write_metadata()
    if opts.sd_checkpoint_autoload:
        load_model()
    else:
        log.debug('Model auto load disabled')

    log.info(f"Startup time: {startup_timer.summary()}")

    if not restart:
        # override all loggers to use the same handlers as the main logger
        for logger in [logging.getLogger(name) for name in logging.root.manager.loggerDict]: # pylint: disable=no-member
            if logger.name.startswith('uvicorn') or logger.name.startswith('sd'):
                continue
            logger.handlers = log.handlers
        # autolaunch only on initial start
        if cmd_opts.autolaunch and local_url is not None:
            cmd_opts.autolaunch = False
            shared.log.info('Launching browser')
            import webbrowser
            webbrowser.open(local_url, new=2, autoraise=True)
    else:
        for module in [module for name, module in sys.modules.items() if name.startswith("modules.ui")]:
            importlib.reload(module)

    return shared.demo.server


def api_only():
    start_common()
    app = FastAPI(**fastapi_args)
    setup_middleware(app, cmd_opts)
    api = create_api(app)
    api.wants_restart = False
    modules.script_callbacks.app_started_callback(None, app)
    modules.sd_models.write_metadata()
    log.info(f"Startup time: {startup_timer.summary()}")
    server = api.launch()
    return server


if __name__ == "__main__":
    if cmd_opts.api_only:
        api_only()
    else:
        webui()
