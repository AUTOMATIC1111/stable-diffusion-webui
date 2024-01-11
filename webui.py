import io
import os
import sys
import glob
import signal
import asyncio
import logging
import importlib
import contextlib
from threading import Thread
import modules.loader
import torch # pylint: disable=wrong-import-order
from modules import timer, errors, paths # pylint: disable=unused-import
from installer import log, git_commit, custom_excepthook
import ldm.modules.encoders.modules # pylint: disable=W0611,C0411,E0401
from modules import shared, extensions, ui_tempdir, modelloader # pylint: disable=ungrouped-imports
from modules import extra_networks, ui_extra_networks # pylint: disable=ungrouped-imports
from modules.paths import create_paths
from modules.call_queue import queue_lock, wrap_queued_call, wrap_gradio_gpu_call # pylint: disable=W0611,C0411,C0412
import modules.devices
import modules.sd_samplers
import modules.lowvram
import modules.scripts
import modules.sd_models
import modules.sd_vae
import modules.progress
import modules.ui
import modules.txt2img
import modules.img2img
import modules.upscaler
import modules.textual_inversion.textual_inversion
import modules.hypernetworks.hypernetwork
import modules.script_callbacks
from modules.middleware import setup_middleware
from modules.shared import cmd_opts, opts


sys.excepthook = custom_excepthook
local_url = None
state = shared.state
backend = shared.backend
if not modules.loader.initialized:
    timer.startup.record("libraries")
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

import modules.sd_hijack
timer.startup.record("ldm")

modules.loader.initialized = True


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
    log.debug('Initializing')
    check_rollback_vae()

    modules.sd_samplers.list_samplers()
    timer.startup.record("samplers")

    modules.sd_vae.refresh_vae_list()
    timer.startup.record("vae")

    extensions.list_extensions()
    timer.startup.record("extensions")

    modelloader.cleanup_models()
    modules.sd_models.setup_model()
    timer.startup.record("models")

    import modules.postprocess.codeformer_model as codeformer
    codeformer.setup_model(opts.codeformer_models_path)
    sys.modules["modules.codeformer_model"] = codeformer
    import modules.postprocess.gfpgan_model as gfpgan
    gfpgan.setup_model(opts.gfpgan_models_path)
    timer.startup.record("face-restore")

    log.debug('Load extensions')
    t_timer, t_total = modules.scripts.load_scripts()
    timer.startup.record("extensions")
    timer.startup.records["extensions"] = t_total # scripts can reset the time
    log.info(f'Extensions time: {t_timer.summary()}')

    modelloader.load_upscalers()
    timer.startup.record("upscalers")

    shared.opts.onchange("sd_vae", wrap_queued_call(lambda: modules.sd_vae.reload_vae_weights()), call=False)
    shared.opts.onchange("temp_dir", ui_tempdir.on_tmpdir_changed)
    timer.startup.record("onchange")

    modules.textual_inversion.textual_inversion.list_textual_inversion_templates()
    shared.reload_hypernetworks()
    shared.prompt_styles.reload()

    ui_extra_networks.initialize()
    ui_extra_networks.register_pages()
    extra_networks.initialize()
    extra_networks.register_default_extra_networks()
    timer.startup.record("extra-networks")

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
        timer.startup.record("tls")

    # make the program just exit at ctrl+c without waiting for anything
    def sigint_handler(_sig, _frame):
        log.info('Exiting')
        try:
            for f in glob.glob("*.lock"):
                os.remove(f)
        except Exception:
            pass
        sys.exit(0)

    signal.signal(signal.SIGINT, sigint_handler)


def load_model():
    if not opts.sd_checkpoint_autoload or (shared.cmd_opts.ckpt is not None and shared.cmd_opts.ckpt.lower() != 'none'):
        log.debug('Model auto load disabled')
    else:
        shared.state.begin('load')
        thread_model = Thread(target=lambda: shared.sd_model)
        thread_model.start()
        thread_refiner = Thread(target=lambda: shared.sd_refiner)
        thread_refiner.start()
        shared.state.end()
        thread_model.join()
        thread_refiner.join()
    shared.opts.onchange("sd_model_checkpoint", wrap_queued_call(lambda: modules.sd_models.reload_model_weights(op='model')), call=False)
    shared.opts.onchange("sd_model_refiner", wrap_queued_call(lambda: modules.sd_models.reload_model_weights(op='refiner')), call=False)
    shared.opts.onchange("sd_model_dict", wrap_queued_call(lambda: modules.sd_models.reload_model_weights(op='dict')), call=False)
    shared.opts.onchange("sd_vae", wrap_queued_call(lambda: modules.sd_vae.reload_vae_weights()), call=False)
    shared.opts.onchange("sd_backend", wrap_queued_call(lambda: modules.sd_models.change_backend()), call=False)
    timer.startup.record("checkpoint")


def create_api(app):
    log.debug('Creating API')
    from modules.api.api import Api
    api = Api(app, queue_lock)
    from modules.api.nvml import nvml_api
    nvml_api(api)
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
            # log.debug(f"Event loop: {self.loop}")

    asyncio.set_event_loop_policy(AnyThreadEventLoopPolicy())


def start_common():
    log.debug('Entering start sequence')
    if shared.cmd_opts.data_dir is not None and len(shared.cmd_opts.data_dir) > 0:
        log.info(f'Using data path: {shared.cmd_opts.data_dir}')
    if shared.cmd_opts.models_dir is not None and len(shared.cmd_opts.models_dir) > 0 and shared.cmd_opts.models_dir != 'models':
        log.info(f'Using models path: {shared.cmd_opts.models_dir}')
    create_paths(opts)
    async_policy()
    initialize()
    if shared.opts.clean_temp_dir_at_start:
        ui_tempdir.cleanup_tmpdr()
        timer.startup.record("cleanup")


def start_ui():
    log.debug('Creating UI')
    modules.script_callbacks.before_ui_callback()
    timer.startup.record("before-ui")
    shared.demo = modules.ui.create_ui(timer.startup)
    timer.startup.record("ui")
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
    if len(gradio_auth_creds) > 0:
        log.info(f'Authentication enabled: users={len(list(gradio_auth_creds))}')

    global local_url # pylint: disable=global-statement
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
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
            show_api=False,
            quiet=True,
            favicon_path='html/logo.ico',
            allowed_paths=[os.path.dirname(__file__), cmd_opts.data_dir],
            app_kwargs=fastapi_args,
            _frontend=True and cmd_opts.share,
        )
    if cmd_opts.data_dir is not None:
        ui_tempdir.register_tmp_file(shared.demo, os.path.join(cmd_opts.data_dir, 'x'))
    shared.log.info(f'Local URL: {local_url}')
    if cmd_opts.docs:
        shared.log.info(f'API Docs: {local_url[:-1]}/docs') # pylint: disable=unsubscriptable-object
    if share_url is not None:
        shared.log.info(f'Share URL: {share_url}')
    shared.log.debug(f'Gradio functions: registered={len(shared.demo.fns)}')
    shared.demo.server.wants_restart = False
    setup_middleware(app, cmd_opts)

    if cmd_opts.subpath:
        import gradio
        gradio.mount_gradio_app(app, shared.demo, path=f"/{cmd_opts.subpath}")
        shared.log.info(f'Redirector mounted: /{cmd_opts.subpath}')

    timer.startup.record("launch")

    modules.progress.setup_progress_api(app)
    create_api(app)
    timer.startup.record("api")

    ui_extra_networks.init_api(app)

    modules.script_callbacks.app_started_callback(shared.demo, app)
    timer.startup.record("app-started")

    time_setup = [f'{k}:{round(v,3)}' for (k,v) in modules.scripts.time_setup.items() if v > 0.005]
    shared.log.debug(f'Scripts setup: {time_setup}')
    time_component = [f'{k}:{round(v,3)}' for (k,v) in modules.scripts.time_component.items() if v > 0.005]
    if len(time_component) > 0:
        shared.log.debug(f'Scripts components: {time_component}')


def webui(restart=False):
    if restart:
        modules.script_callbacks.app_reload_callback()
        modules.script_callbacks.script_unloaded_callback()

    start_common()
    start_ui()
    modules.sd_models.write_metadata()
    load_model()
    shared.opts.save(shared.config_filename)
    if cmd_opts.profile:
        for k, v in modules.script_callbacks.callback_map.items():
            shared.log.debug(f'Registered callbacks: {k}={len(v)} {[c.script for c in v]}')
    log.info(f"Startup time: {timer.startup.summary()}")
    debug = log.trace if os.environ.get('SD_SCRIPT_DEBUG', None) is not None else lambda *args, **kwargs: None
    debug('Trace: SCRIPTS')
    for m in modules.scripts.scripts_data:
        debug(f'  {m}')
    debug('Loaded postprocessing scripts:')
    for m in modules.scripts.postprocessing_scripts_data:
        debug(f'  {m}')
    timer.startup.reset()
    modules.script_callbacks.print_timers()

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
    from fastapi import FastAPI
    app = FastAPI(**fastapi_args)
    setup_middleware(app, cmd_opts)
    api = create_api(app)
    api.wants_restart = False
    modules.script_callbacks.app_started_callback(None, app)
    modules.sd_models.write_metadata()
    log.info(f"Startup time: {timer.startup.summary()}")
    server = api.launch()
    return server


if __name__ == "__main__":
    if cmd_opts.api_only:
        api_only()
    else:
        webui()
