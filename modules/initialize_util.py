import json
import os
import signal
import sys
import re

from modules.timer import startup_timer


def gradio_server_name():
    from modules.shared_cmd_options import cmd_opts

    if cmd_opts.server_name:
        return cmd_opts.server_name
    else:
        return "0.0.0.0" if cmd_opts.listen else None


def fix_torch_version():
    import torch

    # Truncate version number of nightly/local build of PyTorch to not cause exceptions with CodeFormer or Safetensors
    if ".dev" in torch.__version__ or "+git" in torch.__version__:
        torch.__long_version__ = torch.__version__
        torch.__version__ = re.search(r'[\d.]+[\d]', torch.__version__).group(0)


def fix_asyncio_event_loop_policy():
    """
        The default `asyncio` event loop policy only automatically creates
        event loops in the main threads. Other threads must create event
        loops explicitly or `asyncio.get_event_loop` (and therefore
        `.IOLoop.current`) will fail. Installing this policy allows event
        loops to be created automatically on any thread, matching the
        behavior of Tornado versions prior to 5.0 (or 5.0 on Python 2).
    """

    import asyncio

    if sys.platform == "win32" and hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
        # "Any thread" and "selector" should be orthogonal, but there's not a clean
        # interface for composing policies so pick the right base.
        _BasePolicy = asyncio.WindowsSelectorEventLoopPolicy  # type: ignore
    else:
        _BasePolicy = asyncio.DefaultEventLoopPolicy

    class AnyThreadEventLoopPolicy(_BasePolicy):  # type: ignore
        """Event loop policy that allows loop creation on any thread.
        Usage::

            asyncio.set_event_loop_policy(AnyThreadEventLoopPolicy())
        """

        def get_event_loop(self) -> asyncio.AbstractEventLoop:
            try:
                return super().get_event_loop()
            except (RuntimeError, AssertionError):
                # This was an AssertionError in python 3.4.2 (which ships with debian jessie)
                # and changed to a RuntimeError in 3.4.3.
                # "There is no current event loop in thread %r"
                loop = self.new_event_loop()
                self.set_event_loop(loop)
                return loop

    asyncio.set_event_loop_policy(AnyThreadEventLoopPolicy())


def restore_config_state_file():
    from modules import shared, config_states

    config_state_file = shared.opts.restore_config_state_file
    if config_state_file == "":
        return

    shared.opts.restore_config_state_file = ""
    shared.opts.save(shared.config_filename)

    if os.path.isfile(config_state_file):
        print(f"*** About to restore extension state from file: {config_state_file}")
        with open(config_state_file, "r", encoding="utf-8") as f:
            config_state = json.load(f)
            config_states.restore_extension_config(config_state)
        startup_timer.record("restore extension config")
    elif config_state_file:
        print(f"!!! Config state backup not found: {config_state_file}")


def validate_tls_options():
    from modules.shared_cmd_options import cmd_opts

    if not (cmd_opts.tls_keyfile and cmd_opts.tls_certfile):
        return

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


def get_gradio_auth_creds():
    """
    Convert the gradio_auth and gradio_auth_path commandline arguments into
    an iterable of (username, password) tuples.
    """
    from modules.shared_cmd_options import cmd_opts

    def process_credential_line(s):
        s = s.strip()
        if not s:
            return None
        return tuple(s.split(':', 1))

    if cmd_opts.gradio_auth:
        for cred in cmd_opts.gradio_auth.split(','):
            cred = process_credential_line(cred)
            if cred:
                yield cred

    if cmd_opts.gradio_auth_path:
        with open(cmd_opts.gradio_auth_path, 'r', encoding="utf8") as file:
            for line in file.readlines():
                for cred in line.strip().split(','):
                    cred = process_credential_line(cred)
                    if cred:
                        yield cred


def dumpstacks():
    import threading
    import traceback

    id2name = {th.ident: th.name for th in threading.enumerate()}
    code = []
    for threadId, stack in sys._current_frames().items():
        code.append(f"\n# Thread: {id2name.get(threadId, '')}({threadId})")
        for filename, lineno, name, line in traceback.extract_stack(stack):
            code.append(f"""File: "{filename}", line {lineno}, in {name}""")
            if line:
                code.append("  " + line.strip())

    print("\n".join(code))


def configure_sigint_handler():
    # make the program just exit at ctrl+c without waiting for anything
    def sigint_handler(sig, frame):
        print(f'Interrupted with signal {sig} in {frame}')

        dumpstacks()

        os._exit(0)

    if not os.environ.get("COVERAGE_RUN"):
        # Don't install the immediate-quit handler when running under coverage,
        # as then the coverage report won't be generated.
        signal.signal(signal.SIGINT, sigint_handler)


def configure_opts_onchange():
    from modules import shared, sd_models, sd_vae, ui_tempdir, sd_hijack
    from modules.call_queue import wrap_queued_call

    shared.opts.onchange("sd_model_checkpoint", wrap_queued_call(lambda: sd_models.reload_model_weights()), call=False)
    shared.opts.onchange("sd_vae", wrap_queued_call(lambda: sd_vae.reload_vae_weights()), call=False)
    shared.opts.onchange("sd_vae_overrides_per_model_preferences", wrap_queued_call(lambda: sd_vae.reload_vae_weights()), call=False)
    shared.opts.onchange("temp_dir", ui_tempdir.on_tmpdir_changed)
    shared.opts.onchange("gradio_theme", shared.reload_gradio_theme)
    shared.opts.onchange("cross_attention_optimization", wrap_queued_call(lambda: sd_hijack.model_hijack.redo_hijack(shared.sd_model)), call=False)
    startup_timer.record("opts onchange")


def setup_middleware(app):
    from starlette.middleware.gzip import GZipMiddleware

    app.middleware_stack = None  # reset current middleware to allow modifying user provided list
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    configure_cors_middleware(app)
    app.build_middleware_stack()  # rebuild middleware stack on-the-fly


def configure_cors_middleware(app):
    from starlette.middleware.cors import CORSMiddleware
    from modules.shared_cmd_options import cmd_opts

    cors_options = {
        "allow_methods": ["*"],
        "allow_headers": ["*"],
        "allow_credentials": True,
    }
    if cmd_opts.cors_allow_origins:
        cors_options["allow_origins"] = cmd_opts.cors_allow_origins.split(',')
    if cmd_opts.cors_allow_origins_regex:
        cors_options["allow_origin_regex"] = cmd_opts.cors_allow_origins_regex
    app.add_middleware(CORSMiddleware, **cors_options)

