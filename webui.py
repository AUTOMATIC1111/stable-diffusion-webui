import os
import time
from pathlib import Path
from packaging.version import parse
import gradio

from modules import (
    shared,
    ui_tempdir,
    startup_timer,
    initialize_util,
    progress,
    ui,
    ui_extra_networks,
    timer,
    initialize,
    script_callbacks,
    cmd_opts
)

from api import create_api


def limpar_temp_dir():
    if shared.opts.clean_temp_dir_at_start:
        ui_tempdir.cleanup_tmpdr()
        startup_timer.record("cleanup temp dir")


def configurar_ui():
    shared.demo = ui.create_ui()
    startup_timer.record("create ui")
    if not cmd_opts.no_gradio_queue:
        shared.demo.queue(64)


def decidir_autolancamento():
    if os.getenv('SD_WEBUI_RESTARTING') == '1':
        return False
    if shared.opts.auto_launch_browser == "Remote" or cmd_opts.autolaunch:
        return True
    if shared.opts.auto_launch_browser == "Local":
        return not cmd_opts.webui_is_non_local
    return False


def lancar_interface(auto_launch_browser):
    gradio_auth_creds = list(initialize_util.get_gradio_auth_creds()) or None
    return shared.demo.launch(
        share=cmd_opts.share,
        server_name=initialize_util.gradio_server_name(),
        server_port=cmd_opts.port,
        ssl_keyfile=cmd_opts.tls_keyfile,
        ssl_certfile=cmd_opts.tls_certfile,
        ssl_verify=cmd_opts.disable_tls_verify,
        debug=cmd_opts.gradio_debug,
        auth=gradio_auth_creds,
        inbrowser=auto_launch_browser,
        prevent_thread_lock=True,
        allowed_paths=cmd_opts.gradio_allowed_path,
        app_kwargs={"docs_url": "/docs", "redoc_url": "/redoc"},
        root_path=f"/{cmd_opts.subpath}" if cmd_opts.subpath else "",
    )


def proteger_app(app):
    app.user_middleware = [
        x for x in app.user_middleware if x.cls.__name__ != 'CORSMiddleware'
    ]
    initialize_util.setup_middleware(app)


def configurar_apis(app):
    progress.setup_progress_api(app)
    ui.setup_ui_api(app)
    if cmd_opts.api:
        create_api(app)
    ui_extra_networks.add_pages_to_demo(app)


def monitorar_comandos():
    while True:
        server_command = shared.state.wait_for_server_command(timeout=5)
        if server_command:
            if server_command in ("stop", "restart"):
                return server_command
            else:
                print(f"Unknown server command: {server_command}")


def warning_if_invalid_install_dir():
    if parse('3.32.0') <= parse(gradio.__version__) < parse('4'):
        def abspath(path):
            if path.is_absolute():
                return path
            is_symlink = path.is_symlink() or any(parent.is_symlink() for parent in path.parents)
            return Path.cwd() / path if (is_symlink or path == path.resolve()) else path.resolve()

        webui_root = Path(__file__).parent
        if any(part.startswith(".") for part in abspath(webui_root).parts):
            print(f'''{"!"*25} Warning {"!"*25}
WebUI is installed in a directory that has a leading dot (.) in one of its parent directories.
This will prevent WebUI from functioning properly.
Current path: "{webui_root}"
Please move the installation to a different directory.
More info: https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/13292
{"!"*25} Warning {"!"*25}''')


def webui():
    initialize.initialize()
    warning_if_invalid_install_dir()

    while True:
        limpar_temp_dir()
        script_callbacks.before_ui_callback()
        startup_timer.record("scripts before_ui_callback")

        configurar_ui()
        auto_launch_browser = decidir_autolancamento()

        app, _, _ = lancar_interface(auto_launch_browser)
        startup_timer.record("gradio launch")

        proteger_app(app)
        configurar_apis(app)

        startup_timer.record("add APIs")

        with startup_timer.subcategory("app_started_callback"):
            script_callbacks.app_started_callback(shared.demo, app)

        timer.startup_record = startup_timer.dump()
        print(f"Startup time: {startup_timer.summary()}.")

        try:
            server_command = monitorar_comandos()
        except KeyboardInterrupt:
            print('Caught KeyboardInterrupt, stopping...')
            server_command = "stop"

        if server_command == "stop":
            print("Stopping server...")
            shared.demo.close()
            break

        os.environ.setdefault('SD_WEBUI_RESTARTING', '1')
        print('Restarting UI...')
        shared.demo.close()
        time.sleep(0.5)
        startup_timer.reset()
        script_callbacks.app_reload_callback()
        startup_timer.record("app reload callback")
        script_callbacks.script_unloaded_callback()
        startup_timer.record("scripts unloaded callback")
        initialize.initialize_rest(reload_script_modules=True)


if __name__ == "__main__":
    webui()
