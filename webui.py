from __future__ import annotations

import os
import time
from typing import Optional, Tuple

from fastapi import FastAPI
from gradio.routes import App

from modules import (
    timer, initialize_util, initialize, shared, ui_tempdir, 
    script_callbacks, ui, progress, ui_extra_networks
)
from modules.shared_cmd_options import cmd_opts
from modules.api.api import Api
from modules.call_queue import queue_lock

startup_timer = timer.startup_timer
startup_timer.record("launcher")

initialize.imports()
initialize.check_versions()


def create_api(app: FastAPI) -> Api:
    """
    Create and return an API instance.
    
    Args:
        app (FastAPI): The FastAPI application instance.
    
    Returns:
        Api: The created API instance.
    """
    api = Api(app, queue_lock)
    return api


def api_only() -> None:
    """Launch the API-only version of the application."""
    initialize.initialize()

    app = FastAPI()
    initialize_util.setup_middleware(app)
    api = create_api(app)

    script_callbacks.before_ui_callback()
    script_callbacks.app_started_callback(None, app)

    print(f"Startup time: {startup_timer.summary()}.")
    api.launch(
        server_name=initialize_util.gradio_server_name(),
        port=cmd_opts.port or 7861,
        root_path=f"/{cmd_opts.subpath}" if cmd_opts.subpath else ""
    )


def launch_gradio(
    share: bool, 
    server_name: str, 
    server_port: int, 
    auth_creds: Optional[list],
    auto_launch_browser: bool
) -> Tuple[FastAPI, str, Optional[str]]:
    """
    Launch the Gradio interface.
    
    Args:
        share (bool): Whether to create a public link.
        server_name (str): Server name to use.
        server_port (int): Port to run the server on.
        auth_creds (Optional[list]): Authentication credentials.
        auto_launch_browser (bool): Whether to automatically launch the browser.
    
    Returns:
        Tuple[FastAPI, str, Optional[str]]: The FastAPI app, local URL, and share URL (if applicable).
    """
    return shared.demo.launch(
        share=share,
        server_name=server_name,
        server_port=server_port,
        ssl_keyfile=cmd_opts.tls_keyfile,
        ssl_certfile=cmd_opts.tls_certfile,
        ssl_verify=not cmd_opts.disable_tls_verify,
        debug=cmd_opts.gradio_debug,
        auth=auth_creds,
        inbrowser=auto_launch_browser,
        prevent_thread_lock=True,
        allowed_paths=cmd_opts.gradio_allowed_path,
        app_kwargs={
            "docs_url": "/docs",
            "redoc_url": "/redoc",
        },
        root_path=f"/{cmd_opts.subpath}" if cmd_opts.subpath else "",
    )


def setup_app(app: FastAPI, launch_api: bool) -> None:
    """
    Set up the FastAPI application with necessary middleware and APIs.
    
    Args:
        app (FastAPI): The FastAPI application instance.
        launch_api (bool): Whether to launch the API.
    """
    app.user_middleware = [x for x in app.user_middleware if x.cls.__name__ != 'CORSMiddleware']
    initialize_util.setup_middleware(app)
    progress.setup_progress_api(app)
    ui.setup_ui_api(app)
    
    if launch_api:
        create_api(app)
    
    ui_extra_networks.add_pages_to_demo(app)


def webui() -> None:
    """Launch the full web UI version of the application."""
    launch_api = cmd_opts.api
    initialize.initialize()

    while True:
        if shared.opts.clean_temp_dir_at_start:
            ui_tempdir.cleanup_tmpdr()
            startup_timer.record("cleanup temp dir")

        script_callbacks.before_ui_callback()
        startup_timer.record("scripts before_ui_callback")

        shared.demo = ui.create_ui()
        startup_timer.record("create ui")

        if not cmd_opts.no_gradio_queue:
            shared.demo.queue(64)

        gradio_auth_creds = list(initialize_util.get_gradio_auth_creds()) or None

        auto_launch_browser = False
        if os.getenv('SD_WEBUI_RESTARTING') != '1':
            if shared.opts.auto_launch_browser == "Remote" or cmd_opts.autolaunch:
                auto_launch_browser = True
            elif shared.opts.auto_launch_browser == "Local":
                auto_launch_browser = not cmd_opts.webui_is_non_local

        app, local_url, share_url = launch_gradio(
            cmd_opts.share,
            initialize_util.gradio_server_name(),
            cmd_opts.port,
            gradio_auth_creds,
            auto_launch_browser
        )

        startup_timer.record("gradio launch")

        setup_app(app, launch_api)
        startup_timer.record("add APIs")

        with startup_timer.subcategory("app_started_callback"):
            script_callbacks.app_started_callback(shared.demo, app)

        timer.startup_record = startup_timer.dump()
        print(f"Startup time: {startup_timer.summary()}.")

        try:
            server_command = run_server_loop()
            if server_command == "stop":
                break
        except KeyboardInterrupt:
            print('Caught KeyboardInterrupt, stopping...')
            break

        print('Restarting UI...')
        restart_ui()

    print("Stopping server...")
    shared.demo.close()


def run_server_loop() -> Optional[str]:
    """
    Run the main server loop.
    
    Returns:
        Optional[str]: The server command if received, None otherwise.
    """
    while True:
        server_command = shared.state.wait_for_server_command(timeout=5)
        if server_command:
            if server_command in ("stop", "restart"):
                return server_command
            else:
                print(f"Unknown server command: {server_command}")


def restart_ui() -> None:
    """Restart the UI by resetting necessary components."""
    shared.demo.close()
    time.sleep(0.5)
    startup_timer.reset()
    script_callbacks.app_reload_callback()
    startup_timer.record("app reload callback")
    script_callbacks.script_unloaded_callback()
    startup_timer.record("scripts unloaded callback")
    initialize.initialize_rest(reload_script_modules=True)
    
    # Disable auto launch webui in browser for subsequent UI Reload
    os.environ.setdefault('SD_WEBUI_RESTARTING', '1')


if __name__ == "__main__":
    api_only() if cmd_opts.nowebui else webui()
