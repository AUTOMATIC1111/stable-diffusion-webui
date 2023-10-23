# Optimized and Refined Code
from __future__ import annotations

import os
import time
from fastapi import FastAPI
from modules.api.api import Api
from modules.call_queue import queue_lock
from modules import timer, initialize_util, initialize, script_callbacks, shared, ui_tempdir, ui, progress, ui_extra_networks

startup_timer = timer.startup_timer
startup_timer.record("launcher")

initialize.imports()
initialize.check_versions()


def create_api(app):
    api = Api(app, queue_lock)
    return api


def api_only():
    initialize.initialize()
    app = FastAPI()
    initialize_util.setup_middleware(app)
    api = create_api(app)
    script_callbacks.before_ui_callback()
    script_callbacks.app_started_callback(None, app)
    print(f"Startup time: {startup_timer.summary()}.")
    api.launch(server_name="0.0.0.0" if shared.opts.listen else "127.0.0.1",
               port=shared.opts.port if shared.opts.port else 7861,
               root_path=f"/{shared.opts.subpath}" if shared.opts.subpath else "")


def webui():
    launch_api = shared.opts.api
    initialize.initialize()
    while True:
        if shared.opts.clean_temp_dir_at_start:
            ui_tempdir.cleanup_tmpdir()
            startup_timer.record("cleanup temp dir")
        script_callbacks.before_ui_callback()
        startup_timer.record("scripts before_ui_callback")
        shared.demo = ui.create_ui()
        startup_timer.record("create ui")
        if not shared.opts.no_gradio_queue:
            shared.demo.queue(64)
        gradio_auth_creds = list(initialize_util.get_gradio_auth_creds()) or None
        auto_launch_browser = False
        if os.getenv('SD_WEBUI_RESTARTING') != '1':
            if shared.opts.auto_launch_browser == "Remote" or shared.opts.autolaunch:
                auto_launch_browser = True
            elif shared.opts.auto_launch_browser == "Local":
                auto_launch_browser = not any([shared.opts.listen, shared.opts.share, shared.opts.ngrok, shared.opts.server_name])
        app, local_url, share_url = shared.demo.launch(share=shared.opts.share,
                                                       server_name=initialize_util.gradio_server_name(),
                                                       server_port=shared.opts.port,
                                                       ssl_keyfile=shared.opts.tls_keyfile,
                                                       ssl_certfile=shared.opts.tls_certfile,
                                                       ssl_verify=shared.opts.disable_tls_verify,
                                                       debug=shared.opts.gradio_debug,
                                                       auth=gradio_auth_creds,
                                                       inbrowser=auto_launch_browser,
                                                       prevent_thread_lock=True,
                                                       allowed_paths=shared.opts.gradio_allowed_path,
                                                       app_kwargs={
                                                           "docs_url": "/docs",
                                                           "redoc_url": "/redoc",
                                                       },
                                                       root_path=f"/{shared.opts.subpath}" if shared.opts.subpath else "")
        startup_timer.record("gradio launch")
        app.user_middleware = [x for x in app.user_middleware if x.cls.__name__ != 'CORSMiddleware']
        initialize_util.setup_middleware(app)
        progress.setup_progress_api(app)
        ui.setup_ui_api(app)
        if launch_api:
            create_api(app)
        ui_extra_networks.add_pages_to_demo(app)
        startup_timer.record("add APIs")
        with startup_timer.subcategory("app_started_callback"):
            script_callbacks.app_started_callback(shared.demo, app)
        timer.startup_record = startup_timer.dump()
        print(f"Startup time: {startup_timer.summary()}.")
        try:
            while True:
                server_command = shared.state.wait_for_server_command(timeout=5)
                if server_command:
                    if server_command in ("stop", "restart"):
                        break
                    else:
                        print(f"Unknown server command: {server_command}")
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
    if shared.opts.nowebui:
        api_only()
    else:
        webui()
