from __future__ import annotations

import os
import time
import sys
import modules
import subprocess
from modules import timer
from modules import initialize_util
from modules import initialize
from modules import shared
from modules.shared import cmd_opts
from loguru import logger

# 添加sd_scripts 主路径
sys.path.append("sd_scripts")

startup_timer = timer.startup_timer
startup_timer.record("launcher")

initialize.imports()

initialize.check_versions()


def create_api(app):
    from modules.api.api import Api
    from modules.call_queue import queue_lock

    api = Api(app, queue_lock)
    return api


def api_only():
    from fastapi import FastAPI
    from modules.shared_cmd_options import cmd_opts

    initialize.initialize()

    app = FastAPI()
    initialize_util.setup_middleware(app)
    api = create_api(app)

    from modules import script_callbacks
    script_callbacks.before_ui_callback()
    script_callbacks.app_started_callback(None, app)

    print(f"Startup time: {startup_timer.summary()}.")
    api.launch(
        server_name="0.0.0.0" if cmd_opts.listen else "127.0.0.1",
        port=cmd_opts.port if cmd_opts.port else 7861,
        root_path=f"/{cmd_opts.subpath}" if cmd_opts.subpath else ""
    )


def webui():
    from modules.shared_cmd_options import cmd_opts

    launch_api = cmd_opts.api
    initialize.initialize()

    from modules import shared, ui_tempdir, script_callbacks, ui, progress, ui_extra_networks

    from modules import crontab_clear_tmp

    # 定时清除gradio临时文件
    timer = crontab_clear_tmp.clear_gradio_tmp()

    while 1:
        if shared.opts.clean_temp_dir_at_start:
            ui_tempdir.cleanup_tmpdr()
            startup_timer.record("cleanup temp dir")

        script_callbacks.before_ui_callback()
        startup_timer.record("scripts before_ui_callback")

        shared.demo = ui.create_ui()
        startup_timer.record("create ui")

        shared.demo.queue(32)

        gradio_auth_creds = list(initialize_util.get_gradio_auth_creds()) or None

        auto_launch_browser = False
        if os.getenv('SD_WEBUI_RESTARTING') != '1':
            if shared.opts.auto_launch_browser == "Remote" or cmd_opts.autolaunch:
                auto_launch_browser = True
            elif shared.opts.auto_launch_browser == "Local":
                auto_launch_browser = not any([cmd_opts.listen, cmd_opts.share, cmd_opts.ngrok, cmd_opts.server_name])

        # 安装AUTH 脚本
        from modules.user import authorization
        auth = None
        if not cmd_opts.noauth:
            auth = authorization

        app, local_url, share_url = shared.demo.launch(
            share=cmd_opts.share,
            server_name=initialize_util.gradio_server_name(),
            server_port=cmd_opts.port,
            ssl_keyfile=cmd_opts.tls_keyfile,
            ssl_certfile=cmd_opts.tls_certfile,
            ssl_verify=cmd_opts.disable_tls_verify,
            debug=cmd_opts.gradio_debug,
            auth=auth,
            auth_message="美术SD-WEBUI",
            inbrowser=auto_launch_browser,
            prevent_thread_lock=True,
            allowed_paths=cmd_opts.gradio_allowed_path,
            app_kwargs={
                "docs_url": "/docs",
                "redoc_url": "/redoc",
            },
            root_path=f"/{cmd_opts.subpath}" if cmd_opts.subpath else "",
        )

        startup_timer.record("gradio launch")

        # gradio uses a very open CORS policy via app.user_middleware, which makes it possible for
        # an attacker to trick the user into opening a malicious HTML page, which makes a request to the
        # running web ui and do whatever the attacker wants, including installing an extension and
        # running its code. We disable this here. Suggested by RyotaK.
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
            # If we catch a keyboard interrupt, we want to stop the server and exit.
            shared.demo.close()
            break

        # disable auto launch webui in browser for subsequent UI Reload
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

    timer.shutdown()


#### XZ: worker启动相关

def check_resource():
    from scripts.pull_repo_res import pull_res
    pull_res()


def set_pip_index():
    def run(command, desc=None, errdesc=None, custom_env=None, live: bool = False) -> str:
        if desc is not None:
            print(desc)

        run_kwargs = {
            "args": command,
            "shell": True,
            "env": os.environ if custom_env is None else custom_env,
            "encoding": 'utf8',
            "errors": 'ignore',
        }

        if not live:
            run_kwargs["stdout"] = run_kwargs["stderr"] = subprocess.PIPE

        result = subprocess.run(**run_kwargs)

        if result.returncode != 0:
            error_bits = [
                f"{errdesc or 'Error running command'}.",
                f"Command: {command}",
                f"Error code: {result.returncode}",
            ]
            if result.stdout:
                error_bits.append(f"stdout: {result.stdout}")
            if result.stderr:
                error_bits.append(f"stderr: {result.stderr}")
            raise RuntimeError("\n".join(error_bits))

        return (result.stdout or "")

    python = sys.executable
    command = f'''
    {python} -m pip config --site set global.index-url https://mirrors.aliyun.com/pypi/simple/
    '''
    run(command)


def run_worker():
    from consumer import run_executor
    from worker.dumper import dumper

    if cmd_opts.debug_task:
        from worker.task_send import RedisSender, VipLevel

        if cmd_opts.train_only:
            from trainx.typex import PreprocessTask, TrainLoraTask, DigitalDoppelgangerTask
            tasks = [
                # PreprocessTask.debug_task(),
                # TrainLoraTask.debug_task()
                DigitalDoppelgangerTask.debug_task()
            ]
        else:
            from handlers.img2img import Img2ImgTask
            from handlers.extension.controlnet import bind_debug_img_task_args

            tasks = Img2ImgTask.debug_task()
            tasks = bind_debug_img_task_args(*tasks)

        sender = RedisSender()
        sender.push_task(VipLevel.Level_1, *tasks)
    if not cmd_opts.train_only:
        logger.info('initialize...')
        initialize.initialize()
        modules.script_callbacks.before_ui_callback()
        startup_timer.record("scripts before_ui_callback")

        shared.demo = modules.ui.create_ui()

    if cmd_opts.send_task_only:
        dumper.stop()
        return

    logger.debug("[WORKER] worker模式下需要保证config.json中配置controlnet unit限定5个")
    exec = run_executor(shared.sd_model_recorder, train_only=cmd_opts.train_only)
    exec.stop()
    dumper.stop()

 # XZ end


def run_sd_webui():
    import sys
    from tools.mysql import dispose

    print(sys.argv)
    set_pip_index()
    check_resource()

    # if not cmd_opts.skip_install:
    #     from modules.launch_utils import run_extensions_installers
    #     run_extensions_installers(os.path.join(data_path, 'config.json'))

    if cmd_opts.worker:
        run_worker()
    else:
        if cmd_opts.nowebui:
            api_only()
        else:
            webui()

    dispose()


if __name__ == "__main__":
    run_sd_webui()

