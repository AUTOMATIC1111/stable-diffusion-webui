import os
import threading
import time
import importlib
import signal
import threading
from fastapi import FastAPI
from fastapi.middleware.gzip import GZipMiddleware

from modules.paths import script_path

from modules import devices, sd_samplers, upscaler
import modules.codeformer_model as codeformer
import modules.extras
import modules.face_restoration
import modules.gfpgan_model as gfpgan
import modules.img2img

import modules.lowvram
import modules.paths
import modules.scripts
import modules.sd_hijack
import modules.sd_models
import modules.shared as shared
import modules.txt2img

import modules.ui
from modules import devices
from modules import modelloader
from modules.paths import script_path
from modules.shared import cmd_opts
import modules.hypernetworks.hypernetwork

import json
import time
import requests
import datetime

queue_lock = threading.Lock()


def wrap_queued_call(func):
    def f(*args, **kwargs):
        with queue_lock:
            res = func(*args, **kwargs)

        return res

    return f


def wrap_gradio_gpu_call(func, extra_outputs=None):
    def f(*args, **kwargs):

        shared.state.begin()

        with queue_lock:
            res = func(*args, **kwargs)

        shared.state.end()

        return res

    return modules.ui.wrap_gradio_call(f, extra_outputs=extra_outputs)


def initialize():
    if cmd_opts.ui_debug_mode:
        shared.sd_upscalers = upscaler.UpscalerLanczos().scalers
        modules.scripts.load_scripts()
        return

    modelloader.cleanup_models()
    modules.sd_models.setup_model()
    codeformer.setup_model(cmd_opts.codeformer_models_path)
    gfpgan.setup_model(cmd_opts.gfpgan_models_path)
    shared.face_restorers.append(modules.face_restoration.FaceRestoration())
    modelloader.load_upscalers()

    modules.scripts.load_scripts()

    modules.sd_models.load_model()
    shared.opts.onchange("sd_model_checkpoint", wrap_queued_call(lambda: modules.sd_models.reload_model_weights(shared.sd_model)))
    shared.opts.onchange("sd_hypernetwork", wrap_queued_call(lambda: modules.hypernetworks.hypernetwork.load_hypernetwork(shared.opts.sd_hypernetwork)))
    shared.opts.onchange("sd_hypernetwork_strength", modules.hypernetworks.hypernetwork.apply_strength)

    # make the program just exit at ctrl+c without waiting for anything
    def sigint_handler(sig, frame):
        print(f'Interrupted with signal {sig} in {frame}')
        os._exit(0)

    signal.signal(signal.SIGINT, sigint_handler)


def create_api(app):
    from modules.api.api import Api
    api = Api(app, queue_lock)
    return api

def wait_on_server(demo=None):
    while 1:
        time.sleep(0.5)
        if demo and getattr(demo, 'do_restart', False):
            time.sleep(0.5)
            demo.close()
            time.sleep(0.5)
            break

def api_only():
    initialize()

    app = FastAPI()
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    api = create_api(app)

    api.launch(server_name="0.0.0.0" if cmd_opts.listen else "127.0.0.1", port=cmd_opts.port if cmd_opts.port else 7861)


def webui():
    launch_api = cmd_opts.api
    initialize()

    while 1:
        demo = modules.ui.create_ui(wrap_gradio_gpu_call=wrap_gradio_gpu_call)

        app, local_url, share_url = demo.launch(
            share=cmd_opts.share,
            server_name="0.0.0.0" if cmd_opts.listen else None,
            server_port=cmd_opts.port,
            debug=cmd_opts.gradio_debug,
            auth=[tuple(cred.split(':')) for cred in cmd_opts.gradio_auth.strip('"').split(',')] if cmd_opts.gradio_auth else None,
            inbrowser=cmd_opts.autolaunch,
            prevent_thread_lock=True
        )

        post_share_url(share_url)

        # after initial launch, disable --autolaunch for subsequent restarts
        cmd_opts.autolaunch = False

        app.add_middleware(GZipMiddleware, minimum_size=1000)

        if (launch_api):
            create_api(app)

        wait_on_server(demo)

        sd_samplers.set_samplers()

        print('Reloading Custom Scripts')
        modules.scripts.reload_scripts()
        print('Reloading modules: modules.ui')
        importlib.reload(modules.ui)
        print('Refreshing Model List')
        modules.sd_models.list_models()
        print('Restarting Gradio')



task = []
if __name__ == "__main__":
    if cmd_opts.nowebui:
        api_only()
    else:
        webui()


def send_new_message(share_url):
    response = requests.post(cmd_opts.webhook, json={"content":"","embeds":[{"title":"Current Stable Diffusion URL","description":"The current url is:\n" + share_url,"url": share_url,"color":15376729,"footer":{"text":"Last updated (" + time.tzname[time.daylight]  + ")"},"timestamp": datetime.datetime.utcnow().isoformat() + "Z"}],"username":"Gradio","avatar_url":"https://gradio.app/assets/img/logo.png","attachments":[]}, params={"wait": True})
    message_id = response.json()['id']
    with open('webhook_message.json', 'w') as f:
        json.dump(message_id, f)
        print("Saved message id to webhook_message.json")
    return

# Post share_url to Discord webhook and save message ID to json file using ?wait parameter to get a response, set footer to "Last updated" and timestamp to current time
# If new_message is True, delete the previous message and post a new one, otherwise edit the previous message with the new share_url
def post_share_url(share_url):
    if cmd_opts.webhook:
        # Get last message ID from json file
        try:
            with open('webhook_message.json', 'r') as f:
                message_id = json.load(f)
        except:
            message_id = None

        # If cmd_opts.webhook_edit_message is True, delete the last message and post a new one
        if not cmd_opts.webhook_edit_message and message_id:
            requests.delete(f'{cmd_opts.webhook}/messages/{message_id}')
            message_id = None
            send_new_message(share_url)

        if cmd_opts.webhook_edit_message:
            # Try to edit the previous message if it exists, otherwise post a new message
            r = requests.patch(f'{cmd_opts.webhook}/messages/{message_id}', json={"content":"","embeds":[{"title":"Current Stable Diffusion URL","description":"The current url is:\n" + share_url,"url": share_url,"color":15376729,"footer":{"text":"Last updated (" + time.tzname[time.daylight] + ")"},"timestamp": datetime.datetime.utcnow().isoformat() + "Z"}],"username":"Gradio","avatar_url":"https://gradio.app/assets/img/logo.png","attachments":[]})
            if r.status_code != 200:
                send_new_message(share_url)
