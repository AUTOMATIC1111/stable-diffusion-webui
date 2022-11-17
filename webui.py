from fastapi.middleware.cors import CORSMiddleware
import os
import threading
import time
import importlib
import signal
import threading
from fastapi import FastAPI
from fastapi.middleware.gzip import GZipMiddleware

from modules.paths import script_path

from modules import devices, sd_samplers
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
from modules.api.api import Api

queue_lock = threading.Lock()


def initialize():
    gfpgan.setup_model(cmd_opts.gfpgan_models_path)
    shared.face_restorers.append(modules.face_restoration.FaceRestoration())
    modelloader.load_upscalers()

    modules.scripts.load_scripts()

    def sigint_handler(sig, frame):
        print(f'Interrupted with signal {sig} in {frame}')
        os._exit(0)

    signal.signal(signal.SIGINT, sigint_handler)


def create_api(app):
    from modules.api.api import Api
    api = Api(app, queue_lock)
    return api


app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"])
app.add_middleware(GZipMiddleware, minimum_size=1000)
api = create_api(app)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    api.launch(server_name="0.0.0.0", port=port)
