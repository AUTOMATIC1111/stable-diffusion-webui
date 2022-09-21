import os
import sys
import traceback
from collections import namedtuple

from basicsr.utils.download_util import load_file_from_url

import modules.images
from modules import shared
from modules.paths import script_path

LDSRModelInfo = namedtuple("LDSRModelInfo", ["name", "location", "model", "netscale"])

ldsr_models = []
have_ldsr = False
LDSR_obj = None


class UpscalerLDSR(modules.images.Upscaler):
    def __init__(self, steps):
        self.steps = steps
        self.name = "LDSR"

    def do_upscale(self, img):
        return upscale_with_ldsr(img)


def add_lsdr():
    modules.shared.sd_upscalers.append(UpscalerLDSR(100))


def setup_ldsr():
    path = modules.paths.paths.get("LDSR", None)
    if path is None:
        return
    global have_ldsr
    global LDSR_obj
    try:
        from LDSR import LDSR
        model_url = "https://heibox.uni-heidelberg.de/f/578df07c8fc04ffbadf3/?dl=1"
        yaml_url = "https://heibox.uni-heidelberg.de/f/31a76b13ea27482981b4/?dl=1"
        repo_path = 'latent-diffusion/experiments/pretrained_models/'
        model_path = load_file_from_url(url=model_url, model_dir=os.path.join("repositories", repo_path),
                                        progress=True, file_name="model.chkpt")
        yaml_path = load_file_from_url(url=yaml_url, model_dir=os.path.join("repositories", repo_path),
                                       progress=True, file_name="project.yaml")
        have_ldsr = True
        LDSR_obj = LDSR(model_path, yaml_path)


    except Exception:
        print("Error importing LDSR:", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        have_ldsr = False


def upscale_with_ldsr(image):
    setup_ldsr()
    if not have_ldsr or LDSR_obj is None:
        return image

    ddim_steps = shared.opts.ldsr_steps
    pre_scale = shared.opts.ldsr_pre_down
    post_scale = shared.opts.ldsr_post_down

    image = LDSR_obj.super_resolution(image, ddim_steps, pre_scale, post_scale)
    return image
