import os
import sys
import traceback
from collections import namedtuple

from modules import shared, images, modelloader, paths
from modules.paths import models_path

model_dir = "LDSR"
model_path = os.path.join(models_path, model_dir)
cmd_path = None
model_url = "https://heibox.uni-heidelberg.de/f/578df07c8fc04ffbadf3/?dl=1"
yaml_url = "https://heibox.uni-heidelberg.de/f/31a76b13ea27482981b4/?dl=1"

LDSRModelInfo = namedtuple("LDSRModelInfo", ["name", "location", "model", "netscale"])

ldsr_models = []
have_ldsr = False
LDSR_obj = None


class UpscalerLDSR(images.Upscaler):
    def __init__(self, steps):
        self.steps = steps
        self.name = "LDSR"

    def do_upscale(self, img):
        return upscale_with_ldsr(img)


def setup_model(dirname):
    global cmd_path
    global model_path
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    cmd_path = dirname
    shared.sd_upscalers.append(UpscalerLDSR(100))


def prepare_ldsr():
    path = paths.paths.get("LDSR", None)
    if path is None:
        return
    global have_ldsr
    global LDSR_obj
    try:
        from LDSR import LDSR
        model_files = modelloader.load_models(model_path, model_url, cmd_path, dl_name="model.ckpt", ext_filter=[".ckpt"])
        yaml_files = modelloader.load_models(model_path, yaml_url, cmd_path, dl_name="project.yaml", ext_filter=[".yaml"])
        if len(model_files) != 0 and len(yaml_files) != 0:
            model_file = model_files[0]
            yaml_file = yaml_files[0]
            have_ldsr = True
            LDSR_obj = LDSR(model_file, yaml_file)
        else:
            return

    except Exception:
        print("Error importing LDSR:", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        have_ldsr = False


def upscale_with_ldsr(image):
    prepare_ldsr()
    if not have_ldsr or LDSR_obj is None:
        return image

    ddim_steps = shared.opts.ldsr_steps
    pre_scale = shared.opts.ldsr_pre_down
    post_scale = shared.opts.ldsr_post_down

    image = LDSR_obj.super_resolution(image, ddim_steps, pre_scale, post_scale)
    return image
