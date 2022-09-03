import os
import sys
import traceback

from modules.paths import script_path
from modules.shared import cmd_opts


def gfpgan_model_path():
    places = [script_path, '.', os.path.join(cmd_opts.gfpgan_dir, 'experiments/pretrained_models')]
    files = [cmd_opts.gfpgan_model] + [os.path.join(dirname, cmd_opts.gfpgan_model) for dirname in places]
    found = [x for x in files if os.path.exists(x)]

    if len(found) == 0:
        raise Exception("GFPGAN model not found in paths: " + ", ".join(files))

    return found[0]


loaded_gfpgan_model = None


def gfpgan():
    global loaded_gfpgan_model

    if loaded_gfpgan_model is None and gfpgan_constructor is not None:
        loaded_gfpgan_model = gfpgan_constructor(model_path=gfpgan_model_path(), upscale=1, arch='clean', channel_multiplier=2, bg_upsampler=None)

    return loaded_gfpgan_model


def gfpgan_fix_faces(np_image):
    np_image_bgr = np_image[:, :, ::-1]
    cropped_faces, restored_faces, gfpgan_output_bgr = gfpgan().enhance(np_image_bgr, has_aligned=False, only_center_face=False, paste_back=True)
    np_image = gfpgan_output_bgr[:, :, ::-1]

    return np_image


have_gfpgan = False
gfpgan_constructor = None

def setup_gfpgan():
    try:
        gfpgan_model_path()

        if os.path.exists(cmd_opts.gfpgan_dir):
            sys.path.append(os.path.abspath(cmd_opts.gfpgan_dir))
        from gfpgan import GFPGANer

        global have_gfpgan
        have_gfpgan = True

        global gfpgan_constructor
        gfpgan_constructor = GFPGANer
    except Exception:
        print("Error setting up GFPGAN:", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
