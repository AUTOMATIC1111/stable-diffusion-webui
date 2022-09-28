import os
import sys
import traceback
from glob import glob

from modules import shared, devices
from modules.shared import cmd_opts
from modules.paths import script_path
import modules.face_restoration


def gfpgan_model_path():
    from modules.shared import cmd_opts

    filemask = 'GFPGAN*.pth'

    if cmd_opts.gfpgan_model is not None:
        return cmd_opts.gfpgan_model

    places = [script_path, '.', os.path.join(cmd_opts.gfpgan_dir, 'experiments/pretrained_models')]

    filename = None
    for place in places:
        filename = next(iter(glob(os.path.join(place, filemask))), None)
        if filename is not None:
            break

    return filename


loaded_gfpgan_model = None


def gfpgan():
    global loaded_gfpgan_model

    if loaded_gfpgan_model is not None:
        loaded_gfpgan_model.gfpgan.to(shared.device)
        return loaded_gfpgan_model

    if gfpgan_constructor is None:
        return None

    model = gfpgan_constructor(model_path=gfpgan_model_path() or 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth', upscale=1, arch='clean', channel_multiplier=2, bg_upsampler=None)
    model.gfpgan.to(shared.device)
    loaded_gfpgan_model = model

    return model


def gfpgan_fix_faces(np_image):
    model = gfpgan()

    np_image_bgr = np_image[:, :, ::-1]
    cropped_faces, restored_faces, gfpgan_output_bgr = model.enhance(np_image_bgr, has_aligned=False, only_center_face=False, paste_back=True)
    np_image = gfpgan_output_bgr[:, :, ::-1]

    if shared.opts.face_restoration_unload:
        model.gfpgan.to(devices.cpu)

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

        class FaceRestorerGFPGAN(modules.face_restoration.FaceRestoration):
            def name(self):
                return "GFPGAN"

            def restore(self, np_image):
                np_image_bgr = np_image[:, :, ::-1]
                cropped_faces, restored_faces, gfpgan_output_bgr = gfpgan().enhance(np_image_bgr, has_aligned=False, only_center_face=False, paste_back=True)
                np_image = gfpgan_output_bgr[:, :, ::-1]

                return np_image

        shared.face_restorers.append(FaceRestorerGFPGAN())
    except Exception:
        print("Error setting up GFPGAN:", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
