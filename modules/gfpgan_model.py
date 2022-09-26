import os
import sys
import traceback
from glob import glob

from modules import shared, devices
from modules.shared import cmd_opts
from modules.paths import script_path
import modules.face_restoration
from modules import shared, devices, modelloader
from modules.paths import models_path

model_dir = "GFPGAN"
cmd_dir = None
model_path = os.path.join(models_path, model_dir)
model_url = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"

loaded_gfpgan_model = None


def gfpgan():
    global loaded_gfpgan_model
    global model_path
    if loaded_gfpgan_model is not None:
        loaded_gfpgan_model.gfpgan.to(shared.device)
        return loaded_gfpgan_model

    if gfpgan_constructor is None:
        return None

    models = modelloader.load_models(model_path, model_url, cmd_dir)
    if len(models) != 0:
        latest_file = max(models, key=os.path.getctime)
        model_file = latest_file
    else:
        print("Unable to load gfpgan model!")
        return None
    model = gfpgan_constructor(model_path=model_file, model_dir=model_path, upscale=1, arch='clean', channel_multiplier=2,
                               bg_upsampler=None)
    model.gfpgan.to(shared.device)
    loaded_gfpgan_model = model

    return model


def gfpgan_fix_faces(np_image):
    model = gfpgan()
    if model is None:
        return np_image
    np_image_bgr = np_image[:, :, ::-1]
    cropped_faces, restored_faces, gfpgan_output_bgr = model.enhance(np_image_bgr, has_aligned=False, only_center_face=False, paste_back=True)
    np_image = gfpgan_output_bgr[:, :, ::-1]

    if shared.opts.face_restoration_unload:
        model.gfpgan.to(devices.cpu)

    return np_image


have_gfpgan = False
gfpgan_constructor = None


def setup_model(dirname):
    global model_path
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    try:
        from modules.gfpgan_model_arch import GFPGANerr
        global cmd_dir
        global have_gfpgan
        global gfpgan_constructor

        cmd_dir = dirname
        have_gfpgan = True
        gfpgan_constructor = GFPGANerr

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
