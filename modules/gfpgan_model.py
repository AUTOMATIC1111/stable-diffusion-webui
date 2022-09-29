import os
import sys
import traceback

import facexlib
import gfpgan

import modules.face_restoration
from modules import shared, devices, modelloader
from modules.paths import models_path

model_dir = "GFPGAN"
user_path = None
model_path = os.path.join(models_path, model_dir)
model_url = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
have_gfpgan = False
loaded_gfpgan_model = None


def gfpgann():
    global loaded_gfpgan_model
    global model_path
    if loaded_gfpgan_model is not None:
        loaded_gfpgan_model.gfpgan.to(shared.device)
        return loaded_gfpgan_model

    if gfpgan_constructor is None:
        return None

    models = modelloader.load_models(model_path, model_url, user_path, ext_filter="GFPGAN")
    if len(models) == 1 and "http" in models[0]:
        model_file = models[0]
    elif len(models) != 0:
        latest_file = max(models, key=os.path.getctime)
        model_file = latest_file
    else:
        print("Unable to load gfpgan model!")
        return None
    model = gfpgan_constructor(model_path=model_file, upscale=1, arch='clean', channel_multiplier=2,
                               bg_upsampler=None)
    model.gfpgan.to(shared.device)
    loaded_gfpgan_model = model

    return model


def gfpgan_fix_faces(np_image):
    model = gfpgann()
    if model is None:
        return np_image
    np_image_bgr = np_image[:, :, ::-1]
    cropped_faces, restored_faces, gfpgan_output_bgr = model.enhance(np_image_bgr, has_aligned=False,
                                                                     only_center_face=False, paste_back=True)
    np_image = gfpgan_output_bgr[:, :, ::-1]

    if shared.opts.face_restoration_unload:
        model.gfpgan.to(devices.cpu)

    return np_image


gfpgan_constructor = None


def setup_model(dirname):
    global model_path
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    try:
        from gfpgan import GFPGANer
        from facexlib import detection, parsing
        global user_path
        global have_gfpgan
        global gfpgan_constructor

        load_file_from_url_orig = gfpgan.utils.load_file_from_url
        facex_load_file_from_url_orig = facexlib.detection.load_file_from_url
        facex_load_file_from_url_orig2 = facexlib.parsing.load_file_from_url

        def my_load_file_from_url(**kwargs):
            print("Setting model_dir to " + model_path)
            return load_file_from_url_orig(**dict(kwargs, model_dir=model_path))

        def facex_load_file_from_url(**kwargs):
            return facex_load_file_from_url_orig(**dict(kwargs, save_dir=model_path, model_dir=None))

        def facex_load_file_from_url2(**kwargs):
            return facex_load_file_from_url_orig2(**dict(kwargs, save_dir=model_path, model_dir=None))

        gfpgan.utils.load_file_from_url = my_load_file_from_url
        facexlib.detection.load_file_from_url = facex_load_file_from_url
        facexlib.parsing.load_file_from_url = facex_load_file_from_url2
        user_path = dirname
        print("Have gfpgan should be true?")
        have_gfpgan = True
        gfpgan_constructor = GFPGANer

        class FaceRestorerGFPGAN(modules.face_restoration.FaceRestoration):
            def name(self):
                return "GFPGAN"

            def restore(self, np_image):
                np_image_bgr = np_image[:, :, ::-1]
                cropped_faces, restored_faces, gfpgan_output_bgr = gfpgann().enhance(np_image_bgr, has_aligned=False,
                                                                                     only_center_face=False,
                                                                                     paste_back=True)
                np_image = gfpgan_output_bgr[:, :, ::-1]

                return np_image

        shared.face_restorers.append(FaceRestorerGFPGAN())
    except Exception:
        print("Error setting up GFPGAN:", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
