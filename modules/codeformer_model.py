import os

import cv2
import torch

import modules.face_restoration
import modules.shared
from modules import shared, devices, modelloader, errors
from modules.paths import models_path

model_dir = "Codeformer"
model_path = os.path.join(models_path, model_dir)
model_url = 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth'

codeformer = None


class FaceRestorerCodeFormer(modules.face_restoration.FaceRestoration):
    def name(self):
        return "CodeFormer"

    def __init__(self, dirname):
        self.net = None
        self.face_helper = None
        self.cmd_dir = dirname

    def create_models(self):
        from facexlib.detection import retinaface
        from facexlib.utils.face_restoration_helper import FaceRestoreHelper

        if self.net is not None and self.face_helper is not None:
            self.net.to(devices.device_codeformer)
            return self.net, self.face_helper
        model_paths = modelloader.load_models(
            model_path,
            model_url,
            self.cmd_dir,
            download_name='codeformer-v0.1.0.pth',
            ext_filter=['.pth'],
        )

        if len(model_paths) != 0:
            ckpt_path = model_paths[0]
        else:
            print("Unable to load codeformer model.")
            return None, None
        net = modelloader.load_spandrel_model(ckpt_path, device=devices.device_codeformer)

        if hasattr(retinaface, 'device'):
            retinaface.device = devices.device_codeformer

        face_helper = FaceRestoreHelper(
            upscale_factor=1,
            face_size=512,
            crop_ratio=(1, 1),
            det_model='retinaface_resnet50',
            save_ext='png',
            use_parse=True,
            device=devices.device_codeformer,
        )

        self.net = net
        self.face_helper = face_helper

    def send_model_to(self, device):
        self.net.to(device)
        self.face_helper.face_det.to(device)
        self.face_helper.face_parse.to(device)

    def restore(self, np_image, w=None):
        from torchvision.transforms.functional import normalize
        from basicsr.utils import img2tensor, tensor2img
        np_image = np_image[:, :, ::-1]

        original_resolution = np_image.shape[0:2]

        self.create_models()
        if self.net is None or self.face_helper is None:
            return np_image

        self.send_model_to(devices.device_codeformer)

        self.face_helper.clean_all()
        self.face_helper.read_image(np_image)
        self.face_helper.get_face_landmarks_5(only_center_face=False, resize=640, eye_dist_threshold=5)
        self.face_helper.align_warp_face()

        for cropped_face in self.face_helper.cropped_faces:
            cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
            normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            cropped_face_t = cropped_face_t.unsqueeze(0).to(devices.device_codeformer)

            try:
                with torch.no_grad():
                    res = self.net(cropped_face_t, w=w if w is not None else shared.opts.code_former_weight, adain=True)
                    if isinstance(res, tuple):
                        output = res[0]
                    else:
                        output = res
                    if not isinstance(res, torch.Tensor):
                        raise TypeError(f"Expected torch.Tensor, got {type(res)}")
                    restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
                del output
                devices.torch_gc()
            except Exception:
                errors.report('Failed inference for CodeFormer', exc_info=True)
                restored_face = tensor2img(cropped_face_t, rgb2bgr=True, min_max=(-1, 1))

            restored_face = restored_face.astype('uint8')
            self.face_helper.add_restored_face(restored_face)

        self.face_helper.get_inverse_affine(None)

        restored_img = self.face_helper.paste_faces_to_input_image()
        restored_img = restored_img[:, :, ::-1]

        if original_resolution != restored_img.shape[0:2]:
            restored_img = cv2.resize(
                restored_img,
                (0, 0),
                fx=original_resolution[1]/restored_img.shape[1],
                fy=original_resolution[0]/restored_img.shape[0],
                interpolation=cv2.INTER_LINEAR,
            )

        self.face_helper.clean_all()

        if shared.opts.face_restoration_unload:
            self.send_model_to(devices.cpu)

        return restored_img


def setup_model(dirname):
    os.makedirs(model_path, exist_ok=True)
    try:
        global codeformer
        codeformer = FaceRestorerCodeFormer(dirname)
        shared.face_restorers.append(codeformer)
    except Exception:
        errors.report("Error setting up CodeFormer", exc_info=True)
