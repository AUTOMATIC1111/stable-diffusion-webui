import os
import sys
import traceback

import cv2
import torch

import modules.face_restoration
import modules.shared
from modules import shared, devices, modelloader
from modules.paths import models_path

# codeformer people made a choice to include modified basicsr library to their project which makes
# it utterly impossible to use it alongside with other libraries that also use basicsr, like GFPGAN.
# I am making a choice to include some files from codeformer to work around this issue.
model_dir = "Codeformer"
model_path = os.path.join(models_path, model_dir)
model_url = 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth'

have_codeformer = False
codeformer = None


def setup_model(dirname):
    global model_path
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    path = modules.paths.paths.get("CodeFormer", None)
    if path is None:
        return

    try:
        from torchvision.transforms.functional import normalize
        from modules.codeformer.codeformer_arch import CodeFormer
        from basicsr.utils.download_util import load_file_from_url
        from basicsr.utils import imwrite, img2tensor, tensor2img
        from facelib.utils.face_restoration_helper import FaceRestoreHelper
        from facelib.detection.retinaface import retinaface
        from modules.shared import cmd_opts

        net_class = CodeFormer

        class FaceRestorerCodeFormer(modules.face_restoration.FaceRestoration):
            def name(self):
                return "CodeFormer"

            def __init__(self, dirname):
                self.net = None
                self.face_helper = None
                self.cmd_dir = dirname

            def create_models(self):

                if self.net is not None and self.face_helper is not None:
                    self.net.to(devices.device_codeformer)
                    return self.net, self.face_helper
                model_paths = modelloader.load_models(model_path, model_url, self.cmd_dir, download_name='codeformer-v0.1.0.pth')
                if len(model_paths) != 0:
                    ckpt_path = model_paths[0]
                else:
                    print("Unable to load codeformer model.")
                    return None, None
                net = net_class(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9, connect_list=['32', '64', '128', '256']).to(devices.device_codeformer)
                checkpoint = torch.load(ckpt_path)['params_ema']
                net.load_state_dict(checkpoint)
                net.eval()

                if hasattr(retinaface, 'device'):
                    retinaface.device = devices.device_codeformer
                face_helper = FaceRestoreHelper(1, face_size=512, crop_ratio=(1, 1), det_model='retinaface_resnet50', save_ext='png', use_parse=True, device=devices.device_codeformer)

                self.net = net
                self.face_helper = face_helper

                return net, face_helper

            def send_model_to(self, device):
                self.net.to(device)
                self.face_helper.face_det.to(device)
                self.face_helper.face_parse.to(device)

            def restore(self, np_image, w=None):
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

                for idx, cropped_face in enumerate(self.face_helper.cropped_faces):
                    cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
                    normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
                    cropped_face_t = cropped_face_t.unsqueeze(0).to(devices.device_codeformer)

                    try:
                        with torch.no_grad():
                            output = self.net(cropped_face_t, w=w if w is not None else shared.opts.code_former_weight, adain=True)[0]
                            restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
                        del output
                        torch.cuda.empty_cache()
                    except Exception as error:
                        print(f'\tFailed inference for CodeFormer: {error}', file=sys.stderr)
                        restored_face = tensor2img(cropped_face_t, rgb2bgr=True, min_max=(-1, 1))

                    restored_face = restored_face.astype('uint8')
                    self.face_helper.add_restored_face(restored_face)

                self.face_helper.get_inverse_affine(None)

                restored_img = self.face_helper.paste_faces_to_input_image()
                restored_img = restored_img[:, :, ::-1]

                if original_resolution != restored_img.shape[0:2]:
                    restored_img = cv2.resize(restored_img, (0, 0), fx=original_resolution[1]/restored_img.shape[1], fy=original_resolution[0]/restored_img.shape[0], interpolation=cv2.INTER_LINEAR)

                self.face_helper.clean_all()

                if shared.opts.face_restoration_unload:
                    self.send_model_to(devices.cpu)

                return restored_img

        global have_codeformer
        have_codeformer = True

        global codeformer
        codeformer = FaceRestorerCodeFormer(dirname)
        shared.face_restorers.append(codeformer)

    except Exception:
        print("Error setting up CodeFormer:", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)

   # sys.path = stored_sys_path
