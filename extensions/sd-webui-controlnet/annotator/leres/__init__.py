import cv2
import numpy as np
import torch
import os
from modules import devices, shared
from annotator.annotator_path import models_path
from torchvision.transforms import transforms

# AdelaiDepth/LeReS imports
from .leres.depthmap import estimateleres, estimateboost
from .leres.multi_depth_model_woauxi import RelDepthModel
from .leres.net_tools import strip_prefix_if_present

# pix2pix/merge net imports
from .pix2pix.options.test_options import TestOptions
from .pix2pix.models.pix2pix4depth_model import Pix2Pix4DepthModel

base_model_path = os.path.join(models_path, "leres")
old_modeldir = os.path.dirname(os.path.realpath(__file__))

remote_model_path_leres = "https://huggingface.co/lllyasviel/Annotators/resolve/main/res101.pth"
remote_model_path_pix2pix = "https://huggingface.co/lllyasviel/Annotators/resolve/main/latest_net_G.pth"

model = None
pix2pixmodel = None

def unload_leres_model():
    global model, pix2pixmodel
    if model is not None:
        model = model.cpu()
    if pix2pixmodel is not None:
        pix2pixmodel = pix2pixmodel.unload_network('G')


def apply_leres(input_image, thr_a, thr_b, boost=False):
    global model, pix2pixmodel
    if model is None:
        model_path = os.path.join(base_model_path, "res101.pth")
        old_model_path = os.path.join(old_modeldir, "res101.pth")
        
        if os.path.exists(old_model_path):
            model_path = old_model_path
        elif not os.path.exists(model_path):
            from basicsr.utils.download_util import load_file_from_url
            load_file_from_url(remote_model_path_leres, model_dir=base_model_path)

        if torch.cuda.is_available():
            checkpoint = torch.load(model_path)
        else:
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

        model = RelDepthModel(backbone='resnext101')
        model.load_state_dict(strip_prefix_if_present(checkpoint['depth_model'], "module."), strict=True)
        del checkpoint

    if boost and pix2pixmodel is None:
        pix2pixmodel_path = os.path.join(base_model_path, "latest_net_G.pth")
        if not os.path.exists(pix2pixmodel_path):
            from basicsr.utils.download_util import load_file_from_url
            load_file_from_url(remote_model_path_pix2pix, model_dir=base_model_path)

        opt = TestOptions().parse()
        if not torch.cuda.is_available():
            opt.gpu_ids = []  # cpu mode
        pix2pixmodel = Pix2Pix4DepthModel(opt)
        pix2pixmodel.save_dir = base_model_path
        pix2pixmodel.load_networks('latest')
        pix2pixmodel.eval()
    
    if devices.get_device_for("controlnet").type != 'mps':
        model = model.to(devices.get_device_for("controlnet"))

    assert input_image.ndim == 3
    height, width, dim = input_image.shape

    with torch.no_grad():

        if boost:
            depth = estimateboost(input_image, model, 0, pix2pixmodel, max(width, height))
        else:
            depth = estimateleres(input_image, model, width, height)

        numbytes=2
        depth_min = depth.min()
        depth_max = depth.max()
        max_val = (2**(8*numbytes))-1

        # check output before normalizing and mapping to 16 bit
        if depth_max - depth_min > np.finfo("float").eps:
            out = max_val * (depth - depth_min) / (depth_max - depth_min)
        else:
            out = np.zeros(depth.shape)
        
        # single channel, 16 bit image
        depth_image = out.astype("uint16")

        # convert to uint8
        depth_image = cv2.convertScaleAbs(depth_image, alpha=(255.0/65535.0))

        # remove near
        if thr_a != 0:
            thr_a = ((thr_a/100)*255) 
            depth_image = cv2.threshold(depth_image, thr_a, 255, cv2.THRESH_TOZERO)[1]

        # invert image
        depth_image = cv2.bitwise_not(depth_image)

        # remove bg
        if thr_b != 0:
            thr_b = ((thr_b/100)*255)
            depth_image = cv2.threshold(depth_image, thr_b, 255, cv2.THRESH_TOZERO)[1]

        return depth_image
