# This is an improved version and model of HED edge detection with Apache License, Version 2.0.
# Please use this implementation in your products
# This implementation may produce slightly different results from Saining Xie's official implementations,
# but it generates smoother edges and is more suitable for ControlNet as well as other image-to-image translations.
# Different from official models and other implementations, this is an RGB-input model (rather than BGR)
# and in this way it works better for gradio's RGB protocol

import os
import cv2
import torch
import numpy as np

from einops import rearrange
import os
from modules import devices
from annotator.annotator_path import models_path
from annotator.util import safe_step, nms


class DoubleConvBlock(torch.nn.Module):
    def __init__(self, input_channel, output_channel, layer_number):
        super().__init__()
        self.convs = torch.nn.Sequential()
        self.convs.append(torch.nn.Conv2d(in_channels=input_channel, out_channels=output_channel, kernel_size=(3, 3), stride=(1, 1), padding=1))
        for i in range(1, layer_number):
            self.convs.append(torch.nn.Conv2d(in_channels=output_channel, out_channels=output_channel, kernel_size=(3, 3), stride=(1, 1), padding=1))
        self.projection = torch.nn.Conv2d(in_channels=output_channel, out_channels=1, kernel_size=(1, 1), stride=(1, 1), padding=0)

    def __call__(self, x, down_sampling=False):
        h = x
        if down_sampling:
            h = torch.nn.functional.max_pool2d(h, kernel_size=(2, 2), stride=(2, 2))
        for conv in self.convs:
            h = conv(h)
            h = torch.nn.functional.relu(h)
        return h, self.projection(h)


class ControlNetHED_Apache2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = torch.nn.Parameter(torch.zeros(size=(1, 3, 1, 1)))
        self.block1 = DoubleConvBlock(input_channel=3, output_channel=64, layer_number=2)
        self.block2 = DoubleConvBlock(input_channel=64, output_channel=128, layer_number=2)
        self.block3 = DoubleConvBlock(input_channel=128, output_channel=256, layer_number=3)
        self.block4 = DoubleConvBlock(input_channel=256, output_channel=512, layer_number=3)
        self.block5 = DoubleConvBlock(input_channel=512, output_channel=512, layer_number=3)

    def __call__(self, x):
        h = x - self.norm
        h, projection1 = self.block1(h)
        h, projection2 = self.block2(h, down_sampling=True)
        h, projection3 = self.block3(h, down_sampling=True)
        h, projection4 = self.block4(h, down_sampling=True)
        h, projection5 = self.block5(h, down_sampling=True)
        return projection1, projection2, projection3, projection4, projection5


netNetwork = None
remote_model_path = "https://huggingface.co/lllyasviel/Annotators/resolve/main/ControlNetHED.pth"
modeldir = os.path.join(models_path, "hed")
old_modeldir = os.path.dirname(os.path.realpath(__file__))


def apply_hed(input_image, is_safe=False):
    global netNetwork
    if netNetwork is None:
        modelpath = os.path.join(modeldir, "ControlNetHED.pth")
        old_modelpath = os.path.join(old_modeldir, "ControlNetHED.pth")
        if os.path.exists(old_modelpath):
            modelpath = old_modelpath
        elif not os.path.exists(modelpath):
            from basicsr.utils.download_util import load_file_from_url
            load_file_from_url(remote_model_path, model_dir=modeldir)
        netNetwork = ControlNetHED_Apache2().to(devices.get_device_for("controlnet"))
        netNetwork.load_state_dict(torch.load(modelpath, map_location='cpu'))
    netNetwork.to(devices.get_device_for("controlnet")).float().eval()

    assert input_image.ndim == 3
    H, W, C = input_image.shape
    with torch.no_grad():
        image_hed = torch.from_numpy(input_image.copy()).float().to(devices.get_device_for("controlnet"))
        image_hed = rearrange(image_hed, 'h w c -> 1 c h w')
        edges = netNetwork(image_hed)
        edges = [e.detach().cpu().numpy().astype(np.float32)[0, 0] for e in edges]
        edges = [cv2.resize(e, (W, H), interpolation=cv2.INTER_LINEAR) for e in edges]
        edges = np.stack(edges, axis=2)
        edge = 1 / (1 + np.exp(-np.mean(edges, axis=2).astype(np.float64)))
        if is_safe:
            edge = safe_step(edge)
        edge = (edge * 255.0).clip(0, 255).astype(np.uint8)
        return edge

    
def unload_hed_model():
    global netNetwork
    if netNetwork is not None:
        netNetwork.cpu()
