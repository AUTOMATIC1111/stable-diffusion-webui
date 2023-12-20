import os
import warnings

import cv2
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from huggingface_hub import hf_hub_download
from PIL import Image

from modules.control.util import HWC3, resize_image

norm_layer = nn.InstanceNorm2d


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        norm_layer(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        norm_layer(in_features)
                        ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9, sigmoid=True):
        super(Generator, self).__init__()

        # Initial convolution block
        model0 = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, 64, 7),
                    norm_layer(64),
                    nn.ReLU(inplace=True) ]
        self.model0 = nn.Sequential(*model0)

        # Downsampling
        model1 = []
        in_features = 64
        out_features = in_features*2
        for _ in range(2):
            model1 += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        norm_layer(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2
        self.model1 = nn.Sequential(*model1)

        model2 = []
        # Residual blocks
        for _ in range(n_residual_blocks):
            model2 += [ResidualBlock(in_features)]
        self.model2 = nn.Sequential(*model2)

        # Upsampling
        model3 = []
        out_features = in_features//2
        for _ in range(2):
            model3 += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        norm_layer(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2
        self.model3 = nn.Sequential(*model3)

        # Output layer
        model4 = [  nn.ReflectionPad2d(3),
                        nn.Conv2d(64, output_nc, 7)]
        if sigmoid:
            model4 += [nn.Sigmoid()]

        self.model4 = nn.Sequential(*model4)

    def forward(self, x, cond=None): # pylint: disable=unused-argument
        out = self.model0(x)
        out = self.model1(out)
        out = self.model2(out)
        out = self.model3(out)
        out = self.model4(out)

        return out


class LineartDetector:
    def __init__(self, model, coarse_model):
        self.model = model
        self.model_coarse = coarse_model

    @classmethod
    def from_pretrained(cls, pretrained_model_or_path, filename=None, coarse_filename=None, cache_dir=None):
        filename = filename or "sk_model.pth"
        coarse_filename = coarse_filename or "sk_model2.pth"

        if os.path.isdir(pretrained_model_or_path):
            model_path = os.path.join(pretrained_model_or_path, filename)
            coarse_model_path = os.path.join(pretrained_model_or_path, coarse_filename)
        else:
            model_path = hf_hub_download(pretrained_model_or_path, filename, cache_dir=cache_dir)
            coarse_model_path = hf_hub_download(pretrained_model_or_path, coarse_filename, cache_dir=cache_dir)

        model = Generator(3, 1, 3)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()

        coarse_model = Generator(3, 1, 3)
        coarse_model.load_state_dict(torch.load(coarse_model_path, map_location=torch.device('cpu')))
        coarse_model.eval()

        return cls(model, coarse_model)

    def to(self, device):
        self.model.to(device)
        self.model_coarse.to(device)
        return self

    def __call__(self, input_image, coarse=False, detect_resolution=512, image_resolution=512, output_type="pil", **kwargs):
        if "return_pil" in kwargs:
            warnings.warn("return_pil is deprecated. Use output_type instead.", DeprecationWarning)
            output_type = "pil" if kwargs["return_pil"] else "np"
        if type(output_type) is bool:
            warnings.warn("Passing `True` or `False` to `output_type` is deprecated and will raise an error in future versions")
            if output_type:
                output_type = "pil"

        device = next(iter(self.model.parameters())).device
        if not isinstance(input_image, np.ndarray):
            input_image = np.array(input_image, dtype=np.uint8)

        input_image = HWC3(input_image)
        input_image = resize_image(input_image, detect_resolution)

        model = self.model_coarse if coarse else self.model
        assert input_image.ndim == 3
        image = input_image
        image = torch.from_numpy(image).float().to(device)
        image = image / 255.0
        image = rearrange(image, 'h w c -> 1 c h w')
        line = model(image)[0][0]

        line = line.cpu().numpy()
        line = (line * 255.0).clip(0, 255).astype(np.uint8)

        detected_map = line

        detected_map = HWC3(detected_map)

        img = resize_image(input_image, image_resolution)
        H, W, _C = img.shape

        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
        detected_map = 255 - detected_map

        if output_type == "pil":
            detected_map = Image.fromarray(detected_map)

        return detected_map
