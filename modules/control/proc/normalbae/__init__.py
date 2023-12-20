import os
import types
import warnings

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from einops import rearrange
from huggingface_hub import hf_hub_download
from PIL import Image

from modules.control.util import HWC3, resize_image
from .nets.NNET import NNET


# load model
def load_checkpoint(fpath, model):
    ckpt = torch.load(fpath, map_location='cpu')['model']

    load_dict = {}
    for k, v in ckpt.items():
        if k.startswith('module.'):
            k_ = k.replace('module.', '')
            load_dict[k_] = v
        else:
            load_dict[k] = v

    model.load_state_dict(load_dict)
    return model

class NormalBaeDetector:
    def __init__(self, model):
        self.model = model
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    @classmethod
    def from_pretrained(cls, pretrained_model_or_path, filename=None, cache_dir=None):
        filename = filename or "scannet.pt"

        if os.path.isdir(pretrained_model_or_path):
            model_path = os.path.join(pretrained_model_or_path, filename)
        else:
            model_path = hf_hub_download(pretrained_model_or_path, filename, cache_dir=cache_dir)

        args = types.SimpleNamespace()
        args.mode = 'client'
        args.architecture = 'BN'
        args.pretrained = 'scannet'
        args.sampling_ratio = 0.4
        args.importance_ratio = 0.7
        model = NNET(args)
        model = load_checkpoint(model_path, model)
        model.eval()

        return cls(model)

    def to(self, device):
        self.model.to(device)
        return self


    def __call__(self, input_image, detect_resolution=512, image_resolution=512, output_type="pil", **kwargs):
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

        assert input_image.ndim == 3
        image_normal = input_image
        image_normal = torch.from_numpy(image_normal).float().to(device)
        image_normal = image_normal / 255.0
        image_normal = rearrange(image_normal, 'h w c -> 1 c h w')
        image_normal = self.norm(image_normal)

        normal = self.model(image_normal)
        normal = normal[0][-1][:, :3]
        # d = torch.sum(normal ** 2.0, dim=1, keepdim=True) ** 0.5
        # d = torch.maximum(d, torch.ones_like(d) * 1e-5)
        # normal /= d
        normal = ((normal + 1) * 0.5).clip(0, 1)

        normal = rearrange(normal[0], 'c h w -> h w c').cpu().numpy()
        normal_image = (normal * 255.0).clip(0, 255).astype(np.uint8)

        detected_map = normal_image
        detected_map = HWC3(detected_map)

        img = resize_image(input_image, image_resolution)
        H, W, _C = img.shape

        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)

        if output_type == "pil":
            detected_map = Image.fromarray(detected_map)

        return detected_map
