import os
import cv2
import numpy as np
import torch

from einops import rearrange
from .zoedepth.models.zoedepth.zoedepth_v1 import ZoeDepth
from .zoedepth.utils.config import get_config
from modules import devices
from annotator.annotator_path import models_path


class ZoeDetector:
    model_dir = os.path.join(models_path, "zoedepth")

    def __init__(self):
        self.model = None
        self.device = devices.get_device_for("controlnet")

    def load_model(self):
        remote_model_path = "https://huggingface.co/lllyasviel/Annotators/resolve/main/ZoeD_M12_N.pt"
        modelpath = os.path.join(self.model_dir, "ZoeD_M12_N.pt")
        if not os.path.exists(modelpath):
            from basicsr.utils.download_util import load_file_from_url
            load_file_from_url(remote_model_path, model_dir=self.model_dir)
        conf = get_config("zoedepth", "infer")
        model = ZoeDepth.build_from_config(conf)
        model.load_state_dict(torch.load(modelpath, map_location=model.device)['model'])
        model.eval()
        self.model = model.to(self.device)

    def unload_model(self):
        if self.model is not None:
            self.model.cpu()

    def __call__(self, input_image):
        if self.model is None:
            self.load_model()
        self.model.to(self.device)

        assert input_image.ndim == 3
        image_depth = input_image
        with torch.no_grad():
            image_depth = torch.from_numpy(image_depth).float().to(self.device)
            image_depth = image_depth / 255.0
            image_depth = rearrange(image_depth, 'h w c -> 1 c h w')
            depth = self.model.infer(image_depth)

            depth = depth[0, 0].cpu().numpy()

            vmin = np.percentile(depth, 2)
            vmax = np.percentile(depth, 85)

            depth -= vmin
            depth /= vmax - vmin
            depth = 1.0 - depth
            depth_image = (depth * 255.0).clip(0, 255).astype(np.uint8)

            return depth_image
