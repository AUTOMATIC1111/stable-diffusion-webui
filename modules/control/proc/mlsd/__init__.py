import os
import warnings

import cv2
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from modules.control.util import HWC3, resize_image
from .models.mbv2_mlsd_large import MobileV2_MLSD_Large
from .utils import pred_lines


class MLSDdetector:
    def __init__(self, model):
        self.model = model

    @classmethod
    def from_pretrained(cls, pretrained_model_or_path, filename=None, cache_dir=None):
        if pretrained_model_or_path == "lllyasviel/ControlNet":
            filename = filename or "annotator/ckpts/mlsd_large_512_fp32.pth"
        else:
            filename = filename or "mlsd_large_512_fp32.pth"

        if os.path.isdir(pretrained_model_or_path):
            model_path = os.path.join(pretrained_model_or_path, filename)
        else:
            model_path = hf_hub_download(pretrained_model_or_path, filename, cache_dir=cache_dir)

        model = MobileV2_MLSD_Large()
        model.load_state_dict(torch.load(model_path), strict=True)
        model.eval()

        return cls(model)

    def to(self, device):
        self.model.to(device)
        return self

    def __call__(self, input_image, thr_v=0.1, thr_d=0.1, detect_resolution=512, image_resolution=512, output_type="pil", **kwargs):
        if "return_pil" in kwargs:
            warnings.warn("return_pil is deprecated. Use output_type instead.", DeprecationWarning)
            output_type = "pil" if kwargs["return_pil"] else "np"
        if type(output_type) is bool:
            warnings.warn("Passing `True` or `False` to `output_type` is deprecated and will raise an error in future versions")
            if output_type:
                output_type = "pil"

        if not isinstance(input_image, np.ndarray):
            input_image = np.array(input_image, dtype=np.uint8)

        input_image = HWC3(input_image)
        input_image = resize_image(input_image, detect_resolution)

        assert input_image.ndim == 3
        img = input_image
        img_output = np.zeros_like(img)
        try:
            lines = pred_lines(img, self.model, [img.shape[0], img.shape[1]], thr_v, thr_d)
            for line in lines:
                x_start, y_start, x_end, y_end = [int(val) for val in line]
                cv2.line(img_output, (x_start, y_start), (x_end, y_end), [255, 255, 255], 1)
        except Exception:
            pass

        detected_map = img_output[:, :, 0]
        detected_map = HWC3(detected_map)

        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape

        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)

        if output_type == "pil":
            detected_map = Image.fromarray(detected_map)

        return detected_map
