import os
import cv2
import numpy as np
import torch
from einops import rearrange
from huggingface_hub import hf_hub_download
from PIL import Image
from modules import devices
from modules.shared import opts
from modules.control.util import HWC3, nms, resize_image, safe_step
from .pidi_model import pidinet


class PidiNetDetector:
    def __init__(self, model):
        self.model = model

    @classmethod
    def from_pretrained(cls, pretrained_model_or_path, filename=None, cache_dir=None):
        filename = filename or "table5_pidinet.pth"
        if os.path.isdir(pretrained_model_or_path):
            model_path = os.path.join(pretrained_model_or_path, filename)
        else:
            model_path = hf_hub_download(pretrained_model_or_path, filename, cache_dir=cache_dir)
        model = pidinet()
        model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(model_path)['state_dict'].items()})
        model.eval()
        return cls(model)

    def to(self, device):
        self.model.to(device)
        return self

    def __call__(self, input_image, detect_resolution=512, image_resolution=512, safe=False, output_type="pil", scribble=False, apply_filter=False, **kwargs):
        self.model.to(devices.device)
        device = next(iter(self.model.parameters())).device
        if not isinstance(input_image, np.ndarray):
            input_image = np.array(input_image, dtype=np.uint8)
        input_image = HWC3(input_image)
        input_image = resize_image(input_image, detect_resolution)
        assert input_image.ndim == 3
        input_image = input_image[:, :, ::-1].copy()
        image_pidi = torch.from_numpy(input_image).float().to(device)
        image_pidi = image_pidi / 255.0
        image_pidi = rearrange(image_pidi, 'h w c -> 1 c h w')
        edge = self.model(image_pidi)[-1]
        edge = edge.cpu().numpy()
        if apply_filter:
            edge = edge > 0.5
        if safe:
            edge = safe_step(edge)
        edge = (edge * 255.0).clip(0, 255).astype(np.uint8)
        detected_map = edge[0, 0]
        detected_map = HWC3(detected_map)
        img = resize_image(input_image, image_resolution)
        H, W, _C = img.shape
        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
        if scribble:
            detected_map = nms(detected_map, 127, 3.0)
            detected_map = cv2.GaussianBlur(detected_map, (0, 0), 3.0)
            detected_map[detected_map > 4] = 255
            detected_map[detected_map < 255] = 0
        if opts.control_move_processor:
            self.model.to('cpu')
        if output_type == "pil":
            detected_map = Image.fromarray(detected_map)
        return detected_map
