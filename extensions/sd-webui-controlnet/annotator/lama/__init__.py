# https://github.com/advimman/lama

import yaml
import torch
from omegaconf import OmegaConf
import numpy as np

from einops import rearrange
import os
from modules import devices
from annotator.annotator_path import models_path
from annotator.lama.saicinpainting.training.trainers import load_checkpoint


class LamaInpainting:
    model_dir = os.path.join(models_path, "lama")

    def __init__(self):
        self.model = None
        self.device = devices.get_device_for("controlnet")

    def load_model(self):
        remote_model_path = "https://huggingface.co/lllyasviel/Annotators/resolve/main/ControlNetLama.pth"
        modelpath = os.path.join(self.model_dir, "ControlNetLama.pth")
        if not os.path.exists(modelpath):
            from basicsr.utils.download_util import load_file_from_url
            load_file_from_url(remote_model_path, model_dir=self.model_dir)
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml')
        cfg = yaml.safe_load(open(config_path, 'rt'))
        cfg = OmegaConf.create(cfg)
        cfg.training_model.predict_only = True
        cfg.visualizer.kind = 'noop'
        self.model = load_checkpoint(cfg, os.path.abspath(modelpath), strict=False, map_location='cpu')
        self.model = self.model.to(self.device)
        self.model.eval()

    def unload_model(self):
        if self.model is not None:
            self.model.cpu()

    def __call__(self, input_image):
        if self.model is None:
            self.load_model()
        self.model.to(self.device)
        color = np.ascontiguousarray(input_image[:, :, 0:3]).astype(np.float32) / 255.0
        mask = np.ascontiguousarray(input_image[:, :, 3:4]).astype(np.float32) / 255.0
        with torch.no_grad():
            color = torch.from_numpy(color).float().to(self.device)
            mask = torch.from_numpy(mask).float().to(self.device)
            mask = (mask > 0.5).float()
            color = color * (1 - mask)
            image_feed = torch.cat([color, mask], dim=2)
            image_feed = rearrange(image_feed, 'h w c -> 1 c h w')
            result = self.model(image_feed)[0]
            result = rearrange(result, 'c h w -> h w c')
            result = result * mask + color * (1 - mask)
            result *= 255.0
            return result.detach().cpu().numpy().clip(0, 255).astype(np.uint8)
