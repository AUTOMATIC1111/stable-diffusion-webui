"""
Hello, welcome on board,
"""
from __future__ import print_function

import os
import cv2
import numpy as np

import torch

from annotator.teed.ted import TED  # TEED architecture
from einops import rearrange
from modules import devices
from annotator.util import load_model,safe_step
from annotator.annotator_path import models_path

class TEEDDector:
    """https://github.com/xavysp/TEED"""

    model_dir = os.path.join(models_path, "TEED")

    def __init__(self):
        self.device = devices.get_device_for("controlnet")
        self.model = TED().to(self.device).eval()
        remote_url = os.environ.get(
            "CONTROLNET_TEED_MODEL_URL",
            "https://huggingface.co/bdsqlsz/qinglong_controlnet-lllite/resolve/main/Annotators/7_model.pth",
        )
        model_path = load_model(
            "7_model.pth", remote_url=remote_url, model_dir=self.model_dir
        )
        self.model.load_state_dict(torch.load(model_path))

    def unload_model(self):
        if self.model is not None:
            self.model.cpu()

    def __call__(self, image: np.ndarray, safe_steps: int = 2) -> np.ndarray:

        self.model.to(self.device)

        H, W, _ = image.shape
        with torch.no_grad():
            image_teed = torch.from_numpy(image.copy()).float().to(self.device)
            image_teed = rearrange(image_teed, 'h w c -> 1 c h w')
            edges = self.model(image_teed)
            edges = [e.detach().cpu().numpy().astype(np.float32)[0, 0] for e in edges]
            edges = [cv2.resize(e, (W, H), interpolation=cv2.INTER_LINEAR) for e in edges]
            edges = np.stack(edges, axis=2)
            edge = 1 / (1 + np.exp(-np.mean(edges, axis=2).astype(np.float64)))
            if safe_steps != 0:
                edge = safe_step(edge, safe_steps)
            edge = (edge * 255.0).clip(0, 255).astype(np.uint8)
            return edge