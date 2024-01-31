import cv2
import numpy as np
import torch

from einops import rearrange
from .api import MiDaSInference
from modules import devices

model = None

def unload_midas_model():
    global model
    if model is not None:
        model = model.cpu()

def apply_midas(input_image, a=np.pi * 2.0, bg_th=0.1):
    global model
    if model is None:
        model = MiDaSInference(model_type="dpt_hybrid")
    if devices.get_device_for("controlnet").type != 'mps':
        model = model.to(devices.get_device_for("controlnet"))
    
    assert input_image.ndim == 3
    image_depth = input_image
    with torch.no_grad():
        image_depth = torch.from_numpy(image_depth).float()
        if devices.get_device_for("controlnet").type != 'mps':
            image_depth = image_depth.to(devices.get_device_for("controlnet"))
        image_depth = image_depth / 127.5 - 1.0
        image_depth = rearrange(image_depth, 'h w c -> 1 c h w')
        depth = model(image_depth)[0]

        depth_pt = depth.clone()
        depth_pt -= torch.min(depth_pt)
        depth_pt /= torch.max(depth_pt)
        depth_pt = depth_pt.cpu().numpy()
        depth_image = (depth_pt * 255.0).clip(0, 255).astype(np.uint8)

        depth_np = depth.cpu().numpy()
        x = cv2.Sobel(depth_np, cv2.CV_32F, 1, 0, ksize=3)
        y = cv2.Sobel(depth_np, cv2.CV_32F, 0, 1, ksize=3)
        z = np.ones_like(x) * a
        x[depth_pt < bg_th] = 0
        y[depth_pt < bg_th] = 0
        normal = np.stack([x, y, z], axis=2)
        normal /= np.sum(normal ** 2.0, axis=2, keepdims=True) ** 0.5
        normal_image = (normal * 127.5 + 127.5).clip(0, 255).astype(np.uint8)[:, :, ::-1]

        return depth_image, normal_image
