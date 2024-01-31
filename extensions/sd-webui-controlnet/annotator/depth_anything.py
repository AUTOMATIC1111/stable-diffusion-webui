import os
import torch
import cv2
import numpy as np
import torch.nn.functional as F
from torchvision.transforms import Compose

from depth_anything.dpt import DPT_DINOv2
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
from .util import load_model
from .annotator_path import models_path


transform = Compose(
    [
        Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method="lower_bound",
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ]
)


class DepthAnythingDetector:
    """https://github.com/LiheYoung/Depth-Anything"""

    model_dir = os.path.join(models_path, "depth_anything")

    def __init__(self, device: torch.device):
        self.device = device
        self.model = (
            DPT_DINOv2(
                encoder="vitl",
                features=256,
                out_channels=[256, 512, 1024, 1024],
                localhub=False,
            )
            .to(device)
            .eval()
        )
        remote_url = os.environ.get(
            "CONTROLNET_DEPTH_ANYTHING_MODEL_URL",
            "https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints/depth_anything_vitl14.pth",
        )
        model_path = load_model(
            "depth_anything_vitl14.pth", remote_url=remote_url, model_dir=self.model_dir
        )
        self.model.load_state_dict(torch.load(model_path))

    def __call__(self, image: np.ndarray, colored: bool = True) -> np.ndarray:
        self.model.to(self.device)
        h, w = image.shape[:2]

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
        image = transform({"image": image})["image"]
        image = torch.from_numpy(image).unsqueeze(0).to(self.device)
        @torch.no_grad()
        def predict_depth(model, image):
            return model(image)
        depth = predict_depth(self.model, image)
        depth = F.interpolate(
            depth[None], (h, w), mode="bilinear", align_corners=False
        )[0, 0]
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.cpu().numpy().astype(np.uint8)
        if colored:
            return cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)[:, :, ::-1]
        else:
            return depth

    def unload_model(self):
        self.model.to("cpu")
