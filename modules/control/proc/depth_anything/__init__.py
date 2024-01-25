import cv2
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from modules import devices, masking
from modules.shared import opts


class DepthAnythingDetector:
    """https://github.com/LiheYoung/Depth-Anything"""
    def __init__(self, model):
        from torchvision.transforms import Compose
        from modules.control.proc.depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
        self.model = model
        self.transform = Compose([
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
            PrepareForNet()])

    @classmethod
    def from_pretrained(cls, pretrained_model_or_path: str, cache_dir: str) -> str:
        from modules.control.proc.depth_anything.dpt import DPT_DINOv2
        import huggingface_hub as hf
        model = (
            DPT_DINOv2(
                encoder="vitl",
                features=256,
                out_channels=[256, 512, 1024, 1024],
                localhub=False,
            )
            .to(devices.device)
            .eval()
        )
        model_path = hf.hf_hub_download(repo_id=pretrained_model_or_path, filename="pytorch_model.bin", cache_dir=cache_dir)
        model_dict = torch.load(model_path)
        model.load_state_dict(model_dict)
        return cls(model)

    def __call__(self, image, color_map: str = "none", output_type: str = 'pil'):
        self.model.to(devices.device)
        if isinstance(image, Image.Image):
            image = np.array(image)
        h, w = image.shape[:2]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
        image = self.transform({ "image": image })["image"]
        image = torch.from_numpy(image).unsqueeze(0).to(devices.device)
        with devices.inference_context():
            depth = self.model(image)
        if opts.control_move_processor:
            self.model.to('cpu')
        depth = F.interpolate(depth[None], (h, w), mode="bilinear", align_corners=False)[0, 0]
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.cpu().numpy().astype(np.uint8)
        if color_map != 'none':
            depth = cv2.applyColorMap(depth, masking.COLORMAP.index(color_map))[:, :, ::-1]
        if output_type == "pil":
            depth = Image.fromarray(depth)
        return depth

    # def unload_model(self):
    #    self.model.to("cpu")
