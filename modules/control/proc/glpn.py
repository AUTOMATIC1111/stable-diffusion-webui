from PIL import Image
import numpy as np
import torch
from transformers import AutoImageProcessor, GLPNForDepthEstimation
from modules import devices
from modules.shared import opts


class GLPNDetector:
    def __init__(self, model=None, processor=None):
        self.model = model
        self.processor = processor

    def __call__(self, input_image=None):
        from modules.control.processors import cache_dir
        if self.processor is None:
            self.processor = AutoImageProcessor.from_pretrained("vinvino02/glpn-kitti", cache_dir=cache_dir)
        if self.model is None:
            self.model = GLPNForDepthEstimation.from_pretrained("vinvino02/glpn-kitti", cache_dir=cache_dir)

        self.model.to(devices.device)
        with devices.inference_context():
            inputs = self.processor(images=input_image, return_tensors="pt")
            inputs.to(devices.device)
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth
            prediction = torch.nn.functional.interpolate(
                predicted_depth.unsqueeze(1),
                size=input_image.size[::-1],
                mode="bicubic",
                align_corners=False,
            )
            output = prediction.squeeze().cpu().numpy()
            formatted = 255 - (output * 255 / np.max(output)).astype("uint8")
        if opts.control_move_processor:
            self.model.to('cpu')
        depth = Image.fromarray(formatted)
        depth = depth.convert('RGB')
        return depth
