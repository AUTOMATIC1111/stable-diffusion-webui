from PIL import Image
import numpy as np
import torch
from transformers import AutoImageProcessor, DPTForDepthEstimation
from modules import devices


image_processor: AutoImageProcessor = None
dpt_model: DPTForDepthEstimation = None


class DPTDetector:
    def __call__(self, input_image=None):
        global image_processor, dpt_model # pylint: disable=global-statement
        from modules.control.processors import cache_dir
        if image_processor is None:
            image_processor = AutoImageProcessor.from_pretrained("Intel/dpt-large", cache_dir=cache_dir)
        if dpt_model is None:
            dpt_model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large", cache_dir=cache_dir)

        with devices.inference_context():
            inputs = image_processor(images=input_image, return_tensors="pt")
            outputs = dpt_model(**inputs)
            predicted_depth = outputs.predicted_depth
            prediction = torch.nn.functional.interpolate(
                predicted_depth.unsqueeze(1),
                size=input_image.size[::-1],
                mode="bicubic",
                align_corners=False,
            )
            output = prediction.squeeze().cpu().numpy()
            formatted = (output * 255 / np.max(output)).astype("uint8")
            depth = Image.fromarray(formatted)
            depth = depth.convert('RGB')
            return depth
