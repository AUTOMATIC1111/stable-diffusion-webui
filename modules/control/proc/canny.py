import warnings
import cv2
import numpy as np
from PIL import Image
from modules.control.util import HWC3, resize_image

class CannyDetector:
    def __call__(self, input_image=None, low_threshold=100, high_threshold=200, detect_resolution=512, image_resolution=512, output_type=None, **kwargs):
        if "img" in kwargs:
            warnings.warn("img is deprecated, please use `input_image=...` instead.", DeprecationWarning)
            input_image = kwargs.pop("img")
        if input_image is None:
            raise ValueError("input_image must be defined.")

        if not isinstance(input_image, np.ndarray):
            input_image = np.array(input_image, dtype=np.uint8)
            output_type = output_type or "pil"
        else:
            output_type = output_type or "np"

        input_image = HWC3(input_image)
        input_image = resize_image(input_image, detect_resolution)

        detected_map = cv2.Canny(input_image, low_threshold, high_threshold)
        detected_map = HWC3(detected_map)

        img = resize_image(input_image, image_resolution)
        H, W, _C = img.shape

        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)

        if output_type == "pil":
            detected_map = Image.fromarray(detected_map)
            detected_map = detected_map.convert('L')

        return detected_map
