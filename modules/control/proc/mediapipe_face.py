import warnings
from typing import Union
import cv2
import numpy as np
from PIL import Image
from modules.control.util import HWC3, resize_image


class MediapipeFaceDetector:
    def __call__(self,
                 input_image: Union[np.ndarray, Image.Image] = None,
                 max_faces: int = 1,
                 min_confidence: float = 0.5,
                 output_type: str = "pil",
                 detect_resolution: int = 512,
                 image_resolution: int = 512,
                 **kwargs):

        from .mediapipe_face_util import generate_annotation
        if "image" in kwargs:
            warnings.warn("image is deprecated, please use `input_image=...` instead.", DeprecationWarning)
            input_image = kwargs.pop("image")
        if input_image is None:
            raise ValueError("input_image must be defined.")

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

        detected_map = generate_annotation(input_image, max_faces, min_confidence)
        detected_map = HWC3(detected_map)

        img = resize_image(input_image, image_resolution)
        H, W, _C = img.shape

        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)

        if output_type == "pil":
            detected_map = Image.fromarray(detected_map)

        return detected_map
