from typing import Union
import cv2
import numpy as np
from PIL import Image
from modules.control.util import HWC3, resize_image


checked_ok = False

def check_dependencies():
    global checked_ok # pylint: disable=global-statement
    from installer import installed, install, log
    packages = [('mediapipe', 'mediapipe')]
    for pkg in packages:
        if not installed(pkg[1], reload=True, quiet=True):
            install(pkg[0], pkg[1], ignore=False)
    try:
        import mediapipe as mp # pylint: disable=unused-import
        checked_ok = True
        return True
    except Exception as e:
        log.error(f'MediaPipe: {e}')
        return False


class MediapipeFaceDetector:
    def __call__(self,
                 input_image: Union[np.ndarray, Image.Image] = None,
                 max_faces: int = 1,
                 min_confidence: float = 0.5,
                 output_type: str = "pil",
                 detect_resolution: int = 512,
                 image_resolution: int = 512,
                 **kwargs):
        if not checked_ok:
            if not check_dependencies():
                return
        from .mediapipe_face_util import generate_annotation
        if input_image is None:
            raise ValueError("input_image must be defined.")
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
