import warnings
import random
import cv2
import numpy as np
from PIL import Image

from modules.control.util import HWC3, img2mask, make_noise_disk, resize_image


class ContentShuffleDetector:
    def __call__(self, input_image, h=None, w=None, f=None, detect_resolution=512, image_resolution=512, output_type="pil", **kwargs):
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

        H, W, _C = input_image.shape
        if h is None:
            h = H
        if w is None:
            w = W
        if f is None:
            f = 256
        x = make_noise_disk(h, w, 1, f) * float(W - 1)
        y = make_noise_disk(h, w, 1, f) * float(H - 1)
        flow = np.concatenate([x, y], axis=2).astype(np.float32)
        detected_map = cv2.remap(input_image, flow, None, cv2.INTER_LINEAR)

        img = resize_image(input_image, image_resolution)
        H, W, _C = img.shape

        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)

        if output_type == "pil":
            detected_map = Image.fromarray(detected_map)

        return detected_map


class ColorShuffleDetector:
    def __call__(self, img):
        H, W, C = img.shape
        F = np.random.randint(64, 384) # noqa
        A = make_noise_disk(H, W, 3, F)
        B = make_noise_disk(H, W, 3, F)
        C = (A + B) / 2.0
        A = (C + (A - C) * 3.0).clip(0, 1)
        B = (C + (B - C) * 3.0).clip(0, 1)
        L = img.astype(np.float32) / 255.0
        Y = A * L + B * (1 - L)
        Y -= np.min(Y, axis=(0, 1), keepdims=True)
        Y /= np.maximum(np.max(Y, axis=(0, 1), keepdims=True), 1e-5)
        Y *= 255.0
        return Y.clip(0, 255).astype(np.uint8)


class GrayDetector:
    def __call__(self, img):
        eps = 1e-5
        X = img.astype(np.float32)
        r, g, b = X[:, :, 0], X[:, :, 1], X[:, :, 2]
        kr, kg, kb = [random.random() + eps for _ in range(3)]
        ks = kr + kg + kb
        kr /= ks
        kg /= ks
        kb /= ks
        Y = r * kr + g * kg + b * kb
        Y = np.stack([Y] * 3, axis=2)
        return Y.clip(0, 255).astype(np.uint8)


class DownSampleDetector:
    def __call__(self, img, level=3, k=16.0):
        h = img.astype(np.float32)
        for _ in range(level):
            h += np.random.normal(loc=0.0, scale=k, size=h.shape) # noqa
            h = cv2.pyrDown(h)
        for _ in range(level):
            h = cv2.pyrUp(h)
            h += np.random.normal(loc=0.0, scale=k, size=h.shape) # noqa
        return h.clip(0, 255).astype(np.uint8)


class Image2MaskShuffleDetector:
    def __init__(self, resolution=(640, 512)):
        self.H, self.W = resolution

    def __call__(self, img):
        m = img2mask(img, self.H, self.W)
        m *= 255.0
        return m.clip(0, 255).astype(np.uint8)
