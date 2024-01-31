import cv2
import numpy as np
from annotator.util import make_noise_disk


class ContentShuffleDetector:
    def __call__(self, img, h=None, w=None, f=None):
        H, W, C = img.shape
        if h is None:
            h = H
        if w is None:
            w = W
        if f is None:
            f = 256
        x = make_noise_disk(h, w, 1, f) * float(W - 1)
        y = make_noise_disk(h, w, 1, f) * float(H - 1)
        flow = np.concatenate([x, y], axis=2).astype(np.float32)
        return cv2.remap(img, flow, None, cv2.INTER_LINEAR)
