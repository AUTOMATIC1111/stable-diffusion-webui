# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import warnings
from typing import Union

import cv2
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from modules.control.util import HWC3, resize_image
from .automatic_mask_generator import SamAutomaticMaskGenerator
from .build_sam import sam_model_registry


class SamDetector:
    def __init__(self, mask_generator: SamAutomaticMaskGenerator = None):
        self.mask_generator = mask_generator

    @classmethod
    def from_pretrained(cls, model_path, filename, model_type, cache_dir=None):
        """
        Possible model_type : vit_h, vit_l, vit_b, vit_t
        download weights from https://github.com/facebookresearch/segment-anything
        """
        model_path = hf_hub_download(model_path, filename, cache_dir=cache_dir)

        sam = sam_model_registry[model_type](checkpoint=model_path)

        if torch.cuda.is_available():
            sam.to("cuda")

        mask_generator = SamAutomaticMaskGenerator(sam)

        return cls(mask_generator)


    def show_anns(self, anns):
        from numpy.random import default_rng
        gen = default_rng()
        if len(anns) == 0:
            return
        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
        h, w =  anns[0]['segmentation'].shape
        final_img = Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8), mode="RGB")
        for ann in sorted_anns:
            m = ann['segmentation']
            img = np.empty((m.shape[0], m.shape[1], 3), dtype=np.uint8)
            for i in range(3):
                img[:,:,i] = gen.integers(255, dtype=np.uint8)
            final_img.paste(Image.fromarray(img, mode="RGB"), (0, 0), Image.fromarray(np.uint8(m*255)))

        return np.array(final_img, dtype=np.uint8)

    def __call__(self, input_image: Union[np.ndarray, Image.Image]=None, detect_resolution=512, image_resolution=512, output_type="pil", **kwargs) -> Image.Image:
        if "image" in kwargs:
            warnings.warn("image is deprecated, please use `input_image=...` instead.", DeprecationWarning)
            input_image = kwargs.pop("image")

        if input_image is None:
            raise ValueError("input_image must be defined.")

        if not isinstance(input_image, np.ndarray):
            input_image = np.array(input_image, dtype=np.uint8)

        input_image = HWC3(input_image)
        input_image = resize_image(input_image, detect_resolution)

        # Generate Masks
        masks = self.mask_generator.generate(input_image)
        # Create map
        image_map = self.show_anns(masks)

        detected_map = image_map
        detected_map = HWC3(detected_map)

        img = resize_image(input_image, image_resolution)
        H, W, _C = img.shape

        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)

        if output_type == "pil":
            detected_map = Image.fromarray(detected_map)

        return detected_map
