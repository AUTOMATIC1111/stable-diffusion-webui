import abc
from typing import Dict, List

import numpy as np
import torch
from skimage import color
from skimage.segmentation import mark_boundaries

from . import colors

COLORS, _ = colors.generate_colors(151) # 151 - max classes for semantic segmentation


class BaseVisualizer:
    @abc.abstractmethod
    def __call__(self, epoch_i, batch_i, batch, suffix='', rank=None):
        """
        Take a batch, make an image from it and visualize
        """
        raise NotImplementedError()


def visualize_mask_and_images(images_dict: Dict[str, np.ndarray], keys: List[str],
                              last_without_mask=True, rescale_keys=None, mask_only_first=None,
                              black_mask=False) -> np.ndarray:
    mask = images_dict['mask'] > 0.5
    result = []
    for i, k in enumerate(keys):
        img = images_dict[k]
        img = np.transpose(img, (1, 2, 0))

        if rescale_keys is not None and k in rescale_keys:
            img = img - img.min()
            img /= img.max() + 1e-5
        if len(img.shape) == 2:
            img = np.expand_dims(img, 2)

        if img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)
        elif (img.shape[2] > 3):
            img_classes = img.argmax(2)
            img = color.label2rgb(img_classes, colors=COLORS)

        if mask_only_first:
            need_mark_boundaries = i == 0
        else:
            need_mark_boundaries = i < len(keys) - 1 or not last_without_mask

        if need_mark_boundaries:
            if black_mask:
                img = img * (1 - mask[0][..., None])
            img = mark_boundaries(img,
                                  mask[0],
                                  color=(1., 0., 0.),
                                  outline_color=(1., 1., 1.),
                                  mode='thick')
        result.append(img)
    return np.concatenate(result, axis=1)


def visualize_mask_and_images_batch(batch: Dict[str, torch.Tensor], keys: List[str], max_items=10,
                                    last_without_mask=True, rescale_keys=None) -> np.ndarray:
    batch = {k: tens.detach().cpu().numpy() for k, tens in batch.items()
             if k in keys or k == 'mask'}

    batch_size = next(iter(batch.values())).shape[0]
    items_to_vis = min(batch_size, max_items)
    result = []
    for i in range(items_to_vis):
        cur_dct = {k: tens[i] for k, tens in batch.items()}
        result.append(visualize_mask_and_images(cur_dct, keys, last_without_mask=last_without_mask,
                                                rescale_keys=rescale_keys))
    return np.concatenate(result, axis=0)
