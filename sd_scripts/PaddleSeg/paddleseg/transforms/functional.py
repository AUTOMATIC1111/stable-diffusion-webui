# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cv2
import numpy as np
from PIL import Image, ImageEnhance
from scipy.ndimage import distance_transform_edt


def crop(img, crop_coordinate):
    x1, y1, x2, y2 = crop_coordinate
    img = img[y1:y2, x1:x2, ...]
    return img


def rescale_size(img_size, target_size):
    scale = min(
        max(target_size) / max(img_size), min(target_size) / min(img_size))
    rescaled_size = [round(i * scale) for i in img_size]
    return rescaled_size, scale


def normalize(im, mean, std):
    im = im.astype(np.float32, copy=False) / 255.0
    im -= mean
    im /= std
    return im


def resize(im, target_size=608, interp=cv2.INTER_LINEAR):
    if isinstance(target_size, list) or isinstance(target_size, tuple):
        w = target_size[0]
        h = target_size[1]
    else:
        w = target_size
        h = target_size
    im = cv2.resize(im, (w, h), interpolation=interp)
    return im


def resize_long(im, long_size=224, interpolation=cv2.INTER_LINEAR):
    value = max(im.shape[0], im.shape[1])
    scale = float(long_size) / float(value)
    resized_width = int(round(im.shape[1] * scale))
    resized_height = int(round(im.shape[0] * scale))

    im = cv2.resize(
        im, (resized_width, resized_height), interpolation=interpolation)
    return im


def resize_short(im, short_size=224, interpolation=cv2.INTER_LINEAR):
    value = min(im.shape[0], im.shape[1])
    scale = float(short_size) / float(value)
    resized_width = int(round(im.shape[1] * scale))
    resized_height = int(round(im.shape[0] * scale))

    im = cv2.resize(
        im, (resized_width, resized_height), interpolation=interpolation)
    return im


def horizontal_flip(im):
    if len(im.shape) == 3:
        im = im[:, ::-1, :]
    elif len(im.shape) == 2:
        im = im[:, ::-1]
    return im


def vertical_flip(im):
    if len(im.shape) == 3:
        im = im[::-1, :, :]
    elif len(im.shape) == 2:
        im = im[::-1, :]
    return im


def brightness(im, brightness_lower, brightness_upper):
    brightness_delta = np.random.uniform(brightness_lower, brightness_upper)
    im = ImageEnhance.Brightness(im).enhance(brightness_delta)
    return im


def contrast(im, contrast_lower, contrast_upper):
    contrast_delta = np.random.uniform(contrast_lower, contrast_upper)
    im = ImageEnhance.Contrast(im).enhance(contrast_delta)
    return im


def saturation(im, saturation_lower, saturation_upper):
    saturation_delta = np.random.uniform(saturation_lower, saturation_upper)
    im = ImageEnhance.Color(im).enhance(saturation_delta)
    return im


def hue(im, hue_lower, hue_upper):
    hue_delta = np.random.uniform(hue_lower, hue_upper)
    im = np.array(im.convert('HSV'))
    im[:, :, 0] = im[:, :, 0] + hue_delta
    im = Image.fromarray(im, mode='HSV').convert('RGB')
    return im


def sharpness(im, sharpness_lower, sharpness_upper):
    sharpness_delta = np.random.uniform(sharpness_lower, sharpness_upper)
    im = ImageEnhance.Sharpness(im).enhance(sharpness_delta)
    return im


def rotate(im, rotate_lower, rotate_upper):
    rotate_delta = np.random.uniform(rotate_lower, rotate_upper)
    im = im.rotate(int(rotate_delta))
    return im


def mask_to_onehot(mask, num_classes):
    """
    Convert a mask (H, W) to onehot (K, H, W).

    Args:
        mask (np.ndarray): Label mask with shape (H, W)
        num_classes (int): Number of classes.

    Returns:
        np.ndarray: Onehot mask with shape(K, H, W).
    """
    _mask = [mask == i for i in range(num_classes)]
    _mask = np.array(_mask).astype(np.uint8)
    return _mask


def onehot_to_binary_edge(mask, radius):
    """
    Convert a onehot mask (K, H, W) to a edge mask.

    Args:
        mask (np.ndarray): Onehot mask with shape (K, H, W)
        radius (int|float): Radius of edge.

    Returns:
        np.ndarray: Edge mask with shape(H, W).
    """
    if radius < 1:
        raise ValueError('`radius` should be greater than or equal to 1')
    num_classes = mask.shape[0]

    edge = np.zeros(mask.shape[1:])
    # pad borders
    mask = np.pad(mask, ((0, 0), (1, 1), (1, 1)),
                  mode='constant',
                  constant_values=0)
    for i in range(num_classes):
        dist = distance_transform_edt(mask[i, :]) + distance_transform_edt(
            1.0 - mask[i, :])
        dist = dist[1:-1, 1:-1]
        dist[dist > radius] = 0
        edge += dist

    edge = np.expand_dims(edge, axis=0)
    edge = (edge > 0).astype(np.uint8)
    return edge


def mask_to_binary_edge(mask, radius, num_classes):
    """
    Convert a segmentic segmentation mask (H, W) to a binary edge mask(H, W).

    Args:
        mask (np.ndarray): Label mask with shape (H, W)
        radius (int|float): Radius of edge.
        num_classes (int): Number of classes.

    Returns:
        np.ndarray: Edge mask with shape(H, W).
    """
    mask = mask.squeeze()
    onehot = mask_to_onehot(mask, num_classes)
    edge = onehot_to_binary_edge(onehot, radius)
    return edge
