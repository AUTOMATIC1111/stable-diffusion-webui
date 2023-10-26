import random
import warnings
from typing import Union

import torch
from torch import Tensor
from torchvision.transforms import RandomCrop, functional as F, CenterCrop, RandomHorizontalFlip, PILToTensor
from torchvision.transforms.functional import _get_image_size as get_image_size

from taming.data.helper_types import BoundingBox, Image

pil_to_tensor = PILToTensor()


def convert_pil_to_tensor(image: Image) -> Tensor:
    with warnings.catch_warnings():
        # to filter PyTorch UserWarning as described here: https://github.com/pytorch/vision/issues/2194
        warnings.simplefilter("ignore")
        return pil_to_tensor(image)


class RandomCrop1dReturnCoordinates(RandomCrop):
    def forward(self, img: Image) -> (BoundingBox, Image):
        """
        Additionally to cropping, returns the relative coordinates of the crop bounding box.
        Args:
            img (PIL Image or Tensor): Image to be cropped.

        Returns:
            Bounding box: x0, y0, w, h
            PIL Image or Tensor: Cropped image.

        Based on:
            torchvision.transforms.RandomCrop, torchvision 1.7.0
        """
        if self.padding is not None:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)

        width, height = get_image_size(img)
        # pad the width if needed
        if self.pad_if_needed and width < self.size[1]:
            padding = [self.size[1] - width, 0]
            img = F.pad(img, padding, self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and height < self.size[0]:
            padding = [0, self.size[0] - height]
            img = F.pad(img, padding, self.fill, self.padding_mode)

        i, j, h, w = self.get_params(img, self.size)
        bbox = (j / width, i / height, w / width, h / height)  # x0, y0, w, h
        return bbox, F.crop(img, i, j, h, w)


class Random2dCropReturnCoordinates(torch.nn.Module):
    """
    Additionally to cropping, returns the relative coordinates of the crop bounding box.
    Args:
        img (PIL Image or Tensor): Image to be cropped.

    Returns:
        Bounding box: x0, y0, w, h
        PIL Image or Tensor: Cropped image.

    Based on:
        torchvision.transforms.RandomCrop, torchvision 1.7.0
    """

    def __init__(self, min_size: int):
        super().__init__()
        self.min_size = min_size

    def forward(self, img: Image) -> (BoundingBox, Image):
        width, height = get_image_size(img)
        max_size = min(width, height)
        if max_size <= self.min_size:
            size = max_size
        else:
            size = random.randint(self.min_size, max_size)
        top = random.randint(0, height - size)
        left = random.randint(0, width - size)
        bbox = left / width, top / height, size / width, size / height
        return bbox, F.crop(img, top, left, size, size)


class CenterCropReturnCoordinates(CenterCrop):
    @staticmethod
    def get_bbox_of_center_crop(width: int, height: int) -> BoundingBox:
        if width > height:
            w = height / width
            h = 1.0
            x0 = 0.5 - w / 2
            y0 = 0.
        else:
            w = 1.0
            h = width / height
            x0 = 0.
            y0 = 0.5 - h / 2
        return x0, y0, w, h

    def forward(self, img: Union[Image, Tensor]) -> (BoundingBox, Union[Image, Tensor]):
        """
        Additionally to cropping, returns the relative coordinates of the crop bounding box.
        Args:
            img (PIL Image or Tensor): Image to be cropped.

        Returns:
            Bounding box: x0, y0, w, h
            PIL Image or Tensor: Cropped image.
        Based on:
            torchvision.transforms.RandomHorizontalFlip (version 1.7.0)
        """
        width, height = get_image_size(img)
        return self.get_bbox_of_center_crop(width, height),  F.center_crop(img, self.size)


class RandomHorizontalFlipReturn(RandomHorizontalFlip):
    def forward(self, img: Image) -> (bool, Image):
        """
        Additionally to flipping, returns a boolean whether it was flipped or not.
        Args:
            img (PIL Image or Tensor): Image to be flipped.

        Returns:
            flipped: whether the image was flipped or not
            PIL Image or Tensor: Randomly flipped image.

        Based on:
            torchvision.transforms.RandomHorizontalFlip (version 1.7.0)
        """
        if torch.rand(1) < self.p:
            return True, F.hflip(img)
        return False, img
