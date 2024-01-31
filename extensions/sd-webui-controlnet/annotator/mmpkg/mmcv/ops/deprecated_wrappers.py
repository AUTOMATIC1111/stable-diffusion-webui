# Copyright (c) OpenMMLab. All rights reserved.
# This file is for backward compatibility.
# Module wrappers for empty tensor have been moved to mmcv.cnn.bricks.
import warnings

from ..cnn.bricks.wrappers import Conv2d, ConvTranspose2d, Linear, MaxPool2d


class Conv2d_deprecated(Conv2d):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        warnings.warn(
            'Importing Conv2d wrapper from "mmcv.ops" will be deprecated in'
            ' the future. Please import them from "mmcv.cnn" instead')


class ConvTranspose2d_deprecated(ConvTranspose2d):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        warnings.warn(
            'Importing ConvTranspose2d wrapper from "mmcv.ops" will be '
            'deprecated in the future. Please import them from "mmcv.cnn" '
            'instead')


class MaxPool2d_deprecated(MaxPool2d):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        warnings.warn(
            'Importing MaxPool2d wrapper from "mmcv.ops" will be deprecated in'
            ' the future. Please import them from "mmcv.cnn" instead')


class Linear_deprecated(Linear):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        warnings.warn(
            'Importing Linear wrapper from "mmcv.ops" will be deprecated in'
            ' the future. Please import them from "mmcv.cnn" instead')
