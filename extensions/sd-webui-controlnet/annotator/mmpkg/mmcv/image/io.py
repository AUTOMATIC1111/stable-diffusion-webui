# Copyright (c) OpenMMLab. All rights reserved.
import io
import os.path as osp
from pathlib import Path

import cv2
import numpy as np
from cv2 import (IMREAD_COLOR, IMREAD_GRAYSCALE, IMREAD_IGNORE_ORIENTATION,
                 IMREAD_UNCHANGED)

from annotator.mmpkg.mmcv.utils import check_file_exist, is_str, mkdir_or_exist

try:
    from turbojpeg import TJCS_RGB, TJPF_BGR, TJPF_GRAY, TurboJPEG
except ImportError:
    TJCS_RGB = TJPF_GRAY = TJPF_BGR = TurboJPEG = None

try:
    from PIL import Image, ImageOps
except ImportError:
    Image = None

try:
    import tifffile
except ImportError:
    tifffile = None

jpeg = None
supported_backends = ['cv2', 'turbojpeg', 'pillow', 'tifffile']

imread_flags = {
    'color': IMREAD_COLOR,
    'grayscale': IMREAD_GRAYSCALE,
    'unchanged': IMREAD_UNCHANGED,
    'color_ignore_orientation': IMREAD_IGNORE_ORIENTATION | IMREAD_COLOR,
    'grayscale_ignore_orientation':
    IMREAD_IGNORE_ORIENTATION | IMREAD_GRAYSCALE
}

imread_backend = 'cv2'


def use_backend(backend):
    """Select a backend for image decoding.

    Args:
        backend (str): The image decoding backend type. Options are `cv2`,
        `pillow`, `turbojpeg` (see https://github.com/lilohuang/PyTurboJPEG)
        and `tifffile`. `turbojpeg` is faster but it only supports `.jpeg`
        file format.
    """
    assert backend in supported_backends
    global imread_backend
    imread_backend = backend
    if imread_backend == 'turbojpeg':
        if TurboJPEG is None:
            raise ImportError('`PyTurboJPEG` is not installed')
        global jpeg
        if jpeg is None:
            jpeg = TurboJPEG()
    elif imread_backend == 'pillow':
        if Image is None:
            raise ImportError('`Pillow` is not installed')
    elif imread_backend == 'tifffile':
        if tifffile is None:
            raise ImportError('`tifffile` is not installed')


def _jpegflag(flag='color', channel_order='bgr'):
    channel_order = channel_order.lower()
    if channel_order not in ['rgb', 'bgr']:
        raise ValueError('channel order must be either "rgb" or "bgr"')

    if flag == 'color':
        if channel_order == 'bgr':
            return TJPF_BGR
        elif channel_order == 'rgb':
            return TJCS_RGB
    elif flag == 'grayscale':
        return TJPF_GRAY
    else:
        raise ValueError('flag must be "color" or "grayscale"')


def _pillow2array(img, flag='color', channel_order='bgr'):
    """Convert a pillow image to numpy array.

    Args:
        img (:obj:`PIL.Image.Image`): The image loaded using PIL
        flag (str): Flags specifying the color type of a loaded image,
            candidates are 'color', 'grayscale' and 'unchanged'.
            Default to 'color'.
        channel_order (str): The channel order of the output image array,
            candidates are 'bgr' and 'rgb'. Default to 'bgr'.

    Returns:
        np.ndarray: The converted numpy array
    """
    channel_order = channel_order.lower()
    if channel_order not in ['rgb', 'bgr']:
        raise ValueError('channel order must be either "rgb" or "bgr"')

    if flag == 'unchanged':
        array = np.array(img)
        if array.ndim >= 3 and array.shape[2] >= 3:  # color image
            array[:, :, :3] = array[:, :, (2, 1, 0)]  # RGB to BGR
    else:
        # Handle exif orientation tag
        if flag in ['color', 'grayscale']:
            img = ImageOps.exif_transpose(img)
        # If the image mode is not 'RGB', convert it to 'RGB' first.
        if img.mode != 'RGB':
            if img.mode != 'LA':
                # Most formats except 'LA' can be directly converted to RGB
                img = img.convert('RGB')
            else:
                # When the mode is 'LA', the default conversion will fill in
                #  the canvas with black, which sometimes shadows black objects
                #  in the foreground.
                #
                # Therefore, a random color (124, 117, 104) is used for canvas
                img_rgba = img.convert('RGBA')
                img = Image.new('RGB', img_rgba.size, (124, 117, 104))
                img.paste(img_rgba, mask=img_rgba.split()[3])  # 3 is alpha
        if flag in ['color', 'color_ignore_orientation']:
            array = np.array(img)
            if channel_order != 'rgb':
                array = array[:, :, ::-1]  # RGB to BGR
        elif flag in ['grayscale', 'grayscale_ignore_orientation']:
            img = img.convert('L')
            array = np.array(img)
        else:
            raise ValueError(
                'flag must be "color", "grayscale", "unchanged", '
                f'"color_ignore_orientation" or "grayscale_ignore_orientation"'
                f' but got {flag}')
    return array


def imread(img_or_path, flag='color', channel_order='bgr', backend=None):
    """Read an image.

    Args:
        img_or_path (ndarray or str or Path): Either a numpy array or str or
            pathlib.Path. If it is a numpy array (loaded image), then
            it will be returned as is.
        flag (str): Flags specifying the color type of a loaded image,
            candidates are `color`, `grayscale`, `unchanged`,
            `color_ignore_orientation` and `grayscale_ignore_orientation`.
            By default, `cv2` and `pillow` backend would rotate the image
            according to its EXIF info unless called with `unchanged` or
            `*_ignore_orientation` flags. `turbojpeg` and `tifffile` backend
            always ignore image's EXIF info regardless of the flag.
            The `turbojpeg` backend only supports `color` and `grayscale`.
        channel_order (str): Order of channel, candidates are `bgr` and `rgb`.
        backend (str | None): The image decoding backend type. Options are
            `cv2`, `pillow`, `turbojpeg`, `tifffile`, `None`.
            If backend is None, the global imread_backend specified by
            ``mmcv.use_backend()`` will be used. Default: None.

    Returns:
        ndarray: Loaded image array.
    """

    if backend is None:
        backend = imread_backend
    if backend not in supported_backends:
        raise ValueError(f'backend: {backend} is not supported. Supported '
                         "backends are 'cv2', 'turbojpeg', 'pillow'")
    if isinstance(img_or_path, Path):
        img_or_path = str(img_or_path)

    if isinstance(img_or_path, np.ndarray):
        return img_or_path
    elif is_str(img_or_path):
        check_file_exist(img_or_path,
                         f'img file does not exist: {img_or_path}')
        if backend == 'turbojpeg':
            with open(img_or_path, 'rb') as in_file:
                img = jpeg.decode(in_file.read(),
                                  _jpegflag(flag, channel_order))
                if img.shape[-1] == 1:
                    img = img[:, :, 0]
            return img
        elif backend == 'pillow':
            img = Image.open(img_or_path)
            img = _pillow2array(img, flag, channel_order)
            return img
        elif backend == 'tifffile':
            img = tifffile.imread(img_or_path)
            return img
        else:
            flag = imread_flags[flag] if is_str(flag) else flag
            img = cv2.imread(img_or_path, flag)
            if flag == IMREAD_COLOR and channel_order == 'rgb':
                cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
            return img
    else:
        raise TypeError('"img" must be a numpy array or a str or '
                        'a pathlib.Path object')


def imfrombytes(content, flag='color', channel_order='bgr', backend=None):
    """Read an image from bytes.

    Args:
        content (bytes): Image bytes got from files or other streams.
        flag (str): Same as :func:`imread`.
        backend (str | None): The image decoding backend type. Options are
            `cv2`, `pillow`, `turbojpeg`, `None`. If backend is None, the
            global imread_backend specified by ``mmcv.use_backend()`` will be
            used. Default: None.

    Returns:
        ndarray: Loaded image array.
    """

    if backend is None:
        backend = imread_backend
    if backend not in supported_backends:
        raise ValueError(f'backend: {backend} is not supported. Supported '
                         "backends are 'cv2', 'turbojpeg', 'pillow'")
    if backend == 'turbojpeg':
        img = jpeg.decode(content, _jpegflag(flag, channel_order))
        if img.shape[-1] == 1:
            img = img[:, :, 0]
        return img
    elif backend == 'pillow':
        buff = io.BytesIO(content)
        img = Image.open(buff)
        img = _pillow2array(img, flag, channel_order)
        return img
    else:
        img_np = np.frombuffer(content, np.uint8)
        flag = imread_flags[flag] if is_str(flag) else flag
        img = cv2.imdecode(img_np, flag)
        if flag == IMREAD_COLOR and channel_order == 'rgb':
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
        return img


def imwrite(img, file_path, params=None, auto_mkdir=True):
    """Write image to file.

    Args:
        img (ndarray): Image array to be written.
        file_path (str): Image file path.
        params (None or list): Same as opencv :func:`imwrite` interface.
        auto_mkdir (bool): If the parent folder of `file_path` does not exist,
            whether to create it automatically.

    Returns:
        bool: Successful or not.
    """
    if auto_mkdir:
        dir_name = osp.abspath(osp.dirname(file_path))
        mkdir_or_exist(dir_name)
    return cv2.imwrite(file_path, img, params)
