import torch
import os
import functools
import time
import base64
import numpy as np
import safetensors.torch
import cv2
import logging

from typing import Any, Callable, Dict, List
from modules.safe import unsafe_torch_load
from scripts.logging import logger


def load_state_dict(ckpt_path, location="cpu"):
    _, extension = os.path.splitext(ckpt_path)
    if extension.lower() == ".safetensors":
        state_dict = safetensors.torch.load_file(ckpt_path, device=location)
    else:
        state_dict = unsafe_torch_load(ckpt_path, map_location=torch.device(location))
    state_dict = get_state_dict(state_dict)
    logger.info(f"Loaded state_dict from [{ckpt_path}]")
    return state_dict


def get_state_dict(d):
    return d.get("state_dict", d)


def ndarray_lru_cache(max_size: int = 128, typed: bool = False):
    """
    Decorator to enable caching for functions with numpy array arguments.
    Numpy arrays are mutable, and thus not directly usable as hash keys.

    The idea here is to wrap the incoming arguments with type `np.ndarray`
    as `HashableNpArray` so that `lru_cache` can correctly handles `np.ndarray`
    arguments.

    `HashableNpArray` functions exactly the same way as `np.ndarray` except
    having `__hash__` and `__eq__` overriden.
    """

    def decorator(func: Callable):
        """The actual decorator that accept function as input."""

        class HashableNpArray(np.ndarray):
            def __new__(cls, input_array):
                # Input array is an instance of ndarray.
                # The view makes the input array and returned array share the same data.
                obj = np.asarray(input_array).view(cls)
                return obj

            def __eq__(self, other) -> bool:
                return np.array_equal(self, other)

            def __hash__(self):
                # Hash the bytes representing the data of the array.
                return hash(self.tobytes())

        @functools.lru_cache(maxsize=max_size, typed=typed)
        def cached_func(*args, **kwargs):
            """This function only accepts `HashableNpArray` as input params."""
            return func(*args, **kwargs)

        # Preserves original function.__name__ and __doc__.
        @functools.wraps(func)
        def decorated_func(*args, **kwargs):
            """The decorated function that delegates the original function."""

            def convert_item(item: Any):
                if isinstance(item, np.ndarray):
                    return HashableNpArray(item)
                if isinstance(item, tuple):
                    return tuple(convert_item(i) for i in item)
                return item

            args = [convert_item(arg) for arg in args]
            kwargs = {k: convert_item(arg) for k, arg in kwargs.items()}
            return cached_func(*args, **kwargs)

        return decorated_func

    return decorator


def timer_decorator(func):
    """Time the decorated function and output the result to debug logger."""
    if logger.level != logging.DEBUG:
        return func

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        # Only report function that are significant enough.
        if duration > 1e-3:
            logger.debug(f"{func.__name__} ran in: {duration:.3f} sec")
        return result

    return wrapper


class TimeMeta(type):
    """ Metaclass to record execution time on all methods of the
    child class. """
    def __new__(cls, name, bases, attrs):
        for attr_name, attr_value in attrs.items():
            if callable(attr_value):
                attrs[attr_name] = timer_decorator(attr_value)
        return super().__new__(cls, name, bases, attrs)


# svgsupports
svgsupport = False
try:
    import io
    from svglib.svglib import svg2rlg
    from reportlab.graphics import renderPM

    svgsupport = True
except ImportError:
    pass


def svg_preprocess(inputs: Dict, preprocess: Callable):
    if not inputs:
        return None

    if inputs["image"].startswith("data:image/svg+xml;base64,") and svgsupport:
        svg_data = base64.b64decode(
            inputs["image"].replace("data:image/svg+xml;base64,", "")
        )
        drawing = svg2rlg(io.BytesIO(svg_data))
        png_data = renderPM.drawToString(drawing, fmt="PNG")
        encoded_string = base64.b64encode(png_data)
        base64_str = str(encoded_string, "utf-8")
        base64_str = "data:image/png;base64," + base64_str
        inputs["image"] = base64_str
    return preprocess(inputs)


def get_unique_axis0(data):
    arr = np.asanyarray(data)
    idxs = np.lexsort(arr.T)
    arr = arr[idxs]
    unique_idxs = np.empty(len(arr), dtype=np.bool_)
    unique_idxs[:1] = True
    unique_idxs[1:] = np.any(arr[:-1, :] != arr[1:, :], axis=-1)
    return arr[unique_idxs]


def read_image(img_path: str) -> str:
    """Read image from specified path and return a base64 string."""
    img = cv2.imread(img_path)
    _, bytes = cv2.imencode(".png", img)
    encoded_image = base64.b64encode(bytes).decode("utf-8")
    return encoded_image


def read_image_dir(img_dir: str, suffixes=('.png', '.jpg', '.jpeg', '.webp')) -> List[str]:
    """Try read all images in given img_dir."""
    images = []
    for filename in os.listdir(img_dir):
        if filename.endswith(suffixes):
            img_path = os.path.join(img_dir, filename)
            try:
                images.append(read_image(img_path))
            except IOError:
                logger.error(f"Error opening {img_path}")
    return images


def align_dim_latent(x: int) -> int:
    """ Align the pixel dimension (w/h) to latent dimension.
    Stable diffusion 1:8 ratio for latent/pixel, i.e.,
    1 latent unit == 8 pixel unit."""
    return (x // 8) * 8