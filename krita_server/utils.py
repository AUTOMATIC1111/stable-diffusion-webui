from __future__ import annotations

import inspect
import logging
import os

import yaml
from PIL import Image
from pydantic import BaseModel
from webui import modules, shared

from .config import MainConfig

log = logging.getLogger(__name__)

CONFIG_PATH = "krita_config.yaml"


def load_config():
    """Load default config (including those not exposed in the API yet) from
    `krita_config.yaml` in the current working directory.

    Will create `krita_config.yaml` if it has yet to exist using `MainConfig` from
    `config.py`.

    Returns:
        MainConfig: config
    """
    if not os.path.isfile(CONFIG_PATH):
        cfg = MainConfig()
        with open(CONFIG_PATH, "w") as f:
            yaml.safe_dump(cfg.dict(), f)

    with open(CONFIG_PATH) as file:
        obj = yaml.safe_load(file)
        return MainConfig.parse_obj(obj)


def merge_default_config(config: BaseModel, default: BaseModel):
    """Replace unset and None fields in opt with values from default with the
    same field name in place.

    Unset fields does not include fields that are explicitly set to None but
    includes fields with a default value due to being unset.

    Args:
        config (BaseModel): Config object.
        default (BaseModel): Default to merge from.

    Returns:
        BaseModel: Modified config.
    """

    for field in config.__fields__:
        if not field in config.__fields_set__ or field is None:
            setattr(config, field, getattr(default, field))

    return config


def optional(*fields):
    """Decorator function used to modify a pydantic model's fields to all be optional.
    Alternatively, you can  also pass the field names that should be made optional as arguments
    to the decorator.
    Taken from https://github.com/samuelcolvin/pydantic/issues/1223#issuecomment-775363074
    """

    def dec(_cls):
        for field in fields:
            _cls.__fields__[field].required = False
        return _cls

    if fields and inspect.isclass(fields[0]) and issubclass(fields[0], BaseModel):
        cls = fields[0]
        fields = cls.__fields__
        return dec(cls)

    return dec


def save_img(image: Image, sample_path: str, filename: str):
    """Saves an image.

    Args:
        image (PIL.Image): Image to save.
        sample_path (str): Folder to save the image in.
        filename (str): Name to save the image as.

    Returns:
        str: Absolute path where the image was saved.
    """
    path = os.path.join(sample_path, filename)
    image.save(path)
    return os.path.abspath(path)


def fix_aspect_ratio(base_size: int, max_size: int, orig_width: int, orig_height: int):
    """Calculate an appropiate image resolution given the base input size of the
    model and max input size allowed.

    The max input size is due to how Stable Diffusion currently handles resolutions
    larger than its base/native input size of 512, which can cause weird issues
    such as duplicated features in the image. Hence, it is typically better to
    render at a smaller appropiate resolution before using other methods to upscale
    to the original resolution.

    Stable Diffusion also messes up for resolutions smaller than 512. In which case,
    it is better to render at the base resolution before downscaling to the original.

    Args:
        base_size (int): Native/base input size of the model.
        max_size (int): Bax input size to accept.
        orig_width (int): Original width requested.
        orig_height (int): Original height requested.

    Returns:
        Tuple[int, int]: Appropiate (width, height) to use for the model.
    """

    def rnd(r, x):
        z = 64
        return z * round(r * x / z)

    ratio = orig_width / orig_height

    if orig_width > orig_height:
        width, height = rnd(ratio, base_size), base_size
        if width > max_size:
            width, height = max_size, rnd(1 / ratio, max_size)
    else:
        width, height = base_size, rnd(1 / ratio, base_size)
        if height > max_size:
            width, height = rnd(ratio, max_size), max_size

    new_ratio = width / height

    log.info(
        f"img size: {orig_width}x{orig_height} -> {width}x{height}, "
        f"aspect ratio: {ratio:.2f} -> {new_ratio:.2f}, {100 * (new_ratio - ratio) / ratio :.2f}% change"
    )
    return width, height


def parse_prompt(val):
    """Parse different representations of prompt/negative prompt.

    Args:
        val (Any): Key containing the prompt to parse.

    Raises:
        SyntaxError: Value of the prompt key cannot be parsed.

    Returns:
        str: Correctly formatted prompt.
    """
    if isinstance(val, str):
        return val
    if isinstance(val, list):
        return ", ".join(val)
    if isinstance(val, dict):
        prompt = ""
        for item, weight in val.items():
            if not prompt == "":
                prompt += " "
            if weight is None:
                prompt += f"{item}"
            else:
                prompt += f"({item}:{weight})"
        return prompt
    raise SyntaxError("prompt field in krita_config.yml is invalid")


def get_sampler_index(sampler_name: str):
    """Get index of sampler by name.

    Args:
        sampler_name (str): Exact name of sampler.

    Raises:
        KeyError: Sampler cannot be found.

    Returns:
        int: Index of sampler.
    """
    for index, sampler in enumerate(modules.sd_samplers.samplers):
        name, constructor, aliases = sampler
        if sampler_name == name or sampler_name in aliases:
            return index
    raise KeyError(f"sampler not found: {sampler_name}")


def get_upscaler_index(upscaler_name: str):
    """Get index of upscaler by name.

    Args:
        upscaler_name (str): Exact name of upscaler.

    Raises:
        KeyError: Upscaler cannot be found.

    Returns:
        int: Index of sampler.
    """
    for index, upscaler in enumerate(shared.sd_upscalers):
        if upscaler.name == upscaler_name:
            return index
    raise KeyError(f"upscaler not found: {upscaler_name}")


def set_face_restorer(face_restorer: str, codeformer_weight: float):
    """Change which face restorer to use.

    Args:
        face_restorer (str): Exact name of face restorer to use.
        codeformer_weight (float): Strength of face restoration when using CodeFormer.
    """
    # the `shared` module handles app state for the underlying codebase
    shared.opts.face_restoration_model = face_restorer
    shared.opts.code_former_weight = codeformer_weight
