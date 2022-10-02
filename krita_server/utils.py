import logging
import os

import yaml
from webui import modules, shared

log = logging.getLogger(__name__)


def load_config():
    """Load default config (including those not exposed in the API yet) from
    `krita_config.yaml` in the current working directory.

    Returns:
        Dict: config
    """
    with open("krita_config.yaml") as file:
        return yaml.safe_load(file)


def save_img(image, sample_path, filename):
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


def fix_aspect_ratio(base_size, max_size, orig_width, orig_height):
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


def collect_prompt(opts, key):
    """Parse prompt/negative prompt keys from `krita_config.yaml`. Is not used for prompt from API.

    Args:
        opts (Dict): Config from `load_config()`.
        key (str): Key containing the prompt to parse.

    Raises:
        SyntaxError: Value of the prompt key cannot be parsed.

    Returns:
        str: Correctly formatted prompt.
    """
    prompts = opts[key]
    if isinstance(prompts, str):
        return prompts
    if isinstance(prompts, list):
        return ", ".join(prompts)
    if isinstance(prompts, dict):
        prompt = ""
        for item, weight in prompts.items():
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
