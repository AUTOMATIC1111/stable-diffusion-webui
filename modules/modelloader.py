from __future__ import annotations

import importlib
import logging
import os
from typing import TYPE_CHECKING
from urllib.parse import urlparse

import torch

from modules import shared
from modules.upscaler import Upscaler, UpscalerLanczos, UpscalerNearest, UpscalerNone

if TYPE_CHECKING:
    import spandrel

logger = logging.getLogger(__name__)


def load_file_from_url(
    url: str,
    *,
    model_dir: str,
    progress: bool = True,
    file_name: str | None = None,
    hash_prefix: str | None = None,
) -> str:
    """Download a file from `url` into `model_dir`, using the file present if possible.

    Returns the path to the downloaded file.
    """
    os.makedirs(model_dir, exist_ok=True)
    if not file_name:
        parts = urlparse(url)
        file_name = os.path.basename(parts.path)
    cached_file = os.path.abspath(os.path.join(model_dir, file_name))
    if not os.path.exists(cached_file):
        print(f'Downloading: "{url}" to {cached_file}\n')
        from torch.hub import download_url_to_file
        download_url_to_file(url, cached_file, progress=progress, hash_prefix=hash_prefix)
    return cached_file


def load_models(model_path: str, model_url: str = None, command_path: str = None, ext_filter=None, download_name=None, ext_blacklist=None, hash_prefix=None) -> list:
    """
    A one-and done loader to try finding the desired models in specified directories.

    @param download_name: Specify to download from model_url immediately.
    @param model_url: If no other models are found, this will be downloaded on upscale.
    @param model_path: The location to store/find models in.
    @param command_path: A command-line argument to search for models in first.
    @param ext_filter: An optional list of filename extensions to filter by
    @param hash_prefix: the expected sha256 of the model_url
    @return: A list of paths containing the desired model(s)
    """
    output = []

    try:
        places = []

        if command_path is not None and command_path != model_path:
            pretrained_path = os.path.join(command_path, 'experiments/pretrained_models')
            if os.path.exists(pretrained_path):
                print(f"Appending path: {pretrained_path}")
                places.append(pretrained_path)
            elif os.path.exists(command_path):
                places.append(command_path)

        places.append(model_path)

        for place in places:
            for full_path in shared.walk_files(place, allowed_extensions=ext_filter):
                if os.path.islink(full_path) and not os.path.exists(full_path):
                    print(f"Skipping broken symlink: {full_path}")
                    continue
                if ext_blacklist is not None and any(full_path.endswith(x) for x in ext_blacklist):
                    continue
                if full_path not in output:
                    output.append(full_path)

        if model_url is not None and len(output) == 0:
            if download_name is not None:
                output.append(load_file_from_url(model_url, model_dir=places[0], file_name=download_name, hash_prefix=hash_prefix))
            else:
                output.append(model_url)

    except Exception:
        pass

    return output


def friendly_name(file: str):
    if file.startswith("http"):
        file = urlparse(file).path

    file = os.path.basename(file)
    model_name, extension = os.path.splitext(file)
    return model_name


def load_upscalers():
    # We can only do this 'magic' method to dynamically load upscalers if they are referenced,
    # so we'll try to import any _model.py files before looking in __subclasses__
    modules_dir = os.path.join(shared.script_path, "modules")
    for file in os.listdir(modules_dir):
        if "_model.py" in file:
            model_name = file.replace("_model.py", "")
            full_model = f"modules.{model_name}_model"
            try:
                importlib.import_module(full_model)
            except Exception:
                pass

    data = []
    commandline_options = vars(shared.cmd_opts)

    # some of upscaler classes will not go away after reloading their modules, and we'll end
    # up with two copies of those classes. The newest copy will always be the last in the list,
    # so we go from end to beginning and ignore duplicates
    used_classes = {}
    for cls in reversed(Upscaler.__subclasses__()):
        classname = str(cls)
        if classname not in used_classes:
            used_classes[classname] = cls

    for cls in reversed(used_classes.values()):
        name = cls.__name__
        cmd_name = f"{name.lower().replace('upscaler', '')}_models_path"
        commandline_model_path = commandline_options.get(cmd_name, None)
        scaler = cls(commandline_model_path)
        scaler.user_path = commandline_model_path
        scaler.model_download_path = commandline_model_path or scaler.model_path
        data += scaler.scalers

    shared.sd_upscalers = sorted(
        data,
        # Special case for UpscalerNone keeps it at the beginning of the list.
        key=lambda x: x.name.lower() if not isinstance(x.scaler, (UpscalerNone, UpscalerLanczos, UpscalerNearest)) else ""
    )

# None: not loaded, False: failed to load, True: loaded
_spandrel_extra_init_state = None


def _init_spandrel_extra_archs() -> None:
    """
    Try to initialize `spandrel_extra_archs` (exactly once).
    """
    global _spandrel_extra_init_state
    if _spandrel_extra_init_state is not None:
        return

    try:
        import spandrel
        import spandrel_extra_arches
        spandrel.MAIN_REGISTRY.add(*spandrel_extra_arches.EXTRA_REGISTRY)
        _spandrel_extra_init_state = True
    except Exception:
        logger.warning("Failed to load spandrel_extra_arches", exc_info=True)
        _spandrel_extra_init_state = False


def load_spandrel_model(
    path: str | os.PathLike,
    *,
    device: str | torch.device | None,
    prefer_half: bool = False,
    dtype: str | torch.dtype | None = None,
    expected_architecture: str | None = None,
) -> spandrel.ModelDescriptor:
    global _spandrel_extra_init_state

    import spandrel
    _init_spandrel_extra_archs()

    model_descriptor = spandrel.ModelLoader(device=device).load_from_file(str(path))
    arch = model_descriptor.architecture
    if expected_architecture and arch.name != expected_architecture:
        logger.warning(
            f"Model {path!r} is not a {expected_architecture!r} model (got {arch.name!r})",
        )
    half = False
    if prefer_half:
        if model_descriptor.supports_half:
            model_descriptor.model.half()
            half = True
        else:
            logger.info("Model %s does not support half precision, ignoring --half", path)
    if dtype:
        model_descriptor.model.to(dtype=dtype)
    model_descriptor.model.eval()
    logger.debug(
        "Loaded %s from %s (device=%s, half=%s, dtype=%s)",
        arch, path, device, half, dtype,
    )
    return model_descriptor
