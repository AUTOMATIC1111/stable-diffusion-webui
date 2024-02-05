import os
import tempfile
from collections import namedtuple
from pathlib import Path

import gradio.components

from PIL import PngImagePlugin

from modules import shared


Savedfile = namedtuple("Savedfile", ["name"])


def register_tmp_file(gradio, filename):
    if hasattr(gradio, 'temp_file_sets'):  # gradio 3.15
        gradio.temp_file_sets[0] = gradio.temp_file_sets[0] | {os.path.abspath(filename)}

    if hasattr(gradio, 'temp_dirs'):  # gradio 3.9
        gradio.temp_dirs = gradio.temp_dirs | {os.path.abspath(os.path.dirname(filename))}


def check_tmp_file(gradio, filename):
    if hasattr(gradio, 'temp_file_sets'):
        return any(filename in fileset for fileset in gradio.temp_file_sets)

    if hasattr(gradio, 'temp_dirs'):
        return any(Path(temp_dir).resolve() in Path(filename).resolve().parents for temp_dir in gradio.temp_dirs)

    return False


def save_pil_to_file(pil_image, cache_dir=None, format="png"):
    already_saved_as = getattr(pil_image, 'already_saved_as', None)
    if already_saved_as and os.path.isfile(already_saved_as):
        register_tmp_file(shared.demo, already_saved_as)
        filename = already_saved_as

        if not shared.opts.save_images_add_number:
            filename += f'?{os.path.getmtime(already_saved_as)}'

        return filename

    if shared.opts.temp_dir:
        dir = shared.opts.temp_dir
    else:
        dir = cache_dir
        os.makedirs(dir, exist_ok=True)

    use_metadata = False
    metadata = PngImagePlugin.PngInfo()
    for key, value in pil_image.info.items():
        if isinstance(key, str) and isinstance(value, str):
            metadata.add_text(key, value)
            use_metadata = True

    file_obj = tempfile.NamedTemporaryFile(delete=False, suffix=".png", dir=dir)
    pil_image.save(file_obj, pnginfo=(metadata if use_metadata else None))
    return file_obj.name


def move_files_to_cache(data, block, postprocess=False):
    """Move files to cache and replace the file path with the cache path.

    Runs after postprocess and before preprocess.

    Copied from gradio's processing_utils.py

    Args:
        data: The input or output data for a component. Can be a dictionary or a dataclass
        block: The component
        postprocess: Whether its running from postprocessing
    """

    from gradio import FileData
    from gradio.processing_utils import move_resource_to_block_cache
    from gradio.data_classes import GradioRootModel
    from gradio.data_classes import GradioModel
    from gradio_client import utils as client_utils

    def _move_to_cache(d: dict):
        payload = FileData(**d)
        # If the gradio app developer is returning a URL from
        # postprocess, it means the component can display a URL
        # without it being served from the gradio server
        # This makes it so that the URL is not downloaded and speeds up event processing

        payload.path = payload.path.rsplit('?', 1)[0]

        if payload.url and postprocess:
            temp_file_path = payload.url
        else:

            if check_tmp_file(shared.demo, payload.path):
                temp_file_path = payload.path
            else:
                temp_file_path = move_resource_to_block_cache(payload.path, block)
        assert temp_file_path is not None
        payload.path = temp_file_path
        return payload.model_dump()

    if isinstance(data, (GradioRootModel, GradioModel)):
        data = data.model_dump()

    return client_utils.traverse(data, _move_to_cache, client_utils.is_file_obj)


def install_ui_tempdir_override():
    """
    override save to file function so that it also writes PNG info.
    override gradio4's move_files_to_cache function to prevent it from writing a copy into a temporary directory.
    """

    gradio.processing_utils.save_pil_to_cache = save_pil_to_file
    gradio.processing_utils.move_files_to_cache = move_files_to_cache


def on_tmpdir_changed():
    if shared.opts.temp_dir == "" or shared.demo is None:
        return

    os.makedirs(shared.opts.temp_dir, exist_ok=True)

    register_tmp_file(shared.demo, os.path.join(shared.opts.temp_dir, "x"))


def cleanup_tmpdr():
    temp_dir = shared.opts.temp_dir
    if temp_dir == "" or not os.path.isdir(temp_dir):
        return

    for root, _, files in os.walk(temp_dir, topdown=False):
        for name in files:
            _, extension = os.path.splitext(name)
            if extension != ".png":
                continue

            filename = os.path.join(root, name)
            os.remove(filename)
