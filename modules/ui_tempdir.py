import os
import tempfile
from collections import namedtuple
from pathlib import Path

import gradio.components
import gradio as gr

from PIL import PngImagePlugin

from modules import shared


Savedfile = namedtuple("Savedfile", ["name"])


def register_tmp_file(gradio_app, filename):
    if hasattr(gradio_app, 'temp_file_sets'):  # gradio 3.15
        if hasattr(gr.utils, 'abspath'):  # gradio 4.19
            filename = gr.utils.abspath(filename)
        else:
            filename = os.path.abspath(filename)

        gradio_app.temp_file_sets[0] = gradio_app.temp_file_sets[0] | {filename}

    if hasattr(gradio_app, 'temp_dirs'):  # gradio 3.9
        gradio_app.temp_dirs = gradio_app.temp_dirs | {os.path.abspath(os.path.dirname(filename))}


def check_tmp_file(gradio_app, filename):
    if hasattr(gradio_app, 'temp_file_sets'):
        if hasattr(gr.utils, 'abspath'):  # gradio 4.19
            filename = gr.utils.abspath(filename)
        else:
            filename = os.path.abspath(filename)

        return any(filename in fileset for fileset in gradio_app.temp_file_sets)

    if hasattr(gradio_app, 'temp_dirs'):
        return any(Path(temp_dir).resolve() in Path(filename).resolve().parents for temp_dir in gradio_app.temp_dirs)

    return False


def save_pil_to_file(pil_image, cache_dir=None, format="png"):
    already_saved_as = getattr(pil_image, 'already_saved_as', None)
    if already_saved_as and os.path.isfile(already_saved_as):
        register_tmp_file(shared.demo, already_saved_as)
        filename_with_mtime = f'{already_saved_as}?{os.path.getmtime(already_saved_as)}'
        register_tmp_file(shared.demo, filename_with_mtime)
        return filename_with_mtime

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


async def async_move_files_to_cache(data, block, postprocess=False, check_in_upload_folder=False, keep_in_cache=False):
    """Move any files in `data` to cache and (optionally), adds URL prefixes (/file=...) needed to access the cached file.
    Also handles the case where the file is on an external Gradio app (/proxy=...).

    Runs after .postprocess() and before .preprocess().

    Copied from gradio's processing_utils.py

    Args:
        data: The input or output data for a component. Can be a dictionary or a dataclass
        block: The component whose data is being processed
        postprocess: Whether its running from postprocessing
        check_in_upload_folder: If True, instead of moving the file to cache, checks if the file is in already in cache (exception if not).
        keep_in_cache: If True, the file will not be deleted from cache when the server is shut down.
    """

    from gradio import FileData
    from gradio.data_classes import GradioRootModel
    from gradio.data_classes import GradioModel
    from gradio_client import utils as client_utils
    from gradio.utils import get_upload_folder, is_in_or_equal, is_static_file

    async def _move_to_cache(d: dict):
        payload = FileData(**d)

        # EDITED
        payload.path = payload.path.rsplit('?', 1)[0]

        # If the gradio app developer is returning a URL from
        # postprocess, it means the component can display a URL
        # without it being served from the gradio server
        # This makes it so that the URL is not downloaded and speeds up event processing
        if payload.url and postprocess and client_utils.is_http_url_like(payload.url):
            payload.path = payload.url
        elif is_static_file(payload):
            pass
        elif not block.proxy_url:
            # EDITED
            if check_tmp_file(shared.demo, payload.path):
                temp_file_path = payload.path
            else:
                # If the file is on a remote server, do not move it to cache.
                if check_in_upload_folder and not client_utils.is_http_url_like(
                    payload.path
                ):
                    path = os.path.abspath(payload.path)
                    if not is_in_or_equal(path, get_upload_folder()):
                        raise ValueError(
                            f"File {path} is not in the upload folder and cannot be accessed."
                        )
                if not payload.is_stream:
                    temp_file_path = await block.async_move_resource_to_block_cache(
                        payload.path
                    )
                    if temp_file_path is None:
                        raise ValueError("Did not determine a file path for the resource.")
                    payload.path = temp_file_path
                    if keep_in_cache:
                        block.keep_in_cache.add(payload.path)

        url_prefix = "/stream/" if payload.is_stream else "/file="
        if block.proxy_url:
            proxy_url = block.proxy_url.rstrip("/")
            url = f"/proxy={proxy_url}{url_prefix}{payload.path}"
        elif client_utils.is_http_url_like(payload.path) or payload.path.startswith(
            f"{url_prefix}"
        ):
            url = payload.path
        else:
            url = f"{url_prefix}{payload.path}"
        payload.url = url

        return payload.model_dump()

    if isinstance(data, (GradioRootModel, GradioModel)):
        data = data.model_dump()

    return await client_utils.async_traverse(
        data, _move_to_cache, client_utils.is_file_obj
    )


def install_ui_tempdir_override():
    """
    override save to file function so that it also writes PNG info.
    override gradio4's move_files_to_cache function to prevent it from writing a copy into a temporary directory.
    """

    gradio.processing_utils.save_pil_to_cache = save_pil_to_file
    gradio.processing_utils.async_move_files_to_cache = async_move_files_to_cache


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


def is_gradio_temp_path(path):
    """
    Check if the path is a temp dir used by gradio
    """
    path = Path(path)
    if shared.opts.temp_dir and path.is_relative_to(shared.opts.temp_dir):
        return True
    if gradio_temp_dir := os.environ.get("GRADIO_TEMP_DIR"):
        if path.is_relative_to(gradio_temp_dir):
            return True
    if path.is_relative_to(Path(tempfile.gettempdir()) / "gradio"):
        return True
    return False
