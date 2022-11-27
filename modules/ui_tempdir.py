import os
import tempfile
from collections import namedtuple

import gradio as gr

from PIL import PngImagePlugin

from modules import shared


Savedfile = namedtuple("Savedfile", ["name"])


def save_pil_to_file(pil_image, dir=None):
    already_saved_as = getattr(pil_image, 'already_saved_as', None)
    if already_saved_as:
        shared.demo.temp_dirs = shared.demo.temp_dirs | {os.path.abspath(os.path.dirname(already_saved_as))}
        file_obj = Savedfile(already_saved_as)
        return file_obj

    if shared.opts.temp_dir != "":
        dir = shared.opts.temp_dir

    use_metadata = False
    metadata = PngImagePlugin.PngInfo()
    for key, value in pil_image.info.items():
        if isinstance(key, str) and isinstance(value, str):
            metadata.add_text(key, value)
            use_metadata = True

    file_obj = tempfile.NamedTemporaryFile(delete=False, suffix=".png", dir=dir)
    pil_image.save(file_obj, pnginfo=(metadata if use_metadata else None))
    return file_obj


# override save to file function so that it also writes PNG info
gr.processing_utils.save_pil_to_file = save_pil_to_file


def on_tmpdir_changed():
    if shared.opts.temp_dir == "" or shared.demo is None:
        return

    os.makedirs(shared.opts.temp_dir, exist_ok=True)

    shared.demo.temp_dirs = shared.demo.temp_dirs | {os.path.abspath(shared.opts.temp_dir)}


def cleanup_tmpdr():
    temp_dir = shared.opts.temp_dir
    if temp_dir == "" or not os.path.isdir(temp_dir):
        return

    for root, dirs, files in os.walk(temp_dir, topdown=False):
        for name in files:
            _, extension = os.path.splitext(name)
            if extension != ".png":
                continue

            filename = os.path.join(root, name)
            os.remove(filename)
