import os

from gradio_client.client import DEFAULT_TEMP_DIR

from modules import shared


def cleanup_tmpdr():
    temp_dir = shared.opts.temp_dir or DEFAULT_TEMP_DIR
    if temp_dir == "" or not os.path.isdir(temp_dir):
        return

    for root, _, files in os.walk(temp_dir, topdown=False):
        for name in files:
            _, extension = os.path.splitext(name)
            if extension != ".png":
                continue

            filename = os.path.join(root, name)
            os.remove(filename)
