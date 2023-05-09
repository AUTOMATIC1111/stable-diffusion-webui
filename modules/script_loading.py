import os
import sys
import traceback
import importlib.util
from types import ModuleType
from pathlib import Path


def load_module(path):
    module_spec = importlib.util.spec_from_file_location(os.path.basename(path), path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)

    return module


def handle_preload_script(preload_script, parser):
    try:
        module = load_module(preload_script)
        if hasattr(module, 'preload'):
            module.preload(parser)

    except Exception:
        print(f"Error running preload() for {preload_script}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)


def preload_extensions(extensions_dir, parser):
    if not os.path.isdir(extensions_dir):
        return

    for dirname in sorted(os.listdir(extensions_dir)):
        preload_script = os.path.join(extensions_dir, dirname, "preload.py")
        if not os.path.isfile(preload_script):
            continue

        handle_preload_script(preload_script, parser)


def preload_local_extensions(parser, local_extensions):
    for extension in local_extensions:
        if extension.is_dir():
            preload_script = extension.joinpath("preload.py")

            if not preload_script.is_file():
                continue

            handle_preload_script(str(preload_script), parser)
