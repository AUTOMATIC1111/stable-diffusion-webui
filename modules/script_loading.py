import os
import importlib.util

from modules import errors


def load_module(path):
    module_spec = importlib.util.spec_from_file_location(os.path.basename(path), path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)

    return module


def preload_extensions(extensions_dir, parser):
    if not os.path.isdir(extensions_dir):
        return

    for dirname in sorted(os.listdir(extensions_dir)):
        preload_script = os.path.join(extensions_dir, dirname, "preload.py")
        if not os.path.isfile(preload_script):
            continue

        try:
            module = load_module(preload_script)
            if hasattr(module, 'preload'):
                module.preload(parser)

        except Exception:
            errors.report(f"Error running preload() for {preload_script}", exc_info=True)
