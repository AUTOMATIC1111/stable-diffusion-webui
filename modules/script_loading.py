import os
import sys
import traceback
from types import ModuleType


def load_module(path):
    with open(path, "r", encoding="utf8") as file:
        text = file.read()

    compiled = compile(text, path, 'exec')
    module = ModuleType(os.path.basename(path))
    exec(compiled, module.__dict__)

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
            print(f"Error running preload() for {preload_script}", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)
