import io
import os
import contextlib
import importlib.util
import modules.errors as errors
from installer import setup_logging


preloaded = []


def load_module(path):
    module_spec = importlib.util.spec_from_file_location(os.path.basename(path), path)
    module = importlib.util.module_from_spec(module_spec)
    try:
        if '/sd-extension-' in path: # safe extensions without stdout intercept
            module_spec.loader.exec_module(module)
        else:
            # stdout = io.StringIO()
            with contextlib.redirect_stdout(io.StringIO()) as stdout:
                module_spec.loader.exec_module(module)
            setup_logging() # reset since scripts can hijaack logging
            for line in stdout.getvalue().splitlines():
                if len(line) > 0:
                    errors.log.info(f"Extension: script='{os.path.relpath(path)}' {line.strip()}")
    except Exception as e:
        errors.display(e, f'Module load: {path}')
    return module


def preload_extensions(extensions_dir, parser):
    if not os.path.isdir(extensions_dir):
        return
    for dirname in sorted(os.listdir(extensions_dir)):
        if dirname in preloaded:
            continue
        preloaded.append(dirname)
        preload_script = os.path.join(extensions_dir, dirname, "preload.py")
        if not os.path.isfile(preload_script):
            continue
        try:
            module = load_module(preload_script)
            if hasattr(module, 'preload'):
                module.preload(parser)
        except Exception as e:
            errors.display(e, f'Extension preload: {preload_script}')
