#!/bin/env python
"""
print module versions
"""

import importlib
import pkg_resources

modules = [
    'diffusers', 'xformers', 'tokenizers', 'accelerate', 'safetensors'
]

def get_torch():
    try:
        torch = importlib.import_module('torch')
        print('torch:', { 'version': torch.__version__ })
        print('cuda:', { 'available': torch.cuda.is_available(), 'version': torch.version.cuda, 'arch': torch.cuda.get_arch_list() })
        print('device:', { 'name': torch.cuda.get_device_name(torch.cuda.current_device()) })
    except Exception as err:
        print('torch:', { 'error': err })


def version(name: str):
    try:
        ver = pkg_resources.get_distribution(name).version
        print(f"{name}: {ver}")
    except Exception as err:
        print(f"{name} error: {err}")

if __name__ == "__main__": # create & train test embedding when used from cli
    get_torch()
    for module in modules:
        version(module)
