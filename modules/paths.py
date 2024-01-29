import os
import sys
from modules.paths_internal import models_path, script_path, data_path, extensions_dir, extensions_builtin_dir, cwd  # noqa: F401

import modules.safe  # noqa: F401


def mute_sdxl_imports():
    """create fake modules that SDXL wants to import but doesn't actually use for our purposes"""

    class Dummy:
        pass

    module = Dummy()
    module.LPIPS = None
    sys.modules['taming.modules.losses.lpips'] = module

    module = Dummy()
    module.StableDataModuleFromConfig = None
    sys.modules['sgm.data'] = module


# data_path = cmd_opts_pre.data
sys.path.insert(0, script_path)

# search for directory of stable diffusion in following places
sd_path = None
possible_sd_paths = [os.path.join(script_path, 'repositories/stable-diffusion-stability-ai'), '.', os.path.dirname(script_path)]
for possible_sd_path in possible_sd_paths:
    if os.path.exists(os.path.join(possible_sd_path, 'ldm/models/diffusion/ddpm.py')):
        sd_path = os.path.abspath(possible_sd_path)
        break

assert sd_path is not None, f"Couldn't find Stable Diffusion in any of: {possible_sd_paths}"

mute_sdxl_imports()

path_dirs = [
    (sd_path, 'ldm', 'Stable Diffusion', []),
    (os.path.join(sd_path, '../generative-models'), 'sgm', 'Stable Diffusion XL', ["sgm"]),
    (os.path.join(sd_path, '../CodeFormer'), 'inference_codeformer.py', 'CodeFormer', []),
    (os.path.join(sd_path, '../BLIP'), 'models/blip.py', 'BLIP', []),
    (os.path.join(sd_path, '../k-diffusion'), 'k_diffusion/sampling.py', 'k_diffusion', ["atstart"]),
]

paths = {}

for d, must_exist, what, options in path_dirs:
    must_exist_path = os.path.abspath(os.path.join(script_path, d, must_exist))
    if not os.path.exists(must_exist_path):
        print(f"Warning: {what} not found at path {must_exist_path}", file=sys.stderr)
    else:
        d = os.path.abspath(d)
        if "atstart" in options:
            sys.path.insert(0, d)
        elif "sgm" in options:
            # Stable Diffusion XL repo has scripts dir with __init__.py in it which ruins every extension's scripts dir, so we
            # import sgm and remove it from sys.path so that when a script imports scripts.something, it doesbn't use sgm's scripts dir.

            sys.path.insert(0, d)
            import sgm  # noqa: F401
            sys.path.pop(0)
        else:
            sys.path.append(d)
        paths[what] = d
