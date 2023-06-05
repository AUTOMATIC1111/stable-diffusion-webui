import os
import sys
import modules.paths_internal

data_path = modules.paths_internal.data_path
script_path = modules.paths_internal.script_path
models_path = modules.paths_internal.models_path
sd_configs_path = modules.paths_internal.sd_configs_path
sd_default_config = modules.paths_internal.sd_default_config
sd_model_file = modules.paths_internal.sd_model_file
default_sd_model_file = modules.paths_internal.default_sd_model_file
extensions_dir = modules.paths_internal.extensions_dir
extensions_builtin_dir = modules.paths_internal.extensions_builtin_dir

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

path_dirs = [
    (sd_path, 'ldm', 'Stable Diffusion', []),
    (os.path.join(sd_path, '../taming-transformers'), 'taming', 'Taming Transformers', []),
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
        # if "atstart" in options:
        #    sys.path.insert(0, d)
        # else:
        #    sys.path.append(d)
        sys.path.append(d)
        paths[what] = d


def create_paths(opts):
    def create_path(folder):
        if folder is None or folder == '':
            return
        if not os.path.exists(folder):
            try:
                os.makedirs(folder, exist_ok=True)
                # print('Creating folder:', folder)
            except:
                pass

    def fix_path(folder):
        if opts.data[folder] is None or opts.data[folder] == '':
            return opts.data[folder]
        if os.path.isabs(opts.data[folder]):
            return opts.data[folder]
        if opts.data[folder].startswith(data_path):
            return opts.data[folder]
        opts.data[folder] = os.path.join(data_path, opts.data[folder])
        return opts.data[folder]

    create_path(fix_path('temp_dir'))
    create_path(extensions_dir)
    create_path(extensions_builtin_dir)
    create_path(fix_path('hypernetwork_dir'))
    create_path(fix_path('ckpt_dir'))
    create_path(fix_path('vae_dir'))
    create_path(fix_path('embeddings_dir'))
    create_path(fix_path('outdir_samples'))
    create_path(fix_path('outdir_txt2img_samples'))
    create_path(fix_path('outdir_img2img_samples'))
    create_path(fix_path('outdir_extras_samples'))
    create_path(fix_path('outdir_grids'))
    create_path(fix_path('outdir_txt2img_grids'))
    create_path(fix_path('outdir_img2img_grids'))
    create_path(fix_path('outdir_save'))


class Prioritize:
    def __init__(self, name):
        self.name = name
        self.path = None

    def __enter__(self):
        self.path = sys.path.copy()
        sys.path = [paths[self.name]] + sys.path

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.path = self.path
        self.path = None
