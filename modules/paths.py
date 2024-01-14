# this module must not have any dependencies as it is a very first import before webui even starts
import os
import sys
import json
import argparse
from modules.errors import log


# parse args, parse again after we have the data-dir and early-read the config file
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--ckpt", type=str, default=os.environ.get("SD_MODEL", None), help="Path to model checkpoint to load immediately, default: %(default)s")
parser.add_argument("--data-dir", type=str, default=os.environ.get("SD_DATADIR", ''), help="Base path where all user data is stored, default: %(default)s")
parser.add_argument("--models-dir", type=str, default=os.environ.get("SD_MODELSDIR", None), help="Base path where all models are stored, default: %(default)s",)
cli = parser.parse_known_args()[0]
parser.add_argument("--config", type=str, default=os.environ.get("SD_CONFIG", os.path.join(cli.data_dir, 'config.json')), help="Use specific server configuration file, default: %(default)s")
cli = parser.parse_known_args()[0]
config_path = cli.config if os.path.isabs(cli.config) else os.path.join(cli.data_dir, cli.config)
try:
    with open(config_path, 'r', encoding='utf8') as f:
        config = json.load(f)
except Exception:
    config = {}

modules_path = os.path.dirname(os.path.realpath(__file__))
script_path = os.path.dirname(modules_path)
data_path = cli.data_dir
models_config = cli.models_dir or config.get('models_dir') or 'models'
models_path = models_config if os.path.isabs(models_config) else os.path.join(data_path, models_config)
extensions_dir = os.path.join(data_path, "extensions")
extensions_builtin_dir = "extensions-builtin"
sd_configs_path = os.path.join(script_path, "configs")
sd_default_config = os.path.join(sd_configs_path, "v1-inference.yaml")
sd_model_file = cli.ckpt or os.path.join(script_path, 'model.ckpt') # not used
default_sd_model_file = sd_model_file # not used
debug = log.trace if os.environ.get('SD_PATH_DEBUG', None) is not None else lambda *args, **kwargs: None
debug('Trace: PATH')
paths = {}

if os.environ.get('SD_PATH_DEBUG', None) is not None:
    print(f'Paths: script-path="{script_path}" data-dir="{data_path}" models-dir="{models_path}" config="{config_path}"')


def register_paths():
    log.debug('Register paths')
    sys.path.insert(0, script_path)
    sd_path = os.path.join(script_path, 'repositories')
    path_dirs = [
        (sd_path, 'ldm', 'ldm', []),
        (sd_path, 'taming', 'Taming Transformers', []),
        (os.path.join(sd_path, 'blip'), 'models/blip.py', 'BLIP', []),
        (os.path.join(sd_path, 'codeformer'), 'inference_codeformer.py', 'CodeFormer', []),
        (os.path.join(modules_path, 'k-diffusion'), 'k_diffusion/sampling.py', 'k_diffusion', ["atstart"]),
    ]
    for d, must_exist, what, _options in path_dirs:
        must_exist_path = os.path.abspath(os.path.join(script_path, d, must_exist))
        if not os.path.exists(must_exist_path):
            log.error(f'Required path not found: path={must_exist_path} item={what}')
        else:
            d = os.path.abspath(d)
            sys.path.append(d)
            paths[what] = d


def create_path(folder):
    if folder is None or folder == '':
        return
    if os.path.exists(folder):
        return
    try:
        os.makedirs(folder, exist_ok=True)
        log.info(f'Create: folder="{folder}"')
    except Exception as e:
        log.error(f'Create failed: folder="{folder}" {e}')


def create_paths(opts):
    def fix_path(folder):
        tgt = opts.data.get(folder, None) or opts.data_labels[folder].default
        if tgt is None or tgt == '':
            return tgt
        fix = tgt
        if not os.path.isabs(tgt) and len(data_path) > 0 and not tgt.startswith(data_path): # path is already relative to data_path
            fix = os.path.join(data_path, fix)
        if fix.startswith('..'):
            fix = os.path.abspath(fix)
        fix = fix if os.path.isabs(fix) else os.path.relpath(fix, script_path)
        opts.data[folder] = fix
        debug(f'Paths: folder="{folder}" original="{tgt}" target="{fix}"')
        return opts.data[folder]

    create_path(data_path)
    create_path(script_path)
    create_path(models_path)
    create_path(sd_configs_path)
    create_path(extensions_dir)
    create_path(extensions_builtin_dir)
    create_path(fix_path('temp_dir'))
    create_path(fix_path('ckpt_dir'))
    create_path(fix_path('diffusers_dir'))
    create_path(fix_path('vae_dir'))
    create_path(fix_path('lora_dir'))
    create_path(fix_path('embeddings_dir'))
    create_path(fix_path('hypernetwork_dir'))
    create_path(fix_path('outdir_samples'))
    create_path(fix_path('outdir_txt2img_samples'))
    create_path(fix_path('outdir_img2img_samples'))
    create_path(fix_path('outdir_control_samples'))
    create_path(fix_path('outdir_extras_samples'))
    create_path(fix_path('outdir_init_images'))
    create_path(fix_path('outdir_grids'))
    create_path(fix_path('outdir_txt2img_grids'))
    create_path(fix_path('outdir_img2img_grids'))
    create_path(fix_path('outdir_control_grids'))
    create_path(fix_path('outdir_save'))
    create_path(fix_path('outdir_video'))
    create_path(fix_path('styles_dir'))


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
