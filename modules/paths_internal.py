# no longer used, all paths are defined in paths.py

from modules.paths import modules_path, script_path, sd_configs_path, sd_default_config, sd_model_file, default_sd_model_file, data_path, models_path, extensions_dir, extensions_builtin_dir # pylint: disable=unused-import

"""
import argparse
import os

modules_path = os.path.dirname(os.path.realpath(__file__))
script_path = os.path.dirname(modules_path)
sd_configs_path = os.path.join(script_path, "configs")
sd_default_config = os.path.join(sd_configs_path, "v1-inference.yaml")

# Parse the --data-dir flag first so we can use it as a base for our other argument default values
parser_pre = argparse.ArgumentParser(add_help=False)
parser_pre.add_argument("--ckpt", type=str, default=os.environ.get("SD_MODEL", None), help="Path to model checkpoint to load immediately, default: %(default)s")
parser_pre.add_argument("--data-dir", type=str, default=os.environ.get("SD_DATADIR", ''), help="Base path where all user data is stored, default: %(default)s")
parser_pre.add_argument("--models-dir", type=str, default=os.environ.get("SD_MODELSDIR", 'models'), help="Base path where all models are stored, default: %(default)s",)
cmd_opts_pre = parser_pre.parse_known_args()[0]

# parser_pre.add_argument("--config", type=str, default=os.environ.get("SD_CONFIG", os.path.join(data_path, 'config.json')), help="Use specific server configuration file, default: %(default)s")

data_path = cmd_opts_pre.data_dir
models_path = cmd_opts_pre.models_dir if os.path.isabs(cmd_opts_pre.models_dir) else os.path.join(data_path, cmd_opts_pre.models_dir)
extensions_dir = os.path.join(data_path, "extensions")
extensions_builtin_dir = "extensions-builtin"

sd_model_file = cmd_opts_pre.ckpt or os.path.join(script_path, 'model.ckpt') # not used
default_sd_model_file = sd_model_file # not used
"""
