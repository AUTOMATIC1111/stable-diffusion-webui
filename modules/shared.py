import logging
import os
import sys

import gradio as gr

from modules import shared_cmd_options, shared_gradio_themes, options, shared_items, sd_models_types
from modules.paths_internal import models_path, script_path, data_path, sd_configs_path, sd_default_config, sd_model_file, default_sd_model_file, extensions_dir, extensions_builtin_dir  # noqa: F401
from modules import util
from collections import OrderedDict


class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = OrderedDict()

    # def get(self, model):
    #     if model in self.cache:
    #         # Move the model to the end to indicate it was recently used
    #         self.cache.move_to_end(model)
    #         return True
    #     else:
    #         return False

    def put(self, model):
        if model in self.cache:
            # If the model is already in the cache, move it to the end
            self.cache.move_to_end(model)
        elif len(self.cache) >= self.capacity:
            if self.capacity is 0:
                return None
            # If the cache is full, remove the first (least recently used) item
            return self.cache.popitem(last=False)
        # Add the model to the cache
        self.cache[model] = None
        return None

    def evict(self):
        if self.capacity is 0:
            return None
        return self.cache.popitem(last=False)


cmd_opts = shared_cmd_options.cmd_opts
parser = shared_cmd_options.parser

batch_cond_uncond = True  # old field, unused now in favor of shared.opts.batch_cond_uncond
parallel_processing_allowed = True
styles_filename = cmd_opts.styles_file
config_filename = cmd_opts.ui_settings_file
hide_dirs = {"visible": not cmd_opts.hide_ui_dir_config}
default_path = '/stable-diffusion-webui/cache/'
ram_lru_model_cache = LRUCache(int(os.environ.get('max_models_on_ram', 0)))
local_storage_lru_model_cache = LRUCache(
    int(os.environ.get('max_models_on_local', 0)))
runpod_volume_lru_model_cache = LRUCache(
    int(os.environ.get('max_models_on_runpod', 0)))
model_name_state_dict_map = {}
demo = None

device = None

weight_load_location = None

xformers_available = False

hypernetworks = {}

loaded_hypernetworks = []

state = None

prompt_styles = None

interrogator = None

face_restorers = []

options_templates = None
opts = None
restricted_opts = None

sd_model: sd_models_types.WebuiSdModel = None

settings_components = None
"""assinged from ui.py, a mapping on setting names to gradio components repsponsible for those settings"""

tab_names = []

latent_upscale_default_mode = "Latent"
latent_upscale_modes = {
    "Latent": {"mode": "bilinear", "antialias": False},
    "Latent (antialiased)": {"mode": "bilinear", "antialias": True},
    "Latent (bicubic)": {"mode": "bicubic", "antialias": False},
    "Latent (bicubic antialiased)": {"mode": "bicubic", "antialias": True},
    "Latent (nearest)": {"mode": "nearest", "antialias": False},
    "Latent (nearest-exact)": {"mode": "nearest-exact", "antialias": False},
}

sd_upscalers = []

clip_model = None

progress_print_out = sys.stdout

gradio_theme = gr.themes.Base()

total_tqdm = None

mem_mon = None

options_section = options.options_section
OptionInfo = options.OptionInfo
OptionHTML = options.OptionHTML

natural_sort_key = util.natural_sort_key
listfiles = util.listfiles
html_path = util.html_path
html = util.html
walk_files = util.walk_files
ldm_print = util.ldm_print

reload_gradio_theme = shared_gradio_themes.reload_gradio_theme

list_checkpoint_tiles = shared_items.list_checkpoint_tiles
refresh_checkpoints = shared_items.refresh_checkpoints
list_samplers = shared_items.list_samplers
reload_hypernetworks = shared_items.reload_hypernetworks

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    stream_handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(stream_handler)
