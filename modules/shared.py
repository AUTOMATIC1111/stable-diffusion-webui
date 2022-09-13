import sys
import argparse
import json
import os

import gradio as gr
import torch
import tqdm

import modules.artists
from modules.paths import script_path, sd_path
from modules.devices import get_optimal_device
import modules.styles
import modules.interrogate

sd_model_file = os.path.join(script_path, 'model.ckpt')
if not os.path.exists(sd_model_file):
    sd_model_file = "models/ldm/stable-diffusion-v1/model.ckpt"

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default=os.path.join(sd_path, "configs/stable-diffusion/v1-inference.yaml"), help="path to config which constructs model",)
parser.add_argument("--ckpt", type=str, default=os.path.join(sd_path, sd_model_file), help="path to checkpoint of model",)
parser.add_argument("--gfpgan-dir", type=str, help="GFPGAN directory", default=('./src/gfpgan' if os.path.exists('./src/gfpgan') else './GFPGAN'))
parser.add_argument("--gfpgan-model", type=str, help="GFPGAN model file name", default='GFPGANv1.3.pth')
parser.add_argument("--no-half", action='store_true', help="do not switch the model to 16-bit floats")
parser.add_argument("--no-progressbar-hiding", action='store_true', help="do not hide progressbar in gradio UI (we hide it because it slows down ML if you have hardware accleration in browser)")
parser.add_argument("--max-batch-count", type=int, default=16, help="maximum batch count value for the UI")
parser.add_argument("--embeddings-dir", type=str, default=os.path.join(script_path, 'embeddings'), help="embeddings directory for textual inversion (default: embeddings)")
parser.add_argument("--allow-code", action='store_true', help="allow custom script execution from webui")
parser.add_argument("--medvram", action='store_true', help="enable stable diffusion model optimizations for sacrificing a little speed for low VRM usage")
parser.add_argument("--lowvram", action='store_true', help="enable stable diffusion model optimizations for sacrificing a lot of speed for very low VRM usage")
parser.add_argument("--always-batch-cond-uncond", action='store_true', help="a workaround test; may help with speed if you use --lowvram")
parser.add_argument("--unload-gfpgan", action='store_true', help="does not do anything.")
parser.add_argument("--precision", type=str, help="evaluate at this precision", choices=["full", "autocast"], default="autocast")
parser.add_argument("--share", action='store_true', help="use share=True for gradio and make the UI accessible through their site (doesn't work for me but you might have better luck)")
parser.add_argument("--esrgan-models-path", type=str, help="path to directory with ESRGAN models", default=os.path.join(script_path, 'ESRGAN'))
parser.add_argument("--opt-split-attention", action='store_true', help="enable optimization that reduce vram usage by a lot for about 10%% decrease in performance")
parser.add_argument("--opt-split-attention-v1", action='store_true', help="enable older version of --opt-split-attention optimization")
parser.add_argument("--listen", action='store_true', help="launch gradio with 0.0.0.0 as server name, allowing to respond to network requests")
parser.add_argument("--port", type=int, help="launch gradio with given server port, you need root/admin rights for ports < 1024, defaults to 7860 if available", default=None)
parser.add_argument("--show-negative-prompt", action='store_true', help="does not do anything", default=False)
parser.add_argument("--ui-config-file", type=str, help="filename to use for ui configuration", default=os.path.join(script_path, 'ui-config.json'))
parser.add_argument("--hide-ui-dir-config", action='store_true', help="hide directory configuration from webui", default=False)
parser.add_argument("--ui-settings-file", type=str, help="filename to use for ui settings", default=os.path.join(script_path, 'config.json'))
parser.add_argument("--gradio-debug",  action='store_true', help="launch gradio with --debug option")
parser.add_argument("--gradio-auth", type=str, help='set gradio authentication like "username:password"; or comma-delimit multiple like "u1:p1,u2:p2,u3:p3"', default=None)
parser.add_argument("--opt-channelslast", action='store_true', help="change memory type for stable diffusion to channels last")

cmd_opts = parser.parse_args()

device = get_optimal_device()

batch_cond_uncond = cmd_opts.always_batch_cond_uncond or not (cmd_opts.lowvram or cmd_opts.medvram)
parallel_processing_allowed = not cmd_opts.lowvram and not cmd_opts.medvram

config_filename = cmd_opts.ui_settings_file

class State:
    interrupted = False
    job = ""
    job_no = 0
    job_count = 0
    sampling_step = 0
    sampling_steps = 0
    current_latent = None
    current_image = None
    current_image_sampling_step = 0

    def interrupt(self):
        self.interrupted = True

    def nextjob(self):
        self.job_no += 1
        self.sampling_step = 0
        self.current_image_sampling_step = 0


state = State()

artist_db = modules.artists.ArtistsDatabase(os.path.join(script_path, 'artists.csv'))

styles_filename = os.path.join(script_path, 'styles.csv')
prompt_styles = modules.styles.load_styles(styles_filename)

interrogator = modules.interrogate.InterrogateModels("interrogate")

face_restorers = []

class Options:
    class OptionInfo:
        def __init__(self, default=None, label="", component=None, component_args=None):
            self.default = default
            self.label = label
            self.component = component
            self.component_args = component_args

    data = None
    hide_dirs = {"visible": False} if cmd_opts.hide_ui_dir_config else None
    data_labels = {
        "samples_filename_pattern": OptionInfo("", "Images filename pattern"),
        "save_to_dirs": OptionInfo(False, "Save images to a subdirectory"),
        "grid_save_to_dirs": OptionInfo(False, "Save grids to subdirectory"),
        "directories_filename_pattern": OptionInfo("", "Directory name pattern"),
        "outdir_samples": OptionInfo("", "Output directory for images; if empty, defaults to two directories below", component_args=hide_dirs),
        "outdir_txt2img_samples": OptionInfo("outputs/txt2img-images", 'Output directory for txt2img images', component_args=hide_dirs),
        "outdir_img2img_samples": OptionInfo("outputs/img2img-images", 'Output directory for img2img images', component_args=hide_dirs),
        "outdir_extras_samples": OptionInfo("outputs/extras-images", 'Output directory for images from extras tab', component_args=hide_dirs),
        "outdir_grids": OptionInfo("", "Output directory for grids; if empty, defaults to two directories below", component_args=hide_dirs),
        "outdir_txt2img_grids": OptionInfo("outputs/txt2img-grids", 'Output directory for txt2img grids', component_args=hide_dirs),
        "outdir_img2img_grids": OptionInfo("outputs/img2img-grids", 'Output directory for img2img grids', component_args=hide_dirs),
        "outdir_save": OptionInfo("log/images", "Directory for saving images using the Save button", component_args=hide_dirs),
        "samples_save": OptionInfo(True, "Save indiviual samples"),
        "samples_format": OptionInfo('png', 'File format for individual samples'),
        "filter_nsfw": OptionInfo(False, "Filter NSFW content"),
        "grid_save": OptionInfo(True, "Save image grids"),
        "return_grid": OptionInfo(True, "Show grid in results for web"),
        "grid_format": OptionInfo('png', 'File format for grids'),
        "grid_extended_filename": OptionInfo(False, "Add extended info (seed, prompt) to filename when saving grid"),
        "grid_only_if_multiple": OptionInfo(True, "Do not save grids consisting of one picture"),
        "n_rows": OptionInfo(-1, "Grid row count; use -1 for autodetect and 0 for it to be same as batch size", gr.Slider, {"minimum": -1, "maximum": 16, "step": 1}),
        "jpeg_quality": OptionInfo(80, "Quality for saved jpeg images", gr.Slider, {"minimum": 1, "maximum": 100, "step": 1}),
        "export_for_4chan": OptionInfo(True, "If PNG image is larger than 4MB or any dimension is larger than 4000, downscale and save copy as JPG"),
        "enable_pnginfo": OptionInfo(True, "Save text information about generation parameters as chunks to png files"),
        "add_model_hash_to_info": OptionInfo(False, "Add model hash to generation information"),
        "img2img_color_correction": OptionInfo(False, "Apply color correction to img2img results to match original colors."),
        "font": OptionInfo("", "Font for image grids that have text"),
        "enable_emphasis": OptionInfo(True, "Use (text) to make model pay more attention to text text and [text] to make it pay less attention"),
        "save_txt": OptionInfo(False, "Create a text file next to every image with generation parameters."),
        "ESRGAN_tile": OptionInfo(192, "Tile size for upscaling. 0 = no tiling.", gr.Slider, {"minimum": 0, "maximum": 512, "step": 16}),
        "ESRGAN_tile_overlap": OptionInfo(8, "Tile overlap, in pixels for upscaling. Low values = visible seam.", gr.Slider, {"minimum": 0, "maximum": 48, "step": 1}),
        "random_artist_categories": OptionInfo([], "Allowed categories for random artists selection when using the Roll button", gr.CheckboxGroup, {"choices": artist_db.categories()}),
        "upscale_at_full_resolution_padding": OptionInfo(16, "Inpainting at full resolution: padding, in pixels, for the masked region.", gr.Slider, {"minimum": 0, "maximum": 128, "step": 4}),
        "show_progressbar": OptionInfo(True, "Show progressbar"),
        "show_progress_every_n_steps": OptionInfo(0, "Show show image creation progress every N sampling steps. Set 0 to disable.", gr.Slider, {"minimum": 0, "maximum": 32, "step": 1}),
        "multiple_tqdm": OptionInfo(True, "Add a second progress bar to the console that shows progress for an entire job. Broken in PyCharm console."),
        "face_restoration_model": OptionInfo(None, "Face restoration model", gr.Radio, lambda: {"choices": [x.name() for x in face_restorers]}),
        "code_former_weight": OptionInfo(0.5, "CodeFormer weight parameter; 0 = maximum effect; 1 = minimum effect", gr.Slider, {"minimum": 0, "maximum": 1, "step": 0.01}),
        "save_images_before_face_restoration": OptionInfo(False, "Save a copy of image before doing face restoration."),
        "face_restoration_unload": OptionInfo(False, "Move face restoration model from VRAM into RAM after processing"),
        "interrogate_keep_models_in_memory": OptionInfo(False, "Interrogate: keep models in VRAM"),
        "interrogate_use_builtin_artists": OptionInfo(True, "Interrogate: use artists from artists.csv"),
        "interrogate_clip_num_beams": OptionInfo(1, "Interrogate: num_beams for BLIP", gr.Slider, {"minimum": 1, "maximum": 16, "step": 1}),
        "interrogate_clip_min_length": OptionInfo(24, "Interrogate: minimum descripton length (excluding artists, etc..)", gr.Slider, {"minimum": 1, "maximum": 128, "step": 1}),
        "interrogate_clip_max_length": OptionInfo(48, "Interrogate: maximum descripton length", gr.Slider, {"minimum": 1, "maximum": 256, "step": 1}),
        "interrogate_clip_dict_limit": OptionInfo(1500, "Interrogate: maximum number of lines in text file (0 = No limit)"),
    }

    def __init__(self):
        self.data = {k: v.default for k, v in self.data_labels.items()}

    def __setattr__(self, key, value):
        if self.data is not None:
            if key in self.data:
                self.data[key] = value

        return super(Options, self).__setattr__(key, value)

    def __getattr__(self, item):
        if self.data is not None:
            if item in self.data:
                return self.data[item]

        if item in self.data_labels:
            return self.data_labels[item].default

        return super(Options, self).__getattribute__(item)

    def save(self, filename):
        with open(filename, "w", encoding="utf8") as file:
            json.dump(self.data, file)

    def load(self, filename):
        with open(filename, "r", encoding="utf8") as file:
            self.data = json.load(file)


opts = Options()
if os.path.exists(config_filename):
    opts.load(config_filename)

sd_upscalers = []

sd_model = None
sd_model_hash = ''

progress_print_out = sys.stdout


class TotalTQDM:
    def __init__(self):
        self._tqdm = None

    def reset(self):
        self._tqdm = tqdm.tqdm(
            desc="Total progress",
            total=state.job_count * state.sampling_steps,
            position=1,
            file=progress_print_out
        )

    def update(self):
        if not opts.multiple_tqdm:
            return
        if self._tqdm is None:
            self.reset()
        self._tqdm.update()

    def clear(self):
        if self._tqdm is not None:
            self._tqdm.close()
            self._tqdm = None


total_tqdm = TotalTQDM()
