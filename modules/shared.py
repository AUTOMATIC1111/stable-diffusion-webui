import argparse
import json
import os

import gradio as gr
import torch

import modules.artists
from modules.paths import script_path, sd_path
import modules.codeformer_model

config_filename = "config.json"

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
parser.add_argument("--embeddings-dir", type=str, default='embeddings', help="embeddings dirtectory for textual inversion (default: embeddings)")
parser.add_argument("--allow-code", action='store_true', help="allow custom script execution from webui")
parser.add_argument("--medvram", action='store_true', help="enable stable diffusion model optimizations for sacrficing a little speed for low VRM usage")
parser.add_argument("--lowvram", action='store_true', help="enable stable diffusion model optimizations for sacrficing a lot of speed for very low VRM usage")
parser.add_argument("--always-batch-cond-uncond", action='store_true', help="a workaround test; may help with speed in you use --lowvram")
parser.add_argument("--unload-gfpgan", action='store_true', help="unload GFPGAN every time after processing images. Warning: seems to cause memory leaks")
parser.add_argument("--precision", type=str, help="evaluate at this precision", choices=["full", "autocast"], default="autocast")
parser.add_argument("--share", action='store_true', help="use share=True for gradio and make the UI accessible through their site (doesn't work for me but you might have better luck)")
parser.add_argument("--esrgan-models-path", type=str, help="path to directory with ESRGAN models", default=os.path.join(script_path, 'ESRGAN'))
parser.add_argument("--opt-split-attention", action='store_true', help="enable optimization that reduced vram usage by a lot for about 10%% decrease in performance")
parser.add_argument("--listen", action='store_true', help="launch gradio with 0.0.0.0 as server name, allowing to respond to network requests")
parser.add_argument("--port", type=int, help="launch gradio with given server port, you need root/admin rights for ports < 1024, defaults to 7860 if available", default=None)
cmd_opts = parser.parse_args()

if torch.has_cuda:
    device = torch.device("cuda")
elif torch.has_mps:
    device = torch.device("mps")
else:
    device = torch.device("cpu")
batch_cond_uncond = cmd_opts.always_batch_cond_uncond or not (cmd_opts.lowvram or cmd_opts.medvram)
parallel_processing_allowed = not cmd_opts.lowvram and not cmd_opts.medvram


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

face_restorers = []

def find_any_font():
    fonts = ['/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf']

    for font in fonts:
        if os.path.exists(font):
            return font

    return "arial.ttf"


class Options:
    class OptionInfo:
        def __init__(self, default=None, label="", component=None, component_args=None):
            self.default = default
            self.label = label
            self.component = component
            self.component_args = component_args

    data = None
    data_labels = {
        "outdir_samples": OptionInfo("", "Output dictectory for images; if empty, defaults to two directories below"),
        "outdir_txt2img_samples": OptionInfo("outputs/txt2img-images", 'Output dictectory for txt2img images'),
        "outdir_img2img_samples": OptionInfo("outputs/img2img-images", 'Output dictectory for img2img images'),
        "outdir_extras_samples": OptionInfo("outputs/extras-images", 'Output dictectory for images from extras tab'),
        "outdir_grids": OptionInfo("", "Output dictectory for grids; if empty, defaults to two directories below"),
        "outdir_txt2img_grids": OptionInfo("outputs/txt2img-grids", 'Output dictectory for txt2img grids'),
        "outdir_img2img_grids": OptionInfo("outputs/img2img-grids", 'Output dictectory for img2img grids'),
        "save_to_dirs": OptionInfo(False, "When writing images/grids, create a directory with name derived from the prompt"),
        "save_to_dirs_prompt_len": OptionInfo(10, "When using above, how many words from prompt to put into directory name", gr.Slider, {"minimum": 1, "maximum": 32, "step": 1}),
        "outdir_save": OptionInfo("log/images", "Directory for saving images using the Save button"),
        "samples_save": OptionInfo(True, "Save indiviual samples"),
        "samples_format": OptionInfo('png', 'File format for indiviual samples'),
        "grid_save": OptionInfo(True, "Save image grids"),
        "return_grid": OptionInfo(True, "Show grid in results for web"),
        "grid_format": OptionInfo('png', 'File format for grids'),
        "grid_extended_filename": OptionInfo(False, "Add extended info (seed, prompt) to filename when saving grid"),
        "grid_only_if_multiple": OptionInfo(True, "Do not save grids consisting of one picture"),
        "n_rows": OptionInfo(-1, "Grid row count; use -1 for autodetect and 0 for it to be same as batch size", gr.Slider, {"minimum": -1, "maximum": 16, "step": 1}),
        "jpeg_quality": OptionInfo(80, "Quality for saved jpeg images", gr.Slider, {"minimum": 1, "maximum": 100, "step": 1}),
        "export_for_4chan": OptionInfo(True, "If PNG image is larger than 4MB or any dimension is larger than 4000, downscale and save copy as JPG"),
        "enable_pnginfo": OptionInfo(True, "Save text information about generation parameters as chunks to png files"),
        "font": OptionInfo(find_any_font(), "Font for image grids  that have text"),
        "enable_emphasis": OptionInfo(True, "Use (text) to make model pay more attention to text text and [text] to make it pay less attention"),
        "save_txt": OptionInfo(False, "Create a text file next to every image with generation parameters."),
        "ESRGAN_tile": OptionInfo(192, "Tile size for upscaling. 0 = no tiling.", gr.Slider, {"minimum": 0, "maximum": 512, "step": 16}),
        "ESRGAN_tile_overlap": OptionInfo(8, "Tile overlap, in pixels for upscaling. Low values = visible seam.", gr.Slider, {"minimum": 0, "maximum": 48, "step": 1}),
        "random_artist_categories": OptionInfo([], "Allowed categories for random artists selection when using the Roll button", gr.CheckboxGroup, {"choices": artist_db.categories()}),
        "upscale_at_full_resolution_padding": OptionInfo(16, "Inpainting at full resolution: padding, in pixels, for the masked region.", gr.Slider, {"minimum": 0, "maximum": 128, "step": 4}),
        "show_progressbar": OptionInfo(True, "Show progressbar"),
        "show_progress_every_n_steps": OptionInfo(0, "Show show image creation progress every N sampling steps. Set 0 to disable.", gr.Slider, {"minimum": 0, "maximum": 32, "step": 1}),
        "face_restoration_model": OptionInfo(None, "Face restoration model", gr.Radio, lambda: {"choices": [x.name() for x in face_restorers]}),
        "code_former_weight": OptionInfo(0.5, "CodeFormer weight parameter; 0 = maximum effect; 1 = minimum effect", gr.Slider, {"minimum": 0, "maximum": 1, "step": 0.01}),
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


