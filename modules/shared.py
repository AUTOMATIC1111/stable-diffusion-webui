import sys
import argparse
import json
import os
from glob import glob
import gradio as gr
import tqdm

import modules.artists
from modules.paths import script_path, sd_path
from modules.devices import get_optimal_device
import modules.styles
import modules.interrogate
import modules.memmon
import modules.sd_models

sd_model_file = os.path.join(script_path, 'model.ckpt')
default_sd_model_file = sd_model_file

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default=os.path.join(sd_path, "configs/stable-diffusion/v1-inference.yaml"), help="path to config which constructs model",)
parser.add_argument("--ckpt", type=str, default=sd_model_file, help="path to checkpoint of stable diffusion model; this checkpoint will be added to the list of checkpoints and loaded by default if you don't have a checkpoint selected in settings",)
parser.add_argument("--ckpt-dir", type=str, default=os.path.join(script_path, 'models'), help="path to directory with stable diffusion checkpoints",)
parser.add_argument("--gfpgan-dir", type=str, help="GFPGAN directory", default=('./src/gfpgan' if os.path.exists('./src/gfpgan') else './GFPGAN'))
parser.add_argument("--gfpgan-model", type=str, help="GFPGAN model file name", default=next(iter(glob('GFPGAN*.pth')), ''))
parser.add_argument("--no-half", action='store_true', help="do not switch the model to 16-bit floats")
parser.add_argument("--no-progressbar-hiding", action='store_true', help="do not hide progressbar in gradio UI (we hide it because it slows down ML if you have hardware acceleration in browser)")
parser.add_argument("--max-batch-count", type=int, default=16, help="maximum batch count value for the UI")
parser.add_argument("--embeddings-dir", type=str, default=os.path.join(script_path, 'embeddings'), help="embeddings directory for textual inversion (default: embeddings)")
parser.add_argument("--allow-code", action='store_true', help="allow custom script execution from webui")
parser.add_argument("--medvram", action='store_true', help="enable stable diffusion model optimizations for sacrificing a little speed for low VRM usage")
parser.add_argument("--lowvram", action='store_true', help="enable stable diffusion model optimizations for sacrificing a lot of speed for very low VRM usage")
parser.add_argument("--always-batch-cond-uncond", action='store_true', help="disables cond/uncond batching that is enabled to save memory with --medvram or --lowvram")
parser.add_argument("--unload-gfpgan", action='store_true', help="does not do anything.")
parser.add_argument("--precision", type=str, help="evaluate at this precision", choices=["full", "autocast"], default="autocast")
parser.add_argument("--share", action='store_true', help="use share=True for gradio and make the UI accessible through their site (doesn't work for me but you might have better luck)")
parser.add_argument("--esrgan-models-path", type=str, help="path to directory with ESRGAN models", default=os.path.join(script_path, 'ESRGAN'))
parser.add_argument("--swinir-models-path", type=str, help="path to directory with SwinIR models", default=os.path.join(script_path, 'SwinIR'))
parser.add_argument("--opt-split-attention", action='store_true', help="does not do anything")
parser.add_argument("--disable-opt-split-attention", action='store_true', help="disable an optimization that reduces vram usage by a lot")
parser.add_argument("--opt-split-attention-v1", action='store_true', help="enable older version of split attention optimization that does not consume all the VRAM it can find")
parser.add_argument("--listen", action='store_true', help="launch gradio with 0.0.0.0 as server name, allowing to respond to network requests")
parser.add_argument("--port", type=int, help="launch gradio with given server port, you need root/admin rights for ports < 1024, defaults to 7860 if available", default=None)
parser.add_argument("--show-negative-prompt", action='store_true', help="does not do anything", default=False)
parser.add_argument("--ui-config-file", type=str, help="filename to use for ui configuration", default=os.path.join(script_path, 'ui-config.json'))
parser.add_argument("--hide-ui-dir-config", action='store_true', help="hide directory configuration from webui", default=False)
parser.add_argument("--ui-settings-file", type=str, help="filename to use for ui settings", default=os.path.join(script_path, 'config.json'))
parser.add_argument("--gradio-debug",  action='store_true', help="launch gradio with --debug option")
parser.add_argument("--gradio-auth", type=str, help='set gradio authentication like "username:password"; or comma-delimit multiple like "u1:p1,u2:p2,u3:p3"', default=None)
parser.add_argument("--opt-channelslast", action='store_true', help="change memory type for stable diffusion to channels last")
parser.add_argument("--styles-file", type=str, help="filename to use for styles", default=os.path.join(script_path, 'styles.csv'))
parser.add_argument("--autolaunch", action='store_true', help="open the webui URL in the system's default browser upon launch", default=False)
parser.add_argument("--use-textbox-seed", action='store_true', help="use textbox for seeds in UI (no up/down, but possible to input long seeds)", default=False)

cmd_opts = parser.parse_args()

if cmd_opts.opt_split_attention:
    print("Information: --opt-split-attention is now the default. To remove this message, remove --opt-split-attention from command line arguments. To disable the optimization, use --disable-opt-split-attention")

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

styles_filename = cmd_opts.styles_file
prompt_styles = modules.styles.StyleDatabase(styles_filename)

interrogator = modules.interrogate.InterrogateModels("interrogate")

face_restorers = []

modules.sd_models.list_models()


class Options:
    class OptionInfo:
        def __init__(self, default=None, label="", component=None, component_args=None, onchange=None):
            self.default = default
            self.label = label
            self.component = component
            self.component_args = component_args
            self.onchange = onchange

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
        "samples_save": OptionInfo(True, "Always save all generated images"),
        "save_selected_only": OptionInfo(False, "When using 'Save' button, only save a single selected image"),
        "samples_format": OptionInfo('png', 'File format for individual samples'),
        "filter_nsfw": OptionInfo(False, "Filter NSFW content"),
        "grid_save": OptionInfo(True, "Always save all generated image grids"),
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
        "img2img_fix_steps": OptionInfo(False, "With img2img, do exactly the amount of steps the slider specifies (normally you'd do less with less denoising)."),
        "enable_quantization": OptionInfo(False, "Enable quantization in K samplers for sharper and cleaner results. This may change existing seeds. Requires restart to apply."),
        "font": OptionInfo("", "Font for image grids that have text"),
        "enable_emphasis": OptionInfo(True, "Use (text) to make model pay more attention to text and [text] to make it pay less attention"),
        "enable_batch_seeds": OptionInfo(True, "Make K-diffusion samplers produce same images in a batch as when making a single image"),
        "save_txt": OptionInfo(False, "Create a text file next to every image with generation parameters."),
        "GAN_tile": OptionInfo(192, "Tile size for all upscalers. 0 = no tiling.", gr.Slider, {"minimum": 0, "maximum": 512, "step": 16}),
        "GAN_tile_overlap": OptionInfo(8, "Tile overlap, in pixels for all upscalers. Low values = visible seam.", gr.Slider, {"minimum": 0, "maximum": 48, "step": 1}),
        "random_artist_categories": OptionInfo([], "Allowed categories for random artists selection when using the Roll button", gr.CheckboxGroup, {"choices": artist_db.categories()}),
        "upscale_at_full_resolution_padding": OptionInfo(16, "Inpainting at full resolution: padding, in pixels, for the masked region.", gr.Slider, {"minimum": 0, "maximum": 128, "step": 4}),
        "upscaler_for_hires_fix": OptionInfo(None, "Upscaler for highres. fix", gr.Radio, lambda: {"choices": [x.name for x in sd_upscalers]}),
        "show_progressbar": OptionInfo(True, "Show progressbar"),
        "show_progress_every_n_steps": OptionInfo(0, "Show show image creation progress every N sampling steps. Set 0 to disable.", gr.Slider, {"minimum": 0, "maximum": 32, "step": 1}),
        "multiple_tqdm": OptionInfo(True, "Add a second progress bar to the console that shows progress for an entire job. Broken in PyCharm console."),
        "memmon_poll_rate": OptionInfo(8, "VRAM usage polls per second during generation. Set to 0 to disable.", gr.Slider, {"minimum": 0, "maximum": 40, "step":1}),
        "face_restoration_model": OptionInfo(None, "Face restoration model", gr.Radio, lambda: {"choices": [x.name() for x in face_restorers]}),
        "code_former_weight": OptionInfo(0.5, "CodeFormer weight parameter; 0 = maximum effect; 1 = minimum effect", gr.Slider, {"minimum": 0, "maximum": 1, "step": 0.01}),
        "save_images_before_face_restoration": OptionInfo(False, "Save a copy of image before doing face restoration."),
        "face_restoration_unload": OptionInfo(False, "Move face restoration model from VRAM into RAM after processing"),
        "interrogate_keep_models_in_memory": OptionInfo(False, "Interrogate: keep models in VRAM"),
        "interrogate_use_builtin_artists": OptionInfo(True, "Interrogate: use artists from artists.csv"),
        "interrogate_clip_num_beams": OptionInfo(1, "Interrogate: num_beams for BLIP", gr.Slider, {"minimum": 1, "maximum": 16, "step": 1}),
        "interrogate_clip_min_length": OptionInfo(24, "Interrogate: minimum description length (excluding artists, etc..)", gr.Slider, {"minimum": 1, "maximum": 128, "step": 1}),
        "interrogate_clip_max_length": OptionInfo(48, "Interrogate: maximum description length", gr.Slider, {"minimum": 1, "maximum": 256, "step": 1}),
        "interrogate_clip_dict_limit": OptionInfo(1500, "Interrogate: maximum number of lines in text file (0 = No limit)"),
        "sd_model_checkpoint": OptionInfo(None, "Stable Diffusion checkpoint", gr.Radio, lambda: {"choices": [x.title for x in modules.sd_models.checkpoints_list.values()]}),
        "js_modal_lightbox": OptionInfo(True, "Enable full page image viewer"),
        "js_modal_lightbox_initialy_zoomed": OptionInfo(True, "Show images zoomed in by default in full page image viewer"),
        "use_original_name_batch": OptionInfo(False, "Use original name for output filename during batch process"),
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

    def onchange(self, key, func):
        item = self.data_labels.get(key)
        item.onchange = func

    def dumpjson(self):
        d = {k: self.data.get(k, self.data_labels.get(k).default) for k in self.data_labels.keys()}
        return json.dumps(d)


opts = Options()
if os.path.exists(config_filename):
    opts.load(config_filename)

sd_upscalers = []

sd_model = None

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

mem_mon = modules.memmon.MemUsageMonitor("MemMon", device, opts)
mem_mon.start()
