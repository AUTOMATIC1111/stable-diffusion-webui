import os
import sys
from collections import namedtuple
from contextlib import nullcontext

import torch
import torch.nn as nn
import numpy as np
import gradio as gr
from omegaconf import OmegaConf
from PIL import Image, ImageFont, ImageDraw, PngImagePlugin, ImageFilter, ImageOps
from torch import autocast
import mimetypes
import random
import math
import html
import time
import importlib
import signal
import threading
from fastapi import FastAPI
from fastapi.middleware.gzip import GZipMiddleware
from datetime import datetime


from modules.paths import script_path
import k_diffusion.sampling
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

try:
    # this silences the annoying "Some weights of the model checkpoint were not used when initializing..." message at start.
    from modules import devices, sd_samplers, upscaler
    import modules.codeformer_model as codeformer
    import modules.extras
    import modules.face_restoration
    import modules.gfpgan_model as gfpgan
    import modules.img2img

    from transformers import logging
    import modules.lowvram
    import modules.paths
    import modules.scripts
    import modules.sd_hijack
    import modules.sd_models
    import modules.shared as shared
    import modules.txt2img
    logging.set_verbosity_error()
except Exception:
    pass

# this is a fix for Windows users. Without it, javascript files will be served with text/html content-type and the bowser will not show any UI
mimetypes.init()
mimetypes.add_type('application/javascript', '.js')
import modules.ui
from modules import devices
from modules import modelloader
from modules.paths import script_path
from modules.shared import cmd_opts
import modules.hypernetworks.hypernetwork

queue_lock = threading.Lock()
# some of those options should not be changed at all because they would break the model, so I removed them from options.
opt_C = 4
opt_f = 8

LANCZOS = (Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS)
invalid_filename_chars = '<>:"/\\|?*\n'
config_filename = "config.json"

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="configs/stable-diffusion/v1-inference.yaml", help="path to config which constructs model", )
parser.add_argument("--ckpt", type=str, default="models/ldm/stable-diffusion-v1/model.ckpt", help="path to checkpoint of model", )
parser.add_argument("--gfpgan-dir", type=str, help="GFPGAN directory", default=('./src/gfpgan' if os.path.exists('./src/gfpgan') else './GFPGAN'))
parser.add_argument("--no-half", action='store_true', help="do not switch the model to 16-bit floats")
parser.add_argument("--no-progressbar-hiding", action='store_true', help="do not hide progressbar in gradio UI (we hide it because it slows down ML if you have hardware accleration in browser)")
parser.add_argument("--max-batch-count", type=int, default=16, help="maximum batch count value for the UI")
parser.add_argument("--embeddings-dir", type=str, default='embeddings', help="embeddings dirtectory for textual inversion (default: embeddings)")
parser.add_argument("--allow-code", action='store_true', help="allow custom script execution from webui")
parser.add_argument("--lowvram", action='store_true', help="enamble stable diffusion model optimizations for low vram")
parser.add_argument("--precision", type=str, help="evaluate at this precision", choices=["full", "autocast"], default="autocast")

cmd_opts = parser.parse_args()

cpu = torch.device("cpu")
gpu = torch.device("cuda")
device = gpu if torch.cuda.is_available() else cpu

css_hide_progressbar = """
.wrap .m-12 svg { display:none!important; }
.wrap .m-12::before { content:"Loading..." }
.progress-bar { display:none!important; }
.meta-text { display:none!important; }
"""

SamplerData = namedtuple('SamplerData', ['name', 'constructor'])
samplers = [
    *[SamplerData(x[0], lambda funcname=x[1]: KDiffusionSampler(funcname)) for x in [
        ('Euler', 'sample_euler'),
        ('Euler ancestral', 'sample_euler_ancestral'),
        ('LMS', 'sample_lms'),
        ('Heun', 'sample_heun'),
        ('DPM 2', 'sample_dpm_2'),
        ('DPM 2 Ancestral', 'sample_dpm_2_ancestral'),
    ] if hasattr(k_diffusion.sampling, x[1])],
    SamplerData('DDIM', lambda: VanillaStableDiffusionSampler(DDIMSampler)),
    SamplerData('PLMS', lambda: VanillaStableDiffusionSampler(PLMSSampler)),
]
samplers_for_img2img = [x for x in samplers if x.name != 'DDIM' and x.name != 'PLMS']

RealesrganModelInfo = namedtuple("RealesrganModelInfo", ["name", "location", "model", "netscale"])

try:
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
    from realesrgan.archs.srvgg_arch import SRVGGNetCompact

    realesrgan_models = [
        RealesrganModelInfo(
            name="Real-ESRGAN 4x plus",
            location="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
            netscale=4, model=lambda: RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        ),
        RealesrganModelInfo(
            name="Real-ESRGAN 4x plus anime 6B",
            location="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
            netscale=4, model=lambda: RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        ),
        RealesrganModelInfo(
            name="Real-ESRGAN 2x plus",
            location="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
            netscale=2, model=lambda: RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        ),
    ]
    have_realesrgan = True
except Exception:
    print("Error importing Real-ESRGAN:", file=sys.stderr)
    print(traceback.format_exc(), file=sys.stderr)

    realesrgan_models = [RealesrganModelInfo('None', '', 0, None)]
    have_realesrgan = False

sd_upscalers = {
    "RealESRGAN": lambda img: upscale_with_realesrgan(img, 2, 0),
    "Lanczos": lambda img: img.resize((img.width * 2, img.height * 2), resample=LANCZOS),
    "None": lambda img: img
}

have_gfpgan = False
if os.path.exists(cmd_opts.gfpgan_dir):
    try:
        sys.path.append(os.path.abspath(cmd_opts.gfpgan_dir))
        from gfpgan import GFPGANer

        have_gfpgan = True
    except:
        print("Error importing GFPGAN:", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)


def gfpgan():
    model_name = 'GFPGANv1.3'
    model_path = os.path.join(cmd_opts.gfpgan_dir, 'experiments/pretrained_models', model_name + '.pth')
    if not os.path.isfile(model_path):
        raise Exception("GFPGAN model not found at path " + model_path)

    return GFPGANer(model_path=model_path, upscale=1, arch='clean', channel_multiplier=2, bg_upsampler=None)


class Options:
    class OptionInfo:
        def __init__(self, default=None, label="", component=None, component_args=None):
            self.default = default
            self.label = label
            self.component = component
            self.component_args = component_args

    data = None
    data_labels = {
        "outdir": OptionInfo("", "Output dictectory; if empty, defaults to 'outputs/*'"),
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
        "font": OptionInfo("arial.ttf", "Font for image grids  that have text"),
        "prompt_matrix_add_to_start": OptionInfo(True, "In prompt matrix, add the variable combination of text to the start of the prompt, rather than the end"),
        "sd_upscale_upscaler_index": OptionInfo("RealESRGAN", "Upscaler to use for SD upscale", gr.Radio, {"choices": list(sd_upscalers.keys())}),
        "sd_upscale_overlap": OptionInfo(64, "Overlap for tiles for SD upscale. The smaller it is, the less smooth transition from one tile to another", gr.Slider, {"minimum": 0, "maximum": 256, "step": 16}),
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


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.eval()
    return model


module_in_gpu = None


def setup_for_low_vram(sd_model):
    parents = {}

    def send_me_to_gpu(module, _):
        """send this module to GPU; send whatever tracked module was previous in GPU to CPU;
        we add this as forward_pre_hook to a lot of modules and this way all but one of them will
        be in CPU
        """
        global module_in_gpu

        module = parents.get(module, module)

        if module_in_gpu == module:
            return

        if module_in_gpu is not None:
            module_in_gpu.to(cpu)

        module.to(gpu)
        module_in_gpu = module

    # see below for register_forward_pre_hook;
    # first_stage_model does not use forward(), it uses encode/decode, so register_forward_pre_hook is
    # useless here, and we just replace those methods
    def first_stage_model_encode_wrap(self, encoder, x):
        send_me_to_gpu(self, None)
        return encoder(x)

    def first_stage_model_decode_wrap(self, decoder, z):
        send_me_to_gpu(self, None)
        return decoder(z)

    # remove three big modules, cond, first_stage, and unet from the model and then
    # send the model to GPU. Then put modules back. the modules will be in CPU.
    stored = sd_model.cond_stage_model.transformer, sd_model.first_stage_model, sd_model.model
    sd_model.cond_stage_model.transformer, sd_model.first_stage_model, sd_model.model = None, None, None
    sd_model.to(device)
    sd_model.cond_stage_model.transformer, sd_model.first_stage_model, sd_model.model = stored

    # register hooks for those the first two models
    sd_model.cond_stage_model.transformer.register_forward_pre_hook(send_me_to_gpu)
    sd_model.first_stage_model.register_forward_pre_hook(send_me_to_gpu)
    sd_model.first_stage_model.encode = lambda x, en=sd_model.first_stage_model.encode: first_stage_model_encode_wrap(sd_model.first_stage_model, en, x)
    sd_model.first_stage_model.decode = lambda z, de=sd_model.first_stage_model.decode: first_stage_model_decode_wrap(sd_model.first_stage_model, de, z)
    parents[sd_model.cond_stage_model.transformer] = sd_model.cond_stage_model

    # the third remaining model is still too big for 4GB, so we also do the same for its submodules
    # so that only one of them is in GPU at a time
    diff_model = sd_model.model.diffusion_model
    stored = diff_model.input_blocks, diff_model.middle_block, diff_model.output_blocks, diff_model.time_embed
    diff_model.input_blocks, diff_model.middle_block, diff_model.output_blocks, diff_model.time_embed = None, None, None, None
    sd_model.model.to(device)
    diff_model.input_blocks, diff_model.middle_block, diff_model.output_blocks, diff_model.time_embed = stored

    # install hooks for bits of third model
    diff_model.time_embed.register_forward_pre_hook(send_me_to_gpu)
    for block in diff_model.input_blocks:
        block.register_forward_pre_hook(send_me_to_gpu)
    diff_model.middle_block.register_forward_pre_hook(send_me_to_gpu)
    for block in diff_model.output_blocks:
        block.register_forward_pre_hook(send_me_to_gpu)


def create_random_tensors(shape, seeds):
    xs = []
    for seed in seeds:
        torch.manual_seed(seed)

        # randn results depend on device; gpu and cpu get different results for same seed;
        # the way I see it, it's better to do this on CPU, so that everyone gets same result;
        # but the original script had it like this so i do not dare change it for now because
        # it will break everyone's seeds.
        xs.append(torch.randn(shape, device=device))
    x = torch.stack(xs)
    return x


def torch_gc():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def save_image(image, path, basename, seed=None, prompt=None, extension='png', info=None, short_filename=False):
    if short_filename or prompt is None or seed is None:
        filename = f"{basename}"
    else:
        filename = f"{basename}-{seed}-{sanitize_filename_part(prompt)[:128]}"

    if extension == 'png' and opts.enable_pnginfo and info is not None:
        pnginfo = PngImagePlugin.PngInfo()
        pnginfo.add_text("parameters", info)
    else:
        pnginfo = None

    os.makedirs(path, exist_ok=True)
    fullfn = os.path.join(path, f"{filename}.{extension}")
    image.save(fullfn, quality=opts.jpeg_quality, pnginfo=pnginfo)

    target_side_length = 4000
    oversize = image.width > target_side_length or image.height > target_side_length
    if opts.export_for_4chan and (oversize or os.stat(fullfn).st_size > 4 * 1024 * 1024):
        ratio = image.width / image.height

        if oversize and ratio > 1:
            image = image.resize((target_side_length, image.height * target_side_length // image.width), LANCZOS)
        elif oversize:
            image = image.resize((image.width * target_side_length // image.height, target_side_length), LANCZOS)

        image.save(os.path.join(path, f"{filename}.jpg"), quality=opts.jpeg_quality, pnginfo=pnginfo)


def sanitize_filename_part(text):
    return text.replace(' ', '_').translate({ord(x): '' for x in invalid_filename_chars})[:128]


def plaintext_to_html(text, klass=None):
    if klass is None:
        text = "".join([f"<p>{html.escape(x)}</p>\n" for x in text.split('\n')])
    else:
        text = "".join([f"<p class=\"{klass}\">{html.escape(x)}</p>\n" for x in text.split('\n')])
    return text


def image_grid(imgs, batch_size=1, rows=None):
    if rows is None:
        if opts.n_rows > 0:
            rows = opts.n_rows
        elif opts.n_rows == 0:
            rows = batch_size
        else:
            rows = math.sqrt(len(imgs))
            rows = round(rows)

    cols = math.ceil(len(imgs) / rows)

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h), color='black')

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))

    return grid


Grid = namedtuple("Grid", ["tiles", "tile_w", "tile_h", "image_w", "image_h", "overlap"])


def split_grid(image, tile_w=512, tile_h=512, overlap=64):
    w = image.width
    h = image.height

    now = tile_w - overlap  # non-overlap width
    noh = tile_h - overlap

    cols = math.ceil((w - overlap) / now)
    rows = math.ceil((h - overlap) / noh)

    grid = Grid([], tile_w, tile_h, w, h, overlap)
    for row in range(rows):
        row_images = []

        y = row * noh

        if y + tile_h >= h:
            y = h - tile_h

        for col in range(cols):
            x = col * now

            if x + tile_w >= w:
                x = w - tile_w

            tile = image.crop((x, y, x + tile_w, y + tile_h))

            row_images.append([x, tile_w, tile])

        grid.tiles.append([y, tile_h, row_images])

    return grid


def combine_grid(grid):
    def make_mask_image(r):
        r = r * 255 / grid.overlap
        r = r.astype(np.uint8)
        return Image.fromarray(r, 'L')

    mask_w = make_mask_image(np.arange(grid.overlap, dtype=np.float).reshape((1, grid.overlap)).repeat(grid.tile_h, axis=0))
    mask_h = make_mask_image(np.arange(grid.overlap, dtype=np.float).reshape((grid.overlap, 1)).repeat(grid.image_w, axis=1))

    combined_image = Image.new("RGB", (grid.image_w, grid.image_h))
    for y, h, row in grid.tiles:
        combined_row = Image.new("RGB", (grid.image_w, h))
        for x, w, tile in row:
            if x == 0:
                combined_row.paste(tile, (0, 0))
                continue

            combined_row.paste(tile.crop((0, 0, grid.overlap, h)), (x, 0), mask=mask_w)
            combined_row.paste(tile.crop((grid.overlap, 0, w, h)), (x + grid.overlap, 0))

        if y == 0:
            combined_image.paste(combined_row, (0, 0))
            continue

        combined_image.paste(combined_row.crop((0, 0, combined_row.width, grid.overlap)), (0, y), mask=mask_h)
        combined_image.paste(combined_row.crop((0, grid.overlap, combined_row.width, h)), (0, y + grid.overlap))

    return combined_image


class GridAnnotation:
    def __init__(self, text='', is_active=True):
        self.text = text
        self.is_active = is_active
        self.size = None


def draw_grid_annotations(im, width, height, hor_texts, ver_texts):
    def wrap(drawing, text, font, line_length):
        lines = ['']
        for word in text.split():
            line = f'{lines[-1]} {word}'.strip()
            if drawing.textlength(line, font=font) <= line_length:
                lines[-1] = line
            else:
                lines.append(word)
        return lines

    def draw_texts(drawing, draw_x, draw_y, lines):
        for i, line in enumerate(lines):
            drawing.multiline_text((draw_x, draw_y + line.size[1] / 2), line.text, font=fnt, fill=color_active if line.is_active else color_inactive, anchor="mm", align="center")

            if not line.is_active:
                drawing.line((draw_x - line.size[0] // 2, draw_y + line.size[1] // 2, draw_x + line.size[0] // 2, draw_y + line.size[1] // 2), fill=color_inactive, width=4)

            draw_y += line.size[1] + line_spacing

    fontsize = (width + height) // 25
    line_spacing = fontsize // 2
    fnt = ImageFont.truetype(opts.font, fontsize)
    color_active = (0, 0, 0)
    color_inactive = (153, 153, 153)

    pad_left = width * 3 // 4 if len(ver_texts) > 1 else 0

    cols = im.width // width
    rows = im.height // height

    assert cols == len(hor_texts), f'bad number of horizontal texts: {len(hor_texts)}; must be {cols}'
    assert rows == len(ver_texts), f'bad number of vertical texts: {len(ver_texts)}; must be {rows}'

    calc_img = Image.new("RGB", (1, 1), "white")
    calc_d = ImageDraw.Draw(calc_img)

    for texts, allowed_width in zip(hor_texts + ver_texts, [width] * len(hor_texts) + [pad_left] * len(ver_texts)):
        items = [] + texts
        texts.clear()

        for line in items:
            wrapped = wrap(calc_d, line.text, fnt, allowed_width)
            texts += [GridAnnotation(x, line.is_active) for x in wrapped]

        for line in texts:
            bbox = calc_d.multiline_textbbox((0, 0), line.text, font=fnt)
            line.size = (bbox[2] - bbox[0], bbox[3] - bbox[1])

    hor_text_heights = [sum([line.size[1] + line_spacing for line in lines]) - line_spacing for lines in hor_texts]
    ver_text_heights = [sum([line.size[1] + line_spacing for line in lines]) - line_spacing * len(lines) for lines in ver_texts]

    pad_top = max(hor_text_heights) + line_spacing * 2

    result = Image.new("RGB", (im.width + pad_left, im.height + pad_top), "white")
    result.paste(im, (pad_left, pad_top))

    d = ImageDraw.Draw(result)

    for col in range(cols):
        x = pad_left + width * col + width / 2
        y = pad_top / 2 - hor_text_heights[col] / 2

        draw_texts(d, x, y, hor_texts[col])

    for row in range(rows):
        x = pad_left / 2
        y = pad_top + height * row + height / 2 - ver_text_heights[row] / 2

        draw_texts(d, x, y, ver_texts[row])

    return result


def draw_prompt_matrix(im, width, height, all_prompts):
    prompts = all_prompts[1:]
    boundary = math.ceil(len(prompts) / 2)

    prompts_horiz = prompts[:boundary]
    prompts_vert = prompts[boundary:]

    hor_texts = [[GridAnnotation(x, is_active=pos & (1 << i) != 0) for i, x in enumerate(prompts_horiz)] for pos in range(1 << len(prompts_horiz))]
    ver_texts = [[GridAnnotation(x, is_active=pos & (1 << i) != 0) for i, x in enumerate(prompts_vert)] for pos in range(1 << len(prompts_vert))]

    return draw_grid_annotations(im, width, height, hor_texts, ver_texts)


def draw_xy_grid(xs, ys, x_label, y_label, cell):
    res = []

    ver_texts = [[GridAnnotation(y_label(y))] for y in ys]
    hor_texts = [[GridAnnotation(x_label(x))] for x in xs]

    for y in ys:
        for x in xs:
            res.append(cell(x, y))

    grid = image_grid(res, rows=len(ys))
    grid = draw_grid_annotations(grid, res[0].width, res[0].height, hor_texts, ver_texts)

    return grid


def resize_image(resize_mode, im, width, height):
    if resize_mode == 0:
        res = im.resize((width, height), resample=LANCZOS)
    elif resize_mode == 1:
        ratio = width / height
        src_ratio = im.width / im.height

        src_w = width if ratio > src_ratio else im.width * height // im.height
        src_h = height if ratio <= src_ratio else im.height * width // im.width

        resized = im.resize((src_w, src_h), resample=LANCZOS)
        res = Image.new("RGB", (width, height))
        res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))
    else:
        ratio = width / height
        src_ratio = im.width / im.height

        src_w = width if ratio < src_ratio else im.width * height // im.height
        src_h = height if ratio >= src_ratio else im.height * width // im.width

        resized = im.resize((src_w, src_h), resample=LANCZOS)
        res = Image.new("RGB", (width, height))
        res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))

        if ratio < src_ratio:
            fill_height = height // 2 - src_h // 2
            res.paste(resized.resize((width, fill_height), box=(0, 0, width, 0)), box=(0, 0))
            res.paste(resized.resize((width, fill_height), box=(0, resized.height, width, resized.height)), box=(0, fill_height + src_h))
        elif ratio > src_ratio:
            fill_width = width // 2 - src_w // 2
            res.paste(resized.resize((fill_width, height), box=(0, 0, 0, height)), box=(0, 0))
            res.paste(resized.resize((fill_width, height), box=(resized.width, 0, resized.width, height)), box=(fill_width + src_w, 0))

        return res

            data = torch.load(path)
            param_dict = data['string_to_param']
            assert len(param_dict) == 1, 'embedding file has multiple terms in it'
            emb = next(iter(param_dict.items()))[1].reshape(768)
            self.word_embeddings[name] = emb
            self.word_embeddings_checksums[name] = f'{const_hash(emb) & 0xffff:04x}'


def initialize():
    if cmd_opts.ui_debug_mode:
        shared.sd_upscalers = upscaler.UpscalerLanczos().scalers
        modules.scripts.load_scripts()
        return

    modelloader.cleanup_models()
    modules.sd_models.setup_model()
    codeformer.setup_model(cmd_opts.codeformer_models_path)
    gfpgan.setup_model(cmd_opts.gfpgan_models_path)
    shared.face_restorers.append(modules.face_restoration.FaceRestoration())
    modelloader.load_upscalers()

    modules.scripts.load_scripts()

    modules.sd_models.load_model()
    shared.opts.onchange("sd_model_checkpoint", wrap_queued_call(lambda: modules.sd_models.reload_model_weights(shared.sd_model)))
    shared.opts.onchange("sd_hypernetwork", wrap_queued_call(lambda: modules.hypernetworks.hypernetwork.load_hypernetwork(shared.opts.sd_hypernetwork)))
    shared.opts.onchange("sd_hypernetwork_strength", modules.hypernetworks.hypernetwork.apply_strength)

    # make the program just exit at ctrl+c without waiting for anything
    def sigint_handler(sig, frame):
        print(f'Interrupted with signal {sig} in {frame}')
        os._exit(0)

    signal.signal(signal.SIGINT, sigint_handler)


def create_api(app):
    from modules.api.api import Api
    api = Api(app, queue_lock)
    return api

def wait_on_server(demo=None):
    while 1:
        time.sleep(0.5)
        if demo and getattr(demo, 'do_restart', False):
            time.sleep(0.5)
            demo.close()
            time.sleep(0.5)
            break

def api_only():
    initialize()

    app = FastAPI()
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    api = create_api(app)

    api.launch(server_name="0.0.0.0" if cmd_opts.listen else "127.0.0.1", port=cmd_opts.port if cmd_opts.port else 7861)


def webui():
    launch_api = cmd_opts.api
    initialize()

                    mult_change = self.token_mults.get(token)
                    if mult_change is not None:
                        mult *= mult_change
                    elif possible_matches is None:
                        remade_tokens.append(token)
                        multipliers.append(mult)
                    else:
                        found = False
                        for ids, word in possible_matches:
                            if tokens[i:i + len(ids)] == ids:
                                fixes.append((len(remade_tokens), word))
                                remade_tokens.append(777)
                                multipliers.append(mult)
                                i += len(ids) - 1
                                found = True
                                used_custom_terms.append((word, self.hijack.word_embeddings_checksums[word]))
                                break

        app, local_url, share_url = demo.launch(
            share=cmd_opts.share,
            server_name="0.0.0.0" if cmd_opts.listen else None,
            server_port=cmd_opts.port,
            debug=cmd_opts.gradio_debug,
            auth=[tuple(cred.split(':')) for cred in cmd_opts.gradio_auth.strip('"').split(',')] if cmd_opts.gradio_auth else None,
            inbrowser=cmd_opts.autolaunch,
            prevent_thread_lock=True
        )
        # after initial launch, disable --autolaunch for subsequent restarts
        cmd_opts.autolaunch = False

        app.add_middleware(GZipMiddleware, minimum_size=1000)

        if (launch_api):
            create_api(app)

        wait_on_server(demo)

                remade_tokens = remade_tokens + [id_end] * (maxlen - 2 - len(remade_tokens))
                remade_tokens = [id_start] + remade_tokens[0:maxlen - 2] + [id_end]
                cache[tuple_tokens] = (remade_tokens, fixes, multipliers)

        print('Reloading Custom Scripts')
        modules.scripts.reload_scripts()
        print('Reloading modules: modules.ui')
        importlib.reload(modules.ui)
        print('Refreshing Model List')
        modules.sd_models.list_models()
        print('Restarting Gradio')



        tokens = torch.asarray(remade_batch_tokens).to(device)
        outputs = self.wrapped.transformer(input_ids=tokens)
        z = outputs.last_hidden_state

        # restoring original mean is likely not correct, but it seems to work well to prevent artifacts that happen otherwise
        batch_multipliers = torch.asarray(np.array(batch_multipliers)).to(device)
        original_mean = z.mean()
        z *= batch_multipliers.reshape(batch_multipliers.shape + (1,)).expand(z.shape)
        new_mean = z.mean()
        z *= original_mean / new_mean

        return z


class EmbeddingsWithFixes(nn.Module):
    def __init__(self, wrapped, embeddings):
        super().__init__()
        self.wrapped = wrapped
        self.embeddings = embeddings

    def forward(self, input_ids):
        batch_fixes = self.embeddings.fixes
        self.embeddings.fixes = None

        inputs_embeds = self.wrapped(input_ids)

        if batch_fixes is not None:
            for fixes, tensor in zip(batch_fixes, inputs_embeds):
                for offset, word in fixes:
                    tensor[offset] = self.embeddings.word_embeddings[word]

        return inputs_embeds


class StableDiffusionProcessing:
    def __init__(self, outpath=None, prompt="", seed=-1, sampler_index=0, batch_size=1, n_iter=1, steps=50, cfg_scale=7.0, width=512, height=512, prompt_matrix=False, use_GFPGAN=False, do_not_save_samples=False, do_not_save_grid=False, strength_GFPGAN=1.0, extra_generation_params=None, overlay_images=None):
        self.outpath: str = outpath
        self.prompt: str = prompt
        self.seed: int = seed
        self.sampler_index: int = sampler_index
        self.batch_size: int = batch_size
        self.n_iter: int = n_iter
        self.steps: int = steps
        self.cfg_scale: float = cfg_scale
        self.width: int = width
        self.height: int = height
        self.prompt_matrix: bool = prompt_matrix
        self.use_GFPGAN: bool = use_GFPGAN
        self.do_not_save_samples: bool = do_not_save_samples
        self.do_not_save_grid: bool = do_not_save_grid
        self.strength_GFPGAN: bool = strength_GFPGAN
        self.extra_generation_params: dict = extra_generation_params
        self.overlay_images = overlay_images

    def init(self):
        pass

    def sample(self, x, conditioning, unconditional_conditioning):
        raise NotImplementedError()


class VanillaStableDiffusionSampler:
    def __init__(self, constructor):
        self.sampler = constructor(sd_model)

    def sample(self, p: StableDiffusionProcessing, x, conditioning, unconditional_conditioning):
        samples_ddim, _ = self.sampler.sample(S=p.steps, conditioning=conditioning, batch_size=int(x.shape[0]), shape=x[0].shape, verbose=False, unconditional_guidance_scale=p.cfg_scale, unconditional_conditioning=unconditional_conditioning, x_T=x)
        return samples_ddim


class CFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, x, sigma, uncond, cond, cond_scale):
        x_in = torch.cat([x] * 2)
        sigma_in = torch.cat([sigma] * 2)
        cond_in = torch.cat([uncond, cond])
        uncond, cond = self.inner_model(x_in, sigma_in, cond=cond_in).chunk(2)
        return uncond + (cond - uncond) * cond_scale


class KDiffusionSampler:
    def __init__(self, funcname):
        self.model_wrap = k_diffusion.external.CompVisDenoiser(sd_model)
        self.funcname = funcname
        self.func = getattr(k_diffusion.sampling, self.funcname)
        self.model_wrap_cfg = CFGDenoiser(self.model_wrap)

    def sample(self, p: StableDiffusionProcessing, x, conditioning, unconditional_conditioning):
        sigmas = self.model_wrap.get_sigmas(p.steps)
        x = x * sigmas[0]

        samples_ddim = self.func(self.model_wrap_cfg, x, sigmas, extra_args={'cond': conditioning, 'uncond': unconditional_conditioning, 'cond_scale': p.cfg_scale}, disable=False)
        return samples_ddim


Processed = namedtuple('Processed', ['images', 'seed', 'info'])


class OutputInfo:
    def __init__(self, prompt: str, params: str, comments: str):
        self.prompt = prompt.strip()
        self.params = params.strip()
        self.comments = comments.strip()

    def __str__(self):
        return '\n'.join([self.prompt, self.params, self.comments])

    def html(self) -> str:
        return f'''
        {plaintext_to_html(self.prompt, "prompt-info")}<br>
        {plaintext_to_html(self.params, "params-info")}
        {plaintext_to_html(self.comments, "comments-info")}
        '''


def process_images(p: StableDiffusionProcessing) -> Processed:
    """this is the main loop that both txt2img and img2img use; it calls func_init once inside all the scopes and func_sample once per batch"""

    prompt = p.prompt
    model = sd_model

    assert p.prompt is not None
    torch_gc()

    seed = int(random.randrange(4294967294) if p.seed == -1 else p.seed)

    sample_path = os.path.join(p.outpath, "samples")
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(p.outpath)) - 1

    comments = []

    prompt_matrix_parts = []
    if p.prompt_matrix:
        all_prompts = []
        prompt_matrix_parts = prompt.split("|")
        combination_count = 2 ** (len(prompt_matrix_parts) - 1)
        for combination_num in range(combination_count):
            selected_prompts = [text.strip().strip(',') for n, text in enumerate(prompt_matrix_parts[1:]) if combination_num & (1 << n)]

            if opts.prompt_matrix_add_to_start:
                selected_prompts = selected_prompts + [prompt_matrix_parts[0]]
            else:
                selected_prompts = [prompt_matrix_parts[0]] + selected_prompts

            all_prompts.append(", ".join(selected_prompts))

        p.n_iter = math.ceil(len(all_prompts) / p.batch_size)
        all_seeds = len(all_prompts) * [seed]

        print(f"Prompt matrix will create {len(all_prompts)} images using a total of {p.n_iter} batches.")
    else:
        all_prompts = p.batch_size * p.n_iter * [prompt]
        all_seeds = [seed + x for x in range(len(all_prompts))]

    generation_params = {
        "Steps": p.steps,
        "Sampler": samplers[p.sampler_index].name,
        "CFG": p.cfg_scale,
        "Seed": seed,
        "GFPGAN": ("GFPGAN" if p.use_GFPGAN else None)
    }

    if p.extra_generation_params is not None:
        generation_params.update(p.extra_generation_params)

    generation_params_text = ", ".join([k if k == v else f'{k}: {v}' for k, v in generation_params.items() if v is not None])

    def infotext():
        return OutputInfo(prompt, generation_params_text, "".join(["\n\n" + x for x in comments]))

    if os.path.exists(cmd_opts.embeddings_dir):
        model_hijack.load_textual_inversion_embeddings(cmd_opts.embeddings_dir, model)

    output_images = []
    precision_scope = autocast if cmd_opts.precision == "autocast" else nullcontext
    ema_scope = (nullcontext if cmd_opts.lowvram else model.ema_scope)
    with torch.no_grad(), precision_scope("cuda"), ema_scope():
        p.init()

        for n in range(p.n_iter):
            prompts = all_prompts[n * p.batch_size:(n + 1) * p.batch_size]
            seeds = all_seeds[n * p.batch_size:(n + 1) * p.batch_size]

            uc = model.get_learned_conditioning(len(prompts) * [""])
            c = model.get_learned_conditioning(prompts)

            if len(model_hijack.comments) > 0:
                comments += model_hijack.comments

            # we manually generate all input noises because each one should have a specific seed
            x = create_random_tensors([opt_C, p.height // opt_f, p.width // opt_f], seeds=seeds)

            samples_ddim = p.sample(x=x, conditioning=c, unconditional_conditioning=uc)

            x_samples_ddim = model.decode_first_stage(samples_ddim)
            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

            if p.prompt_matrix or opts.samples_save or opts.grid_save:
                for i, x_sample in enumerate(x_samples_ddim):
                    # TODO: convert to BGR colorspace?
                    x_sample = 255. * np.moveaxis(x_sample.cpu().numpy(), 0, 2)
                    x_sample = x_sample.astype(np.uint8)

                    if p.use_GFPGAN and have_gfpgan and p.strength_GFPGAN > 0.0:
                        torch_gc()

                        gfpgan_model = gfpgan()
                        x_sample_bgr = x_sample[:, :, ::-1]
                        cropped_faces, restored_faces, gfpgan_output_bgr = gfpgan_model.enhance(x_sample_bgr, has_aligned=False, only_center_face=False, paste_back=True)
                        image = Image.fromarray(gfpgan_output_bgr[:, :, ::-1])

                        if p.strength_GFPGAN < 1.0:
                            image = Image.blend(Image.fromarray(x_sample), image, p.strength_GFPGAN)
                    else:
                        image = Image.fromarray(x_sample)

                    if p.overlay_images is not None and i < len(p.overlay_images):
                        image = image.convert('RGBA')
                        image.alpha_composite(p.overlay_images[i])
                        image = image.convert('RGB')

                    if not p.do_not_save_samples:
                        save_image(image, sample_path, f"{base_count:05}", seeds[i], prompts[i], opts.samples_format, info=str(infotext()))

                    output_images.append(image)
                    base_count += 1

        unwanted_grid_because_of_img_count = len(output_images) < 2 and opts.grid_only_if_multiple
        if (p.prompt_matrix or opts.grid_save) and not p.do_not_save_grid and not unwanted_grid_because_of_img_count:
            return_grid = opts.return_grid

            if p.prompt_matrix:
                grid = image_grid(output_images, p.batch_size, rows=1 << ((len(prompt_matrix_parts) - 1) // 2))

                try:
                    grid = draw_prompt_matrix(grid, p.width, p.height, prompt_matrix_parts)
                except Exception:
                    import traceback
                    print("Error creating prompt_matrix text:", file=sys.stderr)
                    print(traceback.format_exc(), file=sys.stderr)

                return_grid = True
            else:
                grid = image_grid(output_images, p.batch_size)

            if return_grid:
                output_images.insert(0, grid)

            save_image(grid, p.outpath, f"grid-{grid_count:04}", seed, prompt, opts.grid_format, info=str(infotext()), short_filename=not opts.grid_extended_filename)
            grid_count += 1

    torch_gc()
    return Processed(output_images, seed, infotext())


class StableDiffusionProcessingTxt2Img(StableDiffusionProcessing):
    sampler = None

    def init(self):
        self.sampler = samplers[self.sampler_index].constructor()

    def sample(self, x, conditioning, unconditional_conditioning):
        samples_ddim = self.sampler.sample(self, x, conditioning, unconditional_conditioning)
        return samples_ddim


def txt2img(prompt: str, steps: int, sampler_index: int, use_GFPGAN: bool, strength_GFPGAN: float, prompt_matrix: bool, n_iter: int, batch_size: int, cfg_scale: float, seed: int, height: int, width: int, code: str):
    outpath = opts.outdir or "outputs/txt2img-samples"

    p = StableDiffusionProcessingTxt2Img(
        outpath=outpath,
        prompt=prompt,
        seed=seed,
        sampler_index=sampler_index,
        batch_size=batch_size,
        n_iter=n_iter,
        steps=steps,
        cfg_scale=cfg_scale,
        width=width,
        height=height,
        prompt_matrix=prompt_matrix,
        use_GFPGAN=use_GFPGAN,
        strength_GFPGAN=strength_GFPGAN
    )

    if code != '' and cmd_opts.allow_code:
        p.do_not_save_grid = True
        p.do_not_save_samples = True

        display_result_data = [[], -1, OutputInfo("", "", "")]

        def display(imgs, s=display_result_data[1], i=display_result_data[2]):
            display_result_data[0] = imgs
            display_result_data[1] = s
            display_result_data[2] = i

        from types import ModuleType
        compiled = compile(code, '', 'exec')
        module = ModuleType("testmodule")
        module.__dict__.update(globals())
        module.p = p
        module.display = display
        exec(compiled, module.__dict__)

        processed = Processed(*display_result_data)
    else:
        processed = process_images(p)

    return processed.images, processed.info.html()


class Flagging(gr.FlaggingCallback):

    def setup(self, components, flagging_dir: str):
        pass

    def flag(self, flag_data, flag_option=None, flag_index=None, username=None):
        import csv

        os.makedirs("log/images", exist_ok=True)

        # those must match the "txt2img" function
        prompt, steps, sampler_index, use_gfpgan, gfpgan_strength, prompt_matrix, n_iter, batch_size, cfg_scale, seed, height, width, code, images, seed, comment = flag_data

        filenames = []

        with open("log/log.csv", "a", encoding="utf8", newline='') as file:
            import time
            import base64

            at_start = file.tell() == 0
            writer = csv.writer(file)
            if at_start:
                writer.writerow(["prompt", "seed", "width", "height", "cfgs", "steps", "filename"])

            filename_base = str(int(time.time() * 1000))
            for i, filedata in enumerate(images):
                filename = "log/images/" + filename_base + ("" if len(images) == 1 else "-" + str(i + 1)) + ".png"

                if filedata.startswith("data:image/png;base64,"):
                    filedata = filedata[len("data:image/png;base64,"):]

                with open(filename, "wb") as imgfile:
                    imgfile.write(base64.decodebytes(filedata.encode('utf-8')))

                filenames.append(filename)

            writer.writerow([prompt, seed, width, height, cfg_scale, steps, filenames[0]])

        print("Logged:", filenames[0])


def fill(image, mask):
    image_mod = Image.new('RGBA', (image.width, image.height))

    image_masked = Image.new('RGBa', (image.width, image.height))
    image_masked.paste(image.convert("RGBA").convert("RGBa"), mask=ImageOps.invert(mask.convert('L')))

    image_masked = image_masked.convert('RGBa')

    for radius, repeats in [(64, 1), (16, 2), (4, 4), (2, 2), (0, 1)]:
        blurred = image_masked.filter(ImageFilter.GaussianBlur(radius)).convert('RGBA')
        for _ in range(repeats):
            image_mod.alpha_composite(blurred)

    return image_mod.convert("RGB")


class StableDiffusionProcessingImg2Img(StableDiffusionProcessing):
    sampler = None

    def __init__(self, init_images=None, resize_mode=0, denoising_strength=0.75, mask=None, mask_blur=4, inpainting_fill=0, **kwargs):
        super().__init__(**kwargs)

        self.init_images = init_images
        self.resize_mode: int = resize_mode
        self.denoising_strength: float = denoising_strength
        self.init_latent = None
        self.original_mask = mask
        self.mask_blur = mask_blur
        self.inpainting_fill = inpainting_fill
        self.mask = None
        self.nmask = None

    def init(self):
        self.sampler = samplers_for_img2img[self.sampler_index].constructor()

        if self.original_mask is not None:
            self.original_mask = resize_image(self.resize_mode, self.original_mask, self.width, self.height)
            self.overlay_images = []

        imgs = []

        if not self.init_images or None in self.init_images:
            raise Exception('No input image provided for Image-to-Image')

        for img in self.init_images:
            image = img.convert("RGB")
            image = resize_image(self.resize_mode, image, self.width, self.height)

            if self.original_mask is not None:
                if self.inpainting_fill != 1:
                    image = fill(image, self.original_mask)

                image_masked = Image.new('RGBa', (image.width, image.height))
                image_masked.paste(image.convert("RGBA").convert("RGBa"), mask=ImageOps.invert(self.original_mask.convert('L')))

                self.overlay_images.append(image_masked.convert('RGBA'))

            image = np.array(image).astype(np.float32) / 255.0
            image = np.moveaxis(image, 2, 0)

            imgs.append(image)

        if len(imgs) == 1:
            batch_images = np.expand_dims(imgs[0], axis=0).repeat(self.batch_size, axis=0)
            if self.overlay_images is not None:
                self.overlay_images = self.overlay_images * self.batch_size
        elif len(imgs) <= self.batch_size:
            self.batch_size = len(imgs)
            batch_images = np.array(imgs)
        else:
            raise RuntimeError(f"bad number of images passed: {len(imgs)}; expecting {self.batch_size} or less")

        image = torch.from_numpy(batch_images)
        image = 2. * image - 1.
        image = image.to(device)

        self.init_latent = sd_model.get_first_stage_encoding(sd_model.encode_first_stage(image))

        if self.original_mask is not None:
            if self.mask_blur > 0:
                self.original_mask = self.original_mask.filter(ImageFilter.GaussianBlur(self.mask_blur)).convert('L')

            latmask = self.original_mask.convert('RGB').resize((self.init_latent.shape[3], self.init_latent.shape[2]))
            latmask = np.moveaxis(np.array(latmask, dtype=np.float), 2, 0) / 255
            latmask = latmask[0]
            latmask = np.tile(latmask[None], (4, 1, 1))

            self.mask = torch.asarray(1.0 - latmask).to(device).type(sd_model.dtype)
            self.nmask = torch.asarray(latmask).to(device).type(sd_model.dtype)

    def sample(self, x, conditioning, unconditional_conditioning):
        t_enc = int(min(self.denoising_strength, 0.999) * self.steps)

        sigmas = self.sampler.model_wrap.get_sigmas(self.steps)
        noise = x * sigmas[self.steps - t_enc - 1]
        xi = self.init_latent + noise

        if self.mask is not None:
            if self.inpainting_fill == 2:
                xi = xi * self.mask + noise * self.nmask
            elif self.inpainting_fill == 3:
                xi = xi * self.mask

        sigma_sched = sigmas[self.steps - t_enc - 1:]

        def mask_cb(v):
            v["denoised"][:] = v["denoised"][:] * self.nmask + self.init_latent * self.mask

        samples_ddim = self.sampler.func(self.sampler.model_wrap_cfg, xi, sigma_sched, extra_args={'cond': conditioning, 'uncond': unconditional_conditioning, 'cond_scale': self.cfg_scale}, disable=False, callback=mask_cb if self.mask is not None else None)

        if self.mask is not None:
            samples_ddim = samples_ddim * self.nmask + self.init_latent * self.mask

        return samples_ddim


def img2img(prompt: str, init_img, init_img_with_mask, ddim_steps: int, sampler_index: int, mask_blur: int, inpainting_fill: int, use_GFPGAN: bool, strength_GFPGAN: float, prompt_matrix, loopback: bool, sd_upscale: bool, n_iter: int, batch_size: int, cfg_scale: float, denoising_strength: float, seed: int, height: int, width: int, resize_mode: int):
    outpath = opts.outdir or "outputs/img2img-samples"

    if init_img_with_mask is not None:
        image = init_img_with_mask['image']
        mask = init_img_with_mask['mask']
    else:
        image = init_img
        mask = None

    assert 0. <= denoising_strength <= 1., 'can only work with strength in [0.0, 1.0]'

    p = StableDiffusionProcessingImg2Img(
        outpath=outpath,
        prompt=prompt,
        seed=seed,
        sampler_index=sampler_index,
        batch_size=batch_size,
        n_iter=n_iter,
        steps=ddim_steps,
        cfg_scale=cfg_scale,
        width=width,
        height=height,
        prompt_matrix=prompt_matrix,
        use_GFPGAN=use_GFPGAN,
        strength_GFPGAN=strength_GFPGAN,
        init_images=[image],
        mask=mask,
        mask_blur=mask_blur,
        inpainting_fill=inpainting_fill,
        resize_mode=resize_mode,
        denoising_strength=denoising_strength,
        extra_generation_params={"DNS": denoising_strength}
    )

    if loopback:
        output_images, info = None, None
        history = []
        initial_seed = None
        initial_info = None

        for i in range(n_iter):
            p.n_iter = 1
            p.batch_size = 1
            p.do_not_save_grid = True

            processed = process_images(p)

            if initial_seed is None:
                initial_seed = processed.seed
                initial_info = processed.info

            p.init_img = processed.images[0]
            p.seed = processed.seed + 1
            p.denoising_strength = max(p.denoising_strength * 0.95, 0.1)
            history.append(processed.images[0])

        grid_count = len(os.listdir(outpath)) - 1
        grid = image_grid(history, batch_size, rows=1)

        save_image(grid, outpath, f"grid-{grid_count:04}", initial_seed, prompt, opts.grid_format, info=str(info), short_filename=not opts.grid_extended_filename)

        processed = Processed(history, initial_seed, initial_info)

    elif sd_upscale:
        initial_seed = None
        initial_info = None

        upscaler = sd_upscalers[opts.sd_upscale_upscaler_index]
        img = upscaler(init_img)

        torch_gc()

        grid = split_grid(img, tile_w=width, tile_h=height, overlap=opts.sd_upscale_overlap)

        p.n_iter = 1
        p.do_not_save_grid = True
        p.do_not_save_samples = True

        work = []
        work_results = []

        for y, h, row in grid.tiles:
            for tiledata in row:
                work.append(tiledata[2])

        batch_count = math.ceil(len(work) / p.batch_size)
        print(f"SD upscaling will process a total of {len(work)} images tiled as {len(grid.tiles[0][2])}x{len(grid.tiles)} in a total of {batch_count} batches.")

        for i in range(batch_count):
            p.init_images = work[i * p.batch_size:(i + 1) * p.batch_size]

            processed = process_images(p)

            if initial_seed is None:
                initial_seed = processed.seed
                initial_info = processed.info

            p.seed = processed.seed + 1
            work_results += processed.images

        image_index = 0
        for y, h, row in grid.tiles:
            for tiledata in row:
                tiledata[2] = work_results[image_index]
                image_index += 1

        combined_image = combine_grid(grid)

        grid_count = len(os.listdir(outpath)) - 1
        save_image(combined_image, outpath, f"grid-{grid_count:04}", initial_seed, prompt, opts.grid_format, info=str(initial_info), short_filename=not opts.grid_extended_filename)

        processed = Processed([combined_image], initial_seed, initial_info)

    else:
        processed = process_images(p)

    return processed.images, processed.info.html()


def upscale_with_realesrgan(image, RealESRGAN_upscaling, RealESRGAN_model_index):
    info = realesrgan_models[RealESRGAN_model_index]

    model = info.model()
    upsampler = RealESRGANer(
        scale=info.netscale,
        model_path=info.location,
        model=model,
        half=True
    )

    upsampled = upsampler.enhance(np.array(image), outscale=RealESRGAN_upscaling)[0]

    image = Image.fromarray(upsampled)
    return image


def run_extras(image, GFPGAN_strength, RealESRGAN_upscaling, RealESRGAN_model_index):
    torch_gc()

    if not image:
        raise Exception('No input image provided for Post-Processing')

    image = image.convert("RGB")

    outpath = opts.outdir or "outputs/extras-samples"

    if have_gfpgan and GFPGAN_strength > 0:
        gfpgan_model = gfpgan()
        img_data_bgr = np.array(image, dtype=np.uint8)[:, :, ::-1]
        cropped_faces, restored_faces, restored_img = gfpgan_model.enhance(img_data_bgr, has_aligned=False, only_center_face=False, paste_back=True)
        img_data_rgb = restored_img[:, :, ::-1]
        res = Image.fromarray(img_data_rgb)

        if GFPGAN_strength < 1.0:
            res = Image.blend(image, res, GFPGAN_strength)

        image = res

    if have_realesrgan and RealESRGAN_upscaling != 1.0:
        image = upscale_with_realesrgan(image, RealESRGAN_upscaling, RealESRGAN_model_index)

    base_count = len(os.listdir(outpath))
    save_image(image, outpath, f"{base_count:05}", None, '', opts.samples_format, short_filename=True)

    return [image], 0, ''


def run_pnginfo(image):
    info = ''
    for key, text in image.info.items():
        info += f"""
<div>
<p><b>{plaintext_to_html(str(key))}</b></p>
<p>{plaintext_to_html(str(text))}</p>
</div>
""".strip() + "\n"

    if len(info) == 0:
        message = "Nothing found in the image."
        info = f"<div><p>{message}<p></div>"

    return [info]


pnginfo_interface = gr.Interface(
    wrap_gradio_call(run_pnginfo),
    inputs=[
        gr.Image(label="Source", source="upload", interactive=True, type="pil"),
    ],
    outputs=[
        gr.HTML(),
    ],
    allow_flagging="never",
)

opts = Options()
if os.path.exists(config_filename):
    opts.load(config_filename)


def run_settings(*args):
    for key, value in zip(opts.data_labels.keys(), args):
        opts.data[key] = value

    opts.save(config_filename)

    return plaintext_to_html(f'Settings saved @ {datetime.now().strftime("%I:%M:%S")}')


def create_setting_component(key):
    def fun():
        return opts.data[key] if key in opts.data else opts.data_labels[key].default

    info = opts.data_labels[key]
    t = type(info.default)

    if info.component is not None:
        item = info.component(label=info.label, value=fun, **(info.component_args or {}))
    elif t == str:
        item = gr.Textbox(label=info.label, value=fun, lines=1)
    elif t == int:
        item = gr.Number(label=info.label, value=fun)
    elif t == bool:
        item = gr.Checkbox(label=info.label, value=fun)
    else:
        raise Exception(f'bad options item type: {str(t)} for key {key}')

    return item


sd_config = OmegaConf.load(cmd_opts.config)
sd_model = load_model_from_config(sd_config, cmd_opts.ckpt)
sd_model = (sd_model if cmd_opts.no_half else sd_model.half())

if not cmd_opts.lowvram:
    sd_model = sd_model.to(device)

else:
    setup_for_low_vram(sd_model)

model_hijack = StableDiffusionModelHijack()
model_hijack.hijack(sd_model)


def do_generate(
        mode: str,
        prompt: str,
        cfg: float,
        denoise: float,
        sampler_index: int,
        sampler_steps: int,
        batch_count: int,
        batch_size: int,
        input_img,
        resize_mode,
        image_height: int,
        image_width: int,
        custom_code: str,
        input_seed: int,
        facefix: bool,
        facefix_strength: float,
        prompt_matrix: bool,
        loopback: bool,
        upscale: bool,
        inpainting_mask_blur,
        inpainting_mask_content,
        inpainting_image,
):
    if mode == 'Text-to-Image':
        return txt2img(
            prompt=prompt,
            steps=sampler_steps,
            sampler_index=sampler_index,
            use_GFPGAN=facefix,
            strength_GFPGAN=facefix_strength,
            prompt_matrix=prompt_matrix,
            n_iter=batch_count,
            batch_size=batch_size,
            cfg_scale=cfg,
            seed=input_seed,
            height=image_height,
            width=image_width,
            code=custom_code,

        )
    elif mode == 'Image-to-Image' or mode == 'Inpainting':
        return img2img(
            prompt=prompt,
            init_img=input_img,
            init_img_with_mask=inpainting_image,
            ddim_steps=sampler_steps,
            sampler_index=sampler_index,
            mask_blur=inpainting_mask_blur,
            inpainting_fill=inpainting_mask_content,
            use_GFPGAN=facefix,
            strength_GFPGAN=facefix_strength,
            prompt_matrix=prompt_matrix,
            loopback=loopback,
            sd_upscale=upscale,
            n_iter=batch_count,
            batch_size=batch_size,
            cfg_scale=cfg,
            denoising_strength=denoise,
            seed=input_seed,
            height=image_height,
            width=image_width,
            resize_mode=resize_mode,
        )
    elif mode == 'Post-Processing':
        return run_extras(
            image=input_img,
            GFPGAN_strength=facefix_strength,
            RealESRGAN_upscaling=1.0,
            RealESRGAN_model_index=0
        )

    raise Exception('Invalid mode selected')


css_hide_progressbar = \
    """
    .wrap .m-12 svg { display:none!important; }
    .wrap .m-12::before { content:"Loading..." }
    .progress-bar { display:none!important; }
    .meta-text { display:none!important; }
    """

main_css = \
    """
    .output-html p { margin: 0 0.5em; }
    .performance, .params-info, .comments-info { font-size: 0.85em; color: #666; }
    """

# [data-testid="image"] {min-height: 512px !important}
# #generate{width: 100%;}
custom_css = \
    """
    /* hide scrollbars, better scaling for gallery, small padding for main image */
    ::-webkit-scrollbar { display: none }
    #output_gallery {
        min-height: 50vh !important;
        scrollbar-width: none;
    }
    
    #output_gallery > div > img {
        padding-top: 0.5rem;
        padding-right: 0.5rem;
        padding-left: 0.5rem;
    }
    
    /* remove excess padding around prompt textbox, increase font size */
    #prompt_row input { font-size: 16px }
    #prompt_input {
        padding-top: 0.25rem !important;
        padding-bottom: 0rem !important;
        padding-left: 0rem !important;
        padding-right: 0rem !important;
        border-style: none !important;
    }
    
    /* remove excess padding from mode dropdown, change appear to a button */
    #sd_mode {
        padding-top: 0 !important;
        padding-bottom: 0 !important;
        padding-left: 0 !important;
        padding-right: 0 !important;
        border-style: none !important;
    }
    
    #sd_mode > label > select {
        font-weight: 600;
        min-height: 42px;
        max-height: 42px;
        text-align: center;
        font-size: 1rem;
        appearance: none;
        -webkit-appearance: none;
        background-position: left;
        background-size: contain;
        padding-right: 0;
        border-color: rgb(75 85 99 / var(--tw-border-opacity));
    }
    
    /* custom column scaling (odd = right/left, even = center) */
    #body>.col:nth-child(odd) {
        max-width: 450px;
        min-width: 300px;
    }
    #body>.col:nth-child(even) {
        width:250%;
    }
    
    /* better overall scaling + limits */
    .container {
        max-width: min(1600px, 95%);
    }
    """

full_css = main_css + css_hide_progressbar + custom_css

with gr.Blocks(css=full_css, analytics_enabled=False, title='Stable Diffusion WebUI') as demo:
    with gr.Tabs(elem_id='tabs'):
        with gr.TabItem('Stable Diffusion', id='sd_tab'):
            with gr.Row(elem_id='prompt_row'):
                sd_prompt = gr.Textbox(elem_id='prompt_input', placeholder='A corgi wearing a top hat as an oil painting.', lines=1, max_lines=1, show_label=False)

            with gr.Row(elem_id='body').style(equal_height=False):
                # Left Column
                with gr.Column():
                    sd_mode = gr.Dropdown(show_label=False, value='Text-to-Image', choices=['Text-to-Image', 'Image-to-Image', 'Post-Processing', 'Inpainting'], elem_id='sd_mode')

                    sd_image_height = gr.Slider(minimum=64, maximum=2048, step=64, label="Height", elem_id='img_height', value=512)

                    sd_image_width = gr.Slider(minimum=64, maximum=2048, step=64, label="Width", elem_id='img_width', value=512)

                    with gr.Row():
                        sd_batch_count = gr.Number(label='Batch count', precision=0, value=1)
                        sd_batch_size = gr.Number(label='Images per batch', precision=0, value=1)

                    with gr.Group():
                        sd_input_image = gr.Image(label='Input Image', source="upload", interactive=True, type="pil", show_label=True, visible=False)
                        sd_resize_mode = gr.Dropdown(label="Resize mode", choices=["Stretch", "Scale and crop", "Scale and fill"], type="index", value="Stretch", visible=False)

                        sd_inpainting_mask_blur = gr.Slider(label='Inpainting: mask blur', minimum=0, maximum=64, step=1, value=4, visible=False)
                        sd_inpainting_mask_content = gr.Radio(label='Inpainting: masked content', choices=['fill', 'original', 'latent noise', 'latent nothing'], value='fill', type="index", visible=False)

                    with gr.Group():
                        sd_custom_code = gr.Textbox(label="Python script", visible=cmd_opts.allow_code, lines=1)

                # Center Column
                with gr.Column():
                    sd_output_image = gr.Gallery(show_label=False, elem_id='output_gallery').style(grid=3)
                    sd_inpainting_image = gr.Image(label='Inpainting image', source="upload", interactive=True, type="pil", tool="sketch", visible=False)
                    sd_output_html = gr.HTML()

                # Right Column
                with gr.Column():
                    sd_generate = gr.Button('Generate', variant='primary').style(full_width=True)

                    with gr.Group():
                        sd_sampling_method = gr.Dropdown(label='Sampling method', choices=[x.name for x in samplers], value=samplers[0].name, type="index")
                        sd_sampling_steps = gr.Slider(label="Sampling steps", value=18, minimum=1, maximum=100, step=1)
                        sd_cfg = gr.Slider(label='Prompt similarity (CFG)', value=8.0, minimum=1.0, maximum=15.0, step=0.5)
                        sd_denoise = gr.Slider(label='Denoising strength (DNS)', value=0.75, minimum=0.0, maximum=1.0, step=0.01, visible=False)
                        sd_input_seed = gr.Number(label="Seed", value=-1, precision=0)

                    sd_facefix = \
                        gr.Checkbox(label='GFPGAN', value=False, visible=have_gfpgan)
                    sd_facefix_strength = \
                        gr.Slider(minimum=0.0, maximum=1.0, step=0.1, label="Strength", value=1, interactive=have_gfpgan, visible=False)

                    # TODO: Change to 'Enable syntactic prompts'
                    sd_matrix = \
                        gr.Checkbox(label='Create prompt matrix', value=False)

                    sd_loopback = \
                        gr.Checkbox(label='Output loopback', value=False, visible=False)
                    sd_upscale = \
                        gr.Checkbox(label='Stable diffusion upscale', value=False, visible=False)

        with gr.TabItem('Settings', id='settings_tab'):
            # TODO: Add HTML output to indicate settings saved
            sd_settings = [create_setting_component(key) for key in opts.data_labels.keys()]
            sd_save_settings = \
                gr.Button('Save')
            sd_confirm_settings = \
                gr.HTML()


    def mode_change(mode: str, facefix: bool):
        is_img2img = (mode == 'Image-to-Image')
        is_txt2img = (mode == 'Text-to-Image')
        is_inpainting = (mode == 'Inpainting')
        is_pp = (mode == 'Post-Processing')

        return {
            sd_cfg: gr.update(visible=is_img2img or is_txt2img or is_inpainting),
            sd_denoise: gr.update(visible=is_img2img or is_inpainting),
            sd_sampling_method: gr.update(visible=is_img2img or is_txt2img or is_inpainting),
            sd_sampling_steps: gr.update(visible=is_img2img or is_txt2img or is_inpainting),
            sd_batch_count: gr.update(visible=is_img2img or is_txt2img or is_inpainting),
            sd_batch_size: gr.update(visible=is_img2img or is_txt2img or is_inpainting),
            sd_input_image: gr.update(visible=is_img2img or is_pp),
            sd_resize_mode: gr.update(visible=is_img2img or is_inpainting),
            sd_image_height: gr.update(visible=is_img2img or is_txt2img or is_inpainting),
            sd_image_width: gr.update(visible=is_img2img or is_txt2img or is_inpainting),
            sd_custom_code: gr.update(visible=is_txt2img and cmd_opts.allow_code),
            sd_input_seed: gr.update(visible=is_img2img or is_txt2img or is_inpainting),
            sd_facefix: gr.update(visible=True),
            # TODO: see above, but for facefix
            sd_facefix_strength: gr.update(visible=facefix),
            sd_matrix: gr.update(visible=is_img2img or is_txt2img),
            sd_loopback: gr.update(visible=is_img2img),
            sd_upscale: gr.update(visible=is_img2img),
            sd_inpainting_mask_blur: gr.update(visible=is_inpainting),
            sd_inpainting_mask_content: gr.update(visible=is_inpainting),
            sd_inpainting_image: gr.update(visible=is_inpainting),

        }


    sd_mode.change(
        fn=mode_change,
        inputs=[
            sd_mode,
            sd_facefix
        ],
        outputs=[
            sd_cfg,
            sd_denoise,
            sd_sampling_method,
            sd_sampling_steps,
            sd_batch_count,
            sd_batch_size,
            sd_input_image,
            sd_resize_mode,
            sd_image_height,
            sd_image_width,
            sd_custom_code,
            sd_input_seed,
            sd_facefix,
            sd_facefix_strength,
            sd_matrix,
            sd_loopback,
            sd_upscale,
            sd_inpainting_mask_blur,
            sd_inpainting_mask_content,
            sd_inpainting_image,
        ]
    )

    do_generate_args = dict(
        fn=wrap_gradio_call(do_generate),
        inputs=[
            sd_mode,
            sd_prompt,
            sd_cfg,
            sd_denoise,
            sd_sampling_method,
            sd_sampling_steps,
            sd_batch_count,
            sd_batch_size,
            sd_input_image,
            sd_resize_mode,
            sd_image_height,
            sd_image_width,
            sd_custom_code,
            sd_input_seed,
            sd_facefix,
            sd_facefix_strength,
            sd_matrix,
            sd_loopback,
            sd_upscale,
            sd_inpainting_mask_blur,
            sd_inpainting_mask_content,
            sd_inpainting_image,
        ],
        outputs=[
            sd_output_image,
            sd_output_html
        ]
    )

    sd_prompt.submit(**do_generate_args)
    sd_generate.click(**do_generate_args)

    sd_batch_count.submit(
        lambda value: value if value > 0 else 1,
        inputs=sd_batch_count,
        outputs=sd_batch_count
    )

    sd_batch_size.submit(
        lambda value: value if value > 0 else 1,
        inputs=sd_batch_size,
        outputs=sd_batch_size
    )

    sd_facefix.change(
        lambda checked: gr.update(visible=checked),
        inputs=sd_facefix,
        outputs=sd_facefix_strength
    )

    sd_save_settings.click(
        fn=run_settings,
        inputs=sd_settings,
        outputs=sd_confirm_settings
    )

demo.queue(concurrency_count=1)
demo.launch()
