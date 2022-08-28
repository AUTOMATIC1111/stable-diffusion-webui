import argparse
import os
import sys
from collections import namedtuple
import torch
import torch.nn as nn
import numpy as np
import gradio as gr
from omegaconf import OmegaConf
from PIL import Image, ImageFont, ImageDraw, PngImagePlugin
from torch import autocast
import mimetypes
import random
import math
import html
import time
import json
import traceback

import k_diffusion.sampling
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

try:
    # this silences the annoying "Some weights of the model checkpoint were not used when initializing..." message at start.

    from transformers import logging
    logging.set_verbosity_error()
except Exception:
    pass

# this is a fix for Windows users. Without it, javascript files will be served with text/html content-type and the bowser will not show any UI
mimetypes.init()
mimetypes.add_type('application/javascript', '.js')

# some of those options should not be changed at all because they would break the model, so I removed them from options.
opt_C = 4
opt_f = 8

LANCZOS = (Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS)
invalid_filename_chars = '<>:"/\\|?*\n'
config_filename = "config.json"

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="configs/stable-diffusion/v1-inference.yaml", help="path to config which constructs model",)
parser.add_argument("--ckpt", type=str, default="models/ldm/stable-diffusion-v1/model.ckpt", help="path to checkpoint of model",)
parser.add_argument("--gfpgan-dir", type=str, help="GFPGAN directory", default=('./src/gfpgan' if os.path.exists('./src/gfpgan') else './GFPGAN'))
parser.add_argument("--no-half", action='store_true', help="do not switch the model to 16-bit floats")
parser.add_argument("--no-progressbar-hiding", action='store_true', help="do not hide progressbar in gradio UI (we hide it because it slows down ML if you have hardware accleration in browser)")
parser.add_argument("--max-batch-count", type=int, default=16, help="maximum batch count value for the UI")
parser.add_argument("--embeddings-dir", type=str, default='embeddings', help="embeddings dirtectory for textual inversion (default: embeddings)")
parser.add_argument("--allow-code", action='store_true', help="allow custom script execution from webui")

cmd_opts = parser.parse_args()

css_hide_progressbar = """
.wrap .m-12 svg { display:none!important; }
.wrap .m-12::before { content:"Loading..." }
.progress-bar { display:none!important; }
.meta-text { display:none!important; }
"""

SamplerData = namedtuple('SamplerData', ['name', 'constructor'])
samplers = [
    *[SamplerData(x[0], lambda funcname=x[1]: KDiffusionSampler(funcname)) for x in [
        ('LMS', 'sample_lms'),
        ('Heun', 'sample_heun'),
        ('Euler', 'sample_euler'),
        ('Euler ancestral', 'sample_euler_ancestral'),
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
    print("Error loading Real-ESRGAN:", file=sys.stderr)
    print(traceback.format_exc(), file=sys.stderr)

    realesrgan_models = [RealesrganModelInfo('None', '', 0, None)]
    have_realesrgan = False

sd_upscalers = {
    "RealESRGAN": lambda img: upscale_with_realesrgan(img, 2, 0),
    "Lanczos": lambda img: img.resize((img.width*2, img.height*2), resample=LANCZOS),
    "None": lambda img: img
}


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
        "grid_format": OptionInfo('png', 'File format for grids'),
        "grid_extended_filename": OptionInfo(False, "Add extended info (seed, prompt) to filename when saving grid"),
        "n_rows": OptionInfo(-1, "Grid row count; use -1 for autodetect and 0 for it to be same as batch size", gr.Slider, {"minimum": -1, "maximum": 16, "step": 1}),
        "jpeg_quality": OptionInfo(80, "Quality for saved jpeg images", gr.Slider, {"minimum": 1, "maximum": 100, "step": 1}),
        "export_for_4chan": OptionInfo(True, "If PNG image is larger than 4MB or any dimension is larger than 4000, downscale and save copy as JPG"),
        "enable_pnginfo": OptionInfo(True, "Save text information about generation parameters as chunks to png files"),
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

    model.cuda()
    model.eval()
    return model


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


def plaintext_to_html(text):
    text = "".join([f"<p>{html.escape(x)}</p>\n" for x in text.split('\n')])
    return text


def load_gfpgan():
    model_name = 'GFPGANv1.3'
    model_path = os.path.join(cmd_opts.gfpgan_dir, 'experiments/pretrained_models', model_name + '.pth')
    if not os.path.isfile(model_path):
        raise Exception("GFPGAN model not found at path "+model_path)

    sys.path.append(os.path.abspath(cmd_opts.gfpgan_dir))
    from gfpgan import GFPGANer

    return GFPGANer(model_path=model_path, upscale=1, arch='clean', channel_multiplier=2, bg_upsampler=None)


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

            if x+tile_w >= w:
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
                drawing.line((draw_x - line.size[0]//2, draw_y + line.size[1]//2, draw_x + line.size[0]//2, draw_y + line.size[1]//2), fill=color_inactive, width=4)

            draw_y += line.size[1] + line_spacing

    fontsize = (width + height) // 25
    line_spacing = fontsize // 2
    fnt = ImageFont.truetype("arial.ttf", fontsize)
    color_active = (0, 0, 0)
    color_inactive = (153, 153, 153)

    pad_left = width * 3 // 4 if len(hor_texts) > 1 else 0

    cols = im.width // width
    rows = im.height // height

    assert cols == len(hor_texts), f'bad number of horizontal texts: {len(hor_texts)}; must be {cols}'
    assert rows == len(ver_texts), f'bad number of vertical texts: {len(ver_texts)}; must be {rows}'

    calc_img = Image.new("RGB", (1, 1), "white")
    calc_d = ImageDraw.Draw(calc_img)

    for texts in hor_texts + ver_texts:
        items = [] + texts
        texts.clear()

        for line in items:
            wrapped = wrap(calc_d, line.text, fnt, width)
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


def wrap_gradio_call(func):
    def f(*p1, **p2):
        t = time.perf_counter()
        res = list(func(*p1, **p2))
        elapsed = time.perf_counter() - t

        # last item is always HTML
        res[-1] = res[-1] + f"<p class='performance'>Time taken: {elapsed:.2f}s</p>"

        return tuple(res)

    return f


GFPGAN = None
if os.path.exists(cmd_opts.gfpgan_dir):
    try:
        GFPGAN = load_gfpgan()
        print("Loaded GFPGAN")
    except Exception:
        print("Error loading GFPGAN:", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)


class StableDiffusionModelHijack:
    ids_lookup = {}
    word_embeddings = {}
    word_embeddings_checksums = {}
    fixes = None
    comments = None
    dir_mtime = None

    def load_textual_inversion_embeddings(self, dirname, model):
        mt = os.path.getmtime(dirname)
        if self.dir_mtime is not None and mt <= self.dir_mtime:
            return

        self.dir_mtime = mt
        self.ids_lookup.clear()
        self.word_embeddings.clear()

        tokenizer = model.cond_stage_model.tokenizer

        def const_hash(a):
            r = 0
            for v in a:
                r = (r * 281 ^ int(v) * 997) & 0xFFFFFFFF
            return r

        def process_file(path, filename):
            name = os.path.splitext(filename)[0]

            data = torch.load(path)
            param_dict = data['string_to_param']
            assert len(param_dict) == 1, 'embedding file has multiple terms in it'
            emb = next(iter(param_dict.items()))[1].reshape(768)
            self.word_embeddings[name] = emb
            self.word_embeddings_checksums[name] = f'{const_hash(emb)&0xffff:04x}'

            ids = tokenizer([name], add_special_tokens=False)['input_ids'][0]

            first_id = ids[0]
            if first_id not in self.ids_lookup:
                self.ids_lookup[first_id] = []
            self.ids_lookup[first_id].append((ids, name))

        for fn in os.listdir(dirname):
            try:
                process_file(os.path.join(dirname, fn), fn)
            except Exception:
                print(f"Error loading emedding {fn}:", file=sys.stderr)
                print(traceback.format_exc(), file=sys.stderr)
                continue

        print(f"Loaded a total of {len(self.word_embeddings)} text inversion embeddings.")

    def hijack(self, m):
        model_embeddings = m.cond_stage_model.transformer.text_model.embeddings

        model_embeddings.token_embedding = EmbeddingsWithFixes(model_embeddings.token_embedding, self)
        m.cond_stage_model = FrozenCLIPEmbedderWithCustomWords(m.cond_stage_model, self)


class FrozenCLIPEmbedderWithCustomWords(torch.nn.Module):
    def __init__(self, wrapped, hijack):
        super().__init__()
        self.wrapped = wrapped
        self.hijack = hijack
        self.tokenizer = wrapped.tokenizer
        self.max_length = wrapped.max_length
        self.token_mults = {}

        tokens_with_parens = [(k, v) for k, v in self.tokenizer.get_vocab().items() if '(' in k or ')' in k or '[' in k or ']' in k]
        for text, ident in tokens_with_parens:
            mult = 1.0
            for c in text:
                if c == '[':
                    mult /= 1.1
                if c == ']':
                    mult *= 1.1
                if c == '(':
                    mult *= 1.1
                if c == ')':
                    mult /= 1.1

            if mult != 1.0:
                self.token_mults[ident] = mult

    def forward(self, text):
        self.hijack.fixes = []
        self.hijack.comments = []
        remade_batch_tokens = []
        id_start = self.wrapped.tokenizer.bos_token_id
        id_end = self.wrapped.tokenizer.eos_token_id
        maxlen = self.wrapped.max_length - 2
        used_custom_terms = []

        cache = {}
        batch_tokens = self.wrapped.tokenizer(text, truncation=False, add_special_tokens=False)["input_ids"]
        batch_multipliers = []
        for tokens in batch_tokens:
            tuple_tokens = tuple(tokens)

            if tuple_tokens in cache:
                remade_tokens, fixes, multipliers = cache[tuple_tokens]
            else:
                fixes = []
                remade_tokens = []
                multipliers = []
                mult = 1.0

                i = 0
                while i < len(tokens):
                    token = tokens[i]

                    possible_matches = self.hijack.ids_lookup.get(token, None)

                    mult_change = self.token_mults.get(token)
                    if mult_change is not None:
                        mult *= mult_change
                    elif possible_matches is None:
                        remade_tokens.append(token)
                        multipliers.append(mult)
                    else:
                        found = False
                        for ids, word in possible_matches:
                            if tokens[i:i+len(ids)] == ids:
                                fixes.append((len(remade_tokens), word))
                                remade_tokens.append(777)
                                multipliers.append(mult)
                                i += len(ids) - 1
                                found = True
                                used_custom_terms.append((word, self.hijack.word_embeddings_checksums[word]))
                                break

                        if not found:
                            remade_tokens.append(token)
                            multipliers.append(mult)

                    i += 1

                if len(remade_tokens) > maxlen - 2:
                    vocab = {v: k for k, v in self.wrapped.tokenizer.get_vocab().items()}
                    ovf = remade_tokens[maxlen - 2:]
                    overflowing_words = [vocab.get(int(x), "") for x in ovf]
                    overflowing_text = self.wrapped.tokenizer.convert_tokens_to_string(''.join(overflowing_words))

                    self.hijack.comments.append(f"Warning: too many input tokens; some ({len(overflowing_words)}) have been truncated:\n{overflowing_text}\n")

                remade_tokens = remade_tokens + [id_end] * (maxlen - 2 - len(remade_tokens))
                remade_tokens = [id_start] + remade_tokens[0:maxlen-2] + [id_end]
                cache[tuple_tokens] = (remade_tokens, fixes, multipliers)

            multipliers = multipliers + [1.0] * (maxlen - 2 - len(multipliers))
            multipliers = [1.0] + multipliers[0:maxlen - 2] + [1.0]

            remade_batch_tokens.append(remade_tokens)
            self.hijack.fixes.append(fixes)
            batch_multipliers.append(multipliers)

        if len(used_custom_terms) > 0:
            self.hijack.comments.append("Used custom terms: " + ", ".join([f'{word} [{checksum}]' for word, checksum in used_custom_terms]))

        tokens = torch.asarray(remade_batch_tokens).to(self.wrapped.device)
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
    def __init__(self, outpath=None, prompt="", seed=-1, sampler_index=0, batch_size=1, n_iter=1, steps=50, cfg_scale=7.0, width=512, height=512, prompt_matrix=False, use_GFPGAN=False, do_not_save_samples=False, do_not_save_grid=False, extra_generation_params=None):
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
        self.extra_generation_params: dict = extra_generation_params

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


Processed = namedtuple('Processed', ['images','seed', 'info'])


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
        "CFG scale": p.cfg_scale,
        "Seed": seed,
        "GFPGAN": ("GFPGAN" if p.use_GFPGAN and GFPGAN is not None else None)
    }

    if p.extra_generation_params is not None:
        generation_params.update(p.extra_generation_params)

    generation_params_text = ", ".join([k if k == v else f'{k}: {v}' for k, v in generation_params.items() if v is not None])

    def infotext():
        return f"{prompt}\n{generation_params_text}".strip() + "".join(["\n\n" + x for x in comments])

    if os.path.exists(cmd_opts.embeddings_dir):
        model_hijack.load_textual_inversion_embeddings(cmd_opts.embeddings_dir, model)

    output_images = []
    with torch.no_grad(), autocast("cuda"), model.ema_scope():
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
                    x_sample = 255. * np.moveaxis(x_sample.cpu().numpy(), 0, 2)
                    x_sample = x_sample.astype(np.uint8)

                    if p.use_GFPGAN and GFPGAN is not None:
                        torch_gc()
                        cropped_faces, restored_faces, restored_img = GFPGAN.enhance(x_sample, has_aligned=False, only_center_face=False, paste_back=True)
                        x_sample = restored_img

                    image = Image.fromarray(x_sample)

                    if not p.do_not_save_samples:
                        save_image(image, sample_path, f"{base_count:05}", seeds[i], prompts[i], opts.samples_format, info=infotext())

                    output_images.append(image)
                    base_count += 1

        if (p.prompt_matrix or opts.grid_save) and not p.do_not_save_grid:
            if p.prompt_matrix:
                grid = image_grid(output_images, p.batch_size, rows=1 << ((len(prompt_matrix_parts)-1)//2))

                try:
                    grid = draw_prompt_matrix(grid, p.width, p.height, prompt_matrix_parts)
                except Exception:
                    import traceback
                    print("Error creating prompt_matrix text:", file=sys.stderr)
                    print(traceback.format_exc(), file=sys.stderr)

                output_images.insert(0, grid)
            else:
                grid = image_grid(output_images, p.batch_size)

            save_image(grid, p.outpath, f"grid-{grid_count:04}", seed, prompt, opts.grid_format, info=infotext(), short_filename=not opts.grid_extended_filename)
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

def txt2img(prompt: str, steps: int, sampler_index: int, use_GFPGAN: bool, prompt_matrix: bool, n_iter: int, batch_size: int, cfg_scale: float, seed: int, height: int, width: int, code: str):
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
        use_GFPGAN=use_GFPGAN
    )

    if code != '' and cmd_opts.allow_code:
        p.do_not_save_grid = True
        p.do_not_save_samples = True

        display_result_data = [[], -1, ""]
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

    return processed.images, processed.seed, plaintext_to_html(processed.info)


class Flagging(gr.FlaggingCallback):

    def setup(self, components, flagging_dir: str):
        pass

    def flag(self, flag_data, flag_option=None, flag_index=None, username=None):
        import csv

        os.makedirs("log/images", exist_ok=True)

        # those must match the "txt2img" function
        prompt, ddim_steps, sampler_name, use_gfpgan, prompt_matrix, ddim_eta, n_iter, n_samples, cfg_scale, request_seed, height, width, code, images, seed, comment = flag_data

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
                filename = "log/images/"+filename_base + ("" if len(images) == 1 else "-"+str(i+1)) + ".png"

                if filedata.startswith("data:image/png;base64,"):
                    filedata = filedata[len("data:image/png;base64,"):]

                with open(filename, "wb") as imgfile:
                    imgfile.write(base64.decodebytes(filedata.encode('utf-8')))

                filenames.append(filename)

            writer.writerow([prompt, seed, width, height, cfg_scale, ddim_steps, filenames[0]])

        print("Logged:", filenames[0])


txt2img_interface = gr.Interface(
    wrap_gradio_call(txt2img),
    inputs=[
        gr.Textbox(label="Prompt", placeholder="A corgi wearing a top hat as an oil painting.", lines=1),
        gr.Slider(minimum=1, maximum=150, step=1, label="Sampling Steps", value=50),
        gr.Radio(label='Sampling method', choices=[x.name for x in samplers], value=samplers[0].name, type="index"),
        gr.Checkbox(label='Fix faces using GFPGAN', value=False, visible=GFPGAN is not None),
        gr.Checkbox(label='Create prompt matrix (separate multiple prompts using |, and get all combinations of them)', value=False),
        gr.Slider(minimum=1, maximum=cmd_opts.max_batch_count, step=1, label='Batch count (how many batches of images to generate)', value=1),
        gr.Slider(minimum=1, maximum=8, step=1, label='Batch size (how many images are in a batch; memory-hungry)', value=1),
        gr.Slider(minimum=1.0, maximum=15.0, step=0.5, label='Classifier Free Guidance Scale (how strongly the image should follow the prompt)', value=7.0),
        gr.Number(label='Seed', value=-1),
        gr.Slider(minimum=64, maximum=2048, step=64, label="Height", value=512),
        gr.Slider(minimum=64, maximum=2048, step=64, label="Width", value=512),
        gr.Textbox(label="Python script", visible=cmd_opts.allow_code, lines=1)
    ],
    outputs=[
        gr.Gallery(label="Images"),
        gr.Number(label='Seed'),
        gr.HTML(),
    ],
    title="Stable Diffusion Text-to-Image",
    flagging_callback=Flagging()
)


class StableDiffusionProcessingImg2Img(StableDiffusionProcessing):
    sampler = None

    def __init__(self, init_images=None, resize_mode=0, denoising_strength=0.75, **kwargs):
        super().__init__(**kwargs)

        self.init_images = init_images
        self.resize_mode: int = resize_mode
        self.denoising_strength: float = denoising_strength
        self.init_latent = None

    def init(self):
        self.sampler = samplers_for_img2img[self.sampler_index].constructor()

        imgs = []
        for img in self.init_images:
            image = img.convert("RGB")
            image = resize_image(self.resize_mode, image, self.width, self.height)
            image = np.array(image).astype(np.float32) / 255.0
            image = np.moveaxis(image, 2, 0)
            imgs.append(image)

        if len(imgs) == 1:
            batch_images = np.expand_dims(imgs[0], axis=0).repeat(self.batch_size, axis=0)
        elif len(imgs) <= self.batch_size:
            self.batch_size = len(imgs)
            batch_images = np.array(imgs)
        else:
            raise RuntimeError(f"bad number of images passed: {len(imgs)}; expecting {self.batch_size} or less")

        image = torch.from_numpy(batch_images)
        image = 2. * image - 1.
        image = image.to(device)

        self.init_latent = sd_model.get_first_stage_encoding(sd_model.encode_first_stage(image))

    def sample(self, x, conditioning, unconditional_conditioning):
        t_enc = int(self.denoising_strength * self.steps)

        sigmas = self.sampler.model_wrap.get_sigmas(self.steps)
        noise = x * sigmas[self.steps - t_enc - 1]

        xi = self.init_latent + noise
        sigma_sched = sigmas[self.steps - t_enc - 1:]
        samples_ddim = self.sampler.func(self.sampler.model_wrap_cfg, xi, sigma_sched, extra_args={'cond': conditioning, 'uncond': unconditional_conditioning, 'cond_scale': self.cfg_scale}, disable=False)
        return samples_ddim


def img2img(prompt: str, init_img, ddim_steps: int, sampler_index: int, use_GFPGAN: bool, prompt_matrix, loopback: bool, sd_upscale: bool, n_iter: int, batch_size: int, cfg_scale: float, denoising_strength: float, seed: int, height: int, width: int, resize_mode: int):
    outpath = opts.outdir or "outputs/img2img-samples"

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
        init_images=[init_img],
        resize_mode=resize_mode,
        denoising_strength=denoising_strength,
        extra_generation_params={"Denoising Strength": denoising_strength}
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

        save_image(grid, outpath, f"grid-{grid_count:04}", initial_seed, prompt, opts.grid_format, info=info, short_filename=not opts.grid_extended_filename)

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
            p.init_images = work[i*p.batch_size:(i+1)*p.batch_size]

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
        save_image(combined_image, outpath, f"grid-{grid_count:04}", initial_seed, prompt, opts.grid_format, info=initial_info, short_filename=not opts.grid_extended_filename)

        processed = Processed([combined_image], initial_seed, initial_info)

    else:
        processed = process_images(p)

    return processed.images, processed.seed, plaintext_to_html(processed.info)


sample_img2img = "assets/stable-samples/img2img/sketch-mountains-input.jpg"
sample_img2img = sample_img2img if os.path.exists(sample_img2img) else None

img2img_interface = gr.Interface(
    wrap_gradio_call(img2img),
    inputs=[
        gr.Textbox(placeholder="A fantasy landscape, trending on artstation.", lines=1),
        gr.Image(value=sample_img2img, source="upload", interactive=True, type="pil"),
        gr.Slider(minimum=1, maximum=150, step=1, label="Sampling Steps", value=50),
        gr.Radio(label='Sampling method', choices=[x.name for x in samplers_for_img2img], value=samplers_for_img2img[0].name, type="index"),
        gr.Checkbox(label='Fix faces using GFPGAN', value=False, visible=GFPGAN is not None),
        gr.Checkbox(label='Create prompt matrix (separate multiple prompts using |, and get all combinations of them)', value=False),
        gr.Checkbox(label='Loopback (use images from previous batch when creating next batch)', value=False),
        gr.Checkbox(label='Stable Diffusion upscale', value=False),
        gr.Slider(minimum=1, maximum=cmd_opts.max_batch_count, step=1, label='Batch count (how many batches of images to generate)', value=1),
        gr.Slider(minimum=1, maximum=8, step=1, label='Batch size (how many images are in a batch; memory-hungry)', value=1),
        gr.Slider(minimum=1.0, maximum=15.0, step=0.5, label='Classifier Free Guidance Scale (how strongly the image should follow the prompt)', value=7.0),
        gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label='Denoising Strength', value=0.75),
        gr.Number(label='Seed', value=-1),
        gr.Slider(minimum=64, maximum=2048, step=64, label="Height", value=512),
        gr.Slider(minimum=64, maximum=2048, step=64, label="Width", value=512),
        gr.Radio(label="Resize mode", choices=["Just resize", "Crop and resize", "Resize and fill"], type="index", value="Just resize")
    ],
    outputs=[
        gr.Gallery(),
        gr.Number(label='Seed'),
        gr.HTML(),
    ],
    allow_flagging="never",
)


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

    image = image.convert("RGB")

    outpath = opts.outdir or "outputs/extras-samples"

    if GFPGAN is not None and GFPGAN_strength > 0:
        cropped_faces, restored_faces, restored_img = GFPGAN.enhance(np.array(image, dtype=np.uint8), has_aligned=False, only_center_face=False, paste_back=True)
        res = Image.fromarray(restored_img)

        if GFPGAN_strength < 1.0:
            res = Image.blend(image, res, GFPGAN_strength)

        image = res

    if have_realesrgan and RealESRGAN_upscaling != 1.0:
        image = upscale_with_realesrgan(image, RealESRGAN_upscaling, RealESRGAN_model_index)

    base_count = len(os.listdir(outpath))
    save_image(image, outpath, f"{base_count:05}", None, '', opts.samples_format, short_filename=True)

    return image, 0, ''


extras_interface = gr.Interface(
    wrap_gradio_call(run_extras),
    inputs=[
        gr.Image(label="Source", source="upload", interactive=True, type="pil"),
        gr.Slider(minimum=0.0, maximum=1.0, step=0.001, label="GFPGAN strength", value=1, interactive=GFPGAN is not None),
        gr.Slider(minimum=1.0, maximum=4.0, step=0.05, label="Real-ESRGAN upscaling", value=2, interactive=have_realesrgan),
        gr.Radio(label='Real-ESRGAN model', choices=[x.name for x in realesrgan_models], value=realesrgan_models[0].name, type="index", interactive=have_realesrgan),
    ],
    outputs=[
        gr.Image(label="Result"),
        gr.Number(label='Seed', visible=False),
        gr.HTML(),
    ],
    allow_flagging="never",
)

opts = Options()
if os.path.exists(config_filename):
    opts.load(config_filename)


def run_settings(*args):
    up = []

    for key, value, comp in zip(opts.data_labels.keys(), args, settings_interface.input_components):
        opts.data[key] = value
        up.append(comp.update(value=value))

    opts.save(config_filename)

    return 'Settings saved.', ''


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


settings_interface = gr.Interface(
    run_settings,
    inputs=[create_setting_component(key) for key in opts.data_labels.keys()],
    outputs=[
        gr.Textbox(label='Result'),
        gr.HTML(),
    ],
    title=None,
    description=None,
    allow_flagging="never",
)

interfaces = [
    (txt2img_interface, "txt2img"),
    (img2img_interface, "img2img"),
    (extras_interface, "Extras"),
    (settings_interface, "Settings"),
]

sd_config = OmegaConf.load(cmd_opts.config)
sd_model = load_model_from_config(sd_config, cmd_opts.ckpt)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
sd_model = (sd_model if cmd_opts.no_half else sd_model.half()).to(device)

model_hijack = StableDiffusionModelHijack()
model_hijack.hijack(sd_model)

demo = gr.TabbedInterface(
    interface_list=[x[0] for x in interfaces],
    tab_names=[x[1] for x in interfaces],
    css=("" if cmd_opts.no_progressbar_hiding else css_hide_progressbar) + """
.output-html p {margin: 0 0.5em;}
.performance { font-size: 0.85em; color: #444; }
"""
)

demo.launch()
