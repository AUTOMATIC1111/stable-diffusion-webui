import argparse, os, sys, glob
from collections import namedtuple

import torch
import torch.nn as nn
import numpy as np
import gradio as gr
from omegaconf import OmegaConf
from PIL import Image, ImageFont, ImageDraw, PngImagePlugin
from itertools import islice
from einops import rearrange, repeat
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
import ldm.modules.encoders.modules

try:
    # this silences the annoying "Some weights of the model checkpoint were not used when initializing..." message at start.

    from transformers import logging
    logging.set_verbosity_error()
except:
    pass

# this is a fix for Windows users. Without it, javascript files will be served with text/html content-type and the bowser will not show any UI
mimetypes.init()
mimetypes.add_type('application/javascript', '.js')

# some of those options should not be changed at all because they would break the model, so I removed them from options.
opt_C = 4
opt_f = 8

LANCZOS = (Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS)
invalid_filename_chars = '<>:"/\|?*\n'
config_filename = "config.json"

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="configs/stable-diffusion/v1-inference.yaml", help="path to config which constructs model",)
parser.add_argument("--ckpt", type=str, default="models/ldm/stable-diffusion-v1/model.ckpt", help="path to checkpoint of model",)
parser.add_argument("--gfpgan-dir", type=str, help="GFPGAN directory", default=('./src/gfpgan' if os.path.exists('./src/gfpgan') else './GFPGAN')) # i disagree with where you're putting it but since all guidefags are doing it this way, there you go
parser.add_argument("--no-half", action='store_true', help="do not switch the model to 16-bit floats")
parser.add_argument("--no-progressbar-hiding", action='store_true', help="do not hide progressbar in gradio UI (we hide it because it slows down ML if you have hardware accleration in browser)")
parser.add_argument("--max-batch-count", type=int, default=16, help="maximum batch count value for the UI")
parser.add_argument("--embeddings-dir", type=str, default='embeddings', help="embeddings dirtectory for textual inversion (default: embeddings)")

cmd_opts = parser.parse_args()

css_hide_progressbar = """
.wrap .m-12 svg { display:none!important; }
.wrap .m-12::before { content:"Loading..." }
.progress-bar { display:none!important; }
.meta-text { display:none!important; }
"""

SamplerData = namedtuple('SamplerData', ['name', 'constructor'])
samplers = [
    *[SamplerData(x[0], lambda model: KDiffusionSampler(model, x[1])) for x in [
        ('LMS', 'sample_lms'),
        ('Heun', 'sample_heun'),
        ('Euler', 'sample_euler'),
        ('Euler ancestral', 'sample_euler_ancestral'),
        ('DPM 2', 'sample_dpm_2'),
        ('DPM 2 Ancestral', 'sample_dpm_2_ancestral'),
    ] if hasattr(k_diffusion.sampling, x[1])],
    SamplerData('DDIM', lambda model: DDIMSampler(model)),
    SamplerData('PLMS', lambda model: PLMSSampler(model)),
]

RealesrganModelInfo = namedtuple("RealesrganModelInfo", ["name", "location", "model", "netscale"])

try:
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
    from realesrgan.archs.srvgg_arch import SRVGGNetCompact

    realesrgan_models = [
        RealesrganModelInfo(
            name="Real-ESRGAN 2x plus",
            location="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
            netscale=2, model=lambda: RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        ),
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
    ]
    have_realesrgan = True
except:
    print("Error loading Real-ESRGAN:", file=sys.stderr)
    print(traceback.format_exc(), file=sys.stderr)

    realesrgan_models = [RealesrganModelInfo('None', '', 0, None)]
    have_realesrgan = False


class Options:
    data = None
    data_labels = {
        "outdir": ("", "Output dictectory; if empty, defaults to 'outputs/*'"),
        "samples_save": (True, "Save indiviual samples"),
        "samples_format": ('png', 'File format for indiviual samples'),
        "grid_save": (True, "Save image grids"),
        "grid_format": ('png', 'File format for grids'),
        "grid_extended_filename": (False, "Add extended info (seed, prompt) to filename when saving grid"),
        "n_rows": (-1, "Grid row count; use -1 for autodetect and 0 for it to be same as batch size", -1, 16),
        "jpeg_quality": (80, "Quality for saved jpeg images", 1, 100),
        "verify_input": (True, "Check input, and produce warning if it's too long"),
        "enable_pnginfo": (True, "Save text information about generation parameters as chunks to png files"),
        "prompt_matrix_add_to_start": (True, "In prompt matrix, add the variable combination of text to the start of the prompt, rather than the end"),
    }

    def __init__(self):
        self.data = {k: v[0] for k, v in self.data_labels.items()}

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
            return self.data_labels[item][0]

        return super(Options, self).__getattribute__(item)

    def save(self, filename):
        with open(filename, "w", encoding="utf8") as file:
            json.dump(self.data, file)

    def load(self, filename):
        with open(filename, "r", encoding="utf8") as file:
            self.data = json.load(file)


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


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
    def __init__(self, m, funcname):
        self.model = m
        self.model_wrap = k_diffusion.external.CompVisDenoiser(m)
        self.funcname = funcname

    def sample(self, S, conditioning, batch_size, shape, verbose, unconditional_guidance_scale, unconditional_conditioning, eta, x_T):
        sigmas = self.model_wrap.get_sigmas(S)
        x = x_T * sigmas[0]
        model_wrap_cfg = CFGDenoiser(self.model_wrap)

        fun = getattr(k_diffusion.sampling, self.funcname)
        samples_ddim = fun(model_wrap_cfg, x, sigmas, extra_args={'cond': conditioning, 'uncond': unconditional_conditioning, 'cond_scale': unconditional_guidance_scale}, disable=False)

        return samples_ddim, None


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


def save_image(image, path, basename, seed, prompt, extension, info=None, short_filename=False):
    prompt = sanitize_filename_part(prompt)

    if short_filename:
        filename = f"{basename}.{extension}"
    else:
        filename = f"{basename}-{seed}-{prompt[:128]}.{extension}"

    if extension == 'png' and opts.enable_pnginfo and info is not None:
        pnginfo = PngImagePlugin.PngInfo()
        pnginfo.add_text("parameters", info)
    else:
        pnginfo = None

    image.save(os.path.join(path, filename), quality=opts.jpeg_quality, pnginfo=pnginfo)


def sanitize_filename_part(text):
    return text.replace(' ', '_').translate({ord(x): '' for x in invalid_filename_chars})[:128]


def plaintext_to_html(text):
    text = "".join([f"<p>{html.escape(x)}</p>\n" for x in text.split('\n')])
    return text


def load_GFPGAN():
    model_name = 'GFPGANv1.3'
    model_path = os.path.join(cmd_opts.gfpgan_dir, 'experiments/pretrained_models', model_name + '.pth')
    if not os.path.isfile(model_path):
        raise Exception("GFPGAN model not found at path "+model_path)

    sys.path.append(os.path.abspath(cmd_opts.gfpgan_dir))
    from gfpgan import GFPGANer

    return GFPGANer(model_path=model_path, upscale=1, arch='clean', channel_multiplier=2, bg_upsampler=None)


def image_grid(imgs, batch_size, round_down=False, force_n_rows=None):
    if force_n_rows is not None:
        rows = force_n_rows
    elif opts.n_rows > 0:
        rows = opts.n_rows
    elif opts.n_rows == 0:
        rows = batch_size
    else:
        rows = math.sqrt(len(imgs))
        rows = int(rows) if round_down else round(rows)

    cols = math.ceil(len(imgs) / rows)

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h), color='black')

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))

    return grid


def draw_prompt_matrix(im, width, height, all_prompts):
    def wrap(text, d, font, line_length):
        lines = ['']
        for word in text.split():
            line = f'{lines[-1]} {word}'.strip()
            if d.textlength(line, font=font) <= line_length:
                lines[-1] = line
            else:
                lines.append(word)
        return '\n'.join(lines)

    def draw_texts(pos, x, y, texts, sizes):
        for i, (text, size) in enumerate(zip(texts, sizes)):
            active = pos & (1 << i) != 0

            if not active:
                text = '\u0336'.join(text) + '\u0336'

            d.multiline_text((x, y + size[1] / 2), text, font=fnt, fill=color_active if active else color_inactive, anchor="mm", align="center")

            y += size[1] + line_spacing

    fontsize = (width + height) // 25
    line_spacing = fontsize // 2
    fnt = ImageFont.truetype("arial.ttf", fontsize)
    color_active = (0, 0, 0)
    color_inactive = (153, 153, 153)

    pad_top = height // 4
    pad_left = width * 3 // 4 if len(all_prompts) > 2 else 0

    cols = im.width // width
    rows = im.height // height

    prompts = all_prompts[1:]

    result = Image.new("RGB", (im.width + pad_left, im.height + pad_top), "white")
    result.paste(im, (pad_left, pad_top))

    d = ImageDraw.Draw(result)

    boundary = math.ceil(len(prompts) / 2)
    prompts_horiz = [wrap(x, d, fnt, width) for x in prompts[:boundary]]
    prompts_vert = [wrap(x, d, fnt, pad_left) for x in prompts[boundary:]]

    sizes_hor = [(x[2] - x[0], x[3] - x[1]) for x in [d.multiline_textbbox((0, 0), x, font=fnt) for x in prompts_horiz]]
    sizes_ver = [(x[2] - x[0], x[3] - x[1]) for x in [d.multiline_textbbox((0, 0), x, font=fnt) for x in prompts_vert]]
    hor_text_height = sum([x[1] + line_spacing for x in sizes_hor]) - line_spacing
    ver_text_height = sum([x[1] + line_spacing for x in sizes_ver]) - line_spacing

    for col in range(cols):
        x = pad_left + width * col + width / 2
        y = pad_top / 2 - hor_text_height / 2

        draw_texts(col, x, y, prompts_horiz, sizes_hor)

    for row in range(rows):
        x = pad_left / 2
        y = pad_top + height * row + height / 2 - ver_text_height / 2

        draw_texts(row, x, y, prompts_vert, sizes_ver)

    return result


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


def check_prompt_length(prompt, comments):
    """this function tests if prompt is too long, and if so, adds a message to comments"""

    tokenizer = model.cond_stage_model.tokenizer
    max_length = model.cond_stage_model.max_length

    info = model.cond_stage_model.tokenizer([prompt], truncation=True, max_length=max_length, return_overflowing_tokens=True, padding="max_length", return_tensors="pt")
    ovf = info['overflowing_tokens'][0]
    overflowing_count = ovf.shape[0]
    if overflowing_count == 0:
        return

    vocab = {v: k for k, v in tokenizer.get_vocab().items()}
    overflowing_words = [vocab.get(int(x), "") for x in ovf]
    overflowing_text = tokenizer.convert_tokens_to_string(''.join(overflowing_words))

    comments.append(f"Warning: too many input tokens; some ({len(overflowing_words)}) have been truncated:\n{overflowing_text}\n")


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
        GFPGAN = load_GFPGAN()
        print("Loaded GFPGAN")
    except Exception:
        print("Error loading GFPGAN:", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)


class TextInversionEmbeddings:
    ids_lookup = {}
    word_embeddings = {}
    word_embeddings_checksums = {}
    fixes = []
    used_custom_terms = []
    dir_mtime = None

    def load(self, dir, model):
        mt = os.path.getmtime(dir)
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

        for fn in os.listdir(dir):
            try:
                process_file(os.path.join(dir, fn), fn)
            except:
                print(f"Error loading emedding {fn}:", file=sys.stderr)
                print(traceback.format_exc(), file=sys.stderr)
                continue

        print(f"Loaded a total of {len(self.word_embeddings)} text inversion embeddings.")

    def hijack(self, m):
        model_embeddings = m.cond_stage_model.transformer.text_model.embeddings

        model_embeddings.token_embedding = EmbeddingsWithFixes(model_embeddings.token_embedding, self)
        m.cond_stage_model = FrozenCLIPEmbedderWithCustomWords(m.cond_stage_model, self)

class FrozenCLIPEmbedderWithCustomWords(torch.nn.Module):
    def __init__(self, wrapped, embeddings):
        super().__init__()
        self.wrapped = wrapped
        self.embeddings = embeddings
        self.tokenizer = wrapped.tokenizer
        self.max_length = wrapped.max_length

    def forward(self, text):
        self.embeddings.fixes = []
        self.embeddings.used_custom_terms = []
        remade_batch_tokens = []
        id_start = self.wrapped.tokenizer.bos_token_id
        id_end = self.wrapped.tokenizer.eos_token_id
        maxlen = self.wrapped.max_length - 2

        cache = {}
        batch_tokens = self.wrapped.tokenizer(text, truncation=False, add_special_tokens=False)["input_ids"]
        for tokens in batch_tokens:
            tuple_tokens = tuple(tokens)

            if tuple_tokens in cache:
                remade_tokens, fixes = cache[tuple_tokens]
            else:
                fixes = []
                remade_tokens = []

                i = 0
                while i < len(tokens):
                    token = tokens[i]

                    possible_matches = self.embeddings.ids_lookup.get(token, None)

                    if possible_matches is None:
                        remade_tokens.append(token)
                    else:
                        found = False
                        for ids, word in possible_matches:
                            if tokens[i:i+len(ids)] == ids:
                                fixes.append((len(remade_tokens), word))
                                remade_tokens.append(777)
                                i += len(ids) - 1
                                found = True
                                self.embeddings.used_custom_terms.append((word, self.embeddings.word_embeddings_checksums[word]))
                                break

                        if not found:
                            remade_tokens.append(token)

                    i += 1

                remade_tokens = remade_tokens + [id_end] * (maxlen - 2 - len(remade_tokens))
                remade_tokens = [id_start] + remade_tokens[0:maxlen-2] + [id_end]
                cache[tuple_tokens] = (remade_tokens, fixes)

            remade_batch_tokens.append(remade_tokens)
            self.embeddings.fixes.append(fixes)

        tokens = torch.asarray(remade_batch_tokens).to(self.wrapped.device)
        outputs = self.wrapped.transformer(input_ids=tokens)
        z = outputs.last_hidden_state
        return z


class EmbeddingsWithFixes(nn.Module):
    def __init__(self, wrapped, embeddings):
        super().__init__()
        self.wrapped = wrapped
        self.embeddings = embeddings

    def forward(self, input_ids):
        batch_fixes = self.embeddings.fixes
        self.embeddings.fixes = []

        inputs_embeds = self.wrapped(input_ids)

        for fixes, tensor in zip(batch_fixes, inputs_embeds):
            for offset, word in fixes:
                tensor[offset] = self.embeddings.word_embeddings[word]

        return inputs_embeds


def get_learned_conditioning_with_embeddings(model, prompts):
    if os.path.exists(cmd_opts.embeddings_dir):
        text_inversion_embeddings.load(cmd_opts.embeddings_dir, model)

    return model.get_learned_conditioning(prompts)


def process_images(outpath, func_init, func_sample, prompt, seed, sampler_index, batch_size, n_iter, steps, cfg_scale, width, height, prompt_matrix, use_GFPGAN, do_not_save_grid=False, extra_generation_params=None):
    """this is the main loop that both txt2img and img2img use; it calls func_init once inside all the scopes and func_sample once per batch"""

    assert prompt is not None
    torch_gc()

    if seed == -1:
        seed = random.randrange(4294967294)
    seed = int(seed)

    os.makedirs(outpath, exist_ok=True)

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1

    comments = []

    prompt_matrix_parts = []
    if prompt_matrix:
        all_prompts = []
        prompt_matrix_parts = prompt.split("|")
        combination_count = 2 ** (len(prompt_matrix_parts) - 1)
        for combination_num in range(combination_count):
            selected_prompts = [text.strip().strip(',') for n, text in enumerate(prompt_matrix_parts[1:]) if combination_num & (1<<n)]

            if opts.prompt_matrix_add_to_start:
                selected_prompts = selected_prompts + [prompt_matrix_parts[0]]
            else:
                selected_prompts = [prompt_matrix_parts[0]] + selected_prompts

            all_prompts.append( ", ".join(selected_prompts))

        n_iter = math.ceil(len(all_prompts) / batch_size)
        all_seeds = len(all_prompts) * [seed]

        print(f"Prompt matrix will create {len(all_prompts)} images using a total of {n_iter} batches.")
    else:

        if opts.verify_input:
            try:
                check_prompt_length(prompt, comments)
            except:
                import traceback
                print("Error verifying input:", file=sys.stderr)
                print(traceback.format_exc(), file=sys.stderr)

        all_prompts = batch_size * n_iter * [prompt]
        all_seeds = [seed + x for x in range(len(all_prompts))]

    generation_params = {
        "Steps": steps,
        "Sampler": samplers[sampler_index].name,
        "CFG scale": cfg_scale,
        "Seed": seed,
        "GFPGAN": ("GFPGAN" if use_GFPGAN and GFPGAN is not None else None)
    }

    if extra_generation_params is not None:
        generation_params.update(extra_generation_params)

    generation_params_text = ", ".join([k if k == v else f'{k}: {v}' for k, v in generation_params.items() if v is not None])

    def infotext():
        return f"{prompt}\n{generation_params_text}".strip() + "".join(["\n\n" + x for x in comments])

    if os.path.exists(cmd_opts.embeddings_dir):
        text_inversion_embeddings.load(cmd_opts.embeddings_dir, model)

    output_images = []
    with torch.no_grad(), autocast("cuda"), model.ema_scope():
        init_data = func_init()

        for n in range(n_iter):
            prompts = all_prompts[n * batch_size:(n + 1) * batch_size]
            seeds = all_seeds[n * batch_size:(n + 1) * batch_size]

            uc = model.get_learned_conditioning(len(prompts) * [""])
            c = model.get_learned_conditioning(prompts)

            if len(text_inversion_embeddings.used_custom_terms) > 0:
                comments.append("Used custom terms: " + ", ".join([f'{word} [{checksum}]' for word, checksum in text_inversion_embeddings.used_custom_terms]))

            # we manually generate all input noises because each one should have a specific seed
            x = create_random_tensors([opt_C, height // opt_f, width // opt_f], seeds=seeds)

            samples_ddim = func_sample(init_data=init_data, x=x, conditioning=c, unconditional_conditioning=uc)

            x_samples_ddim = model.decode_first_stage(samples_ddim)
            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

            if prompt_matrix or opts.samples_save or opts.grid_save:
                for i, x_sample in enumerate(x_samples_ddim):
                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                    x_sample = x_sample.astype(np.uint8)

                    if use_GFPGAN and GFPGAN is not None:
                        cropped_faces, restored_faces, restored_img = GFPGAN.enhance(x_sample, has_aligned=False, only_center_face=False, paste_back=True)
                        x_sample = restored_img

                    image = Image.fromarray(x_sample)
                    save_image(image, sample_path, f"{base_count:05}", seeds[i], prompts[i], opts.samples_format, info=infotext())

                    output_images.append(image)
                    base_count += 1

        if (prompt_matrix or opts.grid_save) and not do_not_save_grid:
            grid = image_grid(output_images, batch_size, round_down=prompt_matrix)

            if prompt_matrix:

                try:
                    grid = draw_prompt_matrix(grid, width, height, prompt_matrix_parts)
                except Exception:
                    import traceback
                    print("Error creating prompt_matrix text:", file=sys.stderr)
                    print(traceback.format_exc(), file=sys.stderr)

                output_images.insert(0, grid)

            save_image(grid, outpath, f"grid-{grid_count:04}", seed, prompt, opts.grid_format, info=infotext(), short_filename=not opts.grid_extended_filename)
            grid_count += 1



    torch_gc()
    return output_images, seed, infotext()


def txt2img(prompt: str, ddim_steps: int, sampler_index: int, use_GFPGAN: bool, prompt_matrix: bool, ddim_eta: float, n_iter: int, batch_size: int, cfg_scale: float, seed: int, height: int, width: int):
    outpath = opts.outdir or "outputs/txt2img-samples"

    sampler = samplers[sampler_index].constructor(model)

    def init():
        pass

    def sample(init_data, x, conditioning, unconditional_conditioning):
        samples_ddim, _ = sampler.sample(S=ddim_steps, conditioning=conditioning, batch_size=int(x.shape[0]), shape=x[0].shape, verbose=False, unconditional_guidance_scale=cfg_scale, unconditional_conditioning=unconditional_conditioning, eta=ddim_eta, x_T=x)
        return samples_ddim

    output_images, seed, info = process_images(
        outpath=outpath,
        func_init=init,
        func_sample=sample,
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
        use_GFPGAN=use_GFPGAN
    )

    del sampler

    return output_images, seed, plaintext_to_html(info)


class Flagging(gr.FlaggingCallback):

    def setup(self, components, flagging_dir: str):
        pass

    def flag(self, flag_data, flag_option=None, flag_index=None, username=None):
        import csv

        os.makedirs("log/images", exist_ok=True)

        # those must match the "txt2img" function
        prompt, ddim_steps, sampler_name, use_GFPGAN, prompt_matrix, ddim_eta, n_iter, n_samples, cfg_scale, request_seed, height, width, images, seed, comment = flag_data

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
        gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="DDIM ETA", value=0.0, visible=False),
        gr.Slider(minimum=1, maximum=cmd_opts.max_batch_count, step=1, label='Batch count (how many batches of images to generate)', value=1),
        gr.Slider(minimum=1, maximum=8, step=1, label='Batch size (how many images are in a batch; memory-hungry)', value=1),
        gr.Slider(minimum=1.0, maximum=15.0, step=0.5, label='Classifier Free Guidance Scale (how strongly the image should follow the prompt)', value=7.0),
        gr.Number(label='Seed', value=-1),
        gr.Slider(minimum=64, maximum=2048, step=64, label="Height", value=512),
        gr.Slider(minimum=64, maximum=2048, step=64, label="Width", value=512),
    ],
    outputs=[
        gr.Gallery(label="Images"),
        gr.Number(label='Seed'),
        gr.HTML(),
    ],
    title="Stable Diffusion Text-to-Image",
    flagging_callback=Flagging()
)


def img2img(prompt: str, init_img, ddim_steps: int, use_GFPGAN: bool, prompt_matrix, loopback: bool, n_iter: int, batch_size: int, cfg_scale: float, denoising_strength: float, seed: int, height: int, width: int, resize_mode: int):
    outpath = opts.outdir or "outputs/img2img-samples"

    sampler = KDiffusionSampler(model, 'sample_lms')

    assert 0. <= denoising_strength <= 1., 'can only work with strength in [0.0, 1.0]'

    def init():
        image = init_img.convert("RGB")
        image = resize_image(resize_mode, image, width, height)
        image = np.array(image).astype(np.float32) / 255.0
        image = image[None].transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)

        init_image = 2. * image - 1.
        init_image = init_image.to(device)
        init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
        init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))  # move to latent space

        return init_latent,

    def sample(init_data, x, conditioning, unconditional_conditioning):
        t_enc = int(denoising_strength * ddim_steps)

        x0, = init_data

        sigmas = sampler.model_wrap.get_sigmas(ddim_steps)
        noise = x * sigmas[ddim_steps - t_enc - 1]

        xi = x0 + noise
        sigma_sched = sigmas[ddim_steps - t_enc - 1:]
        model_wrap_cfg = CFGDenoiser(sampler.model_wrap)
        samples_ddim = k_diffusion.sampling.sample_lms(model_wrap_cfg, xi, sigma_sched, extra_args={'cond': conditioning, 'uncond': unconditional_conditioning, 'cond_scale': cfg_scale}, disable=False)
        return samples_ddim

    if loopback:
        output_images, info = None, None
        history = []
        initial_seed = None

        for i in range(n_iter):
            output_images, seed, info = process_images(
                outpath=outpath,
                func_init=init,
                func_sample=sample,
                prompt=prompt,
                seed=seed,
                sampler_index=0,
                batch_size=1,
                n_iter=1,
                steps=ddim_steps,
                cfg_scale=cfg_scale,
                width=width,
                height=height,
                prompt_matrix=prompt_matrix,
                use_GFPGAN=use_GFPGAN,
                do_not_save_grid=True,
                extra_generation_params={"Denoising Strength": denoising_strength},
            )

            if initial_seed is None:
                initial_seed = seed

            init_img = output_images[0]
            seed = seed + 1
            denoising_strength = max(denoising_strength * 0.95, 0.1)
            history.append(init_img)

        grid_count = len(os.listdir(outpath)) - 1
        grid = image_grid(history, batch_size, force_n_rows=1)

        save_image(grid, outpath, f"grid-{grid_count:04}", initial_seed, prompt, opts.grid_format, info=info, short_filename=not opts.grid_extended_filename)

        output_images = history
        seed = initial_seed

    else:
        output_images, seed, info = process_images(
            outpath=outpath,
            func_init=init,
            func_sample=sample,
            prompt=prompt,
            seed=seed,
            sampler_index=0,
            batch_size=batch_size,
            n_iter=n_iter,
            steps=ddim_steps,
            cfg_scale=cfg_scale,
            width=width,
            height=height,
            prompt_matrix=prompt_matrix,
            use_GFPGAN=use_GFPGAN,
            extra_generation_params={"Denoising Strength": denoising_strength},
        )

    del sampler

    return output_images, seed, plaintext_to_html(info)


sample_img2img = "assets/stable-samples/img2img/sketch-mountains-input.jpg"
sample_img2img = sample_img2img if os.path.exists(sample_img2img) else None

img2img_interface = gr.Interface(
    wrap_gradio_call(img2img),
    inputs=[
        gr.Textbox(placeholder="A fantasy landscape, trending on artstation.", lines=1),
        gr.Image(value=sample_img2img, source="upload", interactive=True, type="pil"),
        gr.Slider(minimum=1, maximum=150, step=1, label="Sampling Steps", value=50),
        gr.Checkbox(label='Fix faces using GFPGAN', value=False, visible=GFPGAN is not None),
        gr.Checkbox(label='Create prompt matrix (separate multiple prompts using |, and get all combinations of them)', value=False),
        gr.Checkbox(label='Loopback (use images from previous batch when creating next batch)', value=False),
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


def run_extras(image, GFPGAN_strength, RealESRGAN_upscaling, RealESRGAN_model_index):
    image = image.convert("RGB")

    outpath = opts.outdir or "outputs/extras-samples"

    if GFPGAN is not None and GFPGAN_strength > 0:
        cropped_faces, restored_faces, restored_img = GFPGAN.enhance(np.array(image, dtype=np.uint8), has_aligned=False, only_center_face=False, paste_back=True)
        res = Image.fromarray(restored_img)

        if GFPGAN_strength < 1.0:
            res = Image.blend(image, res, GFPGAN_strength)

        image = res

    if have_realesrgan and RealESRGAN_upscaling != 1.0:
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

    os.makedirs(outpath, exist_ok=True)
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
        return opts.data[key] if key in opts.data else opts.data_labels[key][0]

    labelinfo = opts.data_labels[key]
    t = type(labelinfo[0])
    label = labelinfo[1]
    if t == str:
        item = gr.Textbox(label=label, value=fun, lines=1)
    elif t == int:
        if len(labelinfo) == 4:
            item = gr.Slider(minimum=labelinfo[2], maximum=labelinfo[3], step=1, label=label, value=fun)
        else:
            item = gr.Number(label=label, value=fun)
    elif t == bool:
        item = gr.Checkbox(label=label, value=fun)
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

config = OmegaConf.load(cmd_opts.config)
model = load_model_from_config(config, cmd_opts.ckpt)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = (model if cmd_opts.no_half else model.half()).to(device)
text_inversion_embeddings = TextInversionEmbeddings()

if os.path.exists(cmd_opts.embeddings_dir):
    text_inversion_embeddings.hijack(model)

demo = gr.TabbedInterface(
    interface_list=[x[0] for x in interfaces],
    tab_names=[x[1] for x in interfaces],
    css=("" if cmd_opts.no_progressbar_hiding else css_hide_progressbar) + """
.output-html p {margin: 0 0.5em;}
.performance { font-size: 0.85em; color: #444; }
"""
)

demo.launch()
