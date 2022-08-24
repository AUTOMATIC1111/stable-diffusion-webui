import argparse, os, sys, glob
import torch
import torch.nn as nn
import numpy as np
import gradio as gr
from omegaconf import OmegaConf
from PIL import Image, ImageFont, ImageDraw
from itertools import islice
from einops import rearrange, repeat
from torch import autocast
from contextlib import contextmanager, nullcontext
import mimetypes
import random
import math

import k_diffusion as K
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

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

parser = argparse.ArgumentParser()
parser.add_argument("--outdir", type=str, nargs="?", help="dir to write results to", default=None)
parser.add_argument("--skip_grid", action='store_true', help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",)
parser.add_argument("--skip_save", action='store_true', help="do not save indiviual samples. For speed measurements.",)
parser.add_argument("--n_rows", type=int, default=-1, help="rows in the grid; use -1 for autodetect and 0 for n_rows to be same as batch_size (default: -1)",)
parser.add_argument("--config", type=str, default="configs/stable-diffusion/v1-inference.yaml", help="path to config which constructs model",)
parser.add_argument("--ckpt", type=str, default="models/ldm/stable-diffusion-v1/model.ckpt", help="path to checkpoint of model",)
parser.add_argument("--precision", type=str, help="evaluate at this precision", choices=["full", "autocast"], default="autocast")
parser.add_argument("--gfpgan-dir", type=str, help="GFPGAN directory", default=('./src/gfpgan' if os.path.exists('./src/gfpgan') else './GFPGAN')) # i disagree with where you're putting it but since all guidefags are doing it this way, there you go
parser.add_argument("--no-verify-input", action='store_true', help="do not verify input to check if it's too long")
parser.add_argument("--no-half", action='store_true', help="do not switch the model to 16-bit floats")
parser.add_argument("--no-progressbar-hiding", action='store_true', help="do not hide progressbar in gradio UI (we hide it because it slows down ML if you have hardware accleration in browser)")
parser.add_argument("--max-batch-count",  type=int, default=16, help="maximum batch count value for the UI")
parser.add_argument("--grid-format",  type=str, default='png', help="file format for saved grids; can be png or jpg")
opt = parser.parse_args()

GFPGAN_dir = opt.gfpgan_dir

css_hide_progressbar = """
.wrap .m-12 svg { display:none!important; }
.wrap .m-12::before { content:"Loading..." }
.progress-bar { display:none!important; }
.meta-text { display:none!important; }
"""

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
    def __init__(self, m):
        self.model = m
        self.model_wrap = K.external.CompVisDenoiser(m)

    def sample(self, S, conditioning, batch_size, shape, verbose, unconditional_guidance_scale, unconditional_conditioning, eta, x_T):
        sigmas = self.model_wrap.get_sigmas(S)
        x = x_T * sigmas[0]
        model_wrap_cfg = CFGDenoiser(self.model_wrap)

        samples_ddim = K.sampling.sample_lms(model_wrap_cfg, x, sigmas, extra_args={'cond': conditioning, 'uncond': unconditional_conditioning, 'cond_scale': unconditional_guidance_scale}, disable=False)

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


def load_GFPGAN():
    model_name = 'GFPGANv1.3'
    model_path = os.path.join(GFPGAN_dir, 'experiments/pretrained_models', model_name + '.pth')
    if not os.path.isfile(model_path):
        raise Exception("GFPGAN model not found at path "+model_path)

    sys.path.append(os.path.abspath(GFPGAN_dir))
    from gfpgan import GFPGANer

    return GFPGANer(model_path=model_path, upscale=1, arch='clean', channel_multiplier=2, bg_upsampler=None)


GFPGAN = None
if os.path.exists(GFPGAN_dir):
    try:
        GFPGAN = load_GFPGAN()
        print("Loaded GFPGAN")
    except Exception:
        import traceback
        print("Error loading GFPGAN:", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)

config = OmegaConf.load("configs/stable-diffusion/v1-inference.yaml")
model = load_model_from_config(config, "models/ldm/stable-diffusion-v1/model.ckpt")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = (model if opt.no_half else model.half()).to(device)


def image_grid(imgs, batch_size, round_down=False, force_n_rows=None):
    if force_n_rows is not None:
        rows = force_n_rows
    elif opt.n_rows > 0:
        rows = opt.n_rows
    elif opt.n_rows == 0:
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


def process_images(outpath, func_init, func_sample, prompt, seed, sampler_name, batch_size, n_iter, steps, cfg_scale, width, height, prompt_matrix, use_GFPGAN, do_not_save_grid=False):
    """this is the main loop that both txt2img and img2img use; it calls func_init once inside all the scopes and func_sample once per batch"""

    assert prompt is not None
    torch.cuda.empty_cache()

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
            current = prompt_matrix_parts[0]

            for n, text in enumerate(prompt_matrix_parts[1:]):
                if combination_num & (2 ** n) > 0:
                    current += ("" if text.strip().startswith(",") else ", ") + text

            all_prompts.append(current)

        n_iter = math.ceil(len(all_prompts) / batch_size)
        all_seeds = len(all_prompts) * [seed]

        print(f"Prompt matrix will create {len(all_prompts)} images using a total of {n_iter} batches.")
    else:

        if not opt.no_verify_input:
            try:
                check_prompt_length(prompt, comments)
            except:
                import traceback
                print("Error verifying input:", file=sys.stderr)
                print(traceback.format_exc(), file=sys.stderr)

        all_prompts = batch_size * n_iter * [prompt]
        all_seeds = [seed + x for x in range(len(all_prompts))]

    precision_scope = autocast if opt.precision == "autocast" else nullcontext
    output_images = []
    with torch.no_grad(), precision_scope("cuda"), model.ema_scope():
        init_data = func_init()

        for n in range(n_iter):
            prompts = all_prompts[n * batch_size:(n + 1) * batch_size]
            seeds = all_seeds[n * batch_size:(n + 1) * batch_size]

            uc = None
            if cfg_scale != 1.0:
                uc = model.get_learned_conditioning(len(prompts) * [""])
            if isinstance(prompts, tuple):
                prompts = list(prompts)
            c = model.get_learned_conditioning(prompts)

            # we manually generate all input noises because each one should have a specific seed
            x = create_random_tensors([opt_C, height // opt_f, width // opt_f], seeds=seeds)

            samples_ddim = func_sample(init_data=init_data, x=x, conditioning=c, unconditional_conditioning=uc)

            x_samples_ddim = model.decode_first_stage(samples_ddim)
            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

            if prompt_matrix or not opt.skip_save or not opt.skip_grid:
                for i, x_sample in enumerate(x_samples_ddim):
                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                    x_sample = x_sample.astype(np.uint8)

                    if use_GFPGAN and GFPGAN is not None:
                        cropped_faces, restored_faces, restored_img = GFPGAN.enhance(x_sample, has_aligned=False, only_center_face=False, paste_back=True)
                        x_sample = restored_img

                    image = Image.fromarray(x_sample)
                    filename = f"{base_count:05}-{seeds[i]}_{prompts[i].replace(' ', '_').translate({ord(x): '' for x in invalid_filename_chars})[:128]}.png"

                    image.save(os.path.join(sample_path, filename))

                    output_images.append(image)
                    base_count += 1

        if (prompt_matrix or not opt.skip_grid) and not do_not_save_grid:
            grid = image_grid(output_images, batch_size, round_down=prompt_matrix)

            if prompt_matrix:

                try:
                    grid = draw_prompt_matrix(grid, width, height, prompt_matrix_parts)
                except Exception:
                    import traceback
                    print("Error creating prompt_matrix text:", file=sys.stderr)
                    print(traceback.format_exc(), file=sys.stderr)

                output_images.insert(0, grid)

            grid.save(os.path.join(outpath, f'grid-{grid_count:04}.{opt.grid_format}'))
            grid_count += 1

    info = f"""
{prompt}
Steps: {steps}, Sampler: {sampler_name}, CFG scale: {cfg_scale}, Seed: {seed}{', GFPGAN' if use_GFPGAN and GFPGAN is not None else ''}
        """.strip()

    for comment in comments:
        info += "\n\n" + comment

    return output_images, seed, info


def txt2img(prompt: str, ddim_steps: int, sampler_name: str, use_GFPGAN: bool, prompt_matrix: bool, ddim_eta: float, n_iter: int, batch_size: int, cfg_scale: float, seed: int, height: int, width: int):
    outpath = opt.outdir or "outputs/txt2img-samples"

    if sampler_name == 'PLMS':
        sampler = PLMSSampler(model)
    elif sampler_name == 'DDIM':
        sampler = DDIMSampler(model)
    elif sampler_name == 'k-diffusion':
        sampler = KDiffusionSampler(model)
    else:
        raise Exception("Unknown sampler: " + sampler_name)

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
        sampler_name=sampler_name,
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

    return output_images, seed, info


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
    txt2img,
    inputs=[
        gr.Textbox(label="Prompt", placeholder="A corgi wearing a top hat as an oil painting.", lines=1),
        gr.Slider(minimum=1, maximum=150, step=1, label="Sampling Steps", value=50),
        gr.Radio(label='Sampling method', choices=["DDIM", "PLMS", "k-diffusion"], value="k-diffusion"),
        gr.Checkbox(label='Fix faces using GFPGAN', value=False, visible=GFPGAN is not None),
        gr.Checkbox(label='Create prompt matrix (separate multiple prompts using |, and get all combinations of them)', value=False),
        gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="DDIM ETA", value=0.0, visible=False),
        gr.Slider(minimum=1, maximum=opt.max_batch_count, step=1, label='Batch count (how many batches of images to generate)', value=1),
        gr.Slider(minimum=1, maximum=8, step=1, label='Batch size (how many images are in a batch; memory-hungry)', value=1),
        gr.Slider(minimum=1.0, maximum=15.0, step=0.5, label='Classifier Free Guidance Scale (how strongly the image should follow the prompt)', value=7.0),
        gr.Number(label='Seed', value=-1),
        gr.Slider(minimum=64, maximum=2048, step=64, label="Height", value=512),
        gr.Slider(minimum=64, maximum=2048, step=64, label="Width", value=512),
    ],
    outputs=[
        gr.Gallery(label="Images"),
        gr.Number(label='Seed'),
        gr.Textbox(label="Copy-paste generation parameters"),
    ],
    title="Stable Diffusion Text-to-Image K",
    description="Generate images from text with Stable Diffusion (using K-LMS)",
    flagging_callback=Flagging()
)


def img2img(prompt: str, init_img, ddim_steps: int, use_GFPGAN: bool, prompt_matrix, loopback: bool, n_iter: int, batch_size: int, cfg_scale: float, denoising_strength: float, seed: int, height: int, width: int, resize_mode: int):
    outpath = opt.outdir or "outputs/img2img-samples"

    sampler = KDiffusionSampler(model)

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
        samples_ddim = K.sampling.sample_lms(model_wrap_cfg, xi, sigma_sched, extra_args={'cond': conditioning, 'uncond': unconditional_conditioning, 'cond_scale': cfg_scale}, disable=False)
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
                sampler_name='k-diffusion',
                batch_size=1,
                n_iter=1,
                steps=ddim_steps,
                cfg_scale=cfg_scale,
                width=width,
                height=height,
                prompt_matrix=prompt_matrix,
                use_GFPGAN=use_GFPGAN,
                do_not_save_grid=True
            )

            if initial_seed is None:
                initial_seed = seed

            init_img = output_images[0]
            seed = seed + 1
            denoising_strength = max(denoising_strength * 0.95, 0.1)
            history.append(init_img)

        grid_count = len(os.listdir(outpath)) - 1
        grid = image_grid(history, batch_size, force_n_rows=1)
        grid.save(os.path.join(outpath, f'grid-{grid_count:04}.{opt.grid_format}'))

        output_images = history
        seed = initial_seed

    else:
        output_images, seed, info = process_images(
            outpath=outpath,
            func_init=init,
            func_sample=sample,
            prompt=prompt,
            seed=seed,
            sampler_name='k-diffusion',
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

    return output_images, seed, info


sample_img2img = "assets/stable-samples/img2img/sketch-mountains-input.jpg"
sample_img2img = sample_img2img if os.path.exists(sample_img2img) else None

img2img_interface = gr.Interface(
    img2img,
    inputs=[
        gr.Textbox(placeholder="A fantasy landscape, trending on artstation.", lines=1),
        gr.Image(value=sample_img2img, source="upload", interactive=True, type="pil"),
        gr.Slider(minimum=1, maximum=150, step=1, label="Sampling Steps", value=50),
        gr.Checkbox(label='Fix faces using GFPGAN', value=False, visible=GFPGAN is not None),
        gr.Checkbox(label='Create prompt matrix (separate multiple prompts using |, and get all combinations of them)', value=False),
        gr.Checkbox(label='Loopback (use images from previous batch when creating next batch)', value=False),
        gr.Slider(minimum=1, maximum=opt.max_batch_count, step=1, label='Batch count (how many batches of images to generate)', value=1),
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
        gr.Textbox(label="Copy-paste generation parameters"),
    ],
    title="Stable Diffusion Image-to-Image",
    description="Generate images from images with Stable Diffusion",
    allow_flagging="never",
)

interfaces = [
    (txt2img_interface, "txt2img"),
    (img2img_interface, "img2img")
]

def run_GFPGAN(image, strength):
    image = image.convert("RGB")

    cropped_faces, restored_faces, restored_img = GFPGAN.enhance(np.array(image, dtype=np.uint8), has_aligned=False, only_center_face=False, paste_back=True)
    res = Image.fromarray(restored_img)

    if strength < 1.0:
        res = Image.blend(image, res, strength)

    return res


if GFPGAN is not None:
    interfaces.append((gr.Interface(
        run_GFPGAN,
        inputs=[
            gr.Image(label="Source", source="upload", interactive=True, type="pil"),
            gr.Slider(minimum=0.0, maximum=1.0, step=0.001, label="Effect strength", value=100),
        ],
        outputs=[
            gr.Image(label="Result"),
        ],
        title="GFPGAN",
        description="Fix faces on images",
        allow_flagging="never",
    ), "GFPGAN"))

demo = gr.TabbedInterface(
    interface_list=[x[0] for x in interfaces],
    tab_names=[x[1] for x in interfaces],
    css=("" if opt.no_progressbar_hiding else css_hide_progressbar)
)

demo.launch()
