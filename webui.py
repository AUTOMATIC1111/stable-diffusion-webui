import PIL
import argparse, os, sys, glob
import torch
import torch.nn as nn
import numpy as np
import gradio as gr
from omegaconf import OmegaConf
from PIL import Image
from itertools import islice
from einops import rearrange, repeat
from torch import autocast
from contextlib import contextmanager, nullcontext
import mimetypes
import random
import math
import csv

import k_diffusion as K
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

# this is a fix for Windows users. Without it, javascript files will be served with text/html content-type and the bowser will not show any UI
mimetypes.init()
mimetypes.add_type('application/javascript', '.js')

# some of those options should not be changed at all because they would break the model, so I removed them from options.
opt_C = 4
opt_f = 8

invalid_filename_chars = '<>:"/\|?*'

parser = argparse.ArgumentParser()
parser.add_argument("--outdir", type=str, nargs="?", help="dir to write results to", default=None)
parser.add_argument("--skip_grid", action='store_true', help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",)
parser.add_argument("--skip_save", action='store_true', help="do not save indiviual samples. For speed measurements.",)
parser.add_argument("--n_rows", type=int, default=-1, help="rows in the grid; use -1 for autodetect and 0 for n_rows to be same as batch_size (default: -1)",)
parser.add_argument("--config", type=str, default="configs/stable-diffusion/v1-inference.yaml", help="path to config which constructs model",)
parser.add_argument("--ckpt", type=str, default="models/ldm/stable-diffusion-v1/model.ckpt", help="path to checkpoint of model",)
parser.add_argument("--precision", type=str, help="evaluate at this precision", choices=["full", "autocast"], default="autocast")
parser.add_argument("--gfpgan-dir", type=str, help="GFPGAN directory", default='./GFPGAN')
opt = parser.parse_args()

GFPGAN_dir = opt.gfpgan_dir


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


def load_img_pil(img_pil):
    image = img_pil.convert("RGB")
    w, h = image.size
    print(f"loaded input image of size ({w}, {h})")
    w, h = map(lambda x: x - x % 64, (w, h))  # resize to integer multiple of 64
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    print(f"cropped image to size ({w}, {h})")
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2. * image - 1.


def load_img(path):
    return load_img_pil(Image.open(path))


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
model = model.half().to(device)


def image_grid(imgs, batch_size, round_down=False):
    if opt.n_rows > 0:
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


def dream(prompt: str, ddim_steps: int, sampler_name: str, use_GFPGAN: bool, prompt_matrix: bool, ddim_eta: float, n_iter: int, n_samples: int, cfg_scale: float, seed: int, height: int, width: int):
    torch.cuda.empty_cache()

    outpath = opt.outdir or "outputs/txt2img-samples"

    if seed == -1:
        seed = random.randrange(4294967294)

    seed = int(seed)
    keep_same_seed = False

    is_PLMS = sampler_name == 'PLMS'
    is_DDIM = sampler_name == 'DDIM'
    is_Kdif = sampler_name == 'k-diffusion'

    sampler = None
    if is_PLMS:
        sampler = PLMSSampler(model)
    elif is_DDIM:
        sampler = DDIMSampler(model)
    elif is_Kdif:
        pass
    else:
        raise Exception("Unknown sampler: " + sampler_name)

    model_wrap = K.external.CompVisDenoiser(model)

    os.makedirs(outpath, exist_ok=True)

    batch_size = n_samples

    assert prompt is not None
    prompts = batch_size * [prompt]

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1

    prompt_matrix_prompts = []
    comment = ""
    if prompt_matrix:
        keep_same_seed = True
        comment = "Image prompts:\n\n"

        items = prompt.split("|")
        combination_count = 2 ** (len(items)-1)
        for combination_num in range(combination_count):
            current = items[0]
            label = 'A'

            for n, text in enumerate(items[1:]):
                if combination_num & (2**n) > 0:
                    current += ("" if text.strip().startswith(",") else ", ") + text
                    label += chr(ord('B') + n)

            comment += " - " + label + "\n"

            prompt_matrix_prompts.append(current)
        n_iter = math.ceil(len(prompt_matrix_prompts) / batch_size)

        comment += "\nwhere:\n"
        for n, text in enumerate(items):
            comment += "  " + chr(ord('A') + n) + " = " + items[n] + "\n"

    precision_scope = autocast if opt.precision == "autocast" else nullcontext
    output_images = []
    with torch.no_grad(), precision_scope("cuda"), model.ema_scope():
        for n in range(n_iter):
            if prompt_matrix:
                prompts = prompt_matrix_prompts[n*batch_size:(n+1)*batch_size]

            uc = None
            if cfg_scale != 1.0:
                uc = model.get_learned_conditioning(len(prompts) * [""])
            if isinstance(prompts, tuple):
                prompts = list(prompts)
            c = model.get_learned_conditioning(prompts)
            shape = [opt_C, height // opt_f, width // opt_f]

            batch_seed = seed if keep_same_seed else seed + n * len(prompts)

            # we manually generate all input noises because each one should have a specific seed
            xs = []
            for i in range(len(prompts)):
                current_seed = seed if keep_same_seed else batch_seed + i
                torch.manual_seed(current_seed)
                xs.append(torch.randn(shape, device=device))
            x = torch.stack(xs)

            if is_Kdif:
                sigmas = model_wrap.get_sigmas(ddim_steps)
                x = x * sigmas[0]
                model_wrap_cfg = CFGDenoiser(model_wrap)
                samples_ddim = K.sampling.sample_lms(model_wrap_cfg, x, sigmas, extra_args={'cond': c, 'uncond': uc, 'cond_scale': cfg_scale}, disable=False)

            elif sampler is not None:
                samples_ddim, _ = sampler.sample(S=ddim_steps, conditioning=c, batch_size=len(prompts), shape=shape, verbose=False, unconditional_guidance_scale=cfg_scale, unconditional_conditioning=uc, eta=ddim_eta, x_T=x)

            x_samples_ddim = model.decode_first_stage(samples_ddim)
            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

            if not opt.skip_save or not opt.skip_grid:
                for i, x_sample in enumerate(x_samples_ddim):
                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                    x_sample = x_sample.astype(np.uint8)

                    if use_GFPGAN and GFPGAN is not None:
                        cropped_faces, restored_faces, restored_img = GFPGAN.enhance(x_sample, has_aligned=False, only_center_face=False, paste_back=True)
                        x_sample = restored_img

                    image = Image.fromarray(x_sample)
                    filename = f"{base_count:05}-{seed if keep_same_seed else batch_seed + i}_{prompts[i].replace(' ', '_').translate({ord(x): '' for x in invalid_filename_chars})[:128]}.png"

                    image.save(os.path.join(sample_path, filename))

                    output_images.append(image)
                    base_count += 1




        if not opt.skip_grid:
            # additionally, save as grid
            grid = image_grid(output_images, batch_size, round_down=prompt_matrix)
            grid.save(os.path.join(outpath, f'grid-{grid_count:04}.png'))
            grid_count += 1


    if sampler is not None:
        del sampler

    info = f"""
{prompt}
Steps: {ddim_steps}, Sampler: {sampler_name}, CFG scale: {cfg_scale}, Seed: {seed}{', GFPGAN' if use_GFPGAN and GFPGAN is not None else ''}
    """.strip()

    if len(comment) > 0:
        info += "\n\n" + comment

    return output_images, seed, info

class Flagging(gr.FlaggingCallback):

    def setup(self, components, flagging_dir: str):
        pass

    def flag(self, flag_data, flag_option=None, flag_index=None, username=None) -> int:
        os.makedirs("log/images", exist_ok=True)

        # those must match the "dream" function
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


dream_interface = gr.Interface(
    dream,
    inputs=[
        gr.Textbox(label="Prompt", placeholder="A corgi wearing a top hat as an oil painting.", lines=1),
        gr.Slider(minimum=1, maximum=150, step=1, label="Sampling Steps", value=50),
        gr.Radio(label='Sampling method', choices=["DDIM", "PLMS", "k-diffusion"], value="k-diffusion"),
        gr.Checkbox(label='Fix faces using GFPGAN', value=False, visible=GFPGAN is not None),
        gr.Checkbox(label='Create prompt matrix (separate multiple prompts using |, and get all combinations of them)', value=False),
        gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="DDIM ETA", value=0.0, visible=False),
        gr.Slider(minimum=1, maximum=16, step=1, label='Batch count (how many batches of images to generate)', value=1),
        gr.Slider(minimum=1, maximum=4, step=1, label='Batch size (how many images are in a batch; memory-hungry)', value=1),
        gr.Slider(minimum=1.0, maximum=15.0, step=0.5, label='Classifier Free Guidance Scale (how strongly should the image follow the prompt)', value=7.0),
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


def translation(prompt: str, init_img, ddim_steps: int, use_GFPGAN: bool, ddim_eta: float, n_iter: int, n_samples: int, cfg_scale: float, denoising_strength: float, seed: int, height: int, width: int):
    torch.cuda.empty_cache()

    outpath = opt.outdir or "outputs/img2img-samples"

    if seed == -1:
        seed = random.randrange(4294967294)

    model_wrap = K.external.CompVisDenoiser(model)

    os.makedirs(outpath, exist_ok=True)

    batch_size = n_samples

    assert prompt is not None
    data = [batch_size * [prompt]]

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1
    seedit = 0

    image = init_img.convert("RGB")
    image = image.resize((width, height), resample=PIL.Image.Resampling.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)

    output_images = []
    precision_scope = autocast if opt.precision == "autocast" else nullcontext
    with torch.no_grad(), precision_scope("cuda"), model.ema_scope():
        init_image = 2. * image - 1.
        init_image = init_image.to(device)
        init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
        init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))  # move to latent space
        x0 = init_latent

        assert 0. <= denoising_strength <= 1., 'can only work with strength in [0.0, 1.0]'
        t_enc = int(denoising_strength * ddim_steps)

        for n in range(n_iter):
            for batch_index, prompts in enumerate(data):
                uc = None
                if cfg_scale != 1.0:
                    uc = model.get_learned_conditioning(batch_size * [""])
                if isinstance(prompts, tuple):
                    prompts = list(prompts)
                c = model.get_learned_conditioning(prompts)

                sigmas = model_wrap.get_sigmas(ddim_steps)

                current_seed = seed + n * len(data) + batch_index
                torch.manual_seed(current_seed)

                noise = torch.randn_like(x0) * sigmas[ddim_steps - t_enc - 1]  # for GPU draw
                xi = x0 + noise
                sigma_sched = sigmas[ddim_steps - t_enc - 1:]
                model_wrap_cfg = CFGDenoiser(model_wrap)
                extra_args = {'cond': c, 'uncond': uc, 'cond_scale': cfg_scale}

                samples_ddim = K.sampling.sample_lms(model_wrap_cfg, xi, sigma_sched, extra_args=extra_args, disable=False)
                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

                if not opt.skip_save or not opt.skip_grid:
                    for x_sample in x_samples_ddim:
                        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                        x_sample = x_sample.astype(np.uint8)

                        if use_GFPGAN and GFPGAN is not None:
                            cropped_faces, restored_faces, restored_img = GFPGAN.enhance(x_sample, has_aligned=False, only_center_face=False, paste_back=True)
                            x_sample = restored_img

                        image = Image.fromarray(x_sample)
                        image.save(os.path.join(sample_path, f"{base_count:05}-{current_seed}_{prompt.replace(' ', '_').translate({ord(x): '' for x in invalid_filename_chars})[:128]}.png"))

                        output_images.append(image)
                        base_count += 1

        if not opt.skip_grid:
            # additionally, save as grid
            grid = image_grid(output_images, batch_size)
            grid.save(os.path.join(outpath, f'grid-{grid_count:04}.png'))
            grid_count += 1

    return output_images, seed


# prompt, init_img, ddim_steps, plms, ddim_eta, n_iter, n_samples, cfg_scale, denoising_strength, seed

img2img_interface = gr.Interface(
    translation,
    inputs=[
        gr.Textbox(placeholder="A fantasy landscape, trending on artstation.", lines=1),
        gr.Image(value="https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg", source="upload", interactive=True, type="pil"),
        gr.Slider(minimum=1, maximum=150, step=1, label="Sampling Steps", value=50),
        gr.Checkbox(label='Fix faces using GFPGAN', value=False, visible=GFPGAN is not None),
        gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="DDIM ETA", value=0.0, visible=False),
        gr.Slider(minimum=1, maximum=16, step=1, label='Sampling iterations', value=1),
        gr.Slider(minimum=1, maximum=4, step=1, label='Samples per iteration', value=1),
        gr.Slider(minimum=1.0, maximum=15.0, step=0.5, label='Classifier Free Guidance Scale', value=7.0),
        gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label='Denoising Strength', value=0.75),
        gr.Number(label='Seed', value=-1),
        gr.Slider(minimum=64, maximum=2048, step=64, label="Resize Height", value=512),
        gr.Slider(minimum=64, maximum=2048, step=64, label="Resize Width", value=512),
    ],
    outputs=[
        gr.Gallery(),
        gr.Number(label='Seed')
    ],
    title="Stable Diffusion Image-to-Image",
    description="Generate images from images with Stable Diffusion",
)

interfaces = [
    (dream_interface, "Dream"),
    (img2img_interface, "Image Translation")
]

def run_GFPGAN(image, strength):
    image = image.convert("RGB")

    cropped_faces, restored_faces, restored_img = GFPGAN.enhance(np.array(image, dtype=np.uint8), has_aligned=False, only_center_face=False, paste_back=True)
    res = Image.fromarray(restored_img)

    if strength < 1.0:
        res = PIL.Image.blend(image, res, strength)

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
    ), "GFPGAN"))

demo = gr.TabbedInterface(interface_list=[x[0] for x in interfaces], tab_names=[x[1] for x in interfaces])

demo.launch()
