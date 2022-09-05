import contextlib
import json
import math
import os
import sys

import torch
import numpy as np
from PIL import Image, ImageFilter, ImageOps
import random

import modules.sd_hijack
from modules.sd_hijack import model_hijack
from modules.sd_samplers import samplers, samplers_for_img2img
from modules.shared import opts, cmd_opts, state
import modules.shared as shared
import modules.gfpgan_model as gfpgan
import modules.images as images

# some of those options should not be changed at all because they would break the model, so I removed them from options.
opt_C = 4
opt_f = 8


def torch_gc():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


class StableDiffusionProcessing:
    def __init__(self, sd_model=None, outpath_samples=None, outpath_grids=None, prompt="", seed=-1, sampler_index=0, batch_size=1, n_iter=1, steps=50, cfg_scale=7.0, width=512, height=512, use_GFPGAN=False, tiling=False, do_not_save_samples=False, do_not_save_grid=False, extra_generation_params=None, overlay_images=None, negative_prompt=None):
        self.sd_model = sd_model
        self.outpath_samples: str = outpath_samples
        self.outpath_grids: str = outpath_grids
        self.prompt: str = prompt
        self.prompt_for_display: str = None
        self.negative_prompt: str = (negative_prompt or "")
        self.seed: int = seed
        self.sampler_index: int = sampler_index
        self.batch_size: int = batch_size
        self.n_iter: int = n_iter
        self.steps: int = steps
        self.cfg_scale: float = cfg_scale
        self.width: int = width
        self.height: int = height
        self.use_GFPGAN: bool = use_GFPGAN
        self.tiling: bool = tiling
        self.do_not_save_samples: bool = do_not_save_samples
        self.do_not_save_grid: bool = do_not_save_grid
        self.extra_generation_params: dict = extra_generation_params
        self.overlay_images = overlay_images
        self.paste_to = None

    def init(self):
        pass

    def sample(self, x, conditioning, unconditional_conditioning):
        raise NotImplementedError()


class Processed:
    def __init__(self, p: StableDiffusionProcessing, images_list, seed, info):
        self.images = images_list
        self.prompt = p.prompt
        self.seed = seed
        self.info = info
        self.width = p.width
        self.height = p.height
        self.sampler = samplers[p.sampler_index].name
        self.cfg_scale = p.cfg_scale
        self.steps = p.steps

    def js(self):
        obj = {
            "prompt": self.prompt if type(self.prompt) != list else self.prompt[0],
            "seed": int(self.seed if type(self.seed) != list else self.seed[0]),
            "width": self.width,
            "height": self.height,
            "sampler": self.sampler,
            "cfg_scale": self.cfg_scale,
            "steps": self.steps,
        }

        return json.dumps(obj)


def create_random_tensors(shape, seeds):
    xs = []
    for seed in seeds:
        torch.manual_seed(seed)

        # randn results depend on device; gpu and cpu get different results for same seed;
        # the way I see it, it's better to do this on CPU, so that everyone gets same result;
        # but the original script had it like this so I do not dare change it for now because
        # it will break everyone's seeds.
        xs.append(torch.randn(shape, device=shared.device))
    x = torch.stack(xs)
    return x


def process_images(p: StableDiffusionProcessing) -> Processed:
    """this is the main loop that both txt2img and img2img use; it calls func_init once inside all the scopes and func_sample once per batch"""

    prompt = p.prompt

    assert p.prompt is not None
    torch_gc()

    seed = int(random.randrange(4294967294)) if p.seed == -1 else p.seed

    os.makedirs(p.outpath_samples, exist_ok=True)
    os.makedirs(p.outpath_grids, exist_ok=True)

    modules.sd_hijack.model_hijack.apply_circular(p.tiling)

    comments = []

    if type(prompt) == list:
        all_prompts = prompt
    else:
        all_prompts = p.batch_size * p.n_iter * [prompt]

    if type(seed) == list:
        all_seeds = seed
    else:
        all_seeds = [int(seed + x) for x in range(len(all_prompts))]

    def infotext(iteration=0, position_in_batch=0):
        generation_params = {
            "Steps": p.steps,
            "Sampler": samplers[p.sampler_index].name,
            "CFG scale": p.cfg_scale,
            "Seed": all_seeds[position_in_batch + iteration * p.batch_size],
            "GFPGAN": ("GFPGAN" if p.use_GFPGAN else None),
            "Batch size": (None if p.batch_size < 2 else p.batch_size),
            "Batch pos": (None if p.batch_size < 2 else position_in_batch),
        }

        if p.extra_generation_params is not None:
            generation_params.update(p.extra_generation_params)

        generation_params_text = ", ".join([k if k == v else f'{k}: {v}' for k, v in generation_params.items() if v is not None])

        return f"{p.prompt_for_display or prompt}\n{generation_params_text}".strip() + "".join(["\n\n" + x for x in comments])

    if os.path.exists(cmd_opts.embeddings_dir):
        model_hijack.load_textual_inversion_embeddings(cmd_opts.embeddings_dir, p.sd_model)

    output_images = []
    precision_scope = torch.autocast if cmd_opts.precision == "autocast" else contextlib.nullcontext
    ema_scope = (contextlib.nullcontext if cmd_opts.lowvram else p.sd_model.ema_scope)
    with torch.no_grad(), precision_scope("cuda"), ema_scope():
        p.init()

        for n in range(p.n_iter):
            if state.interrupted:
                break

            prompts = all_prompts[n * p.batch_size:(n + 1) * p.batch_size]
            seeds = all_seeds[n * p.batch_size:(n + 1) * p.batch_size]

            uc = p.sd_model.get_learned_conditioning(len(prompts) * [p.negative_prompt])
            c = p.sd_model.get_learned_conditioning(prompts)

            if len(model_hijack.comments) > 0:
                comments += model_hijack.comments

            # we manually generate all input noises because each one should have a specific seed
            x = create_random_tensors([opt_C, p.height // opt_f, p.width // opt_f], seeds=seeds)

            if p.n_iter > 1:
                shared.state.job = f"Batch {n+1} out of {p.n_iter}"

            samples_ddim = p.sample(x=x, conditioning=c, unconditional_conditioning=uc)

            x_samples_ddim = p.sd_model.decode_first_stage(samples_ddim)
            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

            for i, x_sample in enumerate(x_samples_ddim):
                x_sample = 255. * np.moveaxis(x_sample.cpu().numpy(), 0, 2)
                x_sample = x_sample.astype(np.uint8)

                if p.use_GFPGAN:
                    torch_gc()

                    x_sample = gfpgan.gfpgan_fix_faces(x_sample)

                image = Image.fromarray(x_sample)

                if p.overlay_images is not None and i < len(p.overlay_images):
                    overlay = p.overlay_images[i]

                    if p.paste_to is not None:
                        x, y, w, h = p.paste_to
                        base_image = Image.new('RGBA', (overlay.width, overlay.height))
                        image = images.resize_image(1, image, w, h)
                        base_image.paste(image, (x, y))
                        image = base_image

                    image = image.convert('RGBA')
                    image.alpha_composite(overlay)
                    image = image.convert('RGB')

                if opts.samples_save and not p.do_not_save_samples:
                    images.save_image(image, p.outpath_samples, "", seeds[i], prompts[i], opts.samples_format, info=infotext(n, i))

                output_images.append(image)

        unwanted_grid_because_of_img_count = len(output_images) < 2 and opts.grid_only_if_multiple
        if not p.do_not_save_grid and not unwanted_grid_because_of_img_count:
            return_grid = opts.return_grid

            grid = images.image_grid(output_images, p.batch_size)

            if return_grid:
                output_images.insert(0, grid)

            if opts.grid_save:
                images.save_image(grid, p.outpath_grids, "grid", seed, all_prompts[0], opts.grid_format, info=infotext(), short_filename=not opts.grid_extended_filename)

    torch_gc()
    return Processed(p, output_images, seed, infotext())


class StableDiffusionProcessingTxt2Img(StableDiffusionProcessing):
    sampler = None

    def init(self):
        self.sampler = samplers[self.sampler_index].constructor(self.sd_model)

    def sample(self, x, conditioning, unconditional_conditioning):
        samples_ddim = self.sampler.sample(self, x, conditioning, unconditional_conditioning)
        return samples_ddim


def get_crop_region(mask, pad=0):
    h, w = mask.shape

    crop_left = 0
    for i in range(w):
        if not (mask[:, i] == 0).all():
            break
        crop_left += 1

    crop_right = 0
    for i in reversed(range(w)):
        if not (mask[:, i] == 0).all():
            break
        crop_right += 1

    crop_top = 0
    for i in range(h):
        if not (mask[i] == 0).all():
            break
        crop_top += 1

    crop_bottom = 0
    for i in reversed(range(h)):
        if not (mask[i] == 0).all():
            break
        crop_bottom += 1

    return (
        int(max(crop_left-pad, 0)),
        int(max(crop_top-pad, 0)),
        int(min(w - crop_right + pad, w)),
        int(min(h - crop_bottom + pad, h))
    )


def fill(image, mask):
    image_mod = Image.new('RGBA', (image.width, image.height))

    image_masked = Image.new('RGBa', (image.width, image.height))
    image_masked.paste(image.convert("RGBA").convert("RGBa"), mask=ImageOps.invert(mask.convert('L')))

    image_masked = image_masked.convert('RGBa')

    for radius, repeats in [(256, 1), (64, 1), (16, 2), (4, 4), (2, 2), (0, 1)]:
        blurred = image_masked.filter(ImageFilter.GaussianBlur(radius)).convert('RGBA')
        for _ in range(repeats):
            image_mod.alpha_composite(blurred)

    return image_mod.convert("RGB")


class StableDiffusionProcessingImg2Img(StableDiffusionProcessing):
    sampler = None

    def __init__(self, init_images=None, resize_mode=0, denoising_strength=0.75, mask=None, mask_blur=4, inpainting_fill=0, inpaint_full_res=True, inpainting_mask_invert=0, **kwargs):
        super().__init__(**kwargs)

        self.init_images = init_images
        self.resize_mode: int = resize_mode
        self.denoising_strength: float = denoising_strength
        self.init_latent = None
        self.image_mask = mask
        #self.image_unblurred_mask = None
        self.latent_mask = None
        self.mask_for_overlay = None
        self.mask_blur = mask_blur
        self.inpainting_fill = inpainting_fill
        self.inpaint_full_res = inpaint_full_res
        self.inpainting_mask_invert = inpainting_mask_invert
        self.mask = None
        self.nmask = None

    def init(self):
        self.sampler = samplers_for_img2img[self.sampler_index].constructor(self.sd_model)
        crop_region = None

        if self.image_mask is not None:
            self.image_mask = self.image_mask.convert('L')

            if self.inpainting_mask_invert:
                self.image_mask = ImageOps.invert(self.image_mask)

            #self.image_unblurred_mask = self.image_mask

            if self.mask_blur > 0:
                self.image_mask = self.image_mask.filter(ImageFilter.GaussianBlur(self.mask_blur))

            if self.inpaint_full_res:
                self.mask_for_overlay = self.image_mask
                mask = self.image_mask.convert('L')
                crop_region = get_crop_region(np.array(mask), 64)
                x1, y1, x2, y2 = crop_region

                mask = mask.crop(crop_region)
                self.image_mask = images.resize_image(2, mask, self.width, self.height)
                self.paste_to = (x1, y1, x2-x1, y2-y1)
            else:
                self.image_mask = images.resize_image(self.resize_mode, self.image_mask, self.width, self.height)
                self.mask_for_overlay = self.image_mask

            self.overlay_images = []

        imgs = []
        for img in self.init_images:
            image = img.convert("RGB")

            if crop_region is None:
                image = images.resize_image(self.resize_mode, image, self.width, self.height)

            if self.image_mask is not None:
                if self.inpainting_fill != 1:
                    image = fill(image, self.mask_for_overlay)

                image_masked = Image.new('RGBa', (image.width, image.height))
                image_masked.paste(image.convert("RGBA").convert("RGBa"), mask=ImageOps.invert(self.mask_for_overlay.convert('L')))

                self.overlay_images.append(image_masked.convert('RGBA'))

            if crop_region is not None:
                image = image.crop(crop_region)
                image = images.resize_image(2, image, self.width, self.height)

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
        image = image.to(shared.device)

        self.init_latent = self.sd_model.get_first_stage_encoding(self.sd_model.encode_first_stage(image))

        if self.image_mask is not None:
            init_mask = self.latent_mask if self.latent_mask is not None else self.image_mask
            latmask = init_mask.convert('RGB').resize((self.init_latent.shape[3], self.init_latent.shape[2]))
            latmask = np.moveaxis(np.array(latmask, dtype=np.float64), 2, 0) / 255
            latmask = latmask[0]
            latmask = np.tile(latmask[None], (4, 1, 1))

            self.mask = torch.asarray(1.0 - latmask).to(shared.device).type(self.sd_model.dtype)
            self.nmask = torch.asarray(latmask).to(shared.device).type(self.sd_model.dtype)

            if self.inpainting_fill == 2:
                self.init_latent = self.init_latent * self.mask + create_random_tensors(self.init_latent.shape[1:], [self.seed + x + 1 for x in range(self.init_latent.shape[0])]) * self.nmask
            elif self.inpainting_fill == 3:
                self.init_latent = self.init_latent * self.mask

    def sample(self, x, conditioning, unconditional_conditioning):
        samples = self.sampler.sample_img2img(self, self.init_latent, x, conditioning, unconditional_conditioning)

        if self.mask is not None:
            samples = samples * self.nmask + self.init_latent * self.mask

        return samples
