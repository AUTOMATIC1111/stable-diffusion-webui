import contextlib
import json
import math
import os
import sys

import torch
import numpy as np
from PIL import Image, ImageFilter, ImageOps
import random
import cv2
from skimage import exposure

import modules.sd_hijack
from modules import devices, prompt_parser, masking
from modules.sd_hijack import model_hijack
from modules.sd_samplers import samplers, samplers_for_img2img
from modules.shared import opts, cmd_opts, state
import modules.shared as shared
import modules.face_restoration
import modules.images as images
import modules.styles


# some of those options should not be changed at all because they would break the model, so I removed them from options.
opt_C = 4
opt_f = 8


def setup_color_correction(image):
    correction_target = cv2.cvtColor(np.asarray(image.copy()), cv2.COLOR_RGB2LAB)
    return correction_target


def apply_color_correction(correction, image):
    image = Image.fromarray(cv2.cvtColor(exposure.match_histograms(
        cv2.cvtColor(
            np.asarray(image),
            cv2.COLOR_RGB2LAB
        ),
        correction,
        channel_axis=2
    ), cv2.COLOR_LAB2RGB).astype("uint8"))

    return image


class StableDiffusionProcessing:
    def __init__(self, sd_model=None, outpath_samples=None, outpath_grids=None, prompt="", styles=None, seed=-1, subseed=-1, subseed_strength=0, seed_resize_from_h=-1, seed_resize_from_w=-1, sampler_index=0, batch_size=1, n_iter=1, steps=50, cfg_scale=7.0, width=512, height=512, restore_faces=False, tiling=False, do_not_save_samples=False, do_not_save_grid=False, extra_generation_params=None, overlay_images=None, negative_prompt=None):
        self.sd_model = sd_model
        self.outpath_samples: str = outpath_samples
        self.outpath_grids: str = outpath_grids
        self.prompt: str = prompt
        self.prompt_for_display: str = None
        self.negative_prompt: str = (negative_prompt or "")
        self.styles: str = styles
        self.seed: int = seed
        self.subseed: int = subseed
        self.subseed_strength: float = subseed_strength
        self.seed_resize_from_h: int = seed_resize_from_h
        self.seed_resize_from_w: int = seed_resize_from_w
        self.sampler_index: int = sampler_index
        self.batch_size: int = batch_size
        self.n_iter: int = n_iter
        self.steps: int = steps
        self.cfg_scale: float = cfg_scale
        self.width: int = width
        self.height: int = height
        self.restore_faces: bool = restore_faces
        self.tiling: bool = tiling
        self.do_not_save_samples: bool = do_not_save_samples
        self.do_not_save_grid: bool = do_not_save_grid
        self.extra_generation_params: dict = extra_generation_params or {}
        self.overlay_images = overlay_images
        self.paste_to = None
        self.color_corrections = None
        self.denoising_strength: float = 0

    def init(self, all_prompts, all_seeds, all_subseeds):
        pass

    def sample(self, conditioning, unconditional_conditioning, seeds, subseeds, subseed_strength):
        raise NotImplementedError()


class Processed:
    def __init__(self, p: StableDiffusionProcessing, images_list, seed=-1, info="", subseed=None, all_prompts=None, all_seeds=None, all_subseeds=None, index_of_first_image=0):
        self.images = images_list
        self.prompt = p.prompt
        self.negative_prompt = p.negative_prompt
        self.seed = seed
        self.subseed = subseed
        self.subseed_strength = p.subseed_strength
        self.info = info
        self.width = p.width
        self.height = p.height
        self.sampler_index = p.sampler_index
        self.sampler = samplers[p.sampler_index].name
        self.cfg_scale = p.cfg_scale
        self.steps = p.steps
        self.batch_size = p.batch_size
        self.restore_faces = p.restore_faces
        self.face_restoration_model = opts.face_restoration_model if p.restore_faces else None
        self.sd_model_hash = shared.sd_model.sd_model_hash
        self.seed_resize_from_w = p.seed_resize_from_w
        self.seed_resize_from_h = p.seed_resize_from_h
        self.denoising_strength = getattr(p, 'denoising_strength', None)
        self.extra_generation_params = p.extra_generation_params
        self.index_of_first_image = index_of_first_image

        self.prompt = self.prompt if type(self.prompt) != list else self.prompt[0]
        self.negative_prompt = self.negative_prompt if type(self.negative_prompt) != list else self.negative_prompt[0]
        self.seed = int(self.seed if type(self.seed) != list else self.seed[0])
        self.subseed = int(self.subseed if type(self.subseed) != list else self.subseed[0]) if self.subseed is not None else -1

        self.all_prompts = all_prompts or [self.prompt]
        self.all_seeds = all_seeds or [self.seed]
        self.all_subseeds = all_subseeds or [self.subseed]

    def js(self):
        obj = {
            "prompt": self.prompt,
            "all_prompts": self.all_prompts,
            "negative_prompt": self.negative_prompt,
            "seed": self.seed,
            "all_seeds": self.all_seeds,
            "subseed": self.subseed,
            "all_subseeds": self.all_subseeds,
            "subseed_strength": self.subseed_strength,
            "width": self.width,
            "height": self.height,
            "sampler_index": self.sampler_index,
            "sampler": self.sampler,
            "cfg_scale": self.cfg_scale,
            "steps": self.steps,
            "batch_size": self.batch_size,
            "restore_faces": self.restore_faces,
            "face_restoration_model": self.face_restoration_model,
            "sd_model_hash": self.sd_model_hash,
            "seed_resize_from_w": self.seed_resize_from_w,
            "seed_resize_from_h": self.seed_resize_from_h,
            "denoising_strength": self.denoising_strength,
            "extra_generation_params": self.extra_generation_params,
            "index_of_first_image": self.index_of_first_image,
        }

        return json.dumps(obj)

    def infotext(self,  p: StableDiffusionProcessing, index):
        return create_infotext(p, self.all_prompts, self.all_seeds, self.all_subseeds, comments=[], position_in_batch=index % self.batch_size, iteration=index // self.batch_size)


# from https://discuss.pytorch.org/t/help-regarding-slerp-function-for-generative-model-sampling/32475/3
def slerp(val, low, high):
    low_norm = low/torch.norm(low, dim=1, keepdim=True)
    high_norm = high/torch.norm(high, dim=1, keepdim=True)
    dot = (low_norm*high_norm).sum(1)

    if dot.mean() > 0.9995:
        return low * val + high * (1 - val)

    omega = torch.acos(dot)
    so = torch.sin(omega)
    res = (torch.sin((1.0-val)*omega)/so).unsqueeze(1)*low + (torch.sin(val*omega)/so).unsqueeze(1) * high
    return res


def create_random_tensors(shape, seeds, subseeds=None, subseed_strength=0.0, seed_resize_from_h=0, seed_resize_from_w=0, p=None):
    xs = []

    # if we have multiple seeds, this means we are working with batch size>1; this then
    # enables the generation of additional tensors with noise that the sampler will use during its processing.
    # Using those pre-generated tensors instead of simple torch.randn allows a batch with seeds [100, 101] to
    # produce the same images as with two batches [100], [101].
    if p is not None and p.sampler is not None and len(seeds) > 1 and opts.enable_batch_seeds:
        sampler_noises = [[] for _ in range(p.sampler.number_of_needed_noises(p))]
    else:
        sampler_noises = None

    for i, seed in enumerate(seeds):
        noise_shape = shape if seed_resize_from_h <= 0 or seed_resize_from_w <= 0 else (shape[0], seed_resize_from_h//8, seed_resize_from_w//8)

        subnoise = None
        if subseeds is not None:
            subseed = 0 if i >= len(subseeds) else subseeds[i]

            subnoise = devices.randn(subseed, noise_shape)

        # randn results depend on device; gpu and cpu get different results for same seed;
        # the way I see it, it's better to do this on CPU, so that everyone gets same result;
        # but the original script had it like this, so I do not dare change it for now because
        # it will break everyone's seeds.
        noise = devices.randn(seed, noise_shape)

        if subnoise is not None:
            noise = slerp(subseed_strength, noise, subnoise)

        if noise_shape != shape:
            x = devices.randn(seed, shape)
            dx = (shape[2] - noise_shape[2]) // 2
            dy = (shape[1] - noise_shape[1]) // 2
            w = noise_shape[2] if dx >= 0 else noise_shape[2] + 2 * dx
            h = noise_shape[1] if dy >= 0 else noise_shape[1] + 2 * dy
            tx = 0 if dx < 0 else dx
            ty = 0 if dy < 0 else dy
            dx = max(-dx, 0)
            dy = max(-dy, 0)

            x[:, ty:ty+h, tx:tx+w] = noise[:, dy:dy+h, dx:dx+w]
            noise = x

        if sampler_noises is not None:
            cnt = p.sampler.number_of_needed_noises(p)

            for j in range(cnt):
                sampler_noises[j].append(devices.randn_without_seed(tuple(noise_shape)))

        xs.append(noise)

    if sampler_noises is not None:
        p.sampler.sampler_noises = [torch.stack(n).to(shared.device) for n in sampler_noises]

    x = torch.stack(xs).to(shared.device)
    return x


def fix_seed(p):
    p.seed = int(random.randrange(4294967294)) if p.seed is None or p.seed == '' or p.seed == -1 else p.seed
    p.subseed = int(random.randrange(4294967294)) if p.subseed is None or p.subseed == '' or p.subseed == -1 else p.subseed


def create_infotext(p, all_prompts, all_seeds, all_subseeds, comments, iteration=0, position_in_batch=0):
    index = position_in_batch + iteration * p.batch_size

    generation_params = {
        "Steps": p.steps,
        "Sampler": samplers[p.sampler_index].name,
        "CFG scale": p.cfg_scale,
        "Seed": all_seeds[index],
        "Face restoration": (opts.face_restoration_model if p.restore_faces else None),
        "Size": f"{p.width}x{p.height}",
        "Model hash": getattr(p, 'sd_model_hash', None if not opts.add_model_hash_to_info or not shared.sd_model.sd_model_hash else shared.sd_model.sd_model_hash),
        "Batch size": (None if p.batch_size < 2 else p.batch_size),
        "Batch pos": (None if p.batch_size < 2 else position_in_batch),
        "Variation seed": (None if p.subseed_strength == 0 else all_subseeds[index]),
        "Variation seed strength": (None if p.subseed_strength == 0 else p.subseed_strength),
        "Seed resize from": (None if p.seed_resize_from_w == 0 or p.seed_resize_from_h == 0 else f"{p.seed_resize_from_w}x{p.seed_resize_from_h}"),
        "Denoising strength": getattr(p, 'denoising_strength', None),
    }

    generation_params.update(p.extra_generation_params)

    generation_params_text = ", ".join([k if k == v else f'{k}: {v}' for k, v in generation_params.items() if v is not None])

    negative_prompt_text = "\nNegative prompt: " + p.negative_prompt if p.negative_prompt else ""

    return f"{all_prompts[index]}{negative_prompt_text}\n{generation_params_text}".strip() + "".join(["\n\n" + x for x in comments])


def process_images(p: StableDiffusionProcessing) -> Processed:
    """this is the main loop that both txt2img and img2img use; it calls func_init once inside all the scopes and func_sample once per batch"""

    if type(p.prompt) == list:
        assert(len(p.prompt) > 0)
    else:
        assert p.prompt is not None
        
    devices.torch_gc()

    fix_seed(p)

    os.makedirs(p.outpath_samples, exist_ok=True)
    os.makedirs(p.outpath_grids, exist_ok=True)

    modules.sd_hijack.model_hijack.apply_circular(p.tiling)

    comments = {}

    shared.prompt_styles.apply_styles(p)

    if type(p.prompt) == list:
        all_prompts = p.prompt
    else:
        all_prompts = p.batch_size * p.n_iter * [p.prompt]

    if type(p.seed) == list:
        all_seeds = p.seed
    else:
        all_seeds = [int(p.seed) + (x if p.subseed_strength == 0 else 0) for x in range(len(all_prompts))]

    if type(p.subseed) == list:
        all_subseeds = p.subseed
    else:
        all_subseeds = [int(p.subseed) + x for x in range(len(all_prompts))]

    def infotext(iteration=0, position_in_batch=0):
        return create_infotext(p, all_prompts, all_seeds, all_subseeds, comments, iteration, position_in_batch)

    if os.path.exists(cmd_opts.embeddings_dir):
        model_hijack.load_textual_inversion_embeddings(cmd_opts.embeddings_dir, p.sd_model)

    output_images = []
    precision_scope = torch.autocast if cmd_opts.precision == "autocast" else contextlib.nullcontext
    ema_scope = (contextlib.nullcontext if cmd_opts.lowvram else p.sd_model.ema_scope)
    with torch.no_grad(), precision_scope("cuda"), ema_scope():
        p.init(all_prompts, all_seeds, all_subseeds)

        if state.job_count == -1:
            state.job_count = p.n_iter

        for n in range(p.n_iter):
            if state.interrupted:
                break

            prompts = all_prompts[n * p.batch_size:(n + 1) * p.batch_size]
            seeds = all_seeds[n * p.batch_size:(n + 1) * p.batch_size]
            subseeds = all_subseeds[n * p.batch_size:(n + 1) * p.batch_size]

            if (len(prompts) == 0):
                break

            #uc = p.sd_model.get_learned_conditioning(len(prompts) * [p.negative_prompt])
            #c = p.sd_model.get_learned_conditioning(prompts)
            uc = prompt_parser.get_learned_conditioning(len(prompts) * [p.negative_prompt], p.steps)
            c = prompt_parser.get_learned_conditioning(prompts, p.steps)

            if len(model_hijack.comments) > 0:
                for comment in model_hijack.comments:
                    comments[comment] = 1

            if p.n_iter > 1:
                shared.state.job = f"Batch {n+1} out of {p.n_iter}"

            samples_ddim = p.sample(conditioning=c, unconditional_conditioning=uc, seeds=seeds, subseeds=subseeds, subseed_strength=p.subseed_strength)
            if state.interrupted:

                # if we are interruped, sample returns just noise
                # use the image collected previously in sampler loop
                samples_ddim = shared.state.current_latent

            x_samples_ddim = p.sd_model.decode_first_stage(samples_ddim)
            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

            if opts.filter_nsfw:
                import modules.safety as safety
                x_samples_ddim = modules.safety.censor_batch(x_samples_ddim)

            for i, x_sample in enumerate(x_samples_ddim):
                x_sample = 255. * np.moveaxis(x_sample.cpu().numpy(), 0, 2)
                x_sample = x_sample.astype(np.uint8)

                if p.restore_faces:
                    if opts.save and not p.do_not_save_samples and opts.save_images_before_face_restoration:
                        images.save_image(Image.fromarray(x_sample), p.outpath_samples, "", seeds[i], prompts[i], opts.samples_format, info=infotext(n, i), p=p)

                    devices.torch_gc()

                    x_sample = modules.face_restoration.restore_faces(x_sample)

                image = Image.fromarray(x_sample)

                if p.color_corrections is not None and i < len(p.color_corrections):
                    image = apply_color_correction(p.color_corrections[i], image)

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
                    images.save_image(image, p.outpath_samples, "", seeds[i], prompts[i], opts.samples_format, info=infotext(n, i), p=p)

                output_images.append(image)

            state.nextjob()

        p.color_corrections = None

        index_of_first_image = 0
        unwanted_grid_because_of_img_count = len(output_images) < 2 and opts.grid_only_if_multiple
        if (opts.return_grid or opts.grid_save) and not p.do_not_save_grid and not unwanted_grid_because_of_img_count:
            grid = images.image_grid(output_images, p.batch_size)

            if opts.return_grid:
                output_images.insert(0, grid)
                index_of_first_image = 1

            if opts.grid_save:
                images.save_image(grid, p.outpath_grids, "grid", all_seeds[0], all_prompts[0], opts.grid_format, info=infotext(), short_filename=not opts.grid_extended_filename, p=p)

    devices.torch_gc()
    return Processed(p, output_images, all_seeds[0], infotext(), subseed=all_subseeds[0], all_prompts=all_prompts, all_seeds=all_seeds, all_subseeds=all_subseeds, index_of_first_image=index_of_first_image)


class StableDiffusionProcessingTxt2Img(StableDiffusionProcessing):
    sampler = None
    firstphase_width = 0
    firstphase_height = 0
    firstphase_width_truncated = 0
    firstphase_height_truncated = 0

    def __init__(self, enable_hr=False, scale_latent=True, denoising_strength=0.75, **kwargs):
        super().__init__(**kwargs)
        self.enable_hr = enable_hr
        self.scale_latent = scale_latent
        self.denoising_strength = denoising_strength

    def init(self, all_prompts, all_seeds, all_subseeds):
        if self.enable_hr:
            if state.job_count == -1:
                state.job_count = self.n_iter * 2
            else:
                state.job_count = state.job_count * 2

            desired_pixel_count = 512 * 512
            actual_pixel_count = self.width * self.height
            scale = math.sqrt(desired_pixel_count / actual_pixel_count)

            self.firstphase_width = math.ceil(scale * self.width / 64) * 64
            self.firstphase_height = math.ceil(scale * self.height / 64) * 64
            self.firstphase_width_truncated = int(scale * self.width)
            self.firstphase_height_truncated = int(scale * self.height)

    def sample(self, conditioning, unconditional_conditioning, seeds, subseeds, subseed_strength):
        self.sampler = samplers[self.sampler_index].constructor(self.sd_model)

        if not self.enable_hr:
            x = create_random_tensors([opt_C, self.height // opt_f, self.width // opt_f], seeds=seeds, subseeds=subseeds, subseed_strength=self.subseed_strength, seed_resize_from_h=self.seed_resize_from_h, seed_resize_from_w=self.seed_resize_from_w, p=self)
            samples = self.sampler.sample(self, x, conditioning, unconditional_conditioning)
            return samples

        x = create_random_tensors([opt_C, self.firstphase_height // opt_f, self.firstphase_width // opt_f], seeds=seeds, subseeds=subseeds, subseed_strength=self.subseed_strength, seed_resize_from_h=self.seed_resize_from_h, seed_resize_from_w=self.seed_resize_from_w, p=self)
        samples = self.sampler.sample(self, x, conditioning, unconditional_conditioning)

        truncate_x = (self.firstphase_width - self.firstphase_width_truncated) // opt_f
        truncate_y = (self.firstphase_height - self.firstphase_height_truncated) // opt_f

        samples = samples[:, :, truncate_y//2:samples.shape[2]-truncate_y//2, truncate_x//2:samples.shape[3]-truncate_x//2]

        if self.scale_latent:
            samples = torch.nn.functional.interpolate(samples, size=(self.height // opt_f, self.width // opt_f), mode="bilinear")
        else:
            decoded_samples = self.sd_model.decode_first_stage(samples)

            if opts.upscaler_for_hires_fix is None or opts.upscaler_for_hires_fix == "None":
                decoded_samples = torch.nn.functional.interpolate(decoded_samples, size=(self.height, self.width), mode="bilinear")
            else:
                lowres_samples = torch.clamp((decoded_samples + 1.0) / 2.0, min=0.0, max=1.0)

                batch_images = []
                for i, x_sample in enumerate(lowres_samples):
                    x_sample = 255. * np.moveaxis(x_sample.cpu().numpy(), 0, 2)
                    x_sample = x_sample.astype(np.uint8)
                    image = Image.fromarray(x_sample)
                    upscaler = [x for x in shared.sd_upscalers if x.name == opts.upscaler_for_hires_fix][0]
                    image = upscaler.upscale(image, self.width, self.height)
                    image = np.array(image).astype(np.float32) / 255.0
                    image = np.moveaxis(image, 2, 0)
                    batch_images.append(image)

                decoded_samples = torch.from_numpy(np.array(batch_images))
                decoded_samples = decoded_samples.to(shared.device)
                decoded_samples = 2. * decoded_samples - 1.

            samples = self.sd_model.get_first_stage_encoding(self.sd_model.encode_first_stage(decoded_samples))

        shared.state.nextjob()

        self.sampler = samplers[self.sampler_index].constructor(self.sd_model)
        noise = create_random_tensors(samples.shape[1:], seeds=seeds, subseeds=subseeds, subseed_strength=subseed_strength, seed_resize_from_h=self.seed_resize_from_h, seed_resize_from_w=self.seed_resize_from_w, p=self)
        samples = self.sampler.sample_img2img(self, samples, noise, conditioning, unconditional_conditioning, steps=self.steps)

        return samples


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

    def init(self, all_prompts, all_seeds, all_subseeds):
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
                crop_region = masking.get_crop_region(np.array(mask), opts.upscale_at_full_resolution_padding)
                crop_region = masking.expand_crop_region(crop_region, self.width, self.height, mask.width, mask.height)
                x1, y1, x2, y2 = crop_region

                mask = mask.crop(crop_region)
                self.image_mask = images.resize_image(2, mask, self.width, self.height)
                self.paste_to = (x1, y1, x2-x1, y2-y1)
            else:
                self.image_mask = images.resize_image(self.resize_mode, self.image_mask, self.width, self.height)
                np_mask = np.array(self.image_mask)
                np_mask = np.clip((np_mask.astype(np.float32)) * 2, 0, 255).astype(np.uint8)
                self.mask_for_overlay = Image.fromarray(np_mask)

            self.overlay_images = []

        latent_mask = self.latent_mask if self.latent_mask is not None else self.image_mask

        add_color_corrections = opts.img2img_color_correction and self.color_corrections is None
        if add_color_corrections:
            self.color_corrections = []
        imgs = []
        for img in self.init_images:
            image = img.convert("RGB")

            if crop_region is None:
                image = images.resize_image(self.resize_mode, image, self.width, self.height)

            if self.image_mask is not None:
                image_masked = Image.new('RGBa', (image.width, image.height))
                image_masked.paste(image.convert("RGBA").convert("RGBa"), mask=ImageOps.invert(self.mask_for_overlay.convert('L')))

                self.overlay_images.append(image_masked.convert('RGBA'))

            if crop_region is not None:
                image = image.crop(crop_region)
                image = images.resize_image(2, image, self.width, self.height)

            if self.image_mask is not None:
                if self.inpainting_fill != 1:
                    image = masking.fill(image, latent_mask)

            if add_color_corrections:
                self.color_corrections.append(setup_color_correction(image))

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
            init_mask = latent_mask
            latmask = init_mask.convert('RGB').resize((self.init_latent.shape[3], self.init_latent.shape[2]))
            latmask = np.moveaxis(np.array(latmask, dtype=np.float32), 2, 0) / 255
            latmask = latmask[0]
            latmask = np.around(latmask)
            latmask = np.tile(latmask[None], (4, 1, 1))

            self.mask = torch.asarray(1.0 - latmask).to(shared.device).type(self.sd_model.dtype)
            self.nmask = torch.asarray(latmask).to(shared.device).type(self.sd_model.dtype)

            # this needs to be fixed to be done in sample() using actual seeds for batches
            if self.inpainting_fill == 2:
                self.init_latent = self.init_latent * self.mask + create_random_tensors(self.init_latent.shape[1:], all_seeds[0:self.init_latent.shape[0]]) * self.nmask
            elif self.inpainting_fill == 3:
                self.init_latent = self.init_latent * self.mask

    def sample(self, conditioning, unconditional_conditioning, seeds, subseeds, subseed_strength):
        x = create_random_tensors([opt_C, self.height // opt_f, self.width // opt_f], seeds=seeds, subseeds=subseeds, subseed_strength=self.subseed_strength, seed_resize_from_h=self.seed_resize_from_h, seed_resize_from_w=self.seed_resize_from_w, p=self)

        samples = self.sampler.sample_img2img(self, self.init_latent, x, conditioning, unconditional_conditioning)

        if self.mask is not None:
            samples = samples * self.nmask + self.init_latent * self.mask

        return samples
