import json
import math
import os
import sys
import random
import logging
from typing import Any, Dict, List

import torch
import numpy as np
from PIL import Image, ImageFilter, ImageOps
import cv2
from skimage import exposure

from ldm.data.util import AddMiDaS
from ldm.models.diffusion.ddpm import LatentDepth2ImageDiffusion
from einops import repeat, rearrange
from blendmodes.blend import blendLayers, BlendType

import modules.sd_hijack
from modules import devices, prompt_parser, masking, sd_samplers, lowvram, generation_parameters_copypaste, script_callbacks, extra_networks, sd_vae_approx, scripts # pylint: disable=unused-import
from modules.sd_hijack import model_hijack
from modules.shared import opts, cmd_opts, state # pylint: disable=unused-import
import modules.shared as shared
import modules.paths as paths
import modules.face_restoration
import modules.images as images
import modules.styles
import modules.sd_models as sd_models
import modules.sd_vae as sd_vae
import tomesd # pylint: disable=wrong-import-order


# add a logger for the processing module
logger = logging.getLogger(__name__)
# manually set output level here since there is no option to do so yet through launch options
# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(name)s %(message)s')

# some of those options should not be changed at all because they would break the model, so I removed them from options.
opt_C = 4
opt_f = 8


def setup_color_correction(image):
    logging.info("Calibrating color correction.")
    correction_target = cv2.cvtColor(np.asarray(image.copy()), cv2.COLOR_RGB2LAB)
    return correction_target


def apply_color_correction(correction, original_image):
    logging.info("Applying color correction.")
    image = Image.fromarray(cv2.cvtColor(exposure.match_histograms(
        cv2.cvtColor(
            np.asarray(original_image),
            cv2.COLOR_RGB2LAB
        ),
        correction,
        channel_axis=2
    ), cv2.COLOR_LAB2RGB).astype("uint8"))

    image = blendLayers(image, original_image, BlendType.LUMINOSITY)

    return image


def apply_overlay(image, paste_loc, index, overlays):
    if overlays is None or index >= len(overlays):
        return image

    overlay = overlays[index]

    if paste_loc is not None:
        x, y, w, h = paste_loc
        base_image = Image.new('RGBA', (overlay.width, overlay.height))
        image = images.resize_image(1, image, w, h)
        base_image.paste(image, (x, y))
        image = base_image

    image = image.convert('RGBA')
    image.alpha_composite(overlay)
    image = image.convert('RGB')

    return image


def txt2img_image_conditioning(sd_model, x, width, height):
    if sd_model.model.conditioning_key in {'hybrid', 'concat'}: # Inpainting models

        # The "masked-image" in this case will just be all zeros since the entire image is masked.
        image_conditioning = torch.zeros(x.shape[0], 3, height, width, device=x.device)
        image_conditioning = sd_model.get_first_stage_encoding(sd_model.encode_first_stage(image_conditioning))

        # Add the fake full 1s mask to the first dimension.
        image_conditioning = torch.nn.functional.pad(image_conditioning, (0, 0, 0, 0, 1, 0), value=1.0)
        image_conditioning = image_conditioning.to(x.dtype)

        return image_conditioning

    elif sd_model.model.conditioning_key == "crossattn-adm": # UnCLIP models

        return x.new_zeros(x.shape[0], 2*sd_model.noise_augmentor.time_embed.dim, dtype=x.dtype, device=x.device)

    else:
        # Dummy zero conditioning if we're not using inpainting or unclip models.
        # Still takes up a bit of memory, but no encoder call.
        # Pretty sure we can just make this a 1x1 image since its not going to be used besides its batch size.
        return x.new_zeros(x.shape[0], 5, 1, 1, dtype=x.dtype, device=x.device)


class StableDiffusionProcessing:
    """
    The first set of paramaters: sd_models -> do_not_reload_embeddings represent the minimum required to create a StableDiffusionProcessing
    """
    def __init__(self, sd_model=None, outpath_samples=None, outpath_grids=None, prompt: str = "", styles: List[str] = None, seed: int = -1, subseed: int = -1, subseed_strength: float = 0, seed_resize_from_h: int = -1, seed_resize_from_w: int = -1, seed_enable_extras: bool = True, sampler_name: str = None, batch_size: int = 1, n_iter: int = 1, steps: int = 50, cfg_scale: float = 7.0, width: int = 512, height: int = 512, restore_faces: bool = False, tiling: bool = False, do_not_save_samples: bool = False, do_not_save_grid: bool = False, extra_generation_params: Dict[Any, Any] = None, overlay_images: Any = None, negative_prompt: str = None, eta: float = None, do_not_reload_embeddings: bool = False, denoising_strength: float = 0, ddim_discretize: str = None, s_churn: float = 0.0, s_tmax: float = None, s_tmin: float = 0.0, s_noise: float = 1.0, override_settings: Dict[str, Any] = None, override_settings_restore_afterwards: bool = True, sampler_index: int = None, script_args: list = None): # pylint: disable=unused-argument
        if sampler_index is not None:
            print("sampler_index argument for StableDiffusionProcessing does not do anything; use sampler_name", file=sys.stderr)

        self.outpath_samples: str = outpath_samples
        self.outpath_grids: str = outpath_grids
        self.prompt: str = prompt
        self.prompt_for_display: str = None
        self.negative_prompt: str = (negative_prompt or "")
        self.styles: list = styles or []
        self.seed: int = seed
        self.subseed: int = subseed
        self.subseed_strength: float = subseed_strength
        self.seed_resize_from_h: int = seed_resize_from_h
        self.seed_resize_from_w: int = seed_resize_from_w
        self.sampler_name: str = sampler_name
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
        self.eta = eta
        self.do_not_reload_embeddings = do_not_reload_embeddings
        self.paste_to = None
        self.color_corrections = None
        self.denoising_strength: float = denoising_strength
        self.sampler_noise_scheduler_override = None
        self.ddim_discretize = ddim_discretize or opts.ddim_discretize
        self.s_churn = s_churn or opts.s_churn
        self.s_tmin = s_tmin or opts.s_tmin
        self.s_tmax = s_tmax or float('inf')  # not representable as a standard ui option
        self.s_noise = s_noise or opts.s_noise
        self.override_settings = {k: v for k, v in (override_settings or {}).items() if k not in shared.restricted_opts}
        self.override_settings_restore_afterwards = override_settings_restore_afterwards
        self.is_using_inpainting_conditioning = False
        self.disable_extra_networks = False
        if not seed_enable_extras:
            self.subseed = -1
            self.subseed_strength = 0
            self.seed_resize_from_h = 0
            self.seed_resize_from_w = 0
        self.scripts = None
        self.script_args = script_args
        self.all_prompts = None
        self.all_negative_prompts = None
        self.all_seeds = None
        self.all_subseeds = None
        self.iteration = 0

    @property
    def sd_model(self):
        return shared.sd_model

    def txt2img_image_conditioning(self, x, width=None, height=None):
        self.is_using_inpainting_conditioning = self.sd_model.model.conditioning_key in {'hybrid', 'concat'}

        return txt2img_image_conditioning(self.sd_model, x, width or self.width, height or self.height)

    def depth2img_image_conditioning(self, source_image):
        # Use the AddMiDaS helper to Format our source image to suit the MiDaS model
        transformer = AddMiDaS(model_type="dpt_hybrid")
        transformed = transformer({"jpg": rearrange(source_image[0], "c h w -> h w c")})
        midas_in = torch.from_numpy(transformed["midas_in"][None, ...]).to(device=shared.device)
        midas_in = repeat(midas_in, "1 ... -> n ...", n=self.batch_size)

        conditioning_image = self.sd_model.get_first_stage_encoding(self.sd_model.encode_first_stage(source_image))
        conditioning = torch.nn.functional.interpolate(
            self.sd_model.depth_model(midas_in),
            size=conditioning_image.shape[2:],
            mode="bicubic",
            align_corners=False,
        )

        (depth_min, depth_max) = torch.aminmax(conditioning)
        conditioning = 2. * (conditioning - depth_min) / (depth_max - depth_min) - 1.
        return conditioning

    def edit_image_conditioning(self, source_image):
        conditioning_image = self.sd_model.encode_first_stage(source_image).mode()

        return conditioning_image

    def unclip_image_conditioning(self, source_image):
        c_adm = self.sd_model.embedder(source_image)
        if self.sd_model.noise_augmentor is not None:
            noise_level = 0
            c_adm, noise_level_emb = self.sd_model.noise_augmentor(c_adm, noise_level=repeat(torch.tensor([noise_level]).to(c_adm.device), '1 -> b', b=c_adm.shape[0]))
            c_adm = torch.cat((c_adm, noise_level_emb), 1)
        return c_adm

    def inpainting_image_conditioning(self, source_image, latent_image, image_mask=None):
        self.is_using_inpainting_conditioning = True

        # Handle the different mask inputs
        if image_mask is not None:
            if torch.is_tensor(image_mask):
                conditioning_mask = image_mask
            else:
                conditioning_mask = np.array(image_mask.convert("L"))
                conditioning_mask = conditioning_mask.astype(np.float32) / 255.0
                conditioning_mask = torch.from_numpy(conditioning_mask[None, None])

                # Inpainting model uses a discretized mask as input, so we round to either 1.0 or 0.0
                conditioning_mask = torch.round(conditioning_mask)
        else:
            conditioning_mask = source_image.new_ones(1, 1, *source_image.shape[-2:])

        # Create another latent image, this time with a masked version of the original input.
        # Smoothly interpolate between the masked and unmasked latent conditioning image using a parameter.
        conditioning_mask = conditioning_mask.to(device=source_image.device, dtype=source_image.dtype)
        conditioning_image = torch.lerp(
            source_image,
            source_image * (1.0 - conditioning_mask),
            getattr(self, "inpainting_mask_weight", shared.opts.inpainting_mask_weight)
        )

        # Encode the new masked image using first stage of network.
        conditioning_image = self.sd_model.get_first_stage_encoding(self.sd_model.encode_first_stage(conditioning_image))

        # Create the concatenated conditioning tensor to be fed to `c_concat`
        conditioning_mask = torch.nn.functional.interpolate(conditioning_mask, size=latent_image.shape[-2:])
        conditioning_mask = conditioning_mask.expand(conditioning_image.shape[0], -1, -1, -1)
        image_conditioning = torch.cat([conditioning_mask, conditioning_image], dim=1)
        image_conditioning = image_conditioning.to(shared.device).type(self.sd_model.dtype)

        return image_conditioning

    def img2img_image_conditioning(self, source_image, latent_image, image_mask=None):
        source_image = devices.cond_cast_float(source_image)

        # HACK: Using introspection as the Depth2Image model doesn't appear to uniquely
        # identify itself with a field common to all models. The conditioning_key is also hybrid.
        if isinstance(self.sd_model, LatentDepth2ImageDiffusion):
            return self.depth2img_image_conditioning(source_image)

        if self.sd_model.cond_stage_key == "edit":
            return self.edit_image_conditioning(source_image)

        if self.sampler.conditioning_key in {'hybrid', 'concat'}:
            return self.inpainting_image_conditioning(source_image, latent_image, image_mask=image_mask)

        if self.sampler.conditioning_key == "crossattn-adm":
            return self.unclip_image_conditioning(source_image)

        # Dummy zero conditioning if we're not using inpainting or depth model.
        return latent_image.new_zeros(latent_image.shape[0], 5, 1, 1)

    def init(self, all_prompts, all_seeds, all_subseeds):
        pass

    def sample(self, conditioning, unconditional_conditioning, seeds, subseeds, subseed_strength, prompts):
        raise NotImplementedError()

    def close(self):
        self.sampler = None # pylint: disable=attribute-defined-outside-init


class Processed:
    def __init__(self, p: StableDiffusionProcessing, images_list, seed=-1, info="", subseed=None, all_prompts=None, all_negative_prompts=None, all_seeds=None, all_subseeds=None, index_of_first_image=0, infotexts=None, comments=""):
        self.images = images_list
        self.prompt = p.prompt
        self.negative_prompt = p.negative_prompt
        self.seed = seed
        self.subseed = subseed
        self.subseed_strength = p.subseed_strength
        self.info = info
        self.comments = comments
        self.width = p.width
        self.height = p.height
        self.sampler_name = p.sampler_name
        self.cfg_scale = p.cfg_scale
        self.image_cfg_scale = getattr(p, 'image_cfg_scale', None)
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
        self.styles = p.styles
        self.job_timestamp = state.job_timestamp
        self.clip_skip = opts.CLIP_stop_at_last_layers

        self.eta = p.eta
        self.ddim_discretize = p.ddim_discretize
        self.s_churn = p.s_churn
        self.s_tmin = p.s_tmin
        self.s_tmax = p.s_tmax
        self.s_noise = p.s_noise
        self.sampler_noise_scheduler_override = p.sampler_noise_scheduler_override
        self.prompt = self.prompt if type(self.prompt) != list else self.prompt[0]
        self.negative_prompt = self.negative_prompt if type(self.negative_prompt) != list else self.negative_prompt[0]
        self.seed = int(self.seed if type(self.seed) != list else self.seed[0]) if self.seed is not None else -1
        self.subseed = int(self.subseed if type(self.subseed) != list else self.subseed[0]) if self.subseed is not None else -1
        self.is_using_inpainting_conditioning = p.is_using_inpainting_conditioning

        self.all_prompts = all_prompts or p.all_prompts or [self.prompt]
        self.all_negative_prompts = all_negative_prompts or p.all_negative_prompts or [self.negative_prompt]
        self.all_seeds = all_seeds or p.all_seeds or [self.seed]
        self.all_subseeds = all_subseeds or p.all_subseeds or [self.subseed]
        self.infotexts = infotexts or [info]

    def js(self):
        obj = {
            "prompt": self.all_prompts[0],
            "all_prompts": self.all_prompts,
            "negative_prompt": self.all_negative_prompts[0],
            "all_negative_prompts": self.all_negative_prompts,
            "seed": self.seed,
            "all_seeds": self.all_seeds,
            "subseed": self.subseed,
            "all_subseeds": self.all_subseeds,
            "subseed_strength": self.subseed_strength,
            "width": self.width,
            "height": self.height,
            "sampler_name": self.sampler_name,
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
            "infotexts": self.infotexts,
            "styles": self.styles,
            "job_timestamp": self.job_timestamp,
            "clip_skip": self.clip_skip,
            "is_using_inpainting_conditioning": self.is_using_inpainting_conditioning,
        }

        return json.dumps(obj)

    def infotext(self, p: StableDiffusionProcessing, index):
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
    eta_noise_seed_delta = opts.eta_noise_seed_delta or 0
    xs = []

    # if we have multiple seeds, this means we are working with batch size>1; this then
    # enables the generation of additional tensors with noise that the sampler will use during its processing.
    # Using those pre-generated tensors instead of simple torch.randn allows a batch with seeds [100, 101] to
    # produce the same images as with two batches [100], [101].
    if p is not None and p.sampler is not None and (len(seeds) > 1 and opts.enable_batch_seeds or eta_noise_seed_delta > 0):
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

            if eta_noise_seed_delta > 0:
                torch.manual_seed(seed + eta_noise_seed_delta)

            for j in range(cnt):
                sampler_noises[j].append(devices.randn_without_seed(tuple(noise_shape)))

        xs.append(noise)

    if sampler_noises is not None:
        p.sampler.sampler_noises = [torch.stack(n).to(shared.device) for n in sampler_noises]

    x = torch.stack(xs).to(shared.device)
    return x


def decode_first_stage(model, x):
    with devices.autocast(disable = x.dtype==devices.dtype_vae):
        x = model.decode_first_stage(x)
    return x


def get_fixed_seed(seed):
    if seed is None or seed == '' or seed == -1:
        return int(random.randrange(4294967294))

    return seed


def fix_seed(p):
    p.seed = get_fixed_seed(p.seed)
    p.subseed = get_fixed_seed(p.subseed)


def create_infotext(p, all_prompts, all_seeds, all_subseeds, comments=None, iteration=0, position_in_batch=0): # pylint: disable=unused-argument
    index = position_in_batch + iteration * p.batch_size

    clip_skip = getattr(p, 'clip_skip', opts.CLIP_stop_at_last_layers)

    generation_params = {
        "Steps": p.steps,
        "Sampler": p.sampler_name,
        "CFG scale": p.cfg_scale,
        "Image CFG scale": getattr(p, 'image_cfg_scale', None),
        "Seed": all_seeds[index],
        "Face restoration": (opts.face_restoration_model if p.restore_faces else None),
        "Size": f"{p.width}x{p.height}",
        "Model hash": getattr(p, 'sd_model_hash', None if not opts.add_model_hash_to_info or not shared.sd_model.sd_model_hash else shared.sd_model.sd_model_hash),
        "Model": (None if not opts.add_model_name_to_info or not shared.sd_model.sd_checkpoint_info.model_name else shared.sd_model.sd_checkpoint_info.model_name.replace(',', '').replace(':', '')),
        "VAE": (None if not opts.add_model_name_to_info or sd_vae.loaded_vae_file is None else os.path.splitext(os.path.basename(sd_vae.loaded_vae_file))[0]),
        "Variation seed": (None if p.subseed_strength == 0 else all_subseeds[index]),
        "Variation seed strength": (None if p.subseed_strength == 0 else p.subseed_strength),
        "Seed resize from": (None if p.seed_resize_from_w == 0 or p.seed_resize_from_h == 0 else f"{p.seed_resize_from_w}x{p.seed_resize_from_h}"),
        "Denoising strength": getattr(p, 'denoising_strength', None),
        "Conditional mask weight": getattr(p, "inpainting_mask_weight", shared.opts.inpainting_mask_weight) if p.is_using_inpainting_conditioning else None,
        "Clip skip": None if clip_skip <= 1 else clip_skip,
        "ENSD": None if opts.eta_noise_seed_delta == 0 else opts.eta_noise_seed_delta,
        "Token merging ratio": None if not (opts.token_merging or cmd_opts.token_merging) or opts.token_merging_hr_only else opts.token_merging_ratio,
        "Token merging ratio hr": None if not (opts.token_merging or cmd_opts.token_merging) else opts.token_merging_ratio_hr,
        "Token merging random": None if opts.token_merging_random is False else opts.token_merging_random,
        "Token merging merge attention": None if opts.token_merging_merge_attention is True else opts.token_merging_merge_attention,
        "Token merging merge cross attention": None if opts.token_merging_merge_cross_attention is False else opts.token_merging_merge_cross_attention,
        "Token merging merge mlp": None if opts.token_merging_merge_mlp is False else opts.token_merging_merge_mlp,
        "Token merging stride x": None if opts.token_merging_stride_x == 2 else opts.token_merging_stride_x,
        "Token merging stride y": None if opts.token_merging_stride_y == 2 else opts.token_merging_stride_y
    }
    generation_params.update(p.extra_generation_params)

    generation_params_text = ", ".join([k if k == v else f'{k}: {generation_parameters_copypaste.quote(v)}' for k, v in generation_params.items() if v is not None])

    negative_prompt_text = "\nNegative prompt: " + p.all_negative_prompts[index] if p.all_negative_prompts[index] else ""

    return f"{all_prompts[index]}{negative_prompt_text}\n{generation_params_text}".strip()


def process_images(p: StableDiffusionProcessing) -> Processed:
    stored_opts = {k: opts.data[k] for k in p.override_settings.keys()}

    try:
        for k, v in p.override_settings.items():
            setattr(opts, k, v)

            if k == 'sd_model_checkpoint':
                sd_models.reload_model_weights()

            if k == 'sd_vae':
                sd_vae.reload_vae_weights()

        """
        import torch.profiler
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], record_shapes=True, with_modules=True) as prof:
            with torch.profiler.record_function("process_images"):
                res = process_images_inner(p)
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
        """

        if (opts.token_merging or cmd_opts.token_merging) and not opts.token_merging_hr_only:
            sd_models.apply_token_merging(sd_model=p.sd_model, hr=False)
            logger.debug('Token merging applied')

        res = process_images_inner(p)

    finally:
        # undo model optimizations made by tomesd
        if opts.token_merging or cmd_opts.token_merging:
            tomesd.remove_patch(p.sd_model)
            logger.debug('Token merging model optimizations removed')

        # restore opts to original state
        if p.override_settings_restore_afterwards:
            for k, v in stored_opts.items():
                setattr(opts, k, v)
                if k == 'sd_model_checkpoint':
                    sd_models.reload_model_weights()

                if k == 'sd_vae':
                    sd_vae.reload_vae_weights()

    return res


def process_images_inner(p: StableDiffusionProcessing) -> Processed:
    """this is the main loop that both txt2img and img2img use; it calls func_init once inside all the scopes and func_sample once per batch"""

    if type(p.prompt) == list:
        assert len(p.prompt) > 0
    else:
        assert p.prompt is not None

    devices.torch_gc()

    seed = get_fixed_seed(p.seed)
    subseed = get_fixed_seed(p.subseed)

    modules.sd_hijack.model_hijack.apply_circular(p.tiling)
    modules.sd_hijack.model_hijack.clear_comments()

    comments = {}

    if type(p.prompt) == list:
        p.all_prompts = [shared.prompt_styles.apply_styles_to_prompt(x, p.styles) for x in p.prompt]
    else:
        p.all_prompts = p.batch_size * p.n_iter * [shared.prompt_styles.apply_styles_to_prompt(p.prompt, p.styles)]

    if type(p.negative_prompt) == list:
        p.all_negative_prompts = [shared.prompt_styles.apply_negative_styles_to_prompt(x, p.styles) for x in p.negative_prompt]
    else:
        p.all_negative_prompts = p.batch_size * p.n_iter * [shared.prompt_styles.apply_negative_styles_to_prompt(p.negative_prompt, p.styles)]

    if type(seed) == list:
        p.all_seeds = seed
    else:
        p.all_seeds = [int(seed) + (x if p.subseed_strength == 0 else 0) for x in range(len(p.all_prompts))]

    if type(subseed) == list:
        p.all_subseeds = subseed
    else:
        p.all_subseeds = [int(subseed) + x for x in range(len(p.all_prompts))]

    def infotext(iteration=0, position_in_batch=0):
        return create_infotext(p, p.all_prompts, p.all_seeds, p.all_subseeds, comments, iteration, position_in_batch)

    if os.path.exists(opts.embeddings_dir) and not p.do_not_reload_embeddings:
        model_hijack.embedding_db.load_textual_inversion_embeddings()

    if p.scripts is not None:
        p.scripts.process(p)

    infotexts = []
    output_images = []

    cached_uc = [None, None]
    cached_c = [None, None]

    def get_conds_with_caching(function, required_prompts, steps, cache):
        """
        Returns the result of calling function(shared.sd_model, required_prompts, steps)
        using a cache to store the result if the same arguments have been used before.

        cache is an array containing two elements. The first element is a tuple
        representing the previously used arguments, or None if no arguments
        have been used before. The second element is where the previously
        computed result is stored.
        """

        if cache[0] is not None and (required_prompts, steps) == cache[0]:
            return cache[1]

        with devices.autocast():
            cache[1] = function(shared.sd_model, required_prompts, steps)

        cache[0] = (required_prompts, steps)
        return cache[1]

    with torch.no_grad(), p.sd_model.ema_scope():
        with devices.autocast():
            p.init(p.all_prompts, p.all_seeds, p.all_subseeds)

            # for OSX, loading the model during sampling changes the generated picture, so it is loaded here
            if shared.opts.live_previews_enable and opts.show_progress_type == "Approx NN":
                sd_vae_approx.model()

        if state.job_count == -1:
            state.job_count = p.n_iter

        extra_network_data = None
        for n in range(p.n_iter):
            p.iteration = n

            if state.skipped:
                state.skipped = False

            if state.interrupted:
                break

            prompts = p.all_prompts[n * p.batch_size:(n + 1) * p.batch_size]
            negative_prompts = p.all_negative_prompts[n * p.batch_size:(n + 1) * p.batch_size]
            seeds = p.all_seeds[n * p.batch_size:(n + 1) * p.batch_size]
            subseeds = p.all_subseeds[n * p.batch_size:(n + 1) * p.batch_size]

            if p.scripts is not None:
                p.scripts.before_process_batch(p, batch_number=n, prompts=prompts, seeds=seeds, subseeds=subseeds)

            if len(prompts) == 0:
                break

            prompts, extra_network_data = extra_networks.parse_prompts(prompts)

            if not p.disable_extra_networks:
                with devices.autocast():
                    extra_networks.activate(p, extra_network_data)

            if p.scripts is not None:
                p.scripts.process_batch(p, batch_number=n, prompts=prompts, seeds=seeds, subseeds=subseeds)

            # params.txt should be saved after scripts.process_batch, since the
            # infotext could be modified by that callback
            # Example: a wildcard processed by process_batch sets an extra model
            # strength, which is saved as "Model Strength: 1.0" in the infotext
            if n == 0:
                with open(os.path.join(paths.data_path, "params.txt"), "w", encoding="utf8") as file:
                    processed = Processed(p, [], p.seed, "")
                    file.write(processed.infotext(p, 0))

            step_multiplier = 1
            if not shared.opts.dont_fix_second_order_samplers_schedule:
                try:
                    step_multiplier = 2 if sd_samplers.all_samplers_map.get(p.sampler_name).aliases[0] in ['k_dpmpp_2s_a', 'k_dpmpp_2s_a_ka', 'k_dpmpp_sde', 'k_dpmpp_sde_ka', 'k_dpm_2', 'k_dpm_2_a', 'k_heun'] else 1
                except:
                    pass
            uc = get_conds_with_caching(prompt_parser.get_learned_conditioning, negative_prompts, p.steps * step_multiplier, cached_uc)
            c = get_conds_with_caching(prompt_parser.get_multicond_learned_conditioning, prompts, p.steps * step_multiplier, cached_c)

            if len(model_hijack.comments) > 0:
                for comment in model_hijack.comments:
                    comments[comment] = 1

            if p.n_iter > 1:
                shared.state.job = f"Batch {n+1} out of {p.n_iter}"

            with devices.without_autocast() if devices.unet_needs_upcast else devices.autocast():
                samples_ddim = p.sample(conditioning=c, unconditional_conditioning=uc, seeds=seeds, subseeds=subseeds, subseed_strength=p.subseed_strength, prompts=prompts)

            x_samples_ddim = [decode_first_stage(p.sd_model, samples_ddim[i:i+1].to(dtype=devices.dtype_vae))[0].cpu() for i in range(samples_ddim.size(0))]
            try:
                for x in x_samples_ddim:
                    devices.test_for_nans(x, "vae")
            except devices.NansException as e:
                if not shared.cmd_opts.no_half and not shared.cmd_opts.no_half_vae and shared.cmd_opts.rollback_vae:
                    print('\nA tensor with all NaNs was produced in VAE, try converting to bf16.')
                    devices.dtype_vae = torch.bfloat16
                    vae_file, vae_source = sd_vae.resolve_vae(p.sd_model.sd_model_checkpoint)
                    sd_vae.load_vae(p.sd_model, vae_file, vae_source)
                    x_samples_ddim = [decode_first_stage(p.sd_model, samples_ddim[i:i+1].to(dtype=devices.dtype_vae))[0].cpu() for i in range(samples_ddim.size(0))]
                    for x in x_samples_ddim:
                        devices.test_for_nans(x, "vae")
                else:
                    raise e

            x_samples_ddim = torch.stack(x_samples_ddim).float()
            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

            del samples_ddim

            if shared.cmd_opts.lowvram or shared.cmd_opts.medvram:
                lowvram.send_everything_to_cpu()

            devices.torch_gc()

            if p.scripts is not None:
                p.scripts.postprocess_batch(p, x_samples_ddim, batch_number=n)

            for i, x_sample in enumerate(x_samples_ddim):
                x_sample = 255. * np.moveaxis(x_sample.cpu().numpy(), 0, 2)
                x_sample = x_sample.astype(np.uint8)

                if p.restore_faces:
                    if opts.save and not p.do_not_save_samples and opts.save_images_before_face_restoration:
                        images.save_image(Image.fromarray(x_sample), p.outpath_samples, "", seeds[i], prompts[i], opts.samples_format, info=infotext(n, i), p=p, suffix="-before-face-restoration")

                    devices.torch_gc()

                    x_sample = modules.face_restoration.restore_faces(x_sample)
                    devices.torch_gc()

                image = Image.fromarray(x_sample)

                if p.scripts is not None:
                    pp = scripts.PostprocessImageArgs(image)
                    p.scripts.postprocess_image(p, pp)
                    image = pp.image

                if p.color_corrections is not None and i < len(p.color_corrections):
                    if opts.save and not p.do_not_save_samples and opts.save_images_before_color_correction:
                        image_without_cc = apply_overlay(image, p.paste_to, i, p.overlay_images)
                        images.save_image(image_without_cc, p.outpath_samples, "", seeds[i], prompts[i], opts.samples_format, info=infotext(n, i), p=p, suffix="-before-color-correction")
                    image = apply_color_correction(p.color_corrections[i], image)

                image = apply_overlay(image, p.paste_to, i, p.overlay_images)

                if opts.samples_save and not p.do_not_save_samples:
                    images.save_image(image, p.outpath_samples, "", seeds[i], prompts[i], opts.samples_format, info=infotext(n, i), p=p)

                text = infotext(n, i)
                infotexts.append(text)
                if opts.enable_pnginfo:
                    image.info["parameters"] = text
                output_images.append(image)

                if hasattr(p, 'mask_for_overlay') and p.mask_for_overlay and any([opts.save_mask, opts.save_mask_composite, opts.return_mask, opts.return_mask_composite]):
                    image_mask = p.mask_for_overlay.convert('RGB')
                    image_mask_composite = Image.composite(image.convert('RGBA').convert('RGBa'), Image.new('RGBa', image.size), images.resize_image(2, p.mask_for_overlay, image.width, image.height).convert('L')).convert('RGBA')

                    if opts.save_mask:
                        images.save_image(image_mask, p.outpath_samples, "", seeds[i], prompts[i], opts.samples_format, info=infotext(n, i), p=p, suffix="-mask")

                    if opts.save_mask_composite:
                        images.save_image(image_mask_composite, p.outpath_samples, "", seeds[i], prompts[i], opts.samples_format, info=infotext(n, i), p=p, suffix="-mask-composite")

                    if opts.return_mask:
                        output_images.append(image_mask)

                    if opts.return_mask_composite:
                        output_images.append(image_mask_composite)

            del x_samples_ddim

            devices.torch_gc()

            state.nextjob()

        p.color_corrections = None

        index_of_first_image = 0
        unwanted_grid_because_of_img_count = len(output_images) < 2 and opts.grid_only_if_multiple
        if (opts.return_grid or opts.grid_save) and not p.do_not_save_grid and not unwanted_grid_because_of_img_count:
            grid = images.image_grid(output_images, p.batch_size)

            if opts.return_grid:
                text = infotext()
                infotexts.insert(0, text)
                if opts.enable_pnginfo:
                    grid.info["parameters"] = text
                output_images.insert(0, grid)
                index_of_first_image = 1

            if opts.grid_save:
                images.save_image(grid, p.outpath_grids, "grid", p.all_seeds[0], p.all_prompts[0], opts.grid_format, info=infotext(), short_filename=not opts.grid_extended_filename, p=p, grid=True)

    if not p.disable_extra_networks and extra_network_data:
        extra_networks.deactivate(p, extra_network_data)

    devices.torch_gc()

    res = Processed(p, output_images, p.all_seeds[0], infotext(), comments="".join(["\n\n" + x for x in comments]), subseed=p.all_subseeds[0], index_of_first_image=index_of_first_image, infotexts=infotexts)

    if p.scripts is not None:
        p.scripts.postprocess(p, res)

    return res


def old_hires_fix_first_pass_dimensions(width, height):
    """old algorithm for auto-calculating first pass size"""

    desired_pixel_count = 512 * 512
    actual_pixel_count = width * height
    scale = math.sqrt(desired_pixel_count / actual_pixel_count)
    width = math.ceil(scale * width / 64) * 64
    height = math.ceil(scale * height / 64) * 64

    return width, height


class StableDiffusionProcessingTxt2Img(StableDiffusionProcessing):
    sampler = None

    def __init__(self, enable_hr: bool = False, denoising_strength: float = 0.75, firstphase_width: int = 0, firstphase_height: int = 0, hr_scale: float = 2.0, hr_upscaler: str = None, hr_second_pass_steps: int = 0, hr_resize_x: int = 0, hr_resize_y: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.enable_hr = enable_hr
        self.denoising_strength = denoising_strength
        self.hr_scale = hr_scale
        self.hr_upscaler = hr_upscaler
        self.hr_second_pass_steps = hr_second_pass_steps
        self.hr_resize_x = hr_resize_x
        self.hr_resize_y = hr_resize_y
        self.hr_upscale_to_x = hr_resize_x
        self.hr_upscale_to_y = hr_resize_y

        if firstphase_width != 0 or firstphase_height != 0:
            self.hr_upscale_to_x = self.width
            self.hr_upscale_to_y = self.height
            self.width = firstphase_width
            self.height = firstphase_height

        self.truncate_x = 0
        self.truncate_y = 0
        self.applied_old_hires_behavior_to = None

    def init(self, all_prompts, all_seeds, all_subseeds):
        if self.enable_hr:
            if opts.use_old_hires_fix_width_height and self.applied_old_hires_behavior_to != (self.width, self.height):
                self.hr_resize_x = self.width
                self.hr_resize_y = self.height
                self.hr_upscale_to_x = self.width
                self.hr_upscale_to_y = self.height

                self.width, self.height = old_hires_fix_first_pass_dimensions(self.width, self.height)
                self.applied_old_hires_behavior_to = (self.width, self.height)

            if self.hr_resize_x == 0 and self.hr_resize_y == 0:
                self.extra_generation_params["Hires upscale"] = self.hr_scale
                self.hr_upscale_to_x = int(self.width * self.hr_scale)
                self.hr_upscale_to_y = int(self.height * self.hr_scale)
            else:
                self.extra_generation_params["Hires resize"] = f"{self.hr_resize_x}x{self.hr_resize_y}"

                if self.hr_resize_y == 0:
                    self.hr_upscale_to_x = self.hr_resize_x
                    self.hr_upscale_to_y = self.hr_resize_x * self.height // self.width
                elif self.hr_resize_x == 0:
                    self.hr_upscale_to_x = self.hr_resize_y * self.width // self.height
                    self.hr_upscale_to_y = self.hr_resize_y
                else:
                    target_w = self.hr_resize_x
                    target_h = self.hr_resize_y
                    src_ratio = self.width / self.height
                    dst_ratio = self.hr_resize_x / self.hr_resize_y

                    if src_ratio < dst_ratio:
                        self.hr_upscale_to_x = self.hr_resize_x
                        self.hr_upscale_to_y = self.hr_resize_x * self.height // self.width
                    else:
                        self.hr_upscale_to_x = self.hr_resize_y * self.width // self.height
                        self.hr_upscale_to_y = self.hr_resize_y

                    self.truncate_x = (self.hr_upscale_to_x - target_w) // opt_f
                    self.truncate_y = (self.hr_upscale_to_y - target_h) // opt_f

            # special case: the user has chosen to do nothing
            if self.hr_upscale_to_x == self.width and self.hr_upscale_to_y == self.height:
                self.enable_hr = False
                self.denoising_strength = None
                self.extra_generation_params.pop("Hires upscale", None)
                self.extra_generation_params.pop("Hires resize", None)
                return

            if not state.processing_has_refined_job_count:
                if state.job_count == -1:
                    state.job_count = self.n_iter

                shared.total_tqdm.updateTotal((self.steps + (self.hr_second_pass_steps or self.steps)) * state.job_count)
                state.job_count = state.job_count * 2
                state.processing_has_refined_job_count = True

            if self.hr_second_pass_steps:
                self.extra_generation_params["Hires steps"] = self.hr_second_pass_steps

            if self.hr_upscaler is not None:
                self.extra_generation_params["Hires upscaler"] = self.hr_upscaler

    def sample(self, conditioning, unconditional_conditioning, seeds, subseeds, subseed_strength, prompts):
        self.sampler = sd_samplers.create_sampler(self.sampler_name, self.sd_model)

        latent_scale_mode = shared.latent_upscale_modes.get(self.hr_upscaler, None) if self.hr_upscaler is not None else shared.latent_upscale_modes.get(shared.latent_upscale_default_mode, "nearest")
        if self.enable_hr and latent_scale_mode is None:
            assert len([x for x in shared.sd_upscalers if x.name == self.hr_upscaler]) > 0, f"could not find upscaler named {self.hr_upscaler}"

        x = create_random_tensors([opt_C, self.height // opt_f, self.width // opt_f], seeds=seeds, subseeds=subseeds, subseed_strength=self.subseed_strength, seed_resize_from_h=self.seed_resize_from_h, seed_resize_from_w=self.seed_resize_from_w, p=self)
        samples = self.sampler.sample(self, x, conditioning, unconditional_conditioning, image_conditioning=self.txt2img_image_conditioning(x))

        if not self.enable_hr:
            return samples

        target_width = self.hr_upscale_to_x
        target_height = self.hr_upscale_to_y

        def save_intermediate(image, index):
            """saves image before applying hires fix, if enabled in options; takes as an argument either an image or batch with latent space images"""

            if not opts.save or self.do_not_save_samples or not opts.save_images_before_highres_fix:
                return

            if not isinstance(image, Image.Image):
                image = sd_samplers.sample_to_image(image, index, approximation=0)

            info = create_infotext(self, self.all_prompts, self.all_seeds, self.all_subseeds, [], iteration=self.iteration, position_in_batch=index)
            images.save_image(image, self.outpath_samples, "", seeds[index], prompts[index], opts.samples_format, info=info, suffix="-before-highres-fix")

        if latent_scale_mode is not None:
            for i in range(samples.shape[0]):
                save_intermediate(samples, i)

            samples = torch.nn.functional.interpolate(samples, size=(target_height // opt_f, target_width // opt_f), mode=latent_scale_mode["mode"], antialias=latent_scale_mode["antialias"])

            # Avoid making the inpainting conditioning unless necessary as
            # this does need some extra compute to decode / encode the image again.
            if getattr(self, "inpainting_mask_weight", shared.opts.inpainting_mask_weight) < 1.0:
                image_conditioning = self.img2img_image_conditioning(decode_first_stage(self.sd_model, samples), samples)
            else:
                image_conditioning = self.txt2img_image_conditioning(samples)
        else:
            decoded_samples = decode_first_stage(self.sd_model, samples)
            lowres_samples = torch.clamp((decoded_samples + 1.0) / 2.0, min=0.0, max=1.0)

            batch_images = []
            for i, x_sample in enumerate(lowres_samples):
                x_sample = 255. * np.moveaxis(x_sample.cpu().numpy(), 0, 2)
                x_sample = x_sample.astype(np.uint8)
                image = Image.fromarray(x_sample)

                save_intermediate(image, i)

                image = images.resize_image(0, image, target_width, target_height, upscaler_name=self.hr_upscaler)
                image = np.array(image).astype(np.float32) / 255.0
                image = np.moveaxis(image, 2, 0)
                batch_images.append(image)

            decoded_samples = torch.from_numpy(np.array(batch_images))
            decoded_samples = decoded_samples.to(shared.device)
            decoded_samples = 2. * decoded_samples - 1.

            samples = self.sd_model.get_first_stage_encoding(self.sd_model.encode_first_stage(decoded_samples))

            image_conditioning = self.img2img_image_conditioning(decoded_samples, samples)

        shared.state.nextjob()

        img2img_sampler_name = self.sampler_name
        if self.sampler_name in ['PLMS', 'UniPC']:  # PLMS/UniPC do not support img2img so we just silently switch to DDIM
            img2img_sampler_name = 'DDIM'
        self.sampler = sd_samplers.create_sampler(img2img_sampler_name, self.sd_model)

        samples = samples[:, :, self.truncate_y//2:samples.shape[2]-(self.truncate_y+1)//2, self.truncate_x//2:samples.shape[3]-(self.truncate_x+1)//2]

        noise = create_random_tensors(samples.shape[1:], seeds=seeds, subseeds=subseeds, subseed_strength=subseed_strength, p=self)

        # GC now before running the next img2img to prevent running out of memory
        x = None
        devices.torch_gc()

        # apply token merging optimizations from tomesd for high-res pass
        # check if hr_only so we are not redundantly patching
        if (cmd_opts.token_merging or opts.token_merging) and (opts.token_merging_hr_only or opts.token_merging_ratio_hr != opts.token_merging_ratio):
            # case where user wants to use separate merge ratios
            if not opts.token_merging_hr_only:
                # clean patch done by first pass. (clobbering the first patch might be fine? this might be excessive)
                tomesd.remove_patch(self.sd_model)
                logger.debug('Temporarily removed token merging optimizations in preparation for next pass')

            sd_models.apply_token_merging(sd_model=self.sd_model, hr=True)
            logger.debug('Applied token merging for high-res pass')

        samples = self.sampler.sample_img2img(self, samples, noise, conditioning, unconditional_conditioning, steps=self.hr_second_pass_steps or self.steps, image_conditioning=image_conditioning)

        return samples


class StableDiffusionProcessingImg2Img(StableDiffusionProcessing):
    sampler = None

    def __init__(self, init_images: list = None, resize_mode: int = 0, denoising_strength: float = 0.75, image_cfg_scale: float = None, mask: Any = None, mask_blur: int = 4, inpainting_fill: int = 0, inpaint_full_res: bool = True, inpaint_full_res_padding: int = 0, inpainting_mask_invert: int = 0, initial_noise_multiplier: float = None, **kwargs):
        super().__init__(**kwargs)

        self.init_images = init_images
        self.resize_mode: int = resize_mode
        self.denoising_strength: float = denoising_strength
        self.image_cfg_scale: float = image_cfg_scale if shared.sd_model.cond_stage_key == "edit" else None
        self.init_latent = None
        self.image_mask = mask
        self.latent_mask = None
        self.mask_for_overlay = None
        self.mask_blur = mask_blur
        self.inpainting_fill = inpainting_fill
        self.inpaint_full_res = inpaint_full_res
        self.inpaint_full_res_padding = inpaint_full_res_padding
        self.inpainting_mask_invert = inpainting_mask_invert
        self.initial_noise_multiplier = opts.initial_noise_multiplier if initial_noise_multiplier is None else initial_noise_multiplier
        self.mask = None
        self.nmask = None
        self.image_conditioning = None

    def init(self, all_prompts, all_seeds, all_subseeds):
        self.sampler = sd_samplers.create_sampler(self.sampler_name, self.sd_model)
        crop_region = None

        image_mask = self.image_mask

        if image_mask is not None:
            image_mask = image_mask.convert('L')

            if self.inpainting_mask_invert:
                image_mask = ImageOps.invert(image_mask)

            if self.mask_blur > 0:
                image_mask = image_mask.filter(ImageFilter.GaussianBlur(self.mask_blur))

            if self.inpaint_full_res:
                self.mask_for_overlay = image_mask
                mask = image_mask.convert('L')
                crop_region = masking.get_crop_region(np.array(mask), self.inpaint_full_res_padding)
                crop_region = masking.expand_crop_region(crop_region, self.width, self.height, mask.width, mask.height)
                x1, y1, x2, y2 = crop_region

                mask = mask.crop(crop_region)
                image_mask = images.resize_image(2, mask, self.width, self.height)
                self.paste_to = (x1, y1, x2-x1, y2-y1)
            else:
                image_mask = images.resize_image(self.resize_mode, image_mask, self.width, self.height)
                np_mask = np.array(image_mask)
                np_mask = np.clip((np_mask.astype(np.float32)) * 2, 0, 255).astype(np.uint8)
                self.mask_for_overlay = Image.fromarray(np_mask)

            self.overlay_images = []

        latent_mask = self.latent_mask if self.latent_mask is not None else image_mask

        add_color_corrections = opts.img2img_color_correction and self.color_corrections is None
        if add_color_corrections:
            self.color_corrections = []
        imgs = []
        for img in self.init_images:
            image = images.flatten(img, opts.img2img_background_color)

            if crop_region is None and self.resize_mode != 3:
                image = images.resize_image(self.resize_mode, image, self.width, self.height)

            if image_mask is not None:
                image_masked = Image.new('RGBa', (image.width, image.height))
                image_masked.paste(image.convert("RGBA").convert("RGBa"), mask=ImageOps.invert(self.mask_for_overlay.convert('L')))

                self.overlay_images.append(image_masked.convert('RGBA'))

            # crop_region is not None if we are doing inpaint full res
            if crop_region is not None:
                image = image.crop(crop_region)
                image = images.resize_image(2, image, self.width, self.height)

            if image_mask is not None:
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

            if self.color_corrections is not None and len(self.color_corrections) == 1:
                self.color_corrections = self.color_corrections * self.batch_size

        elif len(imgs) <= self.batch_size:
            self.batch_size = len(imgs)
            batch_images = np.array(imgs)
        else:
            raise RuntimeError(f"bad number of images passed: {len(imgs)}; expecting {self.batch_size} or less")

        image = torch.from_numpy(batch_images)
        image = 2. * image - 1.
        image = image.to(shared.device)

        self.init_latent = self.sd_model.get_first_stage_encoding(self.sd_model.encode_first_stage(image))

        if self.resize_mode == 3:
            self.init_latent = torch.nn.functional.interpolate(self.init_latent, size=(self.height // opt_f, self.width // opt_f), mode="bilinear")

        if image_mask is not None:
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

        self.image_conditioning = self.img2img_image_conditioning(image, self.init_latent, image_mask)

    def sample(self, conditioning, unconditional_conditioning, seeds, subseeds, subseed_strength, prompts):
        x = create_random_tensors([opt_C, self.height // opt_f, self.width // opt_f], seeds=seeds, subseeds=subseeds, subseed_strength=self.subseed_strength, seed_resize_from_h=self.seed_resize_from_h, seed_resize_from_w=self.seed_resize_from_w, p=self)

        if self.initial_noise_multiplier != 1.0:
            self.extra_generation_params["Noise multiplier"] = self.initial_noise_multiplier
            x *= self.initial_noise_multiplier

        samples = self.sampler.sample_img2img(self, self.init_latent, x, conditioning, unconditional_conditioning, image_conditioning=self.image_conditioning)

        if self.mask is not None:
            samples = samples * self.nmask + self.init_latent * self.mask

        del x
        devices.torch_gc()

        return samples
