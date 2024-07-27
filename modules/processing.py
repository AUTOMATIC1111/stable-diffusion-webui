from __future__ import annotations
import json
import logging
import math
import os
import sys
import hashlib
from dataclasses import dataclass, field

import torch
import numpy as np
from PIL import Image, ImageOps
import random
import cv2
from skimage import exposure
from typing import Any

import modules.sd_hijack
from modules import devices, prompt_parser, masking, sd_samplers, lowvram, infotext_utils, extra_networks, sd_vae_approx, scripts, sd_samplers_common, sd_unet, errors, rng, profiling
from modules.rng import slerp # noqa: F401
from modules.sd_hijack import model_hijack
from modules.sd_samplers_common import images_tensor_to_samples, decode_first_stage, approximation_indexes
from modules.shared import opts, cmd_opts, state
import modules.shared as shared
import modules.paths as paths
import modules.face_restoration
import modules.images as images
import modules.styles
import modules.sd_models as sd_models
import modules.sd_vae as sd_vae
from ldm.data.util import AddMiDaS
from ldm.models.diffusion.ddpm import LatentDepth2ImageDiffusion

from einops import repeat, rearrange
from blendmodes.blend import blendLayers, BlendType


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

    return image.convert('RGB')


def uncrop(image, dest_size, paste_loc):
    x, y, w, h = paste_loc
    base_image = Image.new('RGBA', dest_size)
    image = images.resize_image(1, image, w, h)
    base_image.paste(image, (x, y))
    image = base_image

    return image


def apply_overlay(image, paste_loc, overlay):
    if overlay is None:
        return image, image.copy()

    if paste_loc is not None:
        image = uncrop(image, (overlay.width, overlay.height), paste_loc)

    original_denoised_image = image.copy()

    image = image.convert('RGBA')
    image.alpha_composite(overlay)
    image = image.convert('RGB')

    return image, original_denoised_image

def create_binary_mask(image, round=True):
    if image.mode == 'RGBA' and image.getextrema()[-1] != (255, 255):
        if round:
            image = image.split()[-1].convert("L").point(lambda x: 255 if x > 128 else 0)
        else:
            image = image.split()[-1].convert("L")
    else:
        image = image.convert('L')
    return image

def txt2img_image_conditioning(sd_model, x, width, height):
    if sd_model.model.conditioning_key in {'hybrid', 'concat'}: # Inpainting models

        # The "masked-image" in this case will just be all 0.5 since the entire image is masked.
        image_conditioning = torch.ones(x.shape[0], 3, height, width, device=x.device) * 0.5
        image_conditioning = images_tensor_to_samples(image_conditioning, approximation_indexes.get(opts.sd_vae_encode_method))

        # Add the fake full 1s mask to the first dimension.
        image_conditioning = torch.nn.functional.pad(image_conditioning, (0, 0, 0, 0, 1, 0), value=1.0)
        image_conditioning = image_conditioning.to(x.dtype)

        return image_conditioning

    elif sd_model.model.conditioning_key == "crossattn-adm": # UnCLIP models

        return x.new_zeros(x.shape[0], 2*sd_model.noise_augmentor.time_embed.dim, dtype=x.dtype, device=x.device)

    else:
        if sd_model.is_sdxl_inpaint:
            # The "masked-image" in this case will just be all 0.5 since the entire image is masked.
            image_conditioning = torch.ones(x.shape[0], 3, height, width, device=x.device) * 0.5
            image_conditioning = images_tensor_to_samples(image_conditioning,
                                                            approximation_indexes.get(opts.sd_vae_encode_method))

            # Add the fake full 1s mask to the first dimension.
            image_conditioning = torch.nn.functional.pad(image_conditioning, (0, 0, 0, 0, 1, 0), value=1.0)
            image_conditioning = image_conditioning.to(x.dtype)

            return image_conditioning

        # Dummy zero conditioning if we're not using inpainting or unclip models.
        # Still takes up a bit of memory, but no encoder call.
        # Pretty sure we can just make this a 1x1 image since its not going to be used besides its batch size.
        return x.new_zeros(x.shape[0], 5, 1, 1, dtype=x.dtype, device=x.device)


@dataclass(repr=False)
class StableDiffusionProcessing:
    sd_model: object = None
    outpath_samples: str = None
    outpath_grids: str = None
    prompt: str = ""
    prompt_for_display: str = None
    negative_prompt: str = ""
    styles: list[str] = None
    seed: int = -1
    subseed: int = -1
    subseed_strength: float = 0
    seed_resize_from_h: int = -1
    seed_resize_from_w: int = -1
    seed_enable_extras: bool = True
    sampler_name: str = None
    scheduler: str = None
    batch_size: int = 1
    n_iter: int = 1
    steps: int = 50
    cfg_scale: float = 7.0
    width: int = 512
    height: int = 512
    restore_faces: bool = None
    tiling: bool = None
    do_not_save_samples: bool = False
    do_not_save_grid: bool = False
    extra_generation_params: dict[str, Any] = None
    overlay_images: list = None
    eta: float = None
    do_not_reload_embeddings: bool = False
    denoising_strength: float = None
    ddim_discretize: str = None
    s_min_uncond: float = None
    s_churn: float = None
    s_tmax: float = None
    s_tmin: float = None
    s_noise: float = None
    override_settings: dict[str, Any] = None
    override_settings_restore_afterwards: bool = True
    sampler_index: int = None
    refiner_checkpoint: str = None
    refiner_switch_at: float = None
    token_merging_ratio = 0
    token_merging_ratio_hr = 0
    disable_extra_networks: bool = False
    firstpass_image: Image = None

    scripts_value: scripts.ScriptRunner = field(default=None, init=False)
    script_args_value: list = field(default=None, init=False)
    scripts_setup_complete: bool = field(default=False, init=False)

    cached_uc = [None, None]
    cached_c = [None, None]

    comments: dict = None
    sampler: sd_samplers_common.Sampler | None = field(default=None, init=False)
    is_using_inpainting_conditioning: bool = field(default=False, init=False)
    paste_to: tuple | None = field(default=None, init=False)

    is_hr_pass: bool = field(default=False, init=False)

    c: tuple = field(default=None, init=False)
    uc: tuple = field(default=None, init=False)

    rng: rng.ImageRNG | None = field(default=None, init=False)
    step_multiplier: int = field(default=1, init=False)
    color_corrections: list = field(default=None, init=False)

    all_prompts: list = field(default=None, init=False)
    all_negative_prompts: list = field(default=None, init=False)
    all_seeds: list = field(default=None, init=False)
    all_subseeds: list = field(default=None, init=False)
    iteration: int = field(default=0, init=False)
    main_prompt: str = field(default=None, init=False)
    main_negative_prompt: str = field(default=None, init=False)

    prompts: list = field(default=None, init=False)
    negative_prompts: list = field(default=None, init=False)
    seeds: list = field(default=None, init=False)
    subseeds: list = field(default=None, init=False)
    extra_network_data: dict = field(default=None, init=False)

    user: str = field(default=None, init=False)

    sd_model_name: str = field(default=None, init=False)
    sd_model_hash: str = field(default=None, init=False)
    sd_vae_name: str = field(default=None, init=False)
    sd_vae_hash: str = field(default=None, init=False)

    is_api: bool = field(default=False, init=False)

    def __post_init__(self):
        if self.sampler_index is not None:
            print("sampler_index argument for StableDiffusionProcessing does not do anything; use sampler_name", file=sys.stderr)

        self.comments = {}

        if self.styles is None:
            self.styles = []

        self.sampler_noise_scheduler_override = None

        self.extra_generation_params = self.extra_generation_params or {}
        self.override_settings = self.override_settings or {}
        self.script_args = self.script_args or {}

        self.refiner_checkpoint_info = None

        if not self.seed_enable_extras:
            self.subseed = -1
            self.subseed_strength = 0
            self.seed_resize_from_h = 0
            self.seed_resize_from_w = 0

        self.cached_uc = StableDiffusionProcessing.cached_uc
        self.cached_c = StableDiffusionProcessing.cached_c

    def fill_fields_from_opts(self):
        self.s_min_uncond = self.s_min_uncond if self.s_min_uncond is not None else opts.s_min_uncond
        self.s_churn = self.s_churn if self.s_churn is not None else opts.s_churn
        self.s_tmin = self.s_tmin if self.s_tmin is not None else opts.s_tmin
        self.s_tmax = (self.s_tmax if self.s_tmax is not None else opts.s_tmax) or float('inf')
        self.s_noise = self.s_noise if self.s_noise is not None else opts.s_noise

    @property
    def sd_model(self):
        return shared.sd_model

    @sd_model.setter
    def sd_model(self, value):
        pass

    @property
    def scripts(self):
        return self.scripts_value

    @scripts.setter
    def scripts(self, value):
        self.scripts_value = value

        if self.scripts_value and self.script_args_value and not self.scripts_setup_complete:
            self.setup_scripts()

    @property
    def script_args(self):
        return self.script_args_value

    @script_args.setter
    def script_args(self, value):
        self.script_args_value = value

        if self.scripts_value and self.script_args_value and not self.scripts_setup_complete:
            self.setup_scripts()

    def setup_scripts(self):
        self.scripts_setup_complete = True

        self.scripts.setup_scrips(self, is_ui=not self.is_api)

    def comment(self, text):
        self.comments[text] = 1

    def txt2img_image_conditioning(self, x, width=None, height=None):
        self.is_using_inpainting_conditioning = self.sd_model.model.conditioning_key in {'hybrid', 'concat'}

        return txt2img_image_conditioning(self.sd_model, x, width or self.width, height or self.height)

    def depth2img_image_conditioning(self, source_image):
        # Use the AddMiDaS helper to Format our source image to suit the MiDaS model
        transformer = AddMiDaS(model_type="dpt_hybrid")
        transformed = transformer({"jpg": rearrange(source_image[0], "c h w -> h w c")})
        midas_in = torch.from_numpy(transformed["midas_in"][None, ...]).to(device=shared.device)
        midas_in = repeat(midas_in, "1 ... -> n ...", n=self.batch_size)

        conditioning_image = images_tensor_to_samples(source_image*0.5+0.5, approximation_indexes.get(opts.sd_vae_encode_method))
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
        conditioning_image = shared.sd_model.encode_first_stage(source_image).mode()

        return conditioning_image

    def unclip_image_conditioning(self, source_image):
        c_adm = self.sd_model.embedder(source_image)
        if self.sd_model.noise_augmentor is not None:
            noise_level = 0 # TODO: Allow other noise levels?
            c_adm, noise_level_emb = self.sd_model.noise_augmentor(c_adm, noise_level=repeat(torch.tensor([noise_level]).to(c_adm.device), '1 -> b', b=c_adm.shape[0]))
            c_adm = torch.cat((c_adm, noise_level_emb), 1)
        return c_adm

    def inpainting_image_conditioning(self, source_image, latent_image, image_mask=None, round_image_mask=True):
        self.is_using_inpainting_conditioning = True

        # Handle the different mask inputs
        if image_mask is not None:
            if torch.is_tensor(image_mask):
                conditioning_mask = image_mask
            else:
                conditioning_mask = np.array(image_mask.convert("L"))
                conditioning_mask = conditioning_mask.astype(np.float32) / 255.0
                conditioning_mask = torch.from_numpy(conditioning_mask[None, None])

                if round_image_mask:
                    # Caller is requesting a discretized mask as input, so we round to either 1.0 or 0.0
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

    def img2img_image_conditioning(self, source_image, latent_image, image_mask=None, round_image_mask=True):
        source_image = devices.cond_cast_float(source_image)

        # HACK: Using introspection as the Depth2Image model doesn't appear to uniquely
        # identify itself with a field common to all models. The conditioning_key is also hybrid.
        if isinstance(self.sd_model, LatentDepth2ImageDiffusion):
            return self.depth2img_image_conditioning(source_image)

        if self.sd_model.cond_stage_key == "edit":
            return self.edit_image_conditioning(source_image)

        if self.sampler.conditioning_key in {'hybrid', 'concat'}:
            return self.inpainting_image_conditioning(source_image, latent_image, image_mask=image_mask, round_image_mask=round_image_mask)

        if self.sampler.conditioning_key == "crossattn-adm":
            return self.unclip_image_conditioning(source_image)

        if self.sampler.model_wrap.inner_model.is_sdxl_inpaint:
            return self.inpainting_image_conditioning(source_image, latent_image, image_mask=image_mask)

        # Dummy zero conditioning if we're not using inpainting or depth model.
        return latent_image.new_zeros(latent_image.shape[0], 5, 1, 1)

    def init(self, all_prompts, all_seeds, all_subseeds):
        pass

    def sample(self, conditioning, unconditional_conditioning, seeds, subseeds, subseed_strength, prompts):
        raise NotImplementedError()

    def close(self):
        self.sampler = None
        self.c = None
        self.uc = None
        if not opts.persistent_cond_cache:
            StableDiffusionProcessing.cached_c = [None, None]
            StableDiffusionProcessing.cached_uc = [None, None]

    def get_token_merging_ratio(self, for_hr=False):
        if for_hr:
            return self.token_merging_ratio_hr or opts.token_merging_ratio_hr or self.token_merging_ratio or opts.token_merging_ratio

        return self.token_merging_ratio or opts.token_merging_ratio

    def setup_prompts(self):
        if isinstance(self.prompt,list):
            self.all_prompts = self.prompt
        elif isinstance(self.negative_prompt, list):
            self.all_prompts = [self.prompt] * len(self.negative_prompt)
        else:
            self.all_prompts = self.batch_size * self.n_iter * [self.prompt]

        if isinstance(self.negative_prompt, list):
            self.all_negative_prompts = self.negative_prompt
        else:
            self.all_negative_prompts = [self.negative_prompt] * len(self.all_prompts)

        if len(self.all_prompts) != len(self.all_negative_prompts):
            raise RuntimeError(f"Received a different number of prompts ({len(self.all_prompts)}) and negative prompts ({len(self.all_negative_prompts)})")

        self.all_prompts = [shared.prompt_styles.apply_styles_to_prompt(x, self.styles) for x in self.all_prompts]
        self.all_negative_prompts = [shared.prompt_styles.apply_negative_styles_to_prompt(x, self.styles) for x in self.all_negative_prompts]

        self.main_prompt = self.all_prompts[0]
        self.main_negative_prompt = self.all_negative_prompts[0]

    def cached_params(self, required_prompts, steps, extra_network_data, hires_steps=None, use_old_scheduling=False):
        """Returns parameters that invalidate the cond cache if changed"""

        return (
            required_prompts,
            steps,
            hires_steps,
            use_old_scheduling,
            opts.CLIP_stop_at_last_layers,
            shared.sd_model.sd_checkpoint_info,
            extra_network_data,
            opts.sdxl_crop_left,
            opts.sdxl_crop_top,
            self.width,
            self.height,
            opts.fp8_storage,
            opts.cache_fp16_weight,
            opts.emphasis,
        )

    def get_conds_with_caching(self, function, required_prompts, steps, caches, extra_network_data, hires_steps=None):
        """
        Returns the result of calling function(shared.sd_model, required_prompts, steps)
        using a cache to store the result if the same arguments have been used before.

        cache is an array containing two elements. The first element is a tuple
        representing the previously used arguments, or None if no arguments
        have been used before. The second element is where the previously
        computed result is stored.

        caches is a list with items described above.
        """

        if shared.opts.use_old_scheduling:
            old_schedules = prompt_parser.get_learned_conditioning_prompt_schedules(required_prompts, steps, hires_steps, False)
            new_schedules = prompt_parser.get_learned_conditioning_prompt_schedules(required_prompts, steps, hires_steps, True)
            if old_schedules != new_schedules:
                self.extra_generation_params["Old prompt editing timelines"] = True

        cached_params = self.cached_params(required_prompts, steps, extra_network_data, hires_steps, shared.opts.use_old_scheduling)

        for cache in caches:
            if cache[0] is not None and cached_params == cache[0]:
                return cache[1]

        cache = caches[0]

        with devices.autocast():
            cache[1] = function(shared.sd_model, required_prompts, steps, hires_steps, shared.opts.use_old_scheduling)

        cache[0] = cached_params
        return cache[1]

    def setup_conds(self):
        prompts = prompt_parser.SdConditioning(self.prompts, width=self.width, height=self.height)
        negative_prompts = prompt_parser.SdConditioning(self.negative_prompts, width=self.width, height=self.height, is_negative_prompt=True)

        sampler_config = sd_samplers.find_sampler_config(self.sampler_name)
        total_steps = sampler_config.total_steps(self.steps) if sampler_config else self.steps
        self.step_multiplier = total_steps // self.steps
        self.firstpass_steps = total_steps

        self.uc = self.get_conds_with_caching(prompt_parser.get_learned_conditioning, negative_prompts, total_steps, [self.cached_uc], self.extra_network_data)
        self.c = self.get_conds_with_caching(prompt_parser.get_multicond_learned_conditioning, prompts, total_steps, [self.cached_c], self.extra_network_data)

    def get_conds(self):
        return self.c, self.uc

    def parse_extra_network_prompts(self):
        self.prompts, self.extra_network_data = extra_networks.parse_prompts(self.prompts)

    def save_samples(self) -> bool:
        """Returns whether generated images need to be written to disk"""
        return opts.samples_save and not self.do_not_save_samples and (opts.save_incomplete_images or not state.interrupted and not state.skipped)


class Processed:
    def __init__(self, p: StableDiffusionProcessing, images_list, seed=-1, info="", subseed=None, all_prompts=None, all_negative_prompts=None, all_seeds=None, all_subseeds=None, index_of_first_image=0, infotexts=None, comments=""):
        self.images = images_list
        self.prompt = p.prompt
        self.negative_prompt = p.negative_prompt
        self.seed = seed
        self.subseed = subseed
        self.subseed_strength = p.subseed_strength
        self.info = info
        self.comments = "".join(f"{comment}\n" for comment in p.comments)
        self.width = p.width
        self.height = p.height
        self.sampler_name = p.sampler_name
        self.cfg_scale = p.cfg_scale
        self.image_cfg_scale = getattr(p, 'image_cfg_scale', None)
        self.steps = p.steps
        self.batch_size = p.batch_size
        self.restore_faces = p.restore_faces
        self.face_restoration_model = opts.face_restoration_model if p.restore_faces else None
        self.sd_model_name = p.sd_model_name
        self.sd_model_hash = p.sd_model_hash
        self.sd_vae_name = p.sd_vae_name
        self.sd_vae_hash = p.sd_vae_hash
        self.seed_resize_from_w = p.seed_resize_from_w
        self.seed_resize_from_h = p.seed_resize_from_h
        self.denoising_strength = getattr(p, 'denoising_strength', None)
        self.extra_generation_params = p.extra_generation_params
        self.index_of_first_image = index_of_first_image
        self.styles = p.styles
        self.job_timestamp = state.job_timestamp
        self.clip_skip = opts.CLIP_stop_at_last_layers
        self.token_merging_ratio = p.token_merging_ratio
        self.token_merging_ratio_hr = p.token_merging_ratio_hr

        self.eta = p.eta
        self.ddim_discretize = p.ddim_discretize
        self.s_churn = p.s_churn
        self.s_tmin = p.s_tmin
        self.s_tmax = p.s_tmax
        self.s_noise = p.s_noise
        self.s_min_uncond = p.s_min_uncond
        self.sampler_noise_scheduler_override = p.sampler_noise_scheduler_override
        self.prompt = self.prompt if not isinstance(self.prompt, list) else self.prompt[0]
        self.negative_prompt = self.negative_prompt if not isinstance(self.negative_prompt, list) else self.negative_prompt[0]
        self.seed = int(self.seed if not isinstance(self.seed, list) else self.seed[0]) if self.seed is not None else -1
        self.subseed = int(self.subseed if not isinstance(self.subseed, list) else self.subseed[0]) if self.subseed is not None else -1
        self.is_using_inpainting_conditioning = p.is_using_inpainting_conditioning

        self.all_prompts = all_prompts or p.all_prompts or [self.prompt]
        self.all_negative_prompts = all_negative_prompts or p.all_negative_prompts or [self.negative_prompt]
        self.all_seeds = all_seeds or p.all_seeds or [self.seed]
        self.all_subseeds = all_subseeds or p.all_subseeds or [self.subseed]
        self.infotexts = infotexts or [info] * len(images_list)
        self.version = program_version()

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
            "sd_model_name": self.sd_model_name,
            "sd_model_hash": self.sd_model_hash,
            "sd_vae_name": self.sd_vae_name,
            "sd_vae_hash": self.sd_vae_hash,
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
            "version": self.version,
        }

        return json.dumps(obj, default=lambda o: None)

    def infotext(self, p: StableDiffusionProcessing, index):
        return create_infotext(p, self.all_prompts, self.all_seeds, self.all_subseeds, comments=[], position_in_batch=index % self.batch_size, iteration=index // self.batch_size)

    def get_token_merging_ratio(self, for_hr=False):
        return self.token_merging_ratio_hr if for_hr else self.token_merging_ratio


def create_random_tensors(shape, seeds, subseeds=None, subseed_strength=0.0, seed_resize_from_h=0, seed_resize_from_w=0, p=None):
    g = rng.ImageRNG(shape, seeds, subseeds=subseeds, subseed_strength=subseed_strength, seed_resize_from_h=seed_resize_from_h, seed_resize_from_w=seed_resize_from_w)
    return g.next()


class DecodedSamples(list):
    already_decoded = True


def decode_latent_batch(model, batch, target_device=None, check_for_nans=False):
    samples = DecodedSamples()

    if check_for_nans:
        devices.test_for_nans(batch, "unet")

    for i in range(batch.shape[0]):
        sample = decode_first_stage(model, batch[i:i + 1])[0]

        if check_for_nans:

            try:
                devices.test_for_nans(sample, "vae")
            except devices.NansException as e:
                if shared.opts.auto_vae_precision_bfloat16:
                    autofix_dtype = torch.bfloat16
                    autofix_dtype_text = "bfloat16"
                    autofix_dtype_setting = "Automatically convert VAE to bfloat16"
                    autofix_dtype_comment = ""
                elif shared.opts.auto_vae_precision:
                    autofix_dtype = torch.float32
                    autofix_dtype_text = "32-bit float"
                    autofix_dtype_setting = "Automatically revert VAE to 32-bit floats"
                    autofix_dtype_comment = "\nTo always start with 32-bit VAE, use --no-half-vae commandline flag."
                else:
                    raise e

                if devices.dtype_vae == autofix_dtype:
                    raise e

                errors.print_error_explanation(
                    "A tensor with all NaNs was produced in VAE.\n"
                    f"Web UI will now convert VAE into {autofix_dtype_text} and retry.\n"
                    f"To disable this behavior, disable the '{autofix_dtype_setting}' setting.{autofix_dtype_comment}"
                )

                devices.dtype_vae = autofix_dtype
                model.first_stage_model.to(devices.dtype_vae)
                batch = batch.to(devices.dtype_vae)

                sample = decode_first_stage(model, batch[i:i + 1])[0]

        if target_device is not None:
            sample = sample.to(target_device)

        samples.append(sample)

    return samples


def get_fixed_seed(seed):
    if seed == '' or seed is None:
        seed = -1
    elif isinstance(seed, str):
        try:
            seed = int(seed)
        except Exception:
            seed = -1

    if seed == -1:
        return int(random.randrange(4294967294))

    return seed


def fix_seed(p):
    p.seed = get_fixed_seed(p.seed)
    p.subseed = get_fixed_seed(p.subseed)


def program_version():
    import launch

    res = launch.git_tag()
    if res == "<none>":
        res = None

    return res


def create_infotext(p, all_prompts, all_seeds, all_subseeds, comments=None, iteration=0, position_in_batch=0, use_main_prompt=False, index=None, all_negative_prompts=None):
    """
    this function is used to generate the infotext that is stored in the generated images, it's contains the parameters that are required to generate the imagee
    Args:
        p: StableDiffusionProcessing
        all_prompts: list[str]
        all_seeds: list[int]
        all_subseeds: list[int]
        comments: list[str]
        iteration: int
        position_in_batch: int
        use_main_prompt: bool
        index: int
        all_negative_prompts: list[str]

    Returns: str

    Extra generation params
    p.extra_generation_params dictionary allows for additional parameters to be added to the infotext
    this can be use by the base webui or extensions.
    To add a new entry, add a new key value pair, the dictionary key will be used as the key of the parameter in the infotext
    the value generation_params can be defined as:
        - str | None
        - List[str|None]
        - callable func(**kwargs) -> str | None

    When defined as a string, it will be used as without extra processing; this is this most common use case.

    Defining as a list allows for parameter that changes across images in the job, for example, the 'Seed' parameter.
    The list should have the same length as the total number of images in the entire job.

    Defining as a callable function allows parameter cannot be generated earlier or when extra logic is required.
    For example 'Hires prompt', due to reasons the hr_prompt might be changed by process in the pipeline or extensions
    and may vary across different images, defining as a static string or list would not work.

    The function takes locals() as **kwargs, as such will have access to variables like 'p' and 'index'.
    the base signature of the function should be:
        func(**kwargs) -> str | None
    optionally it can have additional arguments that will be used in the function:
        func(p, index, **kwargs) -> str | None
    note: for better future compatibility even though this function will have access to all variables in the locals(),
        it is recommended to only use the arguments present in the function signature of create_infotext.
    For actual implementation examples, see StableDiffusionProcessingTxt2Img.init > get_hr_prompt.
    """

    if use_main_prompt:
        index = 0
    elif index is None:
        index = position_in_batch + iteration * p.batch_size

    if all_negative_prompts is None:
        all_negative_prompts = p.all_negative_prompts

    clip_skip = getattr(p, 'clip_skip', opts.CLIP_stop_at_last_layers)
    enable_hr = getattr(p, 'enable_hr', False)
    token_merging_ratio = p.get_token_merging_ratio()
    token_merging_ratio_hr = p.get_token_merging_ratio(for_hr=True)

    prompt_text = p.main_prompt if use_main_prompt else all_prompts[index]
    negative_prompt = p.main_negative_prompt if use_main_prompt else all_negative_prompts[index]

    uses_ensd = opts.eta_noise_seed_delta != 0
    if uses_ensd:
        uses_ensd = sd_samplers_common.is_sampler_using_eta_noise_seed_delta(p)

    generation_params = {
        "Steps": p.steps,
        "Sampler": p.sampler_name,
        "Schedule type": p.scheduler,
        "CFG scale": p.cfg_scale,
        "Image CFG scale": getattr(p, 'image_cfg_scale', None),
        "Seed": p.all_seeds[0] if use_main_prompt else all_seeds[index],
        "Face restoration": opts.face_restoration_model if p.restore_faces else None,
        "Size": f"{p.width}x{p.height}",
        "Model hash": p.sd_model_hash if opts.add_model_hash_to_info else None,
        "Model": p.sd_model_name if opts.add_model_name_to_info else None,
        "FP8 weight": opts.fp8_storage if devices.fp8 else None,
        "Cache FP16 weight for LoRA": opts.cache_fp16_weight if devices.fp8 else None,
        "VAE hash": p.sd_vae_hash if opts.add_vae_hash_to_info else None,
        "VAE": p.sd_vae_name if opts.add_vae_name_to_info else None,
        "Variation seed": (None if p.subseed_strength == 0 else (p.all_subseeds[0] if use_main_prompt else all_subseeds[index])),
        "Variation seed strength": (None if p.subseed_strength == 0 else p.subseed_strength),
        "Seed resize from": (None if p.seed_resize_from_w <= 0 or p.seed_resize_from_h <= 0 else f"{p.seed_resize_from_w}x{p.seed_resize_from_h}"),
        "Denoising strength": p.extra_generation_params.get("Denoising strength"),
        "Conditional mask weight": getattr(p, "inpainting_mask_weight", shared.opts.inpainting_mask_weight) if p.is_using_inpainting_conditioning else None,
        "Clip skip": None if clip_skip <= 1 else clip_skip,
        "ENSD": opts.eta_noise_seed_delta if uses_ensd else None,
        "Token merging ratio": None if token_merging_ratio == 0 else token_merging_ratio,
        "Token merging ratio hr": None if not enable_hr or token_merging_ratio_hr == 0 else token_merging_ratio_hr,
        "Init image hash": getattr(p, 'init_img_hash', None),
        "RNG": opts.randn_source if opts.randn_source != "GPU" else None,
        "Tiling": "True" if p.tiling else None,
        **p.extra_generation_params,
        "Version": program_version() if opts.add_version_to_infotext else None,
        "User": p.user if opts.add_user_name_to_info else None,
    }

    for key, value in generation_params.items():
        try:
            if isinstance(value, list):
                generation_params[key] = value[index]
            elif callable(value):
                generation_params[key] = value(**locals())
        except Exception:
            errors.report(f'Error creating infotext for key "{key}"', exc_info=True)
            generation_params[key] = None

    generation_params_text = ", ".join([k if k == v else f'{k}: {infotext_utils.quote(v)}' for k, v in generation_params.items() if v is not None])

    negative_prompt_text = f"\nNegative prompt: {negative_prompt}" if negative_prompt else ""

    return f"{prompt_text}{negative_prompt_text}\n{generation_params_text}".strip()


def process_images(p: StableDiffusionProcessing) -> Processed:
    if p.scripts is not None:
        p.scripts.before_process(p)

    stored_opts = {k: opts.data[k] if k in opts.data else opts.get_default(k) for k in p.override_settings.keys() if k in opts.data}

    try:
        # if no checkpoint override or the override checkpoint can't be found, remove override entry and load opts checkpoint
        # and if after running refiner, the refiner model is not unloaded - webui swaps back to main model here, if model over is present it will be reloaded afterwards
        if sd_models.checkpoint_aliases.get(p.override_settings.get('sd_model_checkpoint')) is None:
            p.override_settings.pop('sd_model_checkpoint', None)
            sd_models.reload_model_weights()

        for k, v in p.override_settings.items():
            opts.set(k, v, is_api=True, run_callbacks=False)

            if k == 'sd_model_checkpoint':
                sd_models.reload_model_weights()

            if k == 'sd_vae':
                sd_vae.reload_vae_weights()

        sd_models.apply_token_merging(p.sd_model, p.get_token_merging_ratio())

        # backwards compatibility, fix sampler and scheduler if invalid
        sd_samplers.fix_p_invalid_sampler_and_scheduler(p)

        with profiling.Profiler():
            res = process_images_inner(p)

    finally:
        sd_models.apply_token_merging(p.sd_model, 0)

        # restore opts to original state
        if p.override_settings_restore_afterwards:
            for k, v in stored_opts.items():
                setattr(opts, k, v)

                if k == 'sd_vae':
                    sd_vae.reload_vae_weights()

    return res


def process_images_inner(p: StableDiffusionProcessing) -> Processed:
    """this is the main loop that both txt2img and img2img use; it calls func_init once inside all the scopes and func_sample once per batch"""

    if isinstance(p.prompt, list):
        assert(len(p.prompt) > 0)
    else:
        assert p.prompt is not None

    devices.torch_gc()

    seed = get_fixed_seed(p.seed)
    subseed = get_fixed_seed(p.subseed)

    if p.restore_faces is None:
        p.restore_faces = opts.face_restoration

    if p.tiling is None:
        p.tiling = opts.tiling

    if p.refiner_checkpoint not in (None, "", "None", "none"):
        p.refiner_checkpoint_info = sd_models.get_closet_checkpoint_match(p.refiner_checkpoint)
        if p.refiner_checkpoint_info is None:
            raise Exception(f'Could not find checkpoint with name {p.refiner_checkpoint}')

    if hasattr(shared.sd_model, 'fix_dimensions'):
        p.width, p.height = shared.sd_model.fix_dimensions(p.width, p.height)

    p.sd_model_name = shared.sd_model.sd_checkpoint_info.name_for_extra
    p.sd_model_hash = shared.sd_model.sd_model_hash
    p.sd_vae_name = sd_vae.get_loaded_vae_name()
    p.sd_vae_hash = sd_vae.get_loaded_vae_hash()

    modules.sd_hijack.model_hijack.apply_circular(p.tiling)
    modules.sd_hijack.model_hijack.clear_comments()

    p.fill_fields_from_opts()
    p.setup_prompts()

    if isinstance(seed, list):
        p.all_seeds = seed
    else:
        p.all_seeds = [int(seed) + (x if p.subseed_strength == 0 else 0) for x in range(len(p.all_prompts))]

    if isinstance(subseed, list):
        p.all_subseeds = subseed
    else:
        p.all_subseeds = [int(subseed) + x for x in range(len(p.all_prompts))]

    if os.path.exists(cmd_opts.embeddings_dir) and not p.do_not_reload_embeddings:
        model_hijack.embedding_db.load_textual_inversion_embeddings()

    if p.scripts is not None:
        p.scripts.process(p)

    infotexts = []
    output_images = []
    with torch.no_grad(), p.sd_model.ema_scope():
        with devices.autocast():
            p.init(p.all_prompts, p.all_seeds, p.all_subseeds)

            # for OSX, loading the model during sampling changes the generated picture, so it is loaded here
            if shared.opts.live_previews_enable and opts.show_progress_type == "Approx NN":
                sd_vae_approx.model()

            sd_unet.apply_unet()

        if state.job_count == -1:
            state.job_count = p.n_iter

        for n in range(p.n_iter):
            p.iteration = n

            if state.skipped:
                state.skipped = False

            if state.interrupted or state.stopping_generation:
                break

            sd_models.reload_model_weights()  # model can be changed for example by refiner

            p.prompts = p.all_prompts[n * p.batch_size:(n + 1) * p.batch_size]
            p.negative_prompts = p.all_negative_prompts[n * p.batch_size:(n + 1) * p.batch_size]
            p.seeds = p.all_seeds[n * p.batch_size:(n + 1) * p.batch_size]
            p.subseeds = p.all_subseeds[n * p.batch_size:(n + 1) * p.batch_size]

            latent_channels = getattr(shared.sd_model, 'latent_channels', opt_C)
            p.rng = rng.ImageRNG((latent_channels, p.height // opt_f, p.width // opt_f), p.seeds, subseeds=p.subseeds, subseed_strength=p.subseed_strength, seed_resize_from_h=p.seed_resize_from_h, seed_resize_from_w=p.seed_resize_from_w)

            if p.scripts is not None:
                p.scripts.before_process_batch(p, batch_number=n, prompts=p.prompts, seeds=p.seeds, subseeds=p.subseeds)

            if len(p.prompts) == 0:
                break

            p.parse_extra_network_prompts()

            if not p.disable_extra_networks:
                with devices.autocast():
                    extra_networks.activate(p, p.extra_network_data)

            if p.scripts is not None:
                p.scripts.process_batch(p, batch_number=n, prompts=p.prompts, seeds=p.seeds, subseeds=p.subseeds)

            p.setup_conds()

            p.extra_generation_params.update(model_hijack.extra_generation_params)

            # params.txt should be saved after scripts.process_batch, since the
            # infotext could be modified by that callback
            # Example: a wildcard processed by process_batch sets an extra model
            # strength, which is saved as "Model Strength: 1.0" in the infotext
            if n == 0 and not cmd_opts.no_prompt_history:
                with open(os.path.join(paths.data_path, "params.txt"), "w", encoding="utf8") as file:
                    processed = Processed(p, [])
                    file.write(processed.infotext(p, 0))

            for comment in model_hijack.comments:
                p.comment(comment)

            if p.n_iter > 1:
                shared.state.job = f"Batch {n+1} out of {p.n_iter}"

            sd_models.apply_alpha_schedule_override(p.sd_model, p)

            with devices.without_autocast() if devices.unet_needs_upcast else devices.autocast():
                samples_ddim = p.sample(conditioning=p.c, unconditional_conditioning=p.uc, seeds=p.seeds, subseeds=p.subseeds, subseed_strength=p.subseed_strength, prompts=p.prompts)

            if p.scripts is not None:
                ps = scripts.PostSampleArgs(samples_ddim)
                p.scripts.post_sample(p, ps)
                samples_ddim = ps.samples

            if getattr(samples_ddim, 'already_decoded', False):
                x_samples_ddim = samples_ddim
            else:
                devices.test_for_nans(samples_ddim, "unet")

                if opts.sd_vae_decode_method != 'Full':
                    p.extra_generation_params['VAE Decoder'] = opts.sd_vae_decode_method
                x_samples_ddim = decode_latent_batch(p.sd_model, samples_ddim, target_device=devices.cpu, check_for_nans=True)

            x_samples_ddim = torch.stack(x_samples_ddim).float()
            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

            del samples_ddim

            if lowvram.is_enabled(shared.sd_model):
                lowvram.send_everything_to_cpu()

            devices.torch_gc()

            state.nextjob()

            if p.scripts is not None:
                p.scripts.postprocess_batch(p, x_samples_ddim, batch_number=n)

                p.prompts = p.all_prompts[n * p.batch_size:(n + 1) * p.batch_size]
                p.negative_prompts = p.all_negative_prompts[n * p.batch_size:(n + 1) * p.batch_size]

                batch_params = scripts.PostprocessBatchListArgs(list(x_samples_ddim))
                p.scripts.postprocess_batch_list(p, batch_params, batch_number=n)
                x_samples_ddim = batch_params.images

            def infotext(index=0, use_main_prompt=False):
                return create_infotext(p, p.prompts, p.seeds, p.subseeds, use_main_prompt=use_main_prompt, index=index, all_negative_prompts=p.negative_prompts)

            save_samples = p.save_samples()

            for i, x_sample in enumerate(x_samples_ddim):
                p.batch_index = i

                x_sample = 255. * np.moveaxis(x_sample.cpu().numpy(), 0, 2)
                x_sample = x_sample.astype(np.uint8)

                if p.restore_faces:
                    if save_samples and opts.save_images_before_face_restoration:
                        images.save_image(Image.fromarray(x_sample), p.outpath_samples, "", p.seeds[i], p.prompts[i], opts.samples_format, info=infotext(i), p=p, suffix="-before-face-restoration")

                    devices.torch_gc()

                    x_sample = modules.face_restoration.restore_faces(x_sample)
                    devices.torch_gc()

                image = Image.fromarray(x_sample)

                if p.scripts is not None:
                    pp = scripts.PostprocessImageArgs(image)
                    p.scripts.postprocess_image(p, pp)
                    image = pp.image

                mask_for_overlay = getattr(p, "mask_for_overlay", None)

                if not shared.opts.overlay_inpaint:
                    overlay_image = None
                elif getattr(p, "overlay_images", None) is not None and i < len(p.overlay_images):
                    overlay_image = p.overlay_images[i]
                else:
                    overlay_image = None

                if p.scripts is not None:
                    ppmo = scripts.PostProcessMaskOverlayArgs(i, mask_for_overlay, overlay_image)
                    p.scripts.postprocess_maskoverlay(p, ppmo)
                    mask_for_overlay, overlay_image = ppmo.mask_for_overlay, ppmo.overlay_image

                if p.color_corrections is not None and i < len(p.color_corrections):
                    if save_samples and opts.save_images_before_color_correction:
                        image_without_cc, _ = apply_overlay(image, p.paste_to, overlay_image)
                        images.save_image(image_without_cc, p.outpath_samples, "", p.seeds[i], p.prompts[i], opts.samples_format, info=infotext(i), p=p, suffix="-before-color-correction")
                    image = apply_color_correction(p.color_corrections[i], image)

                # If the intention is to show the output from the model
                # that is being composited over the original image,
                # we need to keep the original image around
                # and use it in the composite step.
                image, original_denoised_image = apply_overlay(image, p.paste_to, overlay_image)

                if p.scripts is not None:
                    pp = scripts.PostprocessImageArgs(image)
                    p.scripts.postprocess_image_after_composite(p, pp)
                    image = pp.image

                if save_samples:
                    images.save_image(image, p.outpath_samples, "", p.seeds[i], p.prompts[i], opts.samples_format, info=infotext(i), p=p)

                text = infotext(i)
                infotexts.append(text)
                if opts.enable_pnginfo:
                    image.info["parameters"] = text
                output_images.append(image)

                if mask_for_overlay is not None:
                    if opts.return_mask or opts.save_mask:
                        image_mask = mask_for_overlay.convert('RGB')
                        if save_samples and opts.save_mask:
                            images.save_image(image_mask, p.outpath_samples, "", p.seeds[i], p.prompts[i], opts.samples_format, info=infotext(i), p=p, suffix="-mask")
                        if opts.return_mask:
                            output_images.append(image_mask)

                    if opts.return_mask_composite or opts.save_mask_composite:
                        image_mask_composite = Image.composite(original_denoised_image.convert('RGBA').convert('RGBa'), Image.new('RGBa', image.size), images.resize_image(2, mask_for_overlay, image.width, image.height).convert('L')).convert('RGBA')
                        if save_samples and opts.save_mask_composite:
                            images.save_image(image_mask_composite, p.outpath_samples, "", p.seeds[i], p.prompts[i], opts.samples_format, info=infotext(i), p=p, suffix="-mask-composite")
                        if opts.return_mask_composite:
                            output_images.append(image_mask_composite)

            del x_samples_ddim

            devices.torch_gc()

        if not infotexts:
            infotexts.append(Processed(p, []).infotext(p, 0))

        p.color_corrections = None

        index_of_first_image = 0
        unwanted_grid_because_of_img_count = len(output_images) < 2 and opts.grid_only_if_multiple
        if (opts.return_grid or opts.grid_save) and not p.do_not_save_grid and not unwanted_grid_because_of_img_count:
            grid = images.image_grid(output_images, p.batch_size)

            if opts.return_grid:
                text = infotext(use_main_prompt=True)
                infotexts.insert(0, text)
                if opts.enable_pnginfo:
                    grid.info["parameters"] = text
                output_images.insert(0, grid)
                index_of_first_image = 1
            if opts.grid_save:
                images.save_image(grid, p.outpath_grids, "grid", p.all_seeds[0], p.all_prompts[0], opts.grid_format, info=infotext(use_main_prompt=True), short_filename=not opts.grid_extended_filename, p=p, grid=True)

    if not p.disable_extra_networks and p.extra_network_data:
        extra_networks.deactivate(p, p.extra_network_data)

    devices.torch_gc()

    res = Processed(
        p,
        images_list=output_images,
        seed=p.all_seeds[0],
        info=infotexts[0],
        subseed=p.all_subseeds[0],
        index_of_first_image=index_of_first_image,
        infotexts=infotexts,
    )

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


@dataclass(repr=False)
class StableDiffusionProcessingTxt2Img(StableDiffusionProcessing):
    enable_hr: bool = False
    denoising_strength: float = 0.75
    firstphase_width: int = 0
    firstphase_height: int = 0
    hr_scale: float = 2.0
    hr_upscaler: str = None
    hr_second_pass_steps: int = 0
    hr_resize_x: int = 0
    hr_resize_y: int = 0
    hr_checkpoint_name: str = None
    hr_sampler_name: str = None
    hr_scheduler: str = None
    hr_prompt: str = ''
    hr_negative_prompt: str = ''
    force_task_id: str = None

    cached_hr_uc = [None, None]
    cached_hr_c = [None, None]

    hr_checkpoint_info: dict = field(default=None, init=False)
    hr_upscale_to_x: int = field(default=0, init=False)
    hr_upscale_to_y: int = field(default=0, init=False)
    truncate_x: int = field(default=0, init=False)
    truncate_y: int = field(default=0, init=False)
    applied_old_hires_behavior_to: tuple = field(default=None, init=False)
    latent_scale_mode: dict = field(default=None, init=False)
    hr_c: tuple | None = field(default=None, init=False)
    hr_uc: tuple | None = field(default=None, init=False)
    all_hr_prompts: list = field(default=None, init=False)
    all_hr_negative_prompts: list = field(default=None, init=False)
    hr_prompts: list = field(default=None, init=False)
    hr_negative_prompts: list = field(default=None, init=False)
    hr_extra_network_data: list = field(default=None, init=False)

    def __post_init__(self):
        super().__post_init__()

        if self.firstphase_width != 0 or self.firstphase_height != 0:
            self.hr_upscale_to_x = self.width
            self.hr_upscale_to_y = self.height
            self.width = self.firstphase_width
            self.height = self.firstphase_height

        self.cached_hr_uc = StableDiffusionProcessingTxt2Img.cached_hr_uc
        self.cached_hr_c = StableDiffusionProcessingTxt2Img.cached_hr_c

    def calculate_target_resolution(self):
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

    def init(self, all_prompts, all_seeds, all_subseeds):
        if self.enable_hr:
            self.extra_generation_params["Denoising strength"] = self.denoising_strength

            if self.hr_checkpoint_name and self.hr_checkpoint_name != 'Use same checkpoint':
                self.hr_checkpoint_info = sd_models.get_closet_checkpoint_match(self.hr_checkpoint_name)

                if self.hr_checkpoint_info is None:
                    raise Exception(f'Could not find checkpoint with name {self.hr_checkpoint_name}')

                self.extra_generation_params["Hires checkpoint"] = self.hr_checkpoint_info.short_title

            if self.hr_sampler_name is not None and self.hr_sampler_name != self.sampler_name:
                self.extra_generation_params["Hires sampler"] = self.hr_sampler_name

            def get_hr_prompt(p, index, prompt_text, **kwargs):
                hr_prompt = p.all_hr_prompts[index]
                return hr_prompt if hr_prompt != prompt_text else None

            def get_hr_negative_prompt(p, index, negative_prompt, **kwargs):
                hr_negative_prompt = p.all_hr_negative_prompts[index]
                return hr_negative_prompt if hr_negative_prompt != negative_prompt else None

            self.extra_generation_params["Hires prompt"] = get_hr_prompt
            self.extra_generation_params["Hires negative prompt"] = get_hr_negative_prompt

            self.extra_generation_params["Hires schedule type"] = None  # to be set in sd_samplers_kdiffusion.py

            if self.hr_scheduler is None:
                self.hr_scheduler = self.scheduler

            self.latent_scale_mode = shared.latent_upscale_modes.get(self.hr_upscaler, None) if self.hr_upscaler is not None else shared.latent_upscale_modes.get(shared.latent_upscale_default_mode, "nearest")
            if self.enable_hr and self.latent_scale_mode is None:
                if not any(x.name == self.hr_upscaler for x in shared.sd_upscalers):
                    raise Exception(f"could not find upscaler named {self.hr_upscaler}")

            self.calculate_target_resolution()

            if not state.processing_has_refined_job_count:
                if state.job_count == -1:
                    state.job_count = self.n_iter
                if getattr(self, 'txt2img_upscale', False):
                    total_steps = (self.hr_second_pass_steps or self.steps) * state.job_count
                else:
                    total_steps = (self.steps + (self.hr_second_pass_steps or self.steps)) * state.job_count
                shared.total_tqdm.updateTotal(total_steps)
                state.job_count = state.job_count * 2
                state.processing_has_refined_job_count = True

            if self.hr_second_pass_steps:
                self.extra_generation_params["Hires steps"] = self.hr_second_pass_steps

            if self.hr_upscaler is not None:
                self.extra_generation_params["Hires upscaler"] = self.hr_upscaler

    def sample(self, conditioning, unconditional_conditioning, seeds, subseeds, subseed_strength, prompts):
        self.sampler = sd_samplers.create_sampler(self.sampler_name, self.sd_model)

        if self.firstpass_image is not None and self.enable_hr:
            # here we don't need to generate image, we just take self.firstpass_image and prepare it for hires fix

            if self.latent_scale_mode is None:
                image = np.array(self.firstpass_image).astype(np.float32) / 255.0 * 2.0 - 1.0
                image = np.moveaxis(image, 2, 0)

                samples = None
                decoded_samples = torch.asarray(np.expand_dims(image, 0))

            else:
                image = np.array(self.firstpass_image).astype(np.float32) / 255.0
                image = np.moveaxis(image, 2, 0)
                image = torch.from_numpy(np.expand_dims(image, axis=0))
                image = image.to(shared.device, dtype=devices.dtype_vae)

                if opts.sd_vae_encode_method != 'Full':
                    self.extra_generation_params['VAE Encoder'] = opts.sd_vae_encode_method

                samples = images_tensor_to_samples(image, approximation_indexes.get(opts.sd_vae_encode_method), self.sd_model)
                decoded_samples = None
                devices.torch_gc()

        else:
            # here we generate an image normally

            x = self.rng.next()
            if self.scripts is not None:
                self.scripts.process_before_every_sampling(
                    p=self,
                    x=x,
                    noise=x,
                    c=conditioning,
                    uc=unconditional_conditioning
                )

            samples = self.sampler.sample(self, x, conditioning, unconditional_conditioning, image_conditioning=self.txt2img_image_conditioning(x))
            del x

            if not self.enable_hr:
                return samples

            devices.torch_gc()

            if self.latent_scale_mode is None:
                decoded_samples = torch.stack(decode_latent_batch(self.sd_model, samples, target_device=devices.cpu, check_for_nans=True)).to(dtype=torch.float32)
            else:
                decoded_samples = None

        with sd_models.SkipWritingToConfig():
            sd_models.reload_model_weights(info=self.hr_checkpoint_info)

        return self.sample_hr_pass(samples, decoded_samples, seeds, subseeds, subseed_strength, prompts)

    def sample_hr_pass(self, samples, decoded_samples, seeds, subseeds, subseed_strength, prompts):
        if shared.state.interrupted:
            return samples

        self.is_hr_pass = True
        target_width = self.hr_upscale_to_x
        target_height = self.hr_upscale_to_y

        def save_intermediate(image, index):
            """saves image before applying hires fix, if enabled in options; takes as an argument either an image or batch with latent space images"""

            if not self.save_samples() or not opts.save_images_before_highres_fix:
                return

            if not isinstance(image, Image.Image):
                image = sd_samplers.sample_to_image(image, index, approximation=0)

            info = create_infotext(self, self.all_prompts, self.all_seeds, self.all_subseeds, [], iteration=self.iteration, position_in_batch=index)
            images.save_image(image, self.outpath_samples, "", seeds[index], prompts[index], opts.samples_format, info=info, p=self, suffix="-before-highres-fix")

        img2img_sampler_name = self.hr_sampler_name or self.sampler_name

        self.sampler = sd_samplers.create_sampler(img2img_sampler_name, self.sd_model)

        if self.latent_scale_mode is not None:
            for i in range(samples.shape[0]):
                save_intermediate(samples, i)

            samples = torch.nn.functional.interpolate(samples, size=(target_height // opt_f, target_width // opt_f), mode=self.latent_scale_mode["mode"], antialias=self.latent_scale_mode["antialias"])

            # Avoid making the inpainting conditioning unless necessary as
            # this does need some extra compute to decode / encode the image again.
            if getattr(self, "inpainting_mask_weight", shared.opts.inpainting_mask_weight) < 1.0:
                image_conditioning = self.img2img_image_conditioning(decode_first_stage(self.sd_model, samples), samples)
            else:
                image_conditioning = self.txt2img_image_conditioning(samples)
        else:
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
            decoded_samples = decoded_samples.to(shared.device, dtype=devices.dtype_vae)

            if opts.sd_vae_encode_method != 'Full':
                self.extra_generation_params['VAE Encoder'] = opts.sd_vae_encode_method
            samples = images_tensor_to_samples(decoded_samples, approximation_indexes.get(opts.sd_vae_encode_method))

            image_conditioning = self.img2img_image_conditioning(decoded_samples, samples)

        shared.state.nextjob()

        samples = samples[:, :, self.truncate_y//2:samples.shape[2]-(self.truncate_y+1)//2, self.truncate_x//2:samples.shape[3]-(self.truncate_x+1)//2]

        self.rng = rng.ImageRNG(samples.shape[1:], self.seeds, subseeds=self.subseeds, subseed_strength=self.subseed_strength, seed_resize_from_h=self.seed_resize_from_h, seed_resize_from_w=self.seed_resize_from_w)
        noise = self.rng.next()

        # GC now before running the next img2img to prevent running out of memory
        devices.torch_gc()

        if not self.disable_extra_networks:
            with devices.autocast():
                extra_networks.activate(self, self.hr_extra_network_data)

        with devices.autocast():
            self.calculate_hr_conds()

        sd_models.apply_token_merging(self.sd_model, self.get_token_merging_ratio(for_hr=True))

        if self.scripts is not None:
            self.scripts.before_hr(self)
            self.scripts.process_before_every_sampling(
                p=self,
                x=samples,
                noise=noise,
                c=self.hr_c,
                uc=self.hr_uc,
            )

        samples = self.sampler.sample_img2img(self, samples, noise, self.hr_c, self.hr_uc, steps=self.hr_second_pass_steps or self.steps, image_conditioning=image_conditioning)

        sd_models.apply_token_merging(self.sd_model, self.get_token_merging_ratio())

        self.sampler = None
        devices.torch_gc()

        decoded_samples = decode_latent_batch(self.sd_model, samples, target_device=devices.cpu, check_for_nans=True)

        self.is_hr_pass = False
        return decoded_samples

    def close(self):
        super().close()
        self.hr_c = None
        self.hr_uc = None
        if not opts.persistent_cond_cache:
            StableDiffusionProcessingTxt2Img.cached_hr_uc = [None, None]
            StableDiffusionProcessingTxt2Img.cached_hr_c = [None, None]

    def setup_prompts(self):
        super().setup_prompts()

        if not self.enable_hr:
            return

        if self.hr_prompt == '':
            self.hr_prompt = self.prompt

        if self.hr_negative_prompt == '':
            self.hr_negative_prompt = self.negative_prompt

        if isinstance(self.hr_prompt, list):
            self.all_hr_prompts = self.hr_prompt
        else:
            self.all_hr_prompts = self.batch_size * self.n_iter * [self.hr_prompt]

        if isinstance(self.hr_negative_prompt, list):
            self.all_hr_negative_prompts = self.hr_negative_prompt
        else:
            self.all_hr_negative_prompts = self.batch_size * self.n_iter * [self.hr_negative_prompt]

        self.all_hr_prompts = [shared.prompt_styles.apply_styles_to_prompt(x, self.styles) for x in self.all_hr_prompts]
        self.all_hr_negative_prompts = [shared.prompt_styles.apply_negative_styles_to_prompt(x, self.styles) for x in self.all_hr_negative_prompts]

    def calculate_hr_conds(self):
        if self.hr_c is not None:
            return

        hr_prompts = prompt_parser.SdConditioning(self.hr_prompts, width=self.hr_upscale_to_x, height=self.hr_upscale_to_y)
        hr_negative_prompts = prompt_parser.SdConditioning(self.hr_negative_prompts, width=self.hr_upscale_to_x, height=self.hr_upscale_to_y, is_negative_prompt=True)

        sampler_config = sd_samplers.find_sampler_config(self.hr_sampler_name or self.sampler_name)
        steps = self.hr_second_pass_steps or self.steps
        total_steps = sampler_config.total_steps(steps) if sampler_config else steps

        self.hr_uc = self.get_conds_with_caching(prompt_parser.get_learned_conditioning, hr_negative_prompts, self.firstpass_steps, [self.cached_hr_uc, self.cached_uc], self.hr_extra_network_data, total_steps)
        self.hr_c = self.get_conds_with_caching(prompt_parser.get_multicond_learned_conditioning, hr_prompts, self.firstpass_steps, [self.cached_hr_c, self.cached_c], self.hr_extra_network_data, total_steps)

    def setup_conds(self):
        if self.is_hr_pass:
            # if we are in hr pass right now, the call is being made from the refiner, and we don't need to setup firstpass cons or switch model
            self.hr_c = None
            self.calculate_hr_conds()
            return

        super().setup_conds()

        self.hr_uc = None
        self.hr_c = None

        if self.enable_hr and self.hr_checkpoint_info is None:
            if shared.opts.hires_fix_use_firstpass_conds:
                self.calculate_hr_conds()

            elif lowvram.is_enabled(shared.sd_model) and shared.sd_model.sd_checkpoint_info == sd_models.select_checkpoint():  # if in lowvram mode, we need to calculate conds right away, before the cond NN is unloaded
                with devices.autocast():
                    extra_networks.activate(self, self.hr_extra_network_data)

                self.calculate_hr_conds()

                with devices.autocast():
                    extra_networks.activate(self, self.extra_network_data)

    def get_conds(self):
        if self.is_hr_pass:
            return self.hr_c, self.hr_uc

        return super().get_conds()

    def parse_extra_network_prompts(self):
        res = super().parse_extra_network_prompts()

        if self.enable_hr:
            self.hr_prompts = self.all_hr_prompts[self.iteration * self.batch_size:(self.iteration + 1) * self.batch_size]
            self.hr_negative_prompts = self.all_hr_negative_prompts[self.iteration * self.batch_size:(self.iteration + 1) * self.batch_size]

            self.hr_prompts, self.hr_extra_network_data = extra_networks.parse_prompts(self.hr_prompts)

        return res


@dataclass(repr=False)
class StableDiffusionProcessingImg2Img(StableDiffusionProcessing):
    init_images: list = None
    resize_mode: int = 0
    denoising_strength: float = 0.75
    image_cfg_scale: float = None
    mask: Any = None
    mask_blur_x: int = 4
    mask_blur_y: int = 4
    mask_blur: int = None
    mask_round: bool = True
    inpainting_fill: int = 0
    inpaint_full_res: bool = True
    inpaint_full_res_padding: int = 0
    inpainting_mask_invert: int = 0
    initial_noise_multiplier: float = None
    latent_mask: Image = None
    force_task_id: str = None

    image_mask: Any = field(default=None, init=False)

    nmask: torch.Tensor = field(default=None, init=False)
    image_conditioning: torch.Tensor = field(default=None, init=False)
    init_img_hash: str = field(default=None, init=False)
    mask_for_overlay: Image = field(default=None, init=False)
    init_latent: torch.Tensor = field(default=None, init=False)

    def __post_init__(self):
        super().__post_init__()

        self.image_mask = self.mask
        self.mask = None
        self.initial_noise_multiplier = opts.initial_noise_multiplier if self.initial_noise_multiplier is None else self.initial_noise_multiplier

    @property
    def mask_blur(self):
        if self.mask_blur_x == self.mask_blur_y:
            return self.mask_blur_x
        return None

    @mask_blur.setter
    def mask_blur(self, value):
        if isinstance(value, int):
            self.mask_blur_x = value
            self.mask_blur_y = value

    def init(self, all_prompts, all_seeds, all_subseeds):
        self.extra_generation_params["Denoising strength"] = self.denoising_strength

        self.image_cfg_scale: float = self.image_cfg_scale if shared.sd_model.cond_stage_key == "edit" else None

        self.sampler = sd_samplers.create_sampler(self.sampler_name, self.sd_model)
        crop_region = None

        image_mask = self.image_mask

        if image_mask is not None:
            # image_mask is passed in as RGBA by Gradio to support alpha masks,
            # but we still want to support binary masks.
            image_mask = create_binary_mask(image_mask, round=self.mask_round)

            if self.inpainting_mask_invert:
                image_mask = ImageOps.invert(image_mask)
                self.extra_generation_params["Mask mode"] = "Inpaint not masked"

            if self.mask_blur_x > 0:
                np_mask = np.array(image_mask)
                kernel_size = 2 * int(2.5 * self.mask_blur_x + 0.5) + 1
                np_mask = cv2.GaussianBlur(np_mask, (kernel_size, 1), self.mask_blur_x)
                image_mask = Image.fromarray(np_mask)

            if self.mask_blur_y > 0:
                np_mask = np.array(image_mask)
                kernel_size = 2 * int(2.5 * self.mask_blur_y + 0.5) + 1
                np_mask = cv2.GaussianBlur(np_mask, (1, kernel_size), self.mask_blur_y)
                image_mask = Image.fromarray(np_mask)

            if self.mask_blur_x > 0 or self.mask_blur_y > 0:
                self.extra_generation_params["Mask blur"] = self.mask_blur

            if self.inpaint_full_res:
                self.mask_for_overlay = image_mask
                mask = image_mask.convert('L')
                crop_region = masking.get_crop_region_v2(mask, self.inpaint_full_res_padding)
                if crop_region:
                    crop_region = masking.expand_crop_region(crop_region, self.width, self.height, mask.width, mask.height)
                    x1, y1, x2, y2 = crop_region
                    mask = mask.crop(crop_region)
                    image_mask = images.resize_image(2, mask, self.width, self.height)
                    self.paste_to = (x1, y1, x2-x1, y2-y1)
                    self.extra_generation_params["Inpaint area"] = "Only masked"
                    self.extra_generation_params["Masked area padding"] = self.inpaint_full_res_padding
                else:
                    crop_region = None
                    image_mask = None
                    self.mask_for_overlay = None
                    self.inpaint_full_res = False
                    massage = 'Unable to perform "Inpaint Only mask" because mask is blank, switch to img2img mode.'
                    model_hijack.comments.append(massage)
                    logging.info(massage)
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

            # Save init image
            if opts.save_init_img:
                self.init_img_hash = hashlib.md5(img.tobytes()).hexdigest()
                images.save_image(img, path=opts.outdir_init_images, basename=None, forced_filename=self.init_img_hash, save_to_dirs=False, existing_info=img.info)

            image = images.flatten(img, opts.img2img_background_color)

            if crop_region is None and self.resize_mode != 3:
                image = images.resize_image(self.resize_mode, image, self.width, self.height)

            if image_mask is not None:
                if self.mask_for_overlay.size != (image.width, image.height):
                    self.mask_for_overlay = images.resize_image(self.resize_mode, self.mask_for_overlay, image.width, image.height)
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

                    if self.inpainting_fill == 0:
                        self.extra_generation_params["Masked content"] = 'fill'

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
        image = image.to(shared.device, dtype=devices.dtype_vae)

        if opts.sd_vae_encode_method != 'Full':
            self.extra_generation_params['VAE Encoder'] = opts.sd_vae_encode_method

        self.init_latent = images_tensor_to_samples(image, approximation_indexes.get(opts.sd_vae_encode_method), self.sd_model)
        devices.torch_gc()

        if self.resize_mode == 3:
            self.init_latent = torch.nn.functional.interpolate(self.init_latent, size=(self.height // opt_f, self.width // opt_f), mode="bilinear")

        if image_mask is not None:
            init_mask = latent_mask
            latmask = init_mask.convert('RGB').resize((self.init_latent.shape[3], self.init_latent.shape[2]))
            latmask = np.moveaxis(np.array(latmask, dtype=np.float32), 2, 0) / 255
            latmask = latmask[0]
            if self.mask_round:
                latmask = np.around(latmask)
            latmask = np.tile(latmask[None], (self.init_latent.shape[1], 1, 1))

            self.mask = torch.asarray(1.0 - latmask).to(shared.device).type(devices.dtype)
            self.nmask = torch.asarray(latmask).to(shared.device).type(devices.dtype)

            # this needs to be fixed to be done in sample() using actual seeds for batches
            if self.inpainting_fill == 2:
                self.init_latent = self.init_latent * self.mask + create_random_tensors(self.init_latent.shape[1:], all_seeds[0:self.init_latent.shape[0]]) * self.nmask
                self.extra_generation_params["Masked content"] = 'latent noise'

            elif self.inpainting_fill == 3:
                self.init_latent = self.init_latent * self.mask
                self.extra_generation_params["Masked content"] = 'latent nothing'

        self.image_conditioning = self.img2img_image_conditioning(image * 2 - 1, self.init_latent, image_mask, self.mask_round)

    def sample(self, conditioning, unconditional_conditioning, seeds, subseeds, subseed_strength, prompts):
        x = self.rng.next()

        if self.initial_noise_multiplier != 1.0:
            self.extra_generation_params["Noise multiplier"] = self.initial_noise_multiplier
            x *= self.initial_noise_multiplier

        if self.scripts is not None:
            self.scripts.process_before_every_sampling(
                p=self,
                x=self.init_latent,
                noise=x,
                c=conditioning,
                uc=unconditional_conditioning
            )
        samples = self.sampler.sample_img2img(self, self.init_latent, x, conditioning, unconditional_conditioning, image_conditioning=self.image_conditioning)

        if self.mask is not None:
            blended_samples = samples * self.nmask + self.init_latent * self.mask

            if self.scripts is not None:
                mba = scripts.MaskBlendArgs(samples, self.nmask, self.init_latent, self.mask, blended_samples)
                self.scripts.on_mask_blend(self, mba)
                blended_samples = mba.blended_latent

            samples = blended_samples

        del x
        devices.torch_gc()

        return samples

    def get_token_merging_ratio(self, for_hr=False):
        return self.token_merging_ratio or ("token_merging_ratio" in self.override_settings and opts.token_merging_ratio) or opts.token_merging_ratio_img2img or opts.token_merging_ratio
