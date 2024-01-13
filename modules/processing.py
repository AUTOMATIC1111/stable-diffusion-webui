import os
import json
import math
import time
import hashlib
import random
import warnings
from contextlib import nullcontext
from typing import Any, Dict, List
from dataclasses import dataclass, field
import torch
import numpy as np
import cv2
from PIL import Image, ImageOps
from skimage import exposure
from einops import repeat, rearrange
from blendmodes.blend import blendLayers, BlendType
from installer import git_commit
from modules import shared, devices, errors
import modules.memstats
import modules.lowvram
import modules.masking
import modules.paths
import modules.scripts
import modules.script_callbacks
import modules.prompt_parser
import modules.extra_networks
import modules.face_restoration
import modules.images as images
import modules.styles
import modules.sd_hijack_freeu
import modules.sd_samplers
import modules.sd_samplers_common
import modules.sd_models
import modules.sd_vae
import modules.sd_vae_approx
import modules.taesd.sd_vae_taesd
import modules.generation_parameters_copypaste
from modules.sd_hijack_hypertile import context_hypertile_vae, context_hypertile_unet, hypertile_set


if shared.backend == shared.Backend.ORIGINAL:
    import modules.sd_hijack

opt_C = 4
opt_f = 8
debug = shared.log.trace if os.environ.get('SD_PROCESS_DEBUG', None) is not None else lambda *args, **kwargs: None
debug('Trace: PROCESS')


def setup_color_correction(image):
    debug("Calibrating color correction")
    correction_target = cv2.cvtColor(np.asarray(image.copy()), cv2.COLOR_RGB2LAB)
    return correction_target


def apply_color_correction(correction, original_image):
    shared.log.debug(f"Applying color correction: correction={correction.shape} image={original_image}")
    np_image = np.asarray(original_image)
    np_recolor = cv2.cvtColor(np_image, cv2.COLOR_RGB2LAB)
    np_match = exposure.match_histograms(np_recolor, correction, channel_axis=2)
    np_output = cv2.cvtColor(np_match, cv2.COLOR_LAB2RGB)
    image = Image.fromarray(np_output.astype("uint8"))
    image = blendLayers(image, original_image, BlendType.LUMINOSITY)
    return image


def apply_overlay(image: Image, paste_loc, index, overlays):
    debug(f'Apply overlay: image={image} loc={paste_loc} index={index} overlays={overlays}')
    if overlays is None or index >= len(overlays):
        return image
    overlay = overlays[index]
    if paste_loc is not None:
        x, y, w, h = paste_loc
        if image.width != w or image.height != h or x != 0 or y != 0:
            base_image = Image.new('RGBA', (overlay.width, overlay.height))
            image = images.resize_image(2, image, w, h)
            base_image.paste(image, (x, y))
            image = base_image
    image = image.convert('RGBA')
    image.alpha_composite(overlay)
    image = image.convert('RGB')
    return image


def create_binary_mask(image):
    if image.mode == 'RGBA' and image.getextrema()[-1] != (255, 255):
        image = image.split()[-1].convert("L").point(lambda x: 255 if x > 128 else 0)
    else:
        image = image.convert('L')
    return image


def images_tensor_to_samples(image, approximation=None, model=None): # pylint: disable=unused-argument
    if model is None:
        model = shared.sd_model
    model.first_stage_model.to(devices.dtype_vae)
    image = image.to(shared.device, dtype=devices.dtype_vae)
    image = image * 2 - 1
    if len(image) > 1:
        x_latent = torch.stack([
            model.get_first_stage_encoding(model.encode_first_stage(torch.unsqueeze(img, 0)))[0]
            for img in image
        ])
    else:
        x_latent = model.get_first_stage_encoding(model.encode_first_stage(image))
    return x_latent


def txt2img_image_conditioning(sd_model, x, width, height):
    if sd_model.model.conditioning_key in {'hybrid', 'concat'}: # Inpainting models
        # The "masked-image" in this case will just be all zeros since the entire image is masked.
        image_conditioning = torch.zeros(x.shape[0], 3, height, width, device=x.device)
        image_conditioning = sd_model.get_first_stage_encoding(sd_model.encode_first_stage(image_conditioning))
        # Add the fake full 1s mask to the first dimension.
        image_conditioning = torch.nn.functional.pad(image_conditioning, (0, 0, 0, 0, 1, 0), value=1.0) # pylint: disable=not-callable
        image_conditioning = image_conditioning.to(x.dtype)
        return image_conditioning
    elif sd_model.model.conditioning_key == "crossattn-adm": # UnCLIP models
        return x.new_zeros(x.shape[0], 2*sd_model.noise_augmentor.time_embed.dim, dtype=x.dtype, device=x.device)
    else:
        # Dummy zero conditioning if we're not using inpainting or unclip models.
        # Still takes up a bit of memory, but no encoder call.
        # Pretty sure we can just make this a 1x1 image since its not going to be used besides its batch size.
        return x.new_zeros(x.shape[0], 5, 1, 1, dtype=x.dtype, device=x.device)


def get_sampler_name(sampler_index: int, img: bool = False) -> str:
    if len(modules.sd_samplers.samplers) > sampler_index:
        sampler_name = modules.sd_samplers.samplers[sampler_index].name
    else:
        sampler_name = "UniPC"
        shared.log.warning(f'Sampler not found: index={sampler_index} available={[s.name for s in modules.sd_samplers.samplers]} fallback={sampler_name}')
    if img and sampler_name == "PLMS":
        sampler_name = "UniPC"
        shared.log.warning(f'Sampler not compatible: name=PLMS fallback={sampler_name}')
    return sampler_name


@dataclass(repr=False)
class StableDiffusionProcessing:
    """
    The first set of paramaters: sd_models -> do_not_reload_embeddings represent the minimum required to create a StableDiffusionProcessing
    """
    def __init__(self, sd_model=None, outpath_samples=None, outpath_grids=None, prompt: str = "", styles: List[str] = None, seed: int = -1, subseed: int = -1, subseed_strength: float = 0, seed_resize_from_h: int = -1, seed_resize_from_w: int = -1, seed_enable_extras: bool = True, sampler_name: str = None, hr_sampler_name: str = None, batch_size: int = 1, n_iter: int = 1, steps: int = 50, cfg_scale: float = 7.0, image_cfg_scale: float = None, clip_skip: int = 1, width: int = 512, height: int = 512, full_quality: bool = True, restore_faces: bool = False, tiling: bool = False, do_not_save_samples: bool = False, do_not_save_grid: bool = False, extra_generation_params: Dict[Any, Any] = None, overlay_images: Any = None, negative_prompt: str = None, eta: float = None, do_not_reload_embeddings: bool = False, denoising_strength: float = 0, diffusers_guidance_rescale: float = 0.7, sag_scale: float = 0.0, resize_mode: int = 0, resize_name: str = 'None', scale_by: float = 0, selected_scale_tab: int = 0, hdr_clamp: bool = False, hdr_boundary: float = 4.0, hdr_threshold: float = 3.5, hdr_center: bool = False, hdr_channel_shift: float = 0.8, hdr_full_shift: float = 0.8, hdr_maximize: bool = False, hdr_max_center: float = 0.6, hdr_max_boundry: float = 1.0, override_settings: Dict[str, Any] = None, override_settings_restore_afterwards: bool = True, sampler_index: int = None, script_args: list = None): # pylint: disable=unused-argument
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
        self.hr_sampler_name: str = hr_sampler_name
        self.batch_size: int = batch_size
        self.n_iter: int = n_iter
        self.steps: int = steps
        self.hr_second_pass_steps = 0
        self.cfg_scale: float = cfg_scale
        self.scale_by: float = scale_by
        self.image_cfg_scale = image_cfg_scale
        self.diffusers_guidance_rescale = diffusers_guidance_rescale
        self.sag_scale = sag_scale
        if devices.backend == "ipex" and width == 1024 and height == 1024 and os.environ.get('DISABLE_IPEX_1024_WA', None) is None:
            width = 1080
            height = 1080
        self.width: int = width
        self.height: int = height
        self.full_quality: bool = full_quality
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
        self.override_settings = {k: v for k, v in (override_settings or {}).items() if k not in shared.restricted_opts}
        self.override_settings_restore_afterwards = override_settings_restore_afterwards
        self.is_using_inpainting_conditioning = False
        self.disable_extra_networks = False
        self.token_merging_ratio = 0
        self.token_merging_ratio_hr = 0
        # self.scripts = modules.scripts.ScriptRunner() # set via property
        # self.script_args = script_args or [] # set via property
        self.per_script_args = {}
        self.all_prompts = None
        self.all_negative_prompts = None
        self.all_seeds = None
        self.all_subseeds = None
        self.clip_skip = clip_skip
        self.iteration = 0
        self.is_control = False
        self.is_hr_pass = False
        self.is_refiner_pass = False
        self.hr_force = False
        self.enable_hr = None
        self.hr_scale = None
        self.hr_upscaler = None
        self.hr_resize_x = 0
        self.hr_resize_y = 0
        self.hr_upscale_to_x = 0
        self.hr_upscale_to_y = 0
        self.truncate_x = 0
        self.truncate_y = 0
        self.applied_old_hires_behavior_to = None
        self.refiner_steps = 5
        self.refiner_start = 0
        self.refiner_prompt = ''
        self.refiner_negative = ''
        self.ops = []
        self.resize_mode: int = resize_mode
        self.resize_name: str = resize_name
        self.ddim_discretize = shared.opts.ddim_discretize
        self.s_min_uncond = shared.opts.s_min_uncond
        self.s_churn = shared.opts.s_churn
        self.s_noise = shared.opts.s_noise
        self.s_min = shared.opts.s_min
        self.s_max = shared.opts.s_max
        self.s_tmin = shared.opts.s_tmin
        self.s_tmax = float('inf')  # not representable as a standard ui option
        shared.opts.data['clip_skip'] = clip_skip
        self.task_args = {}
        # a1111 compatibility items
        self.refiner_switch_at = 0
        self.hr_prompt = ''
        self.all_hr_prompts = []
        self.hr_negative_prompt = ''
        self.all_hr_negative_prompts = []
        self.comments = {}
        self.is_api = False
        self.scripts_value: modules.scripts.ScriptRunner = field(default=None, init=False)
        self.script_args_value: list = field(default=None, init=False)
        self.scripts_setup_complete: bool = field(default=False, init=False)
        # hdr
        self.hdr_clamp = hdr_clamp
        self.hdr_boundary = hdr_boundary
        self.hdr_threshold = hdr_threshold
        self.hdr_center = hdr_center
        self.hdr_channel_shift = hdr_channel_shift
        self.hdr_full_shift = hdr_full_shift
        self.hdr_maximize = hdr_maximize
        self.hdr_max_center = hdr_max_center
        self.hdr_max_boundry = hdr_max_boundry
        self.scheduled_prompt: bool = False
        self.prompt_embeds = []
        self.positive_pooleds = []
        self.negative_embeds = []
        self.negative_pooleds = []


    @property
    def sd_model(self):
        return shared.sd_model

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
        from ldm.data.util import AddMiDaS
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
        image_conditioning = image_conditioning.to(device=shared.device, dtype=source_image.dtype)
        return image_conditioning

    def diffusers_image_conditioning(self, _source_image, latent_image, _image_mask=None):
        # shared.log.warning('Diffusers not implemented: img2img_image_conditioning')
        return latent_image.new_zeros(latent_image.shape[0], 5, 1, 1)

    def img2img_image_conditioning(self, source_image, latent_image, image_mask=None):
        from ldm.models.diffusion.ddpm import LatentDepth2ImageDiffusion
        source_image = devices.cond_cast_float(source_image)
        # HACK: Using introspection as the Depth2Image model doesn't appear to uniquely
        # identify itself with a field common to all models. The conditioning_key is also hybrid.
        if shared.backend == shared.Backend.DIFFUSERS:
            return self.diffusers_image_conditioning(source_image, latent_image, image_mask)
        if isinstance(self.sd_model, LatentDepth2ImageDiffusion):
            return self.depth2img_image_conditioning(source_image)
        if hasattr(self.sd_model, 'cond_stage_key') and self.sd_model.cond_stage_key == "edit":
            return self.edit_image_conditioning(source_image)
        if hasattr(self.sampler, 'conditioning_key') and self.sampler.conditioning_key in {'hybrid', 'concat'}:
            return self.inpainting_image_conditioning(source_image, latent_image, image_mask=image_mask)
        if hasattr(self.sampler, 'conditioning_key') and self.sampler.conditioning_key == "crossattn-adm":
            return self.unclip_image_conditioning(source_image)
        # Dummy zero conditioning if we're not using inpainting or depth model.
        return latent_image.new_zeros(latent_image.shape[0], 5, 1, 1)

    def init(self, all_prompts, all_seeds, all_subseeds):
        pass

    def sample(self, conditioning, unconditional_conditioning, seeds, subseeds, subseed_strength, prompts):
        raise NotImplementedError

    def close(self):
        self.sampler = None # pylint: disable=attribute-defined-outside-init

    def get_token_merging_ratio(self, for_hr=False):
        if for_hr:
            return self.token_merging_ratio_hr or shared.opts.token_merging_ratio_hr or self.token_merging_ratio or shared.opts.token_merging_ratio
        return self.token_merging_ratio or shared.opts.token_merging_ratio


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
        self.width = p.width if hasattr(p, 'width') else (self.images[0].width if len(self.images) > 0 else 0)
        self.height = p.height if hasattr(p, 'height') else (self.images[0].height if len(self.images) > 0 else 0)
        self.sampler_name = p.sampler_name
        self.cfg_scale = p.cfg_scale
        self.image_cfg_scale = p.image_cfg_scale
        self.steps = p.steps
        self.batch_size = p.batch_size
        self.restore_faces = p.restore_faces
        self.face_restoration_model = shared.opts.face_restoration_model if p.restore_faces else None
        self.sd_model_hash = getattr(shared.sd_model, 'sd_model_hash', '')
        self.seed_resize_from_w = p.seed_resize_from_w
        self.seed_resize_from_h = p.seed_resize_from_h
        self.denoising_strength = p.denoising_strength
        self.extra_generation_params = p.extra_generation_params
        self.index_of_first_image = index_of_first_image
        self.styles = p.styles
        self.job_timestamp = shared.state.job_timestamp
        self.clip_skip = p.clip_skip
        self.eta = p.eta
        self.ddim_discretize = p.ddim_discretize
        self.s_churn = p.s_churn
        self.s_tmin = p.s_tmin
        self.s_tmax = p.s_tmax
        self.s_noise = p.s_noise
        self.s_min_uncond = p.s_min_uncond
        self.prompt = self.prompt if type(self.prompt) != list else self.prompt[0]
        self.negative_prompt = self.negative_prompt if type(self.negative_prompt) != list else self.negative_prompt[0]
        self.seed = int(self.seed if type(self.seed) != list else self.seed[0]) if self.seed is not None else -1
        self.subseed = int(self.subseed if type(self.subseed) != list else self.subseed[0]) if self.subseed is not None else -1
        self.is_using_inpainting_conditioning = p.is_using_inpainting_conditioning
        self.all_prompts = all_prompts or p.all_prompts or [self.prompt]
        self.all_negative_prompts = all_negative_prompts or p.all_negative_prompts or [self.negative_prompt]
        self.all_seeds = all_seeds or p.all_seeds or [self.seed]
        self.all_subseeds = all_subseeds or p.all_subseeds or [self.subseed]
        self.token_merging_ratio = p.token_merging_ratio
        self.token_merging_ratio_hr = p.token_merging_ratio_hr
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
            # "is_using_inpainting_conditioning": self.is_using_inpainting_conditioning,
        }
        return json.dumps(obj)

    def infotext(self, p: StableDiffusionProcessing, index):
        return create_infotext(p, self.all_prompts, self.all_seeds, self.all_subseeds, comments=[], position_in_batch=index % self.batch_size, iteration=index // self.batch_size)

    def get_token_merging_ratio(self, for_hr=False):
        return self.token_merging_ratio_hr if for_hr else self.token_merging_ratio


def slerp(val, low, high): # from https://discuss.pytorch.org/t/help-regarding-slerp-function-for-generative-model-sampling/32475/3
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
    eta_noise_seed_delta = shared.opts.eta_noise_seed_delta or 0
    xs = []
    # if we have multiple seeds, this means we are working with batch size>1; this then
    # enables the generation of additional tensors with noise that the sampler will use during its processing.
    # Using those pre-generated tensors instead of simple torch.randn allows a batch with seeds [100, 101] to
    # produce the same images as with two batches [100], [101].
    if p is not None and p.sampler is not None and (len(seeds) > 1 and shared.opts.enable_batch_seeds or eta_noise_seed_delta > 0):
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


def decode_first_stage(model, x, full_quality=True):
    if not shared.opts.keep_incomplete and (shared.state.skipped or shared.state.interrupted):
        shared.log.debug(f'Decode VAE: skipped={shared.state.skipped} interrupted={shared.state.interrupted}')
        x_sample = torch.zeros((len(x), 3, x.shape[2] * 8, x.shape[3] * 8), dtype=devices.dtype_vae, device=devices.device)
        return x_sample
    prev_job = shared.state.job
    shared.state.job = 'vae'
    with devices.autocast(disable = x.dtype==devices.dtype_vae):
        try:
            if full_quality:
                if hasattr(model, 'decode_first_stage'):
                    x_sample = model.decode_first_stage(x)
                elif hasattr(model, 'vae'):
                    x_sample = model.vae(x)
                else:
                    x_sample = x
                    shared.log.error('Decode VAE unknown model')
            else:
                x_sample = torch.zeros((len(x), 3, x.shape[2] * 8, x.shape[3] * 8), dtype=devices.dtype_vae, device=devices.device)
                for i in range(len(x_sample)):
                    x_sample[i] = modules.taesd.sd_vae_taesd.decode(x[i])
        except Exception as e:
            x_sample = x
            shared.log.error(f'Decode VAE: {e}')
    shared.state.job = prev_job
    return x_sample


def get_fixed_seed(seed):
    if seed is None or seed == '' or seed == -1:
        return int(random.randrange(4294967294))
    return seed


def fix_seed(p):
    p.seed = get_fixed_seed(p.seed)
    p.subseed = get_fixed_seed(p.subseed)


def create_infotext(p: StableDiffusionProcessing, all_prompts=None, all_seeds=None, all_subseeds=None, comments=None, iteration=0, position_in_batch=0, index=None, all_negative_prompts=None):
    if not hasattr(shared.sd_model, 'sd_checkpoint_info'):
        return ''
    if index is None:
        index = position_in_batch + iteration * p.batch_size
    if all_prompts is None:
        all_prompts = p.all_prompts or [p.prompt]
    if all_negative_prompts is None:
        all_negative_prompts = p.all_negative_prompts or [p.negative_prompt]
    if all_seeds is None:
        all_seeds = p.all_seeds or [p.seed]
    if all_subseeds is None:
        all_subseeds = p.all_subseeds or [p.subseed]
    while len(all_prompts) <= index:
        all_prompts.append(all_prompts[-1])
    while len(all_seeds) <= index:
        all_seeds.append(all_seeds[-1])
    while len(all_subseeds) <= index:
        all_subseeds.append(all_subseeds[-1])
    while len(all_negative_prompts) <= index:
        all_negative_prompts.append(all_negative_prompts[-1])
    comment = ', '.join(comments) if comments is not None and type(comments) is list else None
    ops = list(set(p.ops))
    ops.reverse()
    args = {
        # basic
        "Steps": p.steps,
        "Seed": all_seeds[index],
        "Sampler": p.sampler_name,
        "CFG scale": p.cfg_scale,
        "Size": f"{p.width}x{p.height}" if hasattr(p, 'width') and hasattr(p, 'height') else None,
        "Batch": f'{p.n_iter}x{p.batch_size}' if p.n_iter > 1 or p.batch_size > 1 else None,
        "Index": f'{p.iteration + 1}x{index + 1}' if (p.n_iter > 1 or p.batch_size > 1) and index >= 0 else None,
        "Parser": shared.opts.prompt_attention,
        "Model": None if (not shared.opts.add_model_name_to_info) or (not shared.sd_model.sd_checkpoint_info.model_name) else shared.sd_model.sd_checkpoint_info.model_name.replace(',', '').replace(':', ''),
        "Model hash": getattr(p, 'sd_model_hash', None if (not shared.opts.add_model_hash_to_info) or (not shared.sd_model.sd_model_hash) else shared.sd_model.sd_model_hash),
        "VAE": (None if not shared.opts.add_model_name_to_info or modules.sd_vae.loaded_vae_file is None else os.path.splitext(os.path.basename(modules.sd_vae.loaded_vae_file))[0]) if p.full_quality else 'TAESD',
        "Seed resize from": None if p.seed_resize_from_w == 0 or p.seed_resize_from_h == 0 else f"{p.seed_resize_from_w}x{p.seed_resize_from_h}",
        "Clip skip": p.clip_skip if p.clip_skip > 1 else None,
        "Prompt2": p.refiner_prompt if len(p.refiner_prompt) > 0 else None,
        "Negative2": p.refiner_negative if len(p.refiner_negative) > 0 else None,
        "Styles": "; ".join(p.styles) if p.styles is not None and len(p.styles) > 0 else None,
        "Tiling": p.tiling if p.tiling else None,
        # sdnext
        "Backend": 'Diffusers' if shared.backend == shared.Backend.DIFFUSERS else 'Original',
        "App": 'SD.Next',
        "Version": git_commit,
        "Comment": comment,
        "Operations": '; '.join(ops).replace('"', '') if len(p.ops) > 0 else 'none',
    }
    if 'txt2img' in p.ops:
        pass
    if shared.backend == shared.Backend.ORIGINAL:
        args["Variation seed"] = None if p.subseed_strength == 0 else all_subseeds[index],
        args["Variation strength"] = None if p.subseed_strength == 0 else p.subseed_strength,
    if 'hires' in p.ops or 'upscale' in p.ops:
        args["Second pass"] = p.enable_hr
        args["Hires force"] = p.hr_force
        args["Hires steps"] = p.hr_second_pass_steps
        args["Hires upscaler"] = p.hr_upscaler
        args["Hires upscale"] = p.hr_scale
        args["Hires resize"] = f"{p.hr_resize_x}x{p.hr_resize_y}"
        args["Hires size"] = f"{p.hr_upscale_to_x}x{p.hr_upscale_to_y}"
        args["Denoising strength"] = p.denoising_strength
        args["Hires sampler"] = p.hr_sampler_name
        args["Image CFG scale"] = p.image_cfg_scale
        args["CFG rescale"] = p.diffusers_guidance_rescale
    if 'refine' in p.ops:
        args["Second pass"] = p.enable_hr
        args["Refiner"] = None if (not shared.opts.add_model_name_to_info) or (not shared.sd_refiner) or (not shared.sd_refiner.sd_checkpoint_info.model_name) else shared.sd_refiner.sd_checkpoint_info.model_name.replace(',', '').replace(':', '')
        args['Image CFG scale'] = p.image_cfg_scale
        args['Refiner steps'] = p.refiner_steps
        args['Refiner start'] = p.refiner_start
        args["Hires steps"] = p.hr_second_pass_steps
        args["Hires sampler"] = p.hr_sampler_name
        args["CFG rescale"] = p.diffusers_guidance_rescale
    if 'img2img' in p.ops or 'inpaint' in p.ops:
        args["Init image size"] = f"{getattr(p, 'init_img_width', 0)}x{getattr(p, 'init_img_height', 0)}"
        args["Init image hash"] = getattr(p, 'init_img_hash', None)
        args["Mask weight"] = getattr(p, "inpainting_mask_weight", shared.opts.inpainting_mask_weight) if p.is_using_inpainting_conditioning else None
        args['Resize scale'] = getattr(p, 'scale_by', None)
        args["Mask blur"] = p.mask_blur if getattr(p, 'mask', None) is not None and getattr(p, 'mask_blur', 0) > 0 else None
        args["Denoising strength"] = getattr(p, 'denoising_strength', None)
        if args["Size"] is None:
            args["Size"] = args["Init image size"]
        # lookup by index
        if getattr(p, 'resize_mode', None) is not None:
            args['Resize mode'] = shared.resize_modes[p.resize_mode]
    if 'face' in p.ops:
        args["Face restoration"] = shared.opts.face_restoration_model
    if 'color' in p.ops:
        args["Color correction"] = True
    # embeddings
    if hasattr(modules.sd_hijack.model_hijack, 'embedding_db') and len(modules.sd_hijack.model_hijack.embedding_db.embeddings_used) > 0: # this is for original hijaacked models only, diffusers are handled separately
        args["Embeddings"] = ', '.join(modules.sd_hijack.model_hijack.embedding_db.embeddings_used)
    # samplers
    args["Sampler ENSD"] = shared.opts.eta_noise_seed_delta if shared.opts.eta_noise_seed_delta != 0 and modules.sd_samplers_common.is_sampler_using_eta_noise_seed_delta(p) else None
    args["Sampler ENSM"] = p.initial_noise_multiplier if getattr(p, 'initial_noise_multiplier', 1.0) != 1.0 else None
    args['Sampler order'] = shared.opts.schedulers_solver_order if shared.opts.schedulers_solver_order != shared.opts.data_labels.get('schedulers_solver_order').default else None
    if shared.backend == shared.Backend.DIFFUSERS:
        args['Sampler beta schedule'] = shared.opts.schedulers_beta_schedule if shared.opts.schedulers_beta_schedule != shared.opts.data_labels.get('schedulers_beta_schedule').default else None
        args['Sampler beta start'] = shared.opts.schedulers_beta_start if shared.opts.schedulers_beta_start != shared.opts.data_labels.get('schedulers_beta_start').default else None
        args['Sampler beta end'] = shared.opts.schedulers_beta_end if shared.opts.schedulers_beta_end != shared.opts.data_labels.get('schedulers_beta_end').default else None
        args['Sampler DPM solver'] = shared.opts.schedulers_dpm_solver if shared.opts.schedulers_dpm_solver != shared.opts.data_labels.get('schedulers_dpm_solver').default else None
    if shared.backend == shared.Backend.ORIGINAL:
        args['Sampler brownian'] = shared.opts.schedulers_brownian_noise if shared.opts.schedulers_brownian_noise != shared.opts.data_labels.get('schedulers_brownian_noise').default else None
        args['Sampler discard'] = shared.opts.schedulers_discard_penultimate if shared.opts.schedulers_discard_penultimate != shared.opts.data_labels.get('schedulers_discard_penultimate').default else None
        args['Sampler dyn threshold'] = shared.opts.schedulers_use_thresholding if shared.opts.schedulers_use_thresholding != shared.opts.data_labels.get('schedulers_use_thresholding').default else None
        args['Sampler karras'] = shared.opts.schedulers_use_karras if shared.opts.schedulers_use_karras != shared.opts.data_labels.get('schedulers_use_karras').default else None
        args['Sampler low order'] = shared.opts.schedulers_use_loworder if shared.opts.schedulers_use_loworder != shared.opts.data_labels.get('schedulers_use_loworder').default else None
        args['Sampler quantization'] = shared.opts.enable_quantization if shared.opts.enable_quantization != shared.opts.data_labels.get('enable_quantization').default else None
        args['Sampler sigma'] = shared.opts.schedulers_sigma if shared.opts.schedulers_sigma != shared.opts.data_labels.get('schedulers_sigma').default else None
        args['Sampler sigma min'] = shared.opts.s_min if shared.opts.s_min != shared.opts.data_labels.get('s_min').default else None
        args['Sampler sigma max'] = shared.opts.s_max if shared.opts.s_max != shared.opts.data_labels.get('s_max').default else None
        args['Sampler sigma churn'] = shared.opts.s_churn if shared.opts.s_churn != shared.opts.data_labels.get('s_churn').default else None
        args['Sampler sigma uncond'] = shared.opts.s_churn if shared.opts.s_churn != shared.opts.data_labels.get('s_churn').default else None
        args['Sampler sigma noise'] = shared.opts.s_noise if shared.opts.s_noise != shared.opts.data_labels.get('s_noise').default else None
        args['Sampler sigma tmin'] = shared.opts.s_tmin if shared.opts.s_tmin != shared.opts.data_labels.get('s_tmin').default else None
    # tome
    token_merging_ratio = p.get_token_merging_ratio()
    token_merging_ratio_hr = p.get_token_merging_ratio(for_hr=True) if p.enable_hr else None
    args['ToMe'] = token_merging_ratio if token_merging_ratio != 0 else None
    args['ToMe hires'] = token_merging_ratio_hr if token_merging_ratio_hr != 0 else None

    args.update(p.extra_generation_params)
    params_text = ", ".join([k if k == v else f'{k}: {modules.generation_parameters_copypaste.quote(v)}' for k, v in args.items() if v is not None])
    negative_prompt_text = f"\nNegative prompt: {all_negative_prompts[index]}" if all_negative_prompts[index] else ""
    infotext = f"{all_prompts[index]}{negative_prompt_text}\n{params_text}".strip()
    return infotext


def process_images(p: StableDiffusionProcessing) -> Processed:
    debug(f'Process images: {vars(p)}')
    if not hasattr(p.sd_model, 'sd_checkpoint_info'):
        return None
    if p.scripts is not None and isinstance(p.scripts, modules.scripts.ScriptRunner):
        p.scripts.before_process(p)
    stored_opts = {}
    for k, v in p.override_settings.copy().items():
        if shared.opts.data.get(k, None) is None and shared.opts.data_labels.get(k, None) is None:
            continue
        orig = shared.opts.data.get(k, None) or shared.opts.data_labels[k].default
        if orig == v or (type(orig) == str and os.path.splitext(orig)[0] == v):
            p.override_settings.pop(k, None)
    for k in p.override_settings.keys():
        stored_opts[k] = shared.opts.data.get(k, None) or shared.opts.data_labels[k].default
    res = None
    try:
        # if no checkpoint override or the override checkpoint can't be found, remove override entry and load opts checkpoint
        if p.override_settings.get('sd_model_checkpoint', None) is not None and modules.sd_models.checkpoint_aliases.get(p.override_settings.get('sd_model_checkpoint')) is None:
            shared.log.warning(f"Override not found: checkpoint={p.override_settings.get('sd_model_checkpoint', None)}")
            p.override_settings.pop('sd_model_checkpoint', None)
            modules.sd_models.reload_model_weights()
        if p.override_settings.get('sd_model_refiner', None) is not None and modules.sd_models.checkpoint_aliases.get(p.override_settings.get('sd_model_refiner')) is None:
            shared.log.warning(f"Override not found: refiner={p.override_settings.get('sd_model_refiner', None)}")
            p.override_settings.pop('sd_model_refiner', None)
            modules.sd_models.reload_model_weights()
        if p.override_settings.get('sd_vae', None) is not None:
            if p.override_settings.get('sd_vae', None) == 'TAESD':
                p.full_quality = False
                p.override_settings.pop('sd_vae', None)
        if p.override_settings.get('Hires upscaler', None) is not None:
            p.enable_hr = True
        if len(p.override_settings.keys()) > 0:
            shared.log.debug(f'Override: {p.override_settings}')
        for k, v in p.override_settings.items():
            setattr(shared.opts, k, v)
            if k == 'sd_model_checkpoint':
                modules.sd_models.reload_model_weights()
            if k == 'sd_vae':
                modules.sd_vae.reload_vae_weights()

        shared.prompt_styles.apply_styles_to_extra(p)
        if not shared.opts.cuda_compile:
            modules.sd_models.apply_token_merging(p.sd_model, p.get_token_merging_ratio())
            modules.sd_hijack_freeu.apply_freeu(p, shared.backend == shared.Backend.ORIGINAL)

        modules.script_callbacks.before_process_callback(p)

        if shared.cmd_opts.profile:
            import cProfile
            profile_python = cProfile.Profile()
            profile_python.enable()
            with context_hypertile_vae(p), context_hypertile_unet(p):
                import torch.profiler # pylint: disable=redefined-outer-name
                activities=[torch.profiler.ProfilerActivity.CPU]
                if torch.cuda.is_available():
                    activities.append(torch.profiler.ProfilerActivity.CUDA)
                shared.log.debug(f'Torch profile: activities={activities}')
                if shared.profiler is None:
                    shared.profiler = torch.profiler.profile(activities=activities, profile_memory=True, with_modules=True)
                shared.profiler.start()
                shared.profiler.step()
                res = process_images_inner(p)
                errors.profile_torch(shared.profiler, 'Process')
            errors.profile(profile_python, 'Process')
        else:
            with context_hypertile_vae(p), context_hypertile_unet(p):
                res = process_images_inner(p)

    finally:
        if not shared.opts.cuda_compile:
            modules.sd_models.apply_token_merging(p.sd_model, 0)
        modules.script_callbacks.after_process_callback(p)
        if p.override_settings_restore_afterwards: # restore opts to original state
            for k, v in stored_opts.items():
                setattr(shared.opts, k, v)
                if k == 'sd_model_checkpoint':
                    modules.sd_models.reload_model_weights()
                if k == 'sd_model_refiner':
                    modules.sd_models.reload_model_weights()
                if k == 'sd_vae':
                    modules.sd_vae.reload_vae_weights()
    return res


def validate_sample(tensor):
    if not isinstance(tensor, np.ndarray) and not isinstance(tensor, torch.Tensor):
        return tensor
    if tensor.dtype == torch.bfloat16: # numpy does not support bf16
        tensor = tensor.to(torch.float16)
    if isinstance(tensor, torch.Tensor) and hasattr(tensor, 'detach'):
        sample = tensor.detach().cpu().numpy()
    elif isinstance(tensor, np.ndarray):
        sample = tensor
    else:
        shared.log.warning(f'Unknown sample type: {type(tensor)}')
    sample = 255.0 * np.moveaxis(sample, 0, 2) if shared.backend == shared.Backend.ORIGINAL else 255.0 * sample
    with warnings.catch_warnings(record=True) as w:
        cast = sample.astype(np.uint8)
    if len(w) > 0:
        nans = np.isnan(sample).sum()
        shared.log.error(f'Failed to validate samples: sample={sample.shape} invalid={nans}')
        cast = np.nan_to_num(sample)
        minimum, maximum, mean = np.min(cast), np.max(cast), np.mean(cast)
        cast = cast.astype(np.uint8)
        shared.log.warning(f'Attempted to correct samples: min={minimum:.2f} max={maximum:.2f} mean={mean:.2f}')
    return cast


def process_init(p: StableDiffusionProcessing):
    seed = get_fixed_seed(p.seed)
    subseed = get_fixed_seed(p.subseed)
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


def process_images_inner(p: StableDiffusionProcessing) -> Processed:
    """this is the main loop that both txt2img and img2img use; it calls func_init once inside all the scopes and func_sample once per batch"""

    if type(p.prompt) == list:
        assert len(p.prompt) > 0
    else:
        assert p.prompt is not None

    if shared.backend == shared.Backend.ORIGINAL:
        modules.sd_hijack.model_hijack.apply_circular(p.tiling)
        modules.sd_hijack.model_hijack.clear_comments()
    comments = {}
    infotexts = []
    output_images = []
    cached_uc = [None, None]
    cached_c = [None, None]

    process_init(p)
    if os.path.exists(shared.opts.embeddings_dir) and not p.do_not_reload_embeddings and shared.backend == shared.Backend.ORIGINAL:
        modules.sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings(force_reload=False)
    if p.scripts is not None and isinstance(p.scripts, modules.scripts.ScriptRunner):
        p.scripts.process(p)


    def get_conds_with_caching(function, required_prompts, steps, cache):
        if cache[0] is not None and (required_prompts, steps) == cache[0]:
            return cache[1]
        with devices.autocast():
            cache[1] = function(shared.sd_model, required_prompts, steps)
        cache[0] = (required_prompts, steps)
        return cache[1]

    def infotext(_inxex=0): # dummy function overriden if there are iterations
        return ''

    ema_scope_context = p.sd_model.ema_scope if shared.backend == shared.Backend.ORIGINAL else nullcontext
    shared.state.job_count = p.n_iter
    with devices.inference_context(), ema_scope_context():
        t0 = time.time()
        with devices.autocast():
            p.init(p.all_prompts, p.all_seeds, p.all_subseeds)
        extra_network_data = None
        debug(f'Processing inner: args={vars(p)}')
        for n in range(p.n_iter):
            p.iteration = n
            if shared.state.skipped:
                shared.log.debug(f'Process skipped: {n}/{p.n_iter}')
                shared.state.skipped = False
                continue
            if shared.state.interrupted:
                shared.log.debug(f'Process interrupted: {n}/{p.n_iter}')
                break
            p.prompts = p.all_prompts[n * p.batch_size:(n + 1) * p.batch_size]
            p.negative_prompts = p.all_negative_prompts[n * p.batch_size:(n + 1) * p.batch_size]
            p.seeds = p.all_seeds[n * p.batch_size:(n + 1) * p.batch_size]
            p.subseeds = p.all_subseeds[n * p.batch_size:(n + 1) * p.batch_size]
            if p.scripts is not None and isinstance(p.scripts, modules.scripts.ScriptRunner):
                p.scripts.before_process_batch(p, batch_number=n, prompts=p.prompts, seeds=p.seeds, subseeds=p.subseeds)
            if len(p.prompts) == 0:
                break
            p.prompts, extra_network_data = modules.extra_networks.parse_prompts(p.prompts)
            if not p.disable_extra_networks:
                with devices.autocast():
                    modules.extra_networks.activate(p, extra_network_data)
            if p.scripts is not None and isinstance(p.scripts, modules.scripts.ScriptRunner):
                p.scripts.process_batch(p, batch_number=n, prompts=p.prompts, seeds=p.seeds, subseeds=p.subseeds)
            step_multiplier = 1
            sampler_config = modules.sd_samplers.find_sampler_config(p.sampler_name)
            step_multiplier = 2 if sampler_config and sampler_config.options.get("second_order", False) else 1

            if shared.backend == shared.Backend.ORIGINAL:
                uc = get_conds_with_caching(modules.prompt_parser.get_learned_conditioning, p.negative_prompts, p.steps * step_multiplier, cached_uc)
                c = get_conds_with_caching(modules.prompt_parser.get_multicond_learned_conditioning, p.prompts, p.steps * step_multiplier, cached_c)
                if len(modules.sd_hijack.model_hijack.comments) > 0:
                    for comment in modules.sd_hijack.model_hijack.comments:
                        comments[comment] = 1
                with devices.without_autocast() if devices.unet_needs_upcast else devices.autocast():
                    samples_ddim = p.sample(conditioning=c, unconditional_conditioning=uc, seeds=p.seeds, subseeds=p.subseeds, subseed_strength=p.subseed_strength, prompts=p.prompts)
                x_samples_ddim = [decode_first_stage(p.sd_model, samples_ddim[i:i+1].to(dtype=devices.dtype_vae), p.full_quality)[0].cpu() for i in range(samples_ddim.size(0))]
                try:
                    for x in x_samples_ddim:
                        devices.test_for_nans(x, "vae")
                except devices.NansException as e:
                    if not shared.opts.no_half and not shared.opts.no_half_vae and shared.cmd_opts.rollback_vae:
                        shared.log.warning('Tensor with all NaNs was produced in VAE')
                        devices.dtype_vae = torch.bfloat16
                        vae_file, vae_source = modules.sd_vae.resolve_vae(p.sd_model.sd_model_checkpoint)
                        modules.sd_vae.load_vae(p.sd_model, vae_file, vae_source)
                        x_samples_ddim = [decode_first_stage(p.sd_model, samples_ddim[i:i+1].to(dtype=devices.dtype_vae), p.full_quality)[0].cpu() for i in range(samples_ddim.size(0))]
                        for x in x_samples_ddim:
                            devices.test_for_nans(x, "vae")
                    else:
                        raise e
                x_samples_ddim = torch.stack(x_samples_ddim).float()
                x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                del samples_ddim

            elif shared.backend == shared.Backend.DIFFUSERS:
                from modules.processing_diffusers import process_diffusers
                x_samples_ddim = process_diffusers(p)
            else:
                raise ValueError(f"Unknown backend {shared.backend}")

            if not shared.opts.keep_incomplete and shared.state.interrupted:
                x_samples_ddim = []

            if shared.cmd_opts.lowvram or shared.cmd_opts.medvram and shared.backend == shared.Backend.ORIGINAL:
                modules.lowvram.send_everything_to_cpu()
                devices.torch_gc()
            if p.scripts is not None and isinstance(p.scripts, modules.scripts.ScriptRunner):
                p.scripts.postprocess_batch(p, x_samples_ddim, batch_number=n)
            if p.scripts is not None and isinstance(p.scripts, modules.scripts.ScriptRunner):
                p.prompts = p.all_prompts[n * p.batch_size:(n + 1) * p.batch_size]
                p.negative_prompts = p.all_negative_prompts[n * p.batch_size:(n + 1) * p.batch_size]
                batch_params = modules.scripts.PostprocessBatchListArgs(list(x_samples_ddim))
                p.scripts.postprocess_batch_list(p, batch_params, batch_number=n)
                x_samples_ddim = batch_params.images

            def infotext(index): # pylint: disable=function-redefined # noqa: F811
                return create_infotext(p, p.prompts, p.seeds, p.subseeds, index=index, all_negative_prompts=p.negative_prompts)

            for i, x_sample in enumerate(x_samples_ddim):
                p.batch_index = i
                if type(x_sample) == Image.Image:
                    image = x_sample
                    x_sample = np.array(x_sample)
                else:
                    x_sample = validate_sample(x_sample)
                    image = Image.fromarray(x_sample)
                if p.restore_faces:
                    if shared.opts.save and not p.do_not_save_samples and shared.opts.save_images_before_face_restoration:
                        orig = p.restore_faces
                        p.restore_faces = False
                        info = infotext(i)
                        p.restore_faces = orig
                        images.save_image(Image.fromarray(x_sample), path=p.outpath_samples, basename="", seed=p.seeds[i], prompt=p.prompts[i], extension=shared.opts.samples_format, info=info, p=p, suffix="-before-face-restore")
                    p.ops.append('face')
                    x_sample = modules.face_restoration.restore_faces(x_sample)
                    image = Image.fromarray(x_sample)
                if p.scripts is not None and isinstance(p.scripts, modules.scripts.ScriptRunner):
                    pp = modules.scripts.PostprocessImageArgs(image)
                    p.scripts.postprocess_image(p, pp)
                    image = pp.image
                if p.color_corrections is not None and i < len(p.color_corrections):
                    if shared.opts.save and not p.do_not_save_samples and shared.opts.save_images_before_color_correction:
                        orig = p.color_corrections
                        p.color_corrections = None
                        info = infotext(i)
                        p.color_corrections = orig
                        image_without_cc = apply_overlay(image, p.paste_to, i, p.overlay_images)
                        images.save_image(image_without_cc, path=p.outpath_samples, basename="", seed=p.seeds[i], prompt=p.prompts[i], extension=shared.opts.samples_format, info=info, p=p, suffix="-before-color-correct")
                    p.ops.append('color')
                    image = apply_color_correction(p.color_corrections[i], image)
                image = apply_overlay(image, p.paste_to, i, p.overlay_images)
                text = infotext(i)
                infotexts.append(text)
                image.info["parameters"] = text
                output_images.append(image)
                if shared.opts.samples_save and not p.do_not_save_samples:
                    images.save_image(image, p.outpath_samples, "", p.seeds[i], p.prompts[i], shared.opts.samples_format, info=text, p=p) # main save image
                if hasattr(p, 'mask_for_overlay') and p.mask_for_overlay and any([shared.opts.save_mask, shared.opts.save_mask_composite, shared.opts.return_mask, shared.opts.return_mask_composite]):
                    image_mask = p.mask_for_overlay.convert('RGB')
                    image_mask_composite = Image.composite(image.convert('RGBA').convert('RGBa'), Image.new('RGBa', image.size), images.resize_image(3, p.mask_for_overlay, image.width, image.height).convert('L')).convert('RGBA')
                    if shared.opts.save_mask:
                        images.save_image(image_mask, p.outpath_samples, "", p.seeds[i], p.prompts[i], shared.opts.samples_format, info=text, p=p, suffix="-mask")
                    if shared.opts.save_mask_composite:
                        images.save_image(image_mask_composite, p.outpath_samples, "", p.seeds[i], p.prompts[i], shared.opts.samples_format, info=text, p=p, suffix="-mask-composite")
                    if shared.opts.return_mask:
                        output_images.append(image_mask)
                    if shared.opts.return_mask_composite:
                        output_images.append(image_mask_composite)
            del x_samples_ddim
            devices.torch_gc()

        t1 = time.time()
        shared.log.info(f'Processed: images={len(output_images)} time={t1 - t0:.2f} its={(p.steps * len(output_images)) / (t1 - t0):.2f} memory={modules.memstats.memory_stats()}')

        p.color_corrections = None
        index_of_first_image = 0
        if (shared.opts.return_grid or shared.opts.grid_save) and not p.do_not_save_grid and len(output_images) > 1:
            if images.check_grid_size(output_images):
                grid = images.image_grid(output_images, p.batch_size)
                if shared.opts.return_grid:
                    text = infotext(-1)
                    infotexts.insert(0, text)
                    grid.info["parameters"] = text
                    output_images.insert(0, grid)
                    index_of_first_image = 1
                if shared.opts.grid_save:
                    images.save_image(grid, p.outpath_grids, "", p.all_seeds[0], p.all_prompts[0], shared.opts.grid_format, info=infotext(-1), p=p, grid=True, suffix="-grid") # main save grid

    if not p.disable_extra_networks:
        modules.extra_networks.deactivate(p, extra_network_data)

    res = Processed(
        p,
        images_list=output_images,
        seed=p.all_seeds[0],
        info=infotext(0),
        comments="\n".join(comments),
        subseed=p.all_subseeds[0],
        index_of_first_image=index_of_first_image,
        infotexts=infotexts,
    )
    if p.scripts is not None and isinstance(p.scripts, modules.scripts.ScriptRunner) and not (shared.state.interrupted or shared.state.skipped):
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

    def __init__(self, enable_hr: bool = False, denoising_strength: float = 0.75, firstphase_width: int = 0, firstphase_height: int = 0, hr_scale: float = 2.0, hr_force: bool = False, hr_upscaler: str = None, hr_second_pass_steps: int = 0, hr_resize_x: int = 0, hr_resize_y: int = 0, refiner_steps: int = 5, refiner_start: float = 0, refiner_prompt: str = '', refiner_negative: str = '', **kwargs):

        super().__init__(**kwargs)
        if devices.backend == "ipex" and os.environ.get('DISABLE_IPEX_1024_WA', None) is None:
            width_curse = bool(hr_resize_x == 1024 and self.height * (hr_resize_x / self.width) == 1024)
            height_curse = bool(hr_resize_y == 1024 and self.width * (hr_resize_y / self.height) == 1024)
            if (width_curse != height_curse) or (height_curse and width_curse):
                if width_curse:
                    hr_resize_x = 1080
                if height_curse:
                    hr_resize_y = 1080
            if self.width * hr_scale == 1024 and self.height * hr_scale == 1024:
                hr_scale = 1080 / self.width
            if firstphase_width * hr_scale == 1024 and firstphase_height * hr_scale == 1024:
                hr_scale = 1080 / firstphase_width
        self.enable_hr = enable_hr
        self.denoising_strength = denoising_strength
        self.hr_scale = hr_scale
        self.hr_upscaler = hr_upscaler
        self.hr_force = hr_force
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
        self.refiner_steps = refiner_steps
        self.refiner_start = refiner_start
        self.refiner_prompt = refiner_prompt
        self.refiner_negative = refiner_negative
        self.sampler = None
        self.scripts = None
        self.script_args = []

    def init(self, all_prompts, all_seeds, all_subseeds):
        if shared.backend == shared.Backend.DIFFUSERS:
            shared.sd_model = modules.sd_models.set_diffuser_pipe(self.sd_model, modules.sd_models.DiffusersTaskType.TEXT_2_IMAGE)
        self.width = self.width or 512
        self.height = self.height or 512

    def init_hr(self):
        if self.hr_resize_x == 0 and self.hr_resize_y == 0:
            self.hr_upscale_to_x = int(self.width * self.hr_scale)
            self.hr_upscale_to_y = int(self.height * self.hr_scale)
        else:
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
                self.truncate_x = (self.hr_upscale_to_x - target_w) // 8
                self.truncate_y = (self.hr_upscale_to_y - target_h) // 8
        # special case: the user has chosen to do nothing
        if (self.hr_upscale_to_x == self.width and self.hr_upscale_to_y == self.height) or self.hr_upscaler is None or self.hr_upscaler == 'None':
            self.is_hr_pass = False
            return
        self.is_hr_pass = True
        hypertile_set(self, hr=True)
        shared.state.job_count = 2 * self.n_iter
        shared.log.debug(f'Init hires: upscaler="{self.hr_upscaler}" sampler="{self.hr_sampler_name}" resize={self.hr_resize_x}x{self.hr_resize_y} upscale={self.hr_upscale_to_x}x{self.hr_upscale_to_y}')

    def sample(self, conditioning, unconditional_conditioning, seeds, subseeds, subseed_strength, prompts):

        latent_scale_mode = shared.latent_upscale_modes.get(self.hr_upscaler, None) if self.hr_upscaler is not None else shared.latent_upscale_modes.get(shared.latent_upscale_default_mode, "None")
        if latent_scale_mode is not None:
            self.hr_force = False # no need to force anything
        if self.enable_hr and (latent_scale_mode is None or self.hr_force):
            if len([x for x in shared.sd_upscalers if x.name == self.hr_upscaler]) == 0:
                shared.log.warning(f"Cannot find upscaler for hires: {self.hr_upscaler}")
                self.enable_hr = False

        self.ops.append('txt2img')
        hypertile_set(self)
        self.sampler = modules.sd_samplers.create_sampler(self.sampler_name, self.sd_model)
        if hasattr(self.sampler, "initialize"):
            self.sampler.initialize(self)
        x = create_random_tensors([4, self.height // 8, self.width // 8], seeds=seeds, subseeds=subseeds, subseed_strength=self.subseed_strength, seed_resize_from_h=self.seed_resize_from_h, seed_resize_from_w=self.seed_resize_from_w, p=self)
        samples = self.sampler.sample(self, x, conditioning, unconditional_conditioning, image_conditioning=self.txt2img_image_conditioning(x))
        shared.state.nextjob()
        if not self.enable_hr or shared.state.interrupted or shared.state.skipped:
            return samples

        self.init_hr()
        if self.is_hr_pass:
            prev_job = shared.state.job
            target_width = self.hr_upscale_to_x
            target_height = self.hr_upscale_to_y
            decoded_samples = None
            if shared.opts.save and shared.opts.save_images_before_highres_fix and not self.do_not_save_samples:
                decoded_samples = decode_first_stage(self.sd_model, samples.to(dtype=devices.dtype_vae), self.full_quality)
                decoded_samples = torch.clamp((decoded_samples + 1.0) / 2.0, min=0.0, max=1.0)
                for i, x_sample in enumerate(decoded_samples):
                    x_sample = validate_sample(x_sample)
                    image = Image.fromarray(x_sample)
                    bak_extra_generation_params, bak_restore_faces = self.extra_generation_params, self.restore_faces
                    self.extra_generation_params = {}
                    self.restore_faces = False
                    info = create_infotext(self, self.all_prompts, self.all_seeds, self.all_subseeds, [], iteration=self.iteration, position_in_batch=i)
                    self.extra_generation_params, self.restore_faces = bak_extra_generation_params, bak_restore_faces
                    images.save_image(image, self.outpath_samples, "", seeds[i], prompts[i], shared.opts.samples_format, info=info, suffix="-before-hires")
            if latent_scale_mode is None or self.hr_force: # non-latent upscaling
                shared.state.job = 'upscale'
                if decoded_samples is None:
                    decoded_samples = decode_first_stage(self.sd_model, samples.to(dtype=devices.dtype_vae), self.full_quality)
                    decoded_samples = torch.clamp((decoded_samples + 1.0) / 2.0, min=0.0, max=1.0)
                batch_images = []
                for _i, x_sample in enumerate(decoded_samples):
                    x_sample = validate_sample(x_sample)
                    image = Image.fromarray(x_sample)
                    image = images.resize_image(1, image, target_width, target_height, upscaler_name=self.hr_upscaler)
                    image = np.array(image).astype(np.float32) / 255.0
                    image = np.moveaxis(image, 2, 0)
                    batch_images.append(image)
                resized_samples = torch.from_numpy(np.array(batch_images))
                resized_samples = resized_samples.to(device=shared.device, dtype=devices.dtype_vae)
                resized_samples = 2.0 * resized_samples - 1.0
                if shared.opts.sd_vae_sliced_encode and len(decoded_samples) > 1:
                    samples = torch.stack([self.sd_model.get_first_stage_encoding(self.sd_model.encode_first_stage(torch.unsqueeze(resized_sample, 0)))[0] for resized_sample in resized_samples])
                else:
                    # TODO add TEASD support
                    samples = self.sd_model.get_first_stage_encoding(self.sd_model.encode_first_stage(resized_samples))
                image_conditioning = self.img2img_image_conditioning(resized_samples, samples)
            else:
                samples = torch.nn.functional.interpolate(samples, size=(target_height // 8, target_width // 8), mode=latent_scale_mode["mode"], antialias=latent_scale_mode["antialias"])
                if getattr(self, "inpainting_mask_weight", shared.opts.inpainting_mask_weight) < 1.0:
                    image_conditioning = self.img2img_image_conditioning(decode_first_stage(self.sd_model, samples.to(dtype=devices.dtype_vae), self.full_quality), samples)
                else:
                    image_conditioning = self.txt2img_image_conditioning(samples.to(dtype=devices.dtype_vae))
                if self.hr_sampler_name == "PLMS":
                    self.hr_sampler_name = 'UniPC'
            if self.hr_force or latent_scale_mode is not None:
                shared.state.job = 'hires'
                if self.denoising_strength > 0:
                    self.ops.append('hires')
                    devices.torch_gc() # GC now before running the next img2img to prevent running out of memory
                    self.sampler = modules.sd_samplers.create_sampler(self.hr_sampler_name or self.sampler_name, self.sd_model)
                    if hasattr(self.sampler, "initialize"):
                        self.sampler.initialize(self)
                    samples = samples[:, :, self.truncate_y//2:samples.shape[2]-(self.truncate_y+1)//2, self.truncate_x//2:samples.shape[3]-(self.truncate_x+1)//2]
                    noise = create_random_tensors(samples.shape[1:], seeds=seeds, subseeds=subseeds, subseed_strength=subseed_strength, p=self)
                    modules.sd_models.apply_token_merging(self.sd_model, self.get_token_merging_ratio(for_hr=True))
                    hypertile_set(self, hr=True)
                    samples = self.sampler.sample_img2img(self, samples, noise, conditioning, unconditional_conditioning, steps=self.hr_second_pass_steps or self.steps, image_conditioning=image_conditioning)
                    modules.sd_models.apply_token_merging(self.sd_model, self.get_token_merging_ratio())
                else:
                    self.ops.append('upscale')
            x = None
            self.is_hr_pass = False
            shared.state.job = prev_job
            shared.state.nextjob()

        return samples


class StableDiffusionProcessingImg2Img(StableDiffusionProcessing):

    def __init__(self, init_images: list = None, resize_mode: int = 0, resize_name: str = 'None', denoising_strength: float = 0.3, image_cfg_scale: float = None, mask: Any = None, mask_blur: int = 4, inpainting_fill: int = 0, inpaint_full_res: bool = True, inpaint_full_res_padding: int = 0, inpainting_mask_invert: int = 0, initial_noise_multiplier: float = None, scale_by: float = 1, refiner_steps: int = 5, refiner_start: float = 0, refiner_prompt: str = '', refiner_negative: str = '', **kwargs):
        super().__init__(**kwargs)
        self.init_images = init_images
        self.resize_mode: int = resize_mode
        self.resize_name: str = resize_name
        self.denoising_strength: float = denoising_strength
        self.image_cfg_scale: float = image_cfg_scale
        self.init_latent = None
        self.image_mask = mask
        self.latent_mask = None
        self.mask_for_overlay = None
        self.mask_blur_x = mask_blur # a1111 compatibility item
        self.mask_blur_y = mask_blur # a1111 compatibility item
        self.mask_blur = mask_blur
        self.inpainting_fill = inpainting_fill
        self.inpaint_full_res = inpaint_full_res
        self.inpaint_full_res_padding = inpaint_full_res_padding
        self.inpainting_mask_invert = inpainting_mask_invert
        self.initial_noise_multiplier = shared.opts.initial_noise_multiplier if initial_noise_multiplier is None else initial_noise_multiplier
        self.mask = None
        self.nmask = None
        self.image_conditioning = None
        self.refiner_steps = refiner_steps
        self.refiner_start = refiner_start
        self.refiner_prompt = refiner_prompt
        self.refiner_negative = refiner_negative
        self.enable_hr = None
        self.is_batch = False
        self.scale_by = scale_by
        self.sampler = None
        self.scripts = None
        self.script_args = []

    def init(self, all_prompts, all_seeds, all_subseeds):
        if shared.backend == shared.Backend.DIFFUSERS and self.image_mask is not None and not self.is_control:
            shared.sd_model = modules.sd_models.set_diffuser_pipe(self.sd_model, modules.sd_models.DiffusersTaskType.INPAINTING)
        elif shared.backend == shared.Backend.DIFFUSERS and self.image_mask is None and not self.is_control:
            shared.sd_model = modules.sd_models.set_diffuser_pipe(self.sd_model, modules.sd_models.DiffusersTaskType.IMAGE_2_IMAGE)

        if self.sampler_name == "PLMS":
            self.sampler_name = 'UniPC'
        self.sampler = modules.sd_samplers.create_sampler(self.sampler_name, self.sd_model)
        if hasattr(self.sampler, "initialize"):
            self.sampler.initialize(self)

        if self.image_mask is not None:
            self.ops.append('inpaint')
        else:
            self.ops.append('img2img')
        crop_region = None

        if self.image_mask is not None:
            if type(self.image_mask) == list:
                self.image_mask = self.image_mask[0]
            self.image_mask = create_binary_mask(self.image_mask)
            if self.inpainting_mask_invert:
                self.image_mask = ImageOps.invert(self.image_mask)
            if self.mask_blur > 0:
                np_mask = np.array(self.image_mask)
                kernel_size = 2 * int(2.5 * self.mask_blur + 0.5) + 1
                np_mask = cv2.GaussianBlur(np_mask, (kernel_size, 1), self.mask_blur)
                np_mask = cv2.GaussianBlur(np_mask, (1, kernel_size), self.mask_blur)
                self.image_mask = Image.fromarray(np_mask)
            if self.inpaint_full_res:
                self.mask_for_overlay = self.image_mask
                mask = self.image_mask.convert('L')
                crop_region = modules.masking.get_crop_region(np.array(mask), self.inpaint_full_res_padding)
                crop_region = modules.masking.expand_crop_region(crop_region, self.width, self.height, mask.width, mask.height)
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

        add_color_corrections = shared.opts.img2img_color_correction and self.color_corrections is None
        if add_color_corrections:
            self.color_corrections = []
        processed = []
        if getattr(self, 'init_images', None) is None:
            return
        if not isinstance(self.init_images, list):
            self.init_images = [self.init_images]
        for img in self.init_images:
            if img is None:
                shared.log.warning(f"Skipping empty image: images={self.init_images}")
                continue
            self.init_img_hash = hashlib.sha256(img.tobytes()).hexdigest()[0:8] # pylint: disable=attribute-defined-outside-init
            self.init_img_width = img.width # pylint: disable=attribute-defined-outside-init
            self.init_img_height = img.height # pylint: disable=attribute-defined-outside-init
            if shared.opts.save_init_img:
                images.save_image(img, path=shared.opts.outdir_init_images, basename=None, forced_filename=self.init_img_hash, suffix="-init-image")
            image = images.flatten(img, shared.opts.img2img_background_color)
            if crop_region is None and self.resize_mode != 4 and self.resize_mode > 0:
                if image.width != self.width or image.height != self.height:
                    image = images.resize_image(self.resize_mode, image, self.width, self.height, self.resize_name)
                self.width = image.width
                self.height = image.height
            if self.image_mask is not None:
                try:
                    image_masked = Image.new('RGBa', (image.width, image.height))
                    image_to_paste = image.convert("RGBA").convert("RGBa")
                    image_to_mask = ImageOps.invert(self.mask_for_overlay.convert('L')) if self.mask_for_overlay is not None else None
                    image_to_mask = image_to_mask.resize((image.width, image.height), Image.Resampling.BILINEAR) if image_to_mask is not None else None
                    image_masked.paste(image_to_paste, mask=image_to_mask)
                    self.overlay_images.append(image_masked.convert('RGBA'))
                except Exception as e:
                    shared.log.error(f"Failed to apply mask to image: {e}")
            if crop_region is not None: # crop_region is not None if we are doing inpaint full res
                image = image.crop(crop_region)
                if image.width != self.width or image.height != self.height:
                    image = images.resize_image(3, image, self.width, self.height, self.resize_name)
            if self.image_mask is not None and self.inpainting_fill != 1:
                image = modules.masking.fill(image, latent_mask)
            if add_color_corrections:
                self.color_corrections.append(setup_color_correction(image))
            processed.append(image)
        self.init_images = processed
        self.batch_size = len(self.init_images)
        if self.overlay_images is not None:
            self.overlay_images = self.overlay_images * self.batch_size
        if self.color_corrections is not None and len(self.color_corrections) == 1:
            self.color_corrections = self.color_corrections * self.batch_size
        if shared.backend == shared.Backend.DIFFUSERS:
            return # we've already set self.init_images and self.mask and we dont need any more processing

        self.init_images = [np.moveaxis((np.array(image).astype(np.float32) / 255.0), 2, 0) for image in self.init_images]
        if len(self.init_images) == 1:
            batch_images = np.expand_dims(self.init_images[0], axis=0).repeat(self.batch_size, axis=0)
        elif len(self.init_images) <= self.batch_size:
            batch_images = np.array(self.init_images)
        image = torch.from_numpy(batch_images)
        image = 2. * image - 1.
        image = image.to(device=shared.device, dtype=devices.dtype_vae)
        self.init_latent = self.sd_model.get_first_stage_encoding(self.sd_model.encode_first_stage(image))
        if self.resize_mode == 4:
            self.init_latent = torch.nn.functional.interpolate(self.init_latent, size=(self.height // 8, self.width // 8), mode="bilinear")
        if self.image_mask is not None:
            init_mask = latent_mask
            latmask = init_mask.convert('RGB').resize((self.init_latent.shape[3], self.init_latent.shape[2]))
            latmask = np.moveaxis(np.array(latmask, dtype=np.float32), 2, 0) / 255
            latmask = latmask[0]
            latmask = np.tile(latmask[None], (4, 1, 1))
            latmask = np.around(latmask)
            self.mask = torch.asarray(1.0 - latmask).to(device=shared.device, dtype=self.sd_model.dtype)
            self.nmask = torch.asarray(latmask).to(device=shared.device, dtype=self.sd_model.dtype)
            if self.inpainting_fill == 2:
                self.init_latent = self.init_latent * self.mask + create_random_tensors(self.init_latent.shape[1:], all_seeds[0:self.init_latent.shape[0]]) * self.nmask
            elif self.inpainting_fill == 3:
                self.init_latent = self.init_latent * self.mask
        self.image_conditioning = self.img2img_image_conditioning(image, self.init_latent, self.image_mask)

    def sample(self, conditioning, unconditional_conditioning, seeds, subseeds, subseed_strength, prompts):
        hypertile_set(self)
        x = create_random_tensors([4, self.height // 8, self.width // 8], seeds=seeds, subseeds=subseeds, subseed_strength=self.subseed_strength, seed_resize_from_h=self.seed_resize_from_h, seed_resize_from_w=self.seed_resize_from_w, p=self)
        x *= self.initial_noise_multiplier
        samples = self.sampler.sample_img2img(self, self.init_latent, x, conditioning, unconditional_conditioning, image_conditioning=self.image_conditioning)
        if self.mask is not None:
            samples = samples * self.nmask + self.init_latent * self.mask
        del x
        devices.torch_gc()
        shared.state.nextjob()
        return samples

    def get_token_merging_ratio(self, for_hr=False):
        return self.token_merging_ratio or ("token_merging_ratio" in self.override_settings and shared.opts.token_merging_ratio) or shared.opts.token_merging_ratio_img2img or shared.opts.token_merging_ratio
