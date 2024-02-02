import os
import math
import hashlib
from typing import Any, Dict, List
from dataclasses import dataclass, field
import torch
import numpy as np
import cv2
from PIL import Image, ImageOps
from modules import shared, devices, images, scripts, masking, sd_samplers, sd_models, processing_helpers
from modules.sd_hijack_hypertile import hypertile_set


@dataclass(repr=False)
class StableDiffusionProcessing:
    """
    The first set of paramaters: sd_models -> do_not_reload_embeddings represent the minimum required to create a StableDiffusionProcessing
    """
    def __init__(self, sd_model=None, outpath_samples=None, outpath_grids=None, prompt: str = "", styles: List[str] = None, seed: int = -1, subseed: int = -1, subseed_strength: float = 0, seed_resize_from_h: int = -1, seed_resize_from_w: int = -1, seed_enable_extras: bool = True, sampler_name: str = None, hr_sampler_name: str = None, batch_size: int = 1, n_iter: int = 1, steps: int = 50, cfg_scale: float = 7.0, image_cfg_scale: float = None, clip_skip: int = 1, width: int = 512, height: int = 512, full_quality: bool = True, restore_faces: bool = False, tiling: bool = False, do_not_save_samples: bool = False, do_not_save_grid: bool = False, extra_generation_params: Dict[Any, Any] = None, overlay_images: Any = None, negative_prompt: str = None, eta: float = None, do_not_reload_embeddings: bool = False, denoising_strength: float = 0, diffusers_guidance_rescale: float = 0.7, sag_scale: float = 0.0, cfg_end: float = 1, resize_mode: int = 0, resize_name: str = 'None', scale_by: float = 0, selected_scale_tab: int = 0, hdr_clamp: bool = False, hdr_boundary: float = 4.0, hdr_threshold: float = 3.5, hdr_center: bool = False, hdr_color_correction: float = 0.8, hdr_brightness: float = 0.8, hdr_sharpen: bool = False, hdr_sharpen_ratio: float = 0.0, hdr_sharpen_start: int = 500, hdr_maximize: bool = False, hdr_max_center: float = 0.6, hdr_max_boundry: float = 1.0, override_settings: Dict[str, Any] = None, override_settings_restore_afterwards: bool = True, sampler_index: int = None, script_args: list = None): # pylint: disable=unused-argument
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
        self.cfg_end = cfg_end
        if devices.backend == "ipex" and width == 1024 and height == 1024 and not torch.xpu.has_fp64_dtype() and os.environ.get('DISABLE_IPEX_1024_WA', None) is None:
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
        self.is_using_inpainting_conditioning = False # a111 compatibility
        self.disable_extra_networks = False
        self.token_merging_ratio = 0
        self.token_merging_ratio_hr = 0
        # self.scripts = scripts.ScriptRunner() # set via property
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
        self.scripts_value: scripts.ScriptRunner = field(default=None, init=False)
        self.script_args_value: list = field(default=None, init=False)
        self.scripts_setup_complete: bool = field(default=False, init=False)
        # hdr
        self.hdr_clamp = hdr_clamp
        self.hdr_boundary = hdr_boundary
        self.hdr_threshold = hdr_threshold
        self.hdr_center = hdr_center
        self.hdr_color_correction = hdr_color_correction
        self.hdr_brightness = hdr_brightness
        self.hdr_sharpen = hdr_sharpen
        self.hdr_sharpen_ratio = hdr_sharpen_ratio
        self.hdr_sharpen_start = hdr_sharpen_start
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


class StableDiffusionProcessingTxt2Img(StableDiffusionProcessing):

    def __init__(self, enable_hr: bool = False, denoising_strength: float = 0.75, firstphase_width: int = 0, firstphase_height: int = 0, hr_scale: float = 2.0, hr_force: bool = False, hr_upscaler: str = None, hr_second_pass_steps: int = 0, hr_resize_x: int = 0, hr_resize_y: int = 0, refiner_steps: int = 5, refiner_start: float = 0, refiner_prompt: str = '', refiner_negative: str = '', **kwargs):

        super().__init__(**kwargs)
        if devices.backend == "ipex" and not torch.xpu.has_fp64_dtype() and os.environ.get('DISABLE_IPEX_1024_WA', None) is None:
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
            shared.sd_model = sd_models.set_diffuser_pipe(self.sd_model, sd_models.DiffusersTaskType.TEXT_2_IMAGE)
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
        from modules import processing_original
        return processing_original.sample_txt2img(self, conditioning, unconditional_conditioning, seeds, subseeds, subseed_strength, prompts)


class StableDiffusionProcessingImg2Img(StableDiffusionProcessing):

    def __init__(self, init_images: list = None, resize_mode: int = 0, resize_name: str = 'None', denoising_strength: float = 0.3, image_cfg_scale: float = None, mask: Any = None, mask_blur: int = 4, inpainting_fill: int = 0, inpaint_full_res: bool = False, inpaint_full_res_padding: int = 0, inpainting_mask_invert: int = 0, initial_noise_multiplier: float = None, scale_by: float = 1, refiner_steps: int = 5, refiner_start: float = 0, refiner_prompt: str = '', refiner_negative: str = '', **kwargs):
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
            shared.sd_model = sd_models.set_diffuser_pipe(self.sd_model, sd_models.DiffusersTaskType.INPAINTING)
        elif shared.backend == shared.Backend.DIFFUSERS and self.image_mask is None and not self.is_control:
            shared.sd_model = sd_models.set_diffuser_pipe(self.sd_model, sd_models.DiffusersTaskType.IMAGE_2_IMAGE)

        if self.sampler_name == "PLMS":
            self.sampler_name = 'UniPC'
        self.sampler = sd_samplers.create_sampler(self.sampler_name, self.sd_model)
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
            if shared.backend == shared.Backend.ORIGINAL: # original way of processing mask
                self.image_mask = processing_helpers.create_binary_mask(self.image_mask)
                if self.inpainting_mask_invert:
                    self.image_mask = ImageOps.invert(self.image_mask)
                if self.mask_blur > 0:
                    np_mask = np.array(self.image_mask)
                    kernel_size = 2 * int(2.5 * self.mask_blur + 0.5) + 1
                    np_mask = cv2.GaussianBlur(np_mask, (kernel_size, 1), self.mask_blur)
                    np_mask = cv2.GaussianBlur(np_mask, (1, kernel_size), self.mask_blur)
                    self.image_mask = Image.fromarray(np_mask)
            else:
                if hasattr(self, 'init_images'):
                    self.image_mask = masking.run_mask(
                        input_image=self.init_images,
                        input_mask=self.image_mask,
                        return_type='Grayscale',
                        mask_blur=self.mask_blur,
                        mask_padding=self.inpaint_full_res_padding,
                        segment_enable=False,
                        invert=self.inpainting_mask_invert,
                    )
            if self.inpaint_full_res: # mask only inpaint
                self.mask_for_overlay = self.image_mask
                mask = self.image_mask.convert('L')
                crop_region = masking.get_crop_region(np.array(mask), self.inpaint_full_res_padding)
                crop_region = masking.expand_crop_region(crop_region, self.width, self.height, mask.width, mask.height)
                x1, y1, x2, y2 = crop_region
                crop_mask = mask.crop(crop_region)
                self.image_mask = images.resize_image(2, crop_mask, self.width, self.height)
                self.paste_to = (x1, y1, x2-x1, y2-y1)
            else: # full image inpaint
                self.image_mask = images.resize_image(self.resize_mode, self.image_mask, self.width, self.height)
                np_mask = np.array(self.image_mask)
                np_mask = np.clip((np_mask.astype(np.float32)) * 2, 0, 255).astype(np.uint8)
                self.mask_for_overlay = Image.fromarray(np_mask)
            self.overlay_images = []

        latent_mask = self.latent_mask if self.latent_mask is not None else self.image_mask

        add_color_corrections = shared.opts.img2img_color_correction and self.color_corrections is None
        if add_color_corrections:
            self.color_corrections = []
        processed_images = []
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
            if self.width is None or self.height is None or self.resize_mode == 0:
                self.width, self.height = image.width, image.height
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
                image = masking.fill(image, latent_mask)
            if add_color_corrections:
                self.color_corrections.append(processing_helpers.setup_color_correction(image))
            processed_images.append(image)
        self.init_images = processed_images
        self.batch_size = len(self.init_images)
        if self.overlay_images is not None:
            self.overlay_images = self.overlay_images * self.batch_size
        if self.color_corrections is not None and len(self.color_corrections) == 1:
            self.color_corrections = self.color_corrections * self.batch_size
        if shared.backend == shared.Backend.DIFFUSERS:
            return # we've already set self.init_images and self.mask and we dont need any more processing
        elif shared.backend == shared.Backend.ORIGINAL:
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
                    self.init_latent = self.init_latent * self.mask + processing_helpers.create_random_tensors(self.init_latent.shape[1:], all_seeds[0:self.init_latent.shape[0]]) * self.nmask
                elif self.inpainting_fill == 3:
                    self.init_latent = self.init_latent * self.mask
            self.image_conditioning = processing_helpers.img2img_image_conditioning(self, image, self.init_latent, self.image_mask)

    def sample(self, conditioning, unconditional_conditioning, seeds, subseeds, subseed_strength, prompts):
        from modules import processing_original
        return processing_original.sample_img2img(self, conditioning, unconditional_conditioning, seeds, subseeds, subseed_strength, prompts)

    def get_token_merging_ratio(self, for_hr=False):
        return self.token_merging_ratio or ("token_merging_ratio" in self.override_settings and shared.opts.token_merging_ratio) or shared.opts.token_merging_ratio_img2img or shared.opts.token_merging_ratio

class StableDiffusionProcessingControl(StableDiffusionProcessingImg2Img):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.strength = None
        self.adapter_conditioning_scale = None
        self.adapter_conditioning_factor = None
        self.guess_mode = None
        self.controlnet_conditioning_scale = None
        self.control_guidance_start = None
        self.control_guidance_end = None
        self.reference_attn = None
        self.reference_adain = None
        self.attention_auto_machine_weight = None
        self.gn_auto_machine_weight = None
        self.style_fidelity = None
        self.ref_image = None
        self.image = None
        self.query_weight = None
        self.adain_weight = None
        self.adapter_conditioning_factor = 1.0
        self.attention = 'Attention'
        self.fidelity = 0.5
        self.override = None
        self.ip_adapter_name = None
        self.ip_adapter_scale = 1.0
        self.ip_adapter_image = None

    def sample(self, conditioning, unconditional_conditioning, seeds, subseeds, subseed_strength, prompts): # abstract
        pass

    def init_hr(self):
        if self.resize_name == 'None' or self.scale_by == 1.0:
            return
        self.is_hr_pass = True
        self.hr_force = True
        self.hr_upscaler = self.resize_name
        self.hr_upscale_to_x, self.hr_upscale_to_y = int(self.width * self.scale_by), int(self.height * self.scale_by)
        self.hr_upscale_to_x, self.hr_upscale_to_y = 8 * math.ceil(self.hr_upscale_to_x / 8), 8 * math.ceil(self.hr_upscale_to_y / 8)
        # hypertile_set(self, hr=True)
        shared.state.job_count = 2 * self.n_iter
        shared.log.debug(f'Control hires: upscaler="{self.hr_upscaler}" upscale={self.scale_by} size={self.hr_upscale_to_x}x{self.hr_upscale_to_y}')
