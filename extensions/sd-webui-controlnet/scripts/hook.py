import torch
import hashlib
import numpy as np
import torch.nn as nn
from functools import partial
from typing import Optional, Any

from scripts.logging import logger
from scripts.enums import ControlModelType, AutoMachine, HiResFixOption
from scripts.controlmodel_ipadapter import ImageEmbed
from modules import devices, lowvram, shared, scripts

cond_cast_unet = getattr(devices, 'cond_cast_unet', lambda x: x)

from ldm.modules.diffusionmodules.util import timestep_embedding, make_beta_schedule
from ldm.modules.diffusionmodules.openaimodel import UNetModel
from ldm.modules.attention import BasicTransformerBlock
from ldm.models.diffusion.ddpm import extract_into_tensor

from modules.prompt_parser import MulticondLearnedConditioning, ComposableScheduledPromptConditioning, ScheduledPromptConditioning
from modules.processing import StableDiffusionProcessing


try:
    from sgm.modules.attention import BasicTransformerBlock as BasicTransformerBlockSGM
except:
    print('Warning: ControlNet failed to load SGM - will use LDM instead.')
    BasicTransformerBlockSGM = BasicTransformerBlock


POSITIVE_MARK_TOKEN = 1024
NEGATIVE_MARK_TOKEN = - POSITIVE_MARK_TOKEN
MARK_EPS = 1e-3


def prompt_context_is_marked(x):
    t = x[..., 0, :]
    m = torch.abs(t) - POSITIVE_MARK_TOKEN
    m = torch.mean(torch.abs(m)).detach().cpu().float().numpy()
    return float(m) < MARK_EPS


def mark_prompt_context(x, positive):
    if isinstance(x, list):
        for i in range(len(x)):
            x[i] = mark_prompt_context(x[i], positive)
        return x
    if isinstance(x, MulticondLearnedConditioning):
        x.batch = mark_prompt_context(x.batch, positive)
        return x
    if isinstance(x, ComposableScheduledPromptConditioning):
        x.schedules = mark_prompt_context(x.schedules, positive)
        return x
    if isinstance(x, ScheduledPromptConditioning):
        if isinstance(x.cond, dict):
            cond = x.cond['crossattn']
            if prompt_context_is_marked(cond):
                return x
            mark = POSITIVE_MARK_TOKEN if positive else NEGATIVE_MARK_TOKEN
            cond = torch.cat([torch.zeros_like(cond)[:1] + mark, cond], dim=0)
            return ScheduledPromptConditioning(end_at_step=x.end_at_step, cond=dict(crossattn=cond, vector=x.cond['vector']))
        else:
            cond = x.cond
            if prompt_context_is_marked(cond):
                return x
            mark = POSITIVE_MARK_TOKEN if positive else NEGATIVE_MARK_TOKEN
            cond = torch.cat([torch.zeros_like(cond)[:1] + mark, cond], dim=0)
            return ScheduledPromptConditioning(end_at_step=x.end_at_step, cond=cond)
    return x


disable_controlnet_prompt_warning = True
# You can disable this warning using disable_controlnet_prompt_warning.


def unmark_prompt_context(x):
    if not prompt_context_is_marked(x):
        # ControlNet must know whether a prompt is conditional prompt (positive prompt) or unconditional conditioning prompt (negative prompt).
        # You can use the hook.py's `mark_prompt_context` to mark the prompts that will be seen by ControlNet.
        # Let us say XXX is a MulticondLearnedConditioning or a ComposableScheduledPromptConditioning or a ScheduledPromptConditioning or a list of these components,
        # if XXX is a positive prompt, you should call mark_prompt_context(XXX, positive=True)
        # if XXX is a negative prompt, you should call mark_prompt_context(XXX, positive=False)
        # After you mark the prompts, the ControlNet will know which prompt is cond/uncond and works as expected.
        # After you mark the prompts, the mismatch errors will disappear.
        if not disable_controlnet_prompt_warning:
            logger.warning('ControlNet Error: Failed to detect whether an instance is cond or uncond!')
            logger.warning('ControlNet Error: This is mainly because other extension(s) blocked A1111\'s \"process.sample()\" and deleted ControlNet\'s sample function.')
            logger.warning('ControlNet Error: ControlNet will shift to a backup backend but the results will be worse than expectation.')
            logger.warning('Solution (For extension developers): Take a look at ControlNet\' hook.py '
                  'UnetHook.hook.process_sample and manually call mark_prompt_context to mark cond/uncond prompts.')
        mark_batch = torch.ones(size=(x.shape[0], 1, 1, 1), dtype=x.dtype, device=x.device)
        context = x
        return mark_batch, [], [], context
    mark = x[:, 0, :]
    context = x[:, 1:, :]
    mark = torch.mean(torch.abs(mark - NEGATIVE_MARK_TOKEN), dim=1)
    mark = (mark > MARK_EPS).float()
    mark_batch = mark[:, None, None, None].to(x.dtype).to(x.device)

    mark = mark.detach().cpu().numpy().tolist()
    uc_indices = [i for i, item in enumerate(mark) if item < 0.5]
    c_indices = [i for i, item in enumerate(mark) if not item < 0.5]

    StableDiffusionProcessing.cached_c = [None, None]
    StableDiffusionProcessing.cached_uc = [None, None]

    return mark_batch, uc_indices, c_indices, context


class HackedImageRNG:
    def __init__(self, rng, noise_modifier, sd_model):
        self.rng = rng
        self.noise_modifier = noise_modifier
        self.sd_model = sd_model

    def next(self):
        result = self.rng.next()
        x0 = self.noise_modifier
        if result.shape[2] != x0.shape[2] or result.shape[3] != x0.shape[3]:
            return result
        x0 = x0.to(result.dtype).to(result.device)
        ts = torch.tensor([999] * result.shape[0]).long().to(result.device)
        result = predict_q_sample(self.sd_model, x0, ts, result)
        logger.info(f'[ControlNet] Initial noise hack applied to {result.shape}.')
        return result


class TorchHijackForUnet:
    """
    This is torch, but with cat that resizes tensors to appropriate dimensions if they do not match;
    this makes it possible to create pictures with dimensions that are multiples of 8 rather than 64
    """

    def __getattr__(self, item):
        if item == 'cat':
            return self.cat

        if hasattr(torch, item):
            return getattr(torch, item)

        raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, item))

    def cat(self, tensors, *args, **kwargs):
        if len(tensors) == 2:
            a, b = tensors
            if a.shape[-2:] != b.shape[-2:]:
                a = torch.nn.functional.interpolate(a, b.shape[-2:], mode="nearest")

            tensors = (a, b)

        return torch.cat(tensors, *args, **kwargs)


th = TorchHijackForUnet()


class ControlParams:
    def __init__(
            self,
            control_model,
            preprocessor,
            hint_cond,
            weight,
            guidance_stopped,
            start_guidance_percent,
            stop_guidance_percent,
            advanced_weighting,
            control_model_type,
            hr_hint_cond,
            global_average_pooling,
            soft_injection,
            cfg_injection,
            hr_option: HiResFixOption = HiResFixOption.BOTH,
            control_context_override: Optional[Any] = None,
            **kwargs  # To avoid errors
    ):
        self.control_model = control_model
        self.preprocessor = preprocessor
        self._hint_cond = hint_cond
        self.weight = weight
        self.guidance_stopped = guidance_stopped
        self.start_guidance_percent = start_guidance_percent
        self.stop_guidance_percent = stop_guidance_percent
        self.advanced_weighting = advanced_weighting
        self.control_model_type = control_model_type
        self.global_average_pooling = global_average_pooling
        self.hr_hint_cond = hr_hint_cond
        self.hr_option = hr_option
        self.control_context_override = control_context_override
        self.used_hint_cond = None
        self.used_hint_cond_latent = None
        self.used_hint_inpaint_hijack = None
        self.soft_injection = soft_injection
        self.cfg_injection = cfg_injection
        self.vision_hint_count = None

    @property
    def hint_cond(self):
        return self._hint_cond

    # fix for all the extensions that modify hint_cond,
    # by forcing used_hint_cond to update on the next timestep
    # hr_hint_cond can stay the same, since most extensions dont modify the hires pass
    # but if they do, it will cause problems
    @hint_cond.setter
    def hint_cond(self, new_hint_cond):
        self._hint_cond = new_hint_cond
        self.used_hint_cond = None
        self.used_hint_cond_latent = None
        self.used_hint_inpaint_hijack = None

    def disabled_by_hr_option(self, is_in_high_res_fix: bool) -> bool:
        if self.hr_option == HiResFixOption.BOTH:
            control_disabled = False
        elif self.hr_option == HiResFixOption.LOW_RES_ONLY:
            control_disabled = is_in_high_res_fix
        elif self.hr_option == HiResFixOption.HIGH_RES_ONLY:
            control_disabled = not is_in_high_res_fix
        else:
            assert False, "NOTREACHED"
        return control_disabled


def aligned_adding(base, x, require_channel_alignment):
    if isinstance(x, float):
        if x == 0.0:
            return base
        return base + x

    if require_channel_alignment:
        zeros = torch.zeros_like(base)
        zeros[:, :x.shape[1], ...] = x
        x = zeros

    # resize to sample resolution
    base_h, base_w = base.shape[-2:]
    xh, xw = x.shape[-2:]

    if xh > 1 or xw > 1:
        if base_h != xh or base_w != xw:
            # logger.info('[Warning] ControlNet finds unexpected mis-alignment in tensor shape.')
            x = th.nn.functional.interpolate(x, size=(base_h, base_w), mode="nearest")

    return base + x


# DFS Search for Torch.nn.Module, Written by Lvmin
def torch_dfs(model: torch.nn.Module):
    result = [model]
    for child in model.children():
        result += torch_dfs(child)
    return result


class AbstractLowScaleModel(nn.Module):
    def __init__(self):
        super(AbstractLowScaleModel, self).__init__()
        self.register_schedule()

    def register_schedule(self, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end,
                                   cosine_s=cosine_s)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        return (extract_into_tensor(self.sqrt_alphas_cumprod.to(x_start), t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod.to(x_start), t, x_start.shape) * noise)


def register_schedule(self):
    linear_start = 0.00085
    linear_end = 0.0120
    num_timesteps = 1000

    betas = (torch.linspace(linear_start ** 0.5, linear_end ** 0.5, num_timesteps, dtype=torch.float64) ** 2.0).numpy()

    alphas = 1. - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

    to_torch = partial(torch.tensor, dtype=torch.float32)

    setattr(self, 'betas', to_torch(betas))
    # setattr(self, 'alphas_cumprod', to_torch(alphas_cumprod))  # a1111 already has this
    setattr(self, 'alphas_cumprod_prev', to_torch(alphas_cumprod_prev))
    setattr(self, 'sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
    setattr(self, 'sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
    setattr(self, 'log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
    setattr(self, 'sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
    setattr(self, 'sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))


def predict_q_sample(ldm, x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)
    return extract_into_tensor(ldm.sqrt_alphas_cumprod.to(x_start), t, x_start.shape) * x_start + extract_into_tensor(ldm.sqrt_one_minus_alphas_cumprod.to(x_start), t, x_start.shape) * noise


def predict_start_from_noise(ldm, x_t, t, noise):
    return extract_into_tensor(ldm.sqrt_recip_alphas_cumprod.to(x_t), t, x_t.shape) * x_t - extract_into_tensor(ldm.sqrt_recipm1_alphas_cumprod.to(x_t), t, x_t.shape) * noise


def predict_noise_from_start(ldm, x_t, t, x0):
    return (extract_into_tensor(ldm.sqrt_recip_alphas_cumprod.to(x_t), t, x_t.shape) * x_t - x0) / extract_into_tensor(ldm.sqrt_recipm1_alphas_cumprod.to(x_t), t, x_t.shape)


def blur(x, k):
    y = torch.nn.functional.pad(x, (k, k, k, k), mode='replicate')
    y = torch.nn.functional.avg_pool2d(y, (k*2+1, k*2+1), stride=(1, 1))
    return y


class TorchCache:
    def __init__(self):
        self.cache = {}

    def hash(self, key):
        v = key.detach().cpu().numpy().astype(np.float32)
        v = (v * 1000.0).astype(np.int32)
        v = np.ascontiguousarray(v.copy())
        sha = hashlib.sha1(v).hexdigest()
        return sha

    def get(self, key):
        key = self.hash(key)
        return self.cache.get(key, None)

    def set(self, key, value):
        self.cache[self.hash(key)] = value


class UnetHook(nn.Module):
    def __init__(self, lowvram=False) -> None:
        super().__init__()
        self.lowvram = lowvram
        self.model = None
        self.sd_ldm = None
        self.control_params = None
        self.attention_auto_machine = AutoMachine.Read
        self.attention_auto_machine_weight = 1.0
        self.gn_auto_machine = AutoMachine.Read
        self.gn_auto_machine_weight = 1.0
        self.current_style_fidelity = 0.0
        self.current_uc_indices = []
        self.current_c_indices = []
        self.is_in_high_res_fix = False

    @staticmethod
    def call_vae_using_process(p, x, batch_size=None, mask=None):
        vae_cache = getattr(p, 'controlnet_vae_cache', None)
        if vae_cache is None:
            vae_cache = TorchCache()
            setattr(p, 'controlnet_vae_cache', vae_cache)
        try:
            if x.shape[1] > 3:
                x = x[:, 0:3, :, :]
            x = x * 2.0 - 1.0
            if mask is not None:
                x = x * (1.0 - mask)
            x = x.type(devices.dtype_vae)
            vae_output = vae_cache.get(x)
            if vae_output is None:
                with devices.autocast():
                    vae_output = p.sd_model.encode_first_stage(x)
                    vae_output = p.sd_model.get_first_stage_encoding(vae_output)
                    if torch.all(torch.isnan(vae_output)).item():
                        logger.info(f'ControlNet find Nans in the VAE encoding. \n '
                                    f'Now ControlNet will automatically retry.\n '
                                    f'To always start with 32-bit VAE, use --no-half-vae commandline flag.')
                        devices.dtype_vae = torch.float32
                        x = x.to(devices.dtype_vae)
                        p.sd_model.first_stage_model.to(devices.dtype_vae)
                        vae_output = p.sd_model.encode_first_stage(x)
                        vae_output = p.sd_model.get_first_stage_encoding(vae_output)
                vae_cache.set(x, vae_output)
                logger.info(f'ControlNet used {str(devices.dtype_vae)} VAE to encode {vae_output.shape}.')
            latent = vae_output
            if batch_size is not None and latent.shape[0] != batch_size:
                latent = torch.cat([latent.clone() for _ in range(batch_size)], dim=0)
            latent = latent.type(devices.dtype_unet)
            return latent
        except Exception as e:
            logger.error(e)
            raise ValueError('ControlNet failed to use VAE. Please try to add `--no-half-vae`, `--no-half` and remove `--precision full` in launch cmd.')

    def guidance_schedule_handler(self, x):
        for param in self.control_params:
            current_sampling_percent = (x.sampling_step / x.total_sampling_steps)
            param.guidance_stopped = current_sampling_percent < param.start_guidance_percent or current_sampling_percent > param.stop_guidance_percent
            if self.model is not None:
                self.model.current_sampling_percent = current_sampling_percent

    def hook(self, model, sd_ldm, control_params, process, batch_option_uint_separate=False, batch_option_style_align=False):
        self.model = model
        self.sd_ldm = sd_ldm
        self.control_params = control_params

        model_is_sdxl = getattr(self.sd_ldm, 'is_sdxl', False)

        outer = self

        def process_sample(*args, **kwargs):
            # ControlNet must know whether a prompt is conditional prompt (positive prompt) or unconditional conditioning prompt (negative prompt).
            # You can use the hook.py's `mark_prompt_context` to mark the prompts that will be seen by ControlNet.
            # Let us say XXX is a MulticondLearnedConditioning or a ComposableScheduledPromptConditioning or a ScheduledPromptConditioning or a list of these components,
            # if XXX is a positive prompt, you should call mark_prompt_context(XXX, positive=True)
            # if XXX is a negative prompt, you should call mark_prompt_context(XXX, positive=False)
            # After you mark the prompts, the ControlNet will know which prompt is cond/uncond and works as expected.
            # After you mark the prompts, the mismatch errors will disappear.
            mark_prompt_context(kwargs.get('conditioning', []), positive=True)
            mark_prompt_context(kwargs.get('unconditional_conditioning', []), positive=False)
            mark_prompt_context(getattr(process, 'hr_c', []), positive=True)
            mark_prompt_context(getattr(process, 'hr_uc', []), positive=False)
            return process.sample_before_CN_hack(*args, **kwargs)

        def forward(self, x, timesteps=None, context=None, y=None, **kwargs):
            is_sdxl = y is not None and model_is_sdxl
            total_t2i_adapter_embedding = [0.0] * 4
            if is_sdxl:
                total_controlnet_embedding = [0.0] * 10
            else:
                total_controlnet_embedding = [0.0] * 13
            require_inpaint_hijack = False
            is_in_high_res_fix = False
            batch_size = int(x.shape[0])

            # Handle cond-uncond marker
            cond_mark, outer.current_uc_indices, outer.current_c_indices, context = unmark_prompt_context(context)
            outer.model.cond_mark = cond_mark
            # logger.info(str(cond_mark[:, 0, 0, 0].detach().cpu().numpy().tolist()) + ' - ' + str(outer.current_uc_indices))

            # Revision
            if is_sdxl:
                revision_y1280 = 0

                for param in outer.control_params:
                    if param.guidance_stopped:
                        continue
                    if param.control_model_type == ControlModelType.ReVision:
                        if param.vision_hint_count is None:
                            k = torch.Tensor([int(param.preprocessor['threshold_a'] * 1000)]).to(param.hint_cond).long().clip(0, 999)
                            param.vision_hint_count = outer.revision_q_sampler.q_sample(param.hint_cond, k)
                        revision_emb = param.vision_hint_count
                        if isinstance(revision_emb, torch.Tensor):
                            revision_y1280 += revision_emb * param.weight

                if isinstance(revision_y1280, torch.Tensor):
                    y[:, :1280] = revision_y1280 * cond_mark[:, :, 0, 0]
                    if any('ignore_prompt' in param.preprocessor['name'] for param in outer.control_params) \
                            or (getattr(process, 'prompt', '') == '' and getattr(process, 'negative_prompt', '') == ''):
                        context = torch.zeros_like(context)

            # High-res fix
            for param in outer.control_params:
                # select which hint_cond to use
                if param.used_hint_cond is None:
                    param.used_hint_cond = param.hint_cond
                    param.used_hint_cond_latent = None
                    param.used_hint_inpaint_hijack = None

                # has high-res fix
                if isinstance(param.hr_hint_cond, torch.Tensor) and x.ndim == 4 and param.hint_cond.ndim == 4 and param.hr_hint_cond.ndim == 4:
                    _, _, h_lr, w_lr = param.hint_cond.shape
                    _, _, h_hr, w_hr = param.hr_hint_cond.shape
                    _, _, h, w = x.shape
                    h, w = h * 8, w * 8
                    if abs(h - h_lr) < abs(h - h_hr):
                        is_in_high_res_fix = False
                        if param.used_hint_cond is not param.hint_cond:
                            param.used_hint_cond = param.hint_cond
                            param.used_hint_cond_latent = None
                            param.used_hint_inpaint_hijack = None
                    else:
                        is_in_high_res_fix = True
                        if param.used_hint_cond is not param.hr_hint_cond:
                            param.used_hint_cond = param.hr_hint_cond
                            param.used_hint_cond_latent = None
                            param.used_hint_inpaint_hijack = None

            self.is_in_high_res_fix = is_in_high_res_fix
            outer.is_in_high_res_fix = is_in_high_res_fix

            # Convert control image to latent
            for param in outer.control_params:
                if param.used_hint_cond_latent is not None:
                    continue
                if param.control_model_type not in [ControlModelType.AttentionInjection] \
                        and 'colorfix' not in param.preprocessor['name'] \
                        and 'inpaint_only' not in param.preprocessor['name']:
                    continue
                param.used_hint_cond_latent = outer.call_vae_using_process(process, param.used_hint_cond, batch_size=batch_size)

            # vram
            for param in outer.control_params:
                if getattr(param.control_model, 'disable_memory_management', False):
                    continue

                if param.control_model is not None:
                    if outer.lowvram and is_sdxl and hasattr(param.control_model, 'aggressive_lowvram'):
                        param.control_model.aggressive_lowvram()
                    elif hasattr(param.control_model, 'fullvram'):
                        param.control_model.fullvram()
                    elif hasattr(param.control_model, 'to'):
                        param.control_model.to(devices.get_device_for("controlnet"))

            # handle prompt token control
            for param in outer.control_params:
                if param.guidance_stopped or param.disabled_by_hr_option(self.is_in_high_res_fix):
                    continue

                if param.control_model_type not in [ControlModelType.T2I_StyleAdapter]:
                    continue

                control = param.control_model(x=x, hint=param.used_hint_cond, timesteps=timesteps, context=context)
                control = torch.cat([control.clone() for _ in range(batch_size)], dim=0)
                control *= param.weight
                control *= cond_mark[:, :, :, 0]
                context = torch.cat([context, control.clone()], dim=1)

            # handle ControlNet / T2I_Adapter
            for param_index, param in enumerate(outer.control_params):
                if param.guidance_stopped or param.disabled_by_hr_option(self.is_in_high_res_fix):
                    continue

                if not (
                    param.control_model_type.is_controlnet() or
                    param.control_model_type == ControlModelType.T2I_Adapter
                ):
                    continue

                # inpaint model workaround
                x_in = x
                control_model = param.control_model.control_model

                if param.control_model_type.is_controlnet():
                    if x.shape[1] != control_model.input_blocks[0][0].in_channels and x.shape[1] == 9:
                        # inpaint_model: 4 data + 4 downscaled image + 1 mask
                        x_in = x[:, :4, ...]
                        require_inpaint_hijack = True

                assert param.used_hint_cond is not None, f"Controlnet is enabled but no input image is given"

                hint = param.used_hint_cond
                if param.control_model_type == ControlModelType.InstantID:
                    assert isinstance(param.control_context_override, ImageEmbed)
                    controlnet_context = param.control_context_override.eval(cond_mark).to(x.device, dtype=x.dtype)
                else:
                    controlnet_context = context

                # ControlNet inpaint protocol
                if hint.shape[1] == 4:
                    c = hint[:, 0:3, :, :]
                    m = hint[:, 3:4, :, :]
                    m = (m > 0.5).float()
                    hint = c * (1 - m) - m

                control = param.control_model(
                    x=x_in,
                    hint=hint,
                    timesteps=timesteps,
                    context=controlnet_context,
                    y=y
                )

                if is_sdxl:
                    control_scales = [param.weight] * 10
                else:
                    control_scales = [param.weight] * 13

                if param.cfg_injection or param.global_average_pooling:
                    if param.control_model_type == ControlModelType.T2I_Adapter:
                        control = [torch.cat([c.clone() for _ in range(batch_size)], dim=0) for c in control]
                    control = [c * cond_mark for c in control]

                high_res_fix_forced_soft_injection = False

                if is_in_high_res_fix:
                    if 'canny' in param.preprocessor['name']:
                        high_res_fix_forced_soft_injection = True
                    if 'mlsd' in param.preprocessor['name']:
                        high_res_fix_forced_soft_injection = True

                # if high_res_fix_forced_soft_injection:
                #     logger.info('[ControlNet] Forced soft_injection in high_res_fix in enabled.')

                if param.soft_injection or high_res_fix_forced_soft_injection:
                    # important! use the soft weights with high-res fix can significantly reduce artifacts.
                    if param.control_model_type == ControlModelType.T2I_Adapter:
                        control_scales = [param.weight * x for x in (0.25, 0.62, 0.825, 1.0)]
                    elif param.control_model_type.is_controlnet():
                        control_scales = [param.weight * (0.825 ** float(12 - i)) for i in range(13)]

                if is_sdxl and param.control_model_type.is_controlnet():
                    control_scales = control_scales[:10]

                if param.advanced_weighting is not None:
                    logger.info(f"Advanced weighting enabled. {param.advanced_weighting}")
                    if param.soft_injection or high_res_fix_forced_soft_injection:
                        logger.warn("Advanced weighting overwrites soft_injection effect.")
                    control_scales = param.advanced_weighting

                control = [c * scale for c, scale in zip(control, control_scales)]
                if param.global_average_pooling:
                    control = [torch.mean(c, dim=(2, 3), keepdim=True) for c in control]

                for idx, item in enumerate(control):
                    target = None
                    if param.control_model_type.is_controlnet():
                        target = total_controlnet_embedding
                    if param.control_model_type == ControlModelType.T2I_Adapter:
                        target = total_t2i_adapter_embedding
                    if target is not None:
                        if batch_option_uint_separate:
                            for pi, ci in enumerate(outer.current_c_indices):
                                if pi % len(outer.control_params) != param_index:
                                    item[ci] = 0
                            for pi, ci in enumerate(outer.current_uc_indices):
                                if pi % len(outer.control_params) != param_index:
                                    item[ci] = 0
                            target[idx] = item + target[idx]
                        else:
                            target[idx] = item + target[idx]

            # Replace x_t to support inpaint models
            for param in outer.control_params:
                if not isinstance(param.used_hint_cond, torch.Tensor):
                    continue
                if param.used_hint_cond.ndim < 2 or param.used_hint_cond.shape[1] != 4:
                    continue
                if x.shape[1] != 9:
                    continue
                if param.used_hint_inpaint_hijack is None:
                    mask_pixel = param.used_hint_cond[:, 3:4, :, :]
                    image_pixel = param.used_hint_cond[:, 0:3, :, :]
                    mask_pixel = (mask_pixel > 0.5).to(mask_pixel.dtype)
                    masked_latent = outer.call_vae_using_process(process, image_pixel, batch_size, mask=mask_pixel)
                    mask_latent = torch.nn.functional.max_pool2d(mask_pixel, (8, 8))
                    if mask_latent.shape[0] != batch_size:
                        mask_latent = torch.cat([mask_latent.clone() for _ in range(batch_size)], dim=0)
                    param.used_hint_inpaint_hijack = torch.cat([mask_latent, masked_latent], dim=1)
                    param.used_hint_inpaint_hijack.to(x.dtype).to(x.device)
                x = torch.cat([x[:, :4, :, :], param.used_hint_inpaint_hijack], dim=1)

            # vram
            for param in outer.control_params:
                if param.control_model is not None:
                    if outer.lowvram:
                        param.control_model.to('cpu')

            # A1111 fix for medvram.
            if shared.cmd_opts.medvram or (getattr(shared.cmd_opts, 'medvram_sdxl', False) and is_sdxl):
                try:
                    # Trigger the register_forward_pre_hook
                    outer.sd_ldm.model()
                except:
                    pass

            # Clear attention and AdaIn cache
            for module in outer.attn_module_list:
                module.bank = []
                module.style_cfgs = []
            for module in outer.gn_module_list:
                module.mean_bank = []
                module.var_bank = []
                module.style_cfgs = []

            # Handle attention and AdaIn control
            for param in outer.control_params:
                if param.guidance_stopped or param.disabled_by_hr_option(self.is_in_high_res_fix):
                    continue

                if param.used_hint_cond_latent is None:
                    continue

                if param.control_model_type not in [ControlModelType.AttentionInjection]:
                    continue

                ref_xt = predict_q_sample(outer.sd_ldm, param.used_hint_cond_latent, torch.round(timesteps.float()).long())

                # Inpaint Hijack
                if x.shape[1] == 9:
                    ref_xt = torch.cat([
                        ref_xt,
                        torch.zeros_like(ref_xt)[:, 0:1, :, :],
                        param.used_hint_cond_latent
                    ], dim=1)

                outer.current_style_fidelity = float(param.preprocessor['threshold_a'])
                outer.current_style_fidelity = max(0.0, min(1.0, outer.current_style_fidelity))

                if is_sdxl:
                    # sdxl's attention hacking is highly unstable.
                    # We have no other methods but to reduce the style_fidelity a bit.
                    # By default, 0.5 ** 3.0 = 0.125
                    outer.current_style_fidelity = outer.current_style_fidelity ** 3.0

                if param.cfg_injection:
                    outer.current_style_fidelity = 1.0
                elif param.soft_injection or is_in_high_res_fix:
                    outer.current_style_fidelity = 0.0

                control_name = param.preprocessor['name']

                if control_name in ['reference_only', 'reference_adain+attn']:
                    outer.attention_auto_machine = AutoMachine.Write
                    outer.attention_auto_machine_weight = param.weight

                if control_name in ['reference_adain', 'reference_adain+attn']:
                    outer.gn_auto_machine = AutoMachine.Write
                    outer.gn_auto_machine_weight = param.weight

                if is_sdxl:
                    outer.original_forward(
                        x=ref_xt.to(devices.dtype_unet),
                        timesteps=timesteps.to(devices.dtype_unet),
                        context=context.to(devices.dtype_unet),
                        y=y
                    )
                else:
                    outer.original_forward(
                        x=ref_xt.to(devices.dtype_unet),
                        timesteps=timesteps.to(devices.dtype_unet),
                        context=context.to(devices.dtype_unet)
                    )

                outer.attention_auto_machine = AutoMachine.Read
                outer.gn_auto_machine = AutoMachine.Read

            # U-Net Encoder
            hs = []
            with th.no_grad():
                t_emb = cond_cast_unet(timestep_embedding(timesteps, self.model_channels, repeat_only=False))
                emb = self.time_embed(t_emb)

                if is_sdxl:
                    assert y.shape[0] == x.shape[0]
                    emb = emb + self.label_emb(y)

                h = x
                for i, module in enumerate(self.input_blocks):
                    self.current_h_shape = (h.shape[0], h.shape[1], h.shape[2], h.shape[3])
                    h = module(h, emb, context)

                    t2i_injection = [3, 5, 8] if is_sdxl else [2, 5, 8, 11]

                    if i in t2i_injection:
                        h = aligned_adding(h, total_t2i_adapter_embedding.pop(0), require_inpaint_hijack)

                    hs.append(h)

                self.current_h_shape = (h.shape[0], h.shape[1], h.shape[2], h.shape[3])
                h = self.middle_block(h, emb, context)

            # U-Net Middle Block
            h = aligned_adding(h, total_controlnet_embedding.pop(), require_inpaint_hijack)

            if len(total_t2i_adapter_embedding) > 0 and is_sdxl:
                h = aligned_adding(h, total_t2i_adapter_embedding.pop(0), require_inpaint_hijack)

            # U-Net Decoder
            for i, module in enumerate(self.output_blocks):
                self.current_h_shape = (h.shape[0], h.shape[1], h.shape[2], h.shape[3])
                h = th.cat([h, aligned_adding(hs.pop(), total_controlnet_embedding.pop(), require_inpaint_hijack)], dim=1)
                h = module(h, emb, context)

            # U-Net Output
            h = h.type(x.dtype)
            h = self.out(h)

            # Post-processing for color fix
            for param in outer.control_params:
                if param.used_hint_cond_latent is None:
                    continue
                if 'colorfix' not in param.preprocessor['name']:
                    continue

                k = int(param.preprocessor['threshold_a'])
                if is_in_high_res_fix and not param.disabled_by_hr_option(self.is_in_high_res_fix):
                    k *= 2

                # Inpaint hijack
                xt = x[:, :4, :, :]

                x0_origin = param.used_hint_cond_latent
                t = torch.round(timesteps.float()).long()
                x0_prd = predict_start_from_noise(outer.sd_ldm, xt, t, h)
                x0 = x0_prd - blur(x0_prd, k) + blur(x0_origin, k)

                if '+sharp' in param.preprocessor['name']:
                    detail_weight = float(param.preprocessor['threshold_b']) * 0.01
                    neg = detail_weight * blur(x0, k) + (1 - detail_weight) * x0
                    x0 = cond_mark * x0 + (1 - cond_mark) * neg

                eps_prd = predict_noise_from_start(outer.sd_ldm, xt, t, x0)

                w = max(0.0, min(1.0, float(param.weight)))
                h = eps_prd * w + h * (1 - w)

            # Post-processing for restore
            for param in outer.control_params:
                if param.used_hint_cond_latent is None:
                    continue
                if 'inpaint_only' not in param.preprocessor['name']:
                    continue
                if param.used_hint_cond.shape[1] != 4:
                    continue

                # Inpaint hijack
                xt = x[:, :4, :, :]

                mask = param.used_hint_cond[:, 3:4, :, :]
                mask = torch.nn.functional.max_pool2d(mask, (10, 10), stride=(8, 8), padding=1)

                x0_origin = param.used_hint_cond_latent
                t = torch.round(timesteps.float()).long()
                x0_prd = predict_start_from_noise(outer.sd_ldm, xt, t, h)
                x0 = x0_prd * mask + x0_origin * (1 - mask)
                eps_prd = predict_noise_from_start(outer.sd_ldm, xt, t, x0)

                w = max(0.0, min(1.0, float(param.weight)))
                h = eps_prd * w + h * (1 - w)

            return h

        def move_all_control_model_to_cpu():
            for param in getattr(outer, 'control_params', []) or []:
                if isinstance(param.control_model, torch.nn.Module):
                    param.control_model.to("cpu")

        def forward_webui(*args, **kwargs):
            # webui will handle other compoments 
            try:
                if shared.cmd_opts.lowvram:
                    lowvram.send_everything_to_cpu()
                return forward(*args, **kwargs)
            except Exception as e:
                move_all_control_model_to_cpu()
                raise e
            finally:
                if outer.lowvram:
                    move_all_control_model_to_cpu()

        def hacked_basic_transformer_inner_forward(self, x, context=None):
            x_norm1 = self.norm1(x)
            self_attn1 = None
            if self.disable_self_attn:
                # Do not use self-attention
                self_attn1 = self.attn1(x_norm1, context=context)
            else:
                # Use self-attention
                self_attention_context = x_norm1
                if outer.attention_auto_machine == AutoMachine.Write:
                    if outer.attention_auto_machine_weight > self.attn_weight:
                        self.bank.append(self_attention_context.detach().clone())
                        self.style_cfgs.append(outer.current_style_fidelity)
                if outer.attention_auto_machine == AutoMachine.Read:
                    if len(self.bank) > 0:
                        style_cfg = sum(self.style_cfgs) / float(len(self.style_cfgs))
                        self_attn1_uc = self.attn1(x_norm1, context=torch.cat([self_attention_context] + self.bank, dim=1))
                        self_attn1_c = self_attn1_uc.clone()
                        if len(outer.current_uc_indices) > 0 and style_cfg > 1e-5:
                            self_attn1_c[outer.current_uc_indices] = self.attn1(
                                x_norm1[outer.current_uc_indices],
                                context=self_attention_context[outer.current_uc_indices])
                        self_attn1 = style_cfg * self_attn1_c + (1.0 - style_cfg) * self_attn1_uc
                    self.bank = []
                    self.style_cfgs = []
                if outer.attention_auto_machine == AutoMachine.StyleAlign and not outer.is_in_high_res_fix:
                    # very VRAM hungry - disable at high_res_fix

                    def shared_attn1(inner_x):
                        BB, FF, CC = inner_x.shape
                        return self.attn1(inner_x.reshape(1, BB * FF, CC)).reshape(BB, FF, CC)

                    uc_layer = shared_attn1(x_norm1[outer.current_uc_indices])
                    c_layer = shared_attn1(x_norm1[outer.current_c_indices])
                    self_attn1 = torch.zeros_like(x_norm1).to(uc_layer)
                    self_attn1[outer.current_uc_indices] = uc_layer
                    self_attn1[outer.current_c_indices] = c_layer
                    del uc_layer, c_layer
                if self_attn1 is None:
                    self_attn1 = self.attn1(x_norm1, context=self_attention_context)

            x = self_attn1.to(x.dtype) + x
            x = self.attn2(self.norm2(x), context=context) + x
            x = self.ff(self.norm3(x)) + x
            return x

        def hacked_group_norm_forward(self, *args, **kwargs):
            eps = 1e-6
            x = self.original_forward_cn_hijack(*args, **kwargs)
            y = None
            if outer.gn_auto_machine == AutoMachine.Write:
                if outer.gn_auto_machine_weight > self.gn_weight:
                    var, mean = torch.var_mean(x, dim=(2, 3), keepdim=True, correction=0)
                    self.mean_bank.append(mean)
                    self.var_bank.append(var)
                    self.style_cfgs.append(outer.current_style_fidelity)
            if outer.gn_auto_machine == AutoMachine.Read:
                if len(self.mean_bank) > 0 and len(self.var_bank) > 0:
                    style_cfg = sum(self.style_cfgs) / float(len(self.style_cfgs))
                    var, mean = torch.var_mean(x, dim=(2, 3), keepdim=True, correction=0)
                    std = torch.maximum(var, torch.zeros_like(var) + eps) ** 0.5
                    mean_acc = sum(self.mean_bank) / float(len(self.mean_bank))
                    var_acc = sum(self.var_bank) / float(len(self.var_bank))
                    std_acc = torch.maximum(var_acc, torch.zeros_like(var_acc) + eps) ** 0.5
                    y_uc = (((x - mean) / std) * std_acc) + mean_acc
                    y_c = y_uc.clone()
                    if len(outer.current_uc_indices) > 0 and style_cfg > 1e-5:
                        y_c[outer.current_uc_indices] = x.to(y_c.dtype)[outer.current_uc_indices]
                    y = style_cfg * y_c + (1.0 - style_cfg) * y_uc
                self.mean_bank = []
                self.var_bank = []
                self.style_cfgs = []
            if y is None:
                y = x
            return y.to(x.dtype)

        if getattr(process, 'sample_before_CN_hack', None) is None:
            process.sample_before_CN_hack = process.sample
        process.sample = process_sample

        model._original_forward = model.forward
        outer.original_forward = model.forward
        model.forward = forward_webui.__get__(model, UNetModel)

        if model_is_sdxl:
            register_schedule(sd_ldm)
            outer.revision_q_sampler = AbstractLowScaleModel()

        need_attention_hijack = False

        for param in outer.control_params:
            if param.control_model_type in [ControlModelType.AttentionInjection]:
                need_attention_hijack = True

        if batch_option_style_align:
            need_attention_hijack = True
            outer.attention_auto_machine = AutoMachine.StyleAlign
            outer.gn_auto_machine = AutoMachine.StyleAlign

        all_modules = torch_dfs(model)

        if need_attention_hijack:
            attn_modules = [module for module in all_modules if isinstance(module, BasicTransformerBlock) or isinstance(module, BasicTransformerBlockSGM)]
            attn_modules = sorted(attn_modules, key=lambda x: - x.norm1.normalized_shape[0])

            for i, module in enumerate(attn_modules):
                if getattr(module, '_original_inner_forward_cn_hijack', None) is None:
                    module._original_inner_forward_cn_hijack = module._forward
                module._forward = hacked_basic_transformer_inner_forward.__get__(module, BasicTransformerBlock)
                module.bank = []
                module.style_cfgs = []
                module.attn_weight = float(i) / float(len(attn_modules))

            gn_modules = [model.middle_block]
            model.middle_block.gn_weight = 0

            if model_is_sdxl:
                input_block_indices = [4, 5, 7, 8]
                output_block_indices = [0, 1, 2, 3, 4, 5]
            else:
                input_block_indices = [4, 5, 7, 8, 10, 11]
                output_block_indices = [0, 1, 2, 3, 4, 5, 6, 7]

            for w, i in enumerate(input_block_indices):
                module = model.input_blocks[i]
                module.gn_weight = 1.0 - float(w) / float(len(input_block_indices))
                gn_modules.append(module)

            for w, i in enumerate(output_block_indices):
                module = model.output_blocks[i]
                module.gn_weight = float(w) / float(len(output_block_indices))
                gn_modules.append(module)

            for i, module in enumerate(gn_modules):
                if getattr(module, 'original_forward_cn_hijack', None) is None:
                    module.original_forward_cn_hijack = module.forward
                module.forward = hacked_group_norm_forward.__get__(module, torch.nn.Module)
                module.mean_bank = []
                module.var_bank = []
                module.style_cfgs = []
                module.gn_weight *= 2

            outer.attn_module_list = attn_modules
            outer.gn_module_list = gn_modules
        else:
            for module in all_modules:
                _original_inner_forward_cn_hijack = getattr(module, '_original_inner_forward_cn_hijack', None)
                original_forward_cn_hijack = getattr(module, 'original_forward_cn_hijack', None)
                if _original_inner_forward_cn_hijack is not None:
                    module._forward = _original_inner_forward_cn_hijack
                if original_forward_cn_hijack is not None:
                    module.forward = original_forward_cn_hijack
            outer.attn_module_list = []
            outer.gn_module_list = []

        scripts.script_callbacks.on_cfg_denoiser(self.guidance_schedule_handler)

    def restore(self):
        scripts.script_callbacks.remove_callbacks_for_function(self.guidance_schedule_handler)
        self.control_params = None

        if self.model is not None:
            if hasattr(self.model, "_original_forward"):
                self.model.forward = self.model._original_forward
                del self.model._original_forward
