import contextlib

import os
import safetensors
import torch
import math

import k_diffusion
from transformers import CLIPTokenizer

from modules import shared, devices, modelloader, sd_hijack_clip

from modules.models.sd3.sd3_impls import SDVAE
from modules.models.sd3.sd3_cond import CLIPL_CONFIG, T5_CONFIG, CLIPL_URL, T5_URL, SafetensorsMapping, Sd3T5
from modules.models.sd3.other_impls import SDClipModel, T5XXLModel, SDTokenizer, T5XXLTokenizer
from PIL import Image

from .model import Flux


class FluxTokenizer:
    def __init__(self):
        clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.clip_l = SDTokenizer(tokenizer=clip_tokenizer)
        self.t5xxl = T5XXLTokenizer()

    def tokenize_with_weights(self, text:str):
        out = {}
        out["l"] = self.clip_l.tokenize_with_weights(text)
        out["t5xxl"] = self.t5xxl.tokenize_with_weights(text)
        return out


class Flux1ClipL(sd_hijack_clip.TextConditionalModel):
    def __init__(self, clip_l):
        super().__init__()

        self.clip_l = clip_l

        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

        empty = self.tokenizer('')["input_ids"]
        self.id_start = empty[0]
        self.id_end = empty[1]
        self.id_pad = empty[1]

        self.return_pooled = True

    def tokenize(self, texts):
        return self.tokenizer(texts, truncation=False, add_special_tokens=False)["input_ids"]

    def encode_with_transformers(self, tokens):
        l_out, l_pooled = self.clip_l(tokens)
        l_out = torch.cat([l_out], dim=-1)
        l_out = torch.nn.functional.pad(l_out, (0, 4096 - l_out.shape[-1]))

        vector_out = torch.cat([l_pooled], dim=-1)

        l_out.pooled = vector_out

        return l_out

    def encode_embedding_init_text(self, init_text, nvpt):
        return torch.zeros((nvpt, 768+1280), device=devices.device) # XXX



class FluxCond(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.tokenizer = FluxTokenizer()

        with torch.no_grad():
            self.clip_l = SDClipModel(layer="hidden", layer_idx=-2, device="cpu", dtype=devices.dtype_inference, layer_norm_hidden_state=False, return_projected_pooled=False, textmodel_json_config=CLIPL_CONFIG)

            if shared.opts.flux_enable_t5:
                self.t5xxl = T5XXLModel(T5_CONFIG, device="cpu", dtype=devices.dtype_inference)
            else:
                self.t5xxl = None

            self.model_l = Flux1ClipL(self.clip_l)
            self.model_t5 = Sd3T5(self.t5xxl)

    def forward(self, prompts: list[str]):
        with devices.without_autocast():
            l_out, vector_out = self.model_l(prompts)
            t5_out = self.model_t5(prompts, token_count=l_out.shape[1])
            lt_out = torch.cat([l_out, t5_out], dim=-2)

        return {
            'crossattn': lt_out,
            'vector': vector_out,
        }

    def before_load_weights(self, state_dict):
        clip_path = os.path.join(shared.models_path, "CLIP")

        if 'text_encoders.clip_l.transformer.text_model.embeddings.position_embedding.weight' not in state_dict:
            clip_l_file = modelloader.load_file_from_url(CLIPL_URL, model_dir=clip_path, file_name="clip_l.safetensors")
            with safetensors.safe_open(clip_l_file, framework="pt") as file:
                self.clip_l.transformer.load_state_dict(SafetensorsMapping(file), strict=False)

        if self.t5xxl and 'text_encoders.t5xxl.transformer.encoder.block.0.layer.0.SelfAttention.k.weight' not in state_dict:
            t5_file = modelloader.load_file_from_url(T5_URL, model_dir=clip_path, file_name="t5xxl_fp8_e4m3fn.safetensors")
            with safetensors.safe_open(t5_file, framework="pt") as file:
                self.t5xxl.transformer.load_state_dict(SafetensorsMapping(file), strict=False)

    def encode_embedding_init_text(self, init_text, nvpt):
        return self.model_l.encode_embedding_init_text(init_text, nvpt)

    def tokenize(self, texts):
        return self.model_l.tokenize(texts)

    def medvram_modules(self):
        return [self.clip_l, self.t5xxl]

    def get_token_count(self, text):
        _, token_count = self.model_l.process_texts([text])

        return token_count

    def get_target_prompt_token_count(self, token_count):
        return self.model_l.get_target_prompt_token_count(token_count)

def flux_time_shift(mu: float, sigma: float, t):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)

class ModelSamplingFlux(torch.nn.Module):
    def __init__(self, shift=1.15):
        super().__init__()

        self.set_parameters(shift=shift)

    def set_parameters(self, shift=1.15, timesteps=10000):
        self.shift = shift
        ts = self.sigma((torch.arange(1, timesteps + 1, 1) / timesteps))
        self.register_buffer('sigmas', ts)

    @property
    def sigma_min(self):
        return self.sigmas[0]

    @property
    def sigma_max(self):
        return self.sigmas[-1]

    def timestep(self, sigma):
        return sigma

    def sigma(self, timestep):
        return flux_time_shift(self.shift, 1.0, timestep)

    def percent_to_sigma(self, percent):
        if percent <= 0.0:
            return 1.0
        if percent >= 1.0:
            return 0.0
        return 1.0 - percent

    def calculate_denoised(self, sigma, model_output, model_input):
        sigma = sigma.view(sigma.shape[:1] + (1,) * (model_output.ndim - 1))
        return model_input - model_output * sigma


class BaseModel(torch.nn.Module):
    """Wrapper around the core FLUX model"""
    def __init__(self, shift=1.15, device=None, dtype=torch.float16, state_dict=None, prefix="", **kwargs):
        super().__init__()

        self.diffusion_model = Flux(device=device, dtype=dtype, **kwargs)
        self.model_sampling = ModelSamplingFlux(shift=shift)
        self.depth = kwargs['depth']
        self.depth_single_block = kwargs['depth_single_blocks']

    def apply_model(self, x, sigma, c_crossattn=None, y=None):
        dtype = self.get_dtype()
        timestep = self.model_sampling.timestep(sigma).float()
        guidance = torch.FloatTensor([3.5]).to(device=devices.device, dtype=torch.float32)
        model_output = self.diffusion_model(x.to(dtype), timestep, context=c_crossattn.to(dtype), y=y.to(dtype), guidance=guidance).to(x.dtype)
        return self.model_sampling.calculate_denoised(sigma, model_output, x)

    def forward(self, *args, **kwargs):
        return self.apply_model(*args, **kwargs)

    def get_dtype(self):
        return self.diffusion_model.dtype


class FLUX1LatentFormat:
    """Latents are slightly shifted from center - this class must be called after VAE Decode to correct for the shift"""
    def __init__(self, scale_factor=0.3611, shift_factor=0.1159):
        self.scale_factor = scale_factor
        self.shift_factor = shift_factor

    def process_in(self, latent):
        return (latent - self.shift_factor) * self.scale_factor

    def process_out(self, latent):
        return (latent / self.scale_factor) + self.shift_factor

    def decode_latent_to_preview(self, x0):
        """Quick RGB approximate preview of sd3 latents"""
        factors = torch.tensor([
            [-0.0404,  0.0159,  0.0609], [ 0.0043,  0.0298,  0.0850],
            [ 0.0328, -0.0749, -0.0503], [-0.0245,  0.0085,  0.0549],
            [ 0.0966,  0.0894,  0.0530], [ 0.0035,  0.0399,  0.0123],
            [ 0.0583,  0.1184,  0.1262], [-0.0191, -0.0206, -0.0306],
            [-0.0324,  0.0055,  0.1001], [ 0.0955,  0.0659, -0.0545],
            [-0.0504,  0.0231, -0.0013], [ 0.0500, -0.0008, -0.0088],
            [ 0.0982,  0.0941,  0.0976], [-0.1233, -0.0280, -0.0897],
            [-0.0005, -0.0530, -0.0020], [-0.1273, -0.0932, -0.0680],
        ], device="cpu")
        latent_image = x0[0].permute(1, 2, 0).cpu() @ factors

        latents_ubyte = (((latent_image + 1) / 2)
                            .clamp(0, 1)  # change scale from -1..1 to 0..1
                            .mul(0xFF)  # to 0..255
                            .byte()).cpu()

        return Image.fromarray(latents_ubyte.numpy())


class FLUX1Denoiser(k_diffusion.external.DiscreteSchedule):
    def __init__(self, inner_model, sigmas):
        super().__init__(sigmas, quantize=shared.opts.enable_quantization)
        self.inner_model = inner_model

    def forward(self, input, sigma, **kwargs):
        return self.inner_model.apply_model(input, sigma, **kwargs)


class FLUX1Inferencer(torch.nn.Module):
    def __init__(self, state_dict, use_ema=False):
        super().__init__()

        params = dict(
            image_model="flux",
            in_channels=16,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10000,
            qkv_bias=True,
            guidance_embed=True,
        )

        # detect model_prefix
        diffusion_model_prefix = "model.diffusion_model."
        if "model.diffusion_model.double_blocks.0.img_attn.norm.key_norm.scale" in state_dict:
            diffusion_model_prefix = "model.diffusion_model."
        elif "double_blocks.0.img_attn.norm.key_norm.scale" in state_dict:
            diffusion_model_prefix = ""

        shift=1.15
        # check guidance_in to detect Flux schnell
        if f"{diffusion_model_prefix}guidance_in.in_layer.weight" not in state_dict:
            print("Flux schnell detected")
            params.update(dict(guidance_embed=False,))
            shift=1.0

        with torch.no_grad():
            self.model = BaseModel(shift=shift, state_dict=state_dict, prefix=diffusion_model_prefix, device="cpu", dtype=devices.dtype_inference, **params)
            self.first_stage_model = SDVAE(device="cpu", dtype=devices.dtype_vae)
            self.first_stage_model.dtype = devices.dtype_vae
            self.vae = self.first_stage_model # real vae

        self.alphas_cumprod = 1 / (self.model.model_sampling.sigmas ** 2 + 1)

        self.text_encoders = FluxCond()
        self.cond_stage_key = 'txt'

        self.parameterization = "eps"
        self.model.conditioning_key = "crossattn"

        self.latent_format = FLUX1LatentFormat()
        self.latent_channels = 16

    @property
    def cond_stage_model(self):
        return self.text_encoders

    def before_load_weights(self, state_dict):
        self.cond_stage_model.before_load_weights(state_dict)

    def ema_scope(self):
        return contextlib.nullcontext()

    def get_learned_conditioning(self, batch: list[str]):
        return self.cond_stage_model(batch)

    def apply_model(self, x, t, cond):
        return self.model(x, t, c_crossattn=cond['crossattn'], y=cond['vector'])

    def decode_first_stage(self, latent):
        latent = self.latent_format.process_out(latent)
        x = self.first_stage_model.decode(latent)
        if x.dtype == torch.float16:
            x = torch.nan_to_num(x, nan=0.0, posinf=65504, neginf=-65504)
        return x

    def encode_first_stage(self, image):
        latent = self.first_stage_model.encode(image)
        return self.latent_format.process_in(latent)

    def get_first_stage_encoding(self, x):
        return x

    def create_denoiser(self):
        return FLUX1Denoiser(self, self.model.model_sampling.sigmas)

    def medvram_fields(self):
        return [
            (self, 'first_stage_model'),
            (self, 'text_encoders'),
            (self, 'model'),
        ]

    def add_noise_to_latent(self, x, noise, amount):
        return x * (1 - amount) + noise * amount

    def fix_dimensions(self, width, height):
        return width // 16 * 16, height // 16 * 16

    def diffusers_weight_mapping(self):
        # https://github.com/kohya-ss/sd-scripts/blob/a61cf73a5cb5209c3f4d1a3688dd276a4dfd1ecb/networks/convert_flux_lora.py
        # please see also https://github.com/huggingface/diffusers/blob/main/src/diffusers/loaders/lora_conversion_utils.py
        for i in range(self.model.depth):
            yield f"transformer.transformer_blocks.{i}.attn.add_k_proj", f"diffusion_model_double_blocks_{i}_txt_attn_qkv_k_proj"
            yield f"transformer.transformer_blocks.{i}.attn.add_q_proj", f"diffusion_model_double_blocks_{i}_txt_attn_qkv_q_proj"
            yield f"transformer.transformer_blocks.{i}.attn.add_v_proj", f"diffusion_model_double_blocks_{i}_txt_attn_qkv_v_proj"

            yield f"transformer.transformer_blocks.{i}.attn.to_add_out", f"diffusion_model_double_blocks_{i}_txt_attn_proj"

            yield f"transformer.transformer_blocks.{i}.attn.to_k", f"diffusion_model_double_blocks_{i}_img_attn_qkv_k_proj"
            yield f"transformer.transformer_blocks.{i}.attn.to_q", f"diffusion_model_double_blocks_{i}_img_attn_qkv_q_proj"
            yield f"transformer.transformer_blocks.{i}.attn.to_v", f"diffusion_model_double_blocks_{i}_img_attn_qkv_v_proj"

            yield f"transformer.transformer_blocks.{i}.attn.to_out.0", f"diffusion_model_double_blocks_{i}_img_attn_proj"

            yield f"transformer.transformer_blocks.{i}.ff.net.0.proj", f"diffusion_model_double_blocks_{i}_img_mlp_0"
            yield f"transformer.transformer_blocks.{i}.ff.net.2", f"diffusion_model_double_blocks_{i}_img_mlp_2"
            yield f"transformer.transformer_blocks.{i}.ff_context.net.0.proj", f"diffusion_model_double_blocks_{i}_txt_mlp_0"
            yield f"transformer.transformer_blocks.{i}.ff_context.net.2", f"diffusion_model_double_blocks_{i}_txt_mlp_2"
            yield f"transformer.transformer_blocks.{i}.norm1.linear", f"diffusion_model_double_blocks_{i}_img_mod_lin"
            yield f"transformer.transformer_blocks.{i}.norm1_context.linear", f"diffusion_model_double_blocks_{i}_txt_mod_lin"

        for i in range(self.model.depth_single_block):
            yield f"transformer.single_transformer_blocks.{i}.attn.to_q", f"diffusion_model_single_blocks_{i}_linear1_q_proj"
            yield f"transformer.single_transformer_blocks.{i}.attn.to_k", f"diffusion_model_single_blocks_{i}_linear1_k_proj"
            yield f"transformer.single_transformer_blocks.{i}.attn.to_v", f"diffusion_model_single_blocks_{i}_linear1_v_proj"
            yield f"transformer.single_transformer_blocks.{i}.proj_mlp",  f"diffusion_model_single_blocks_{i}_linear1_mlp_proj"

            yield f"transformer.single_transformer_blocks.{i}.proj_out", f"diffusion_model_single_blocks_{i}_linear2"
            yield f"transformer.single_transformer_blocks.{i}.norm.linear", f"diffusion_model_single_blocks_{i}_modulation_lin"
