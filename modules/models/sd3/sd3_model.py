import contextlib
import os
from typing import Mapping

import safetensors
import torch

import k_diffusion
from modules.models.sd3.other_impls import SDClipModel, SDXLClipG, T5XXLModel, SD3Tokenizer
from modules.models.sd3.sd3_impls import BaseModel, SDVAE, SD3LatentFormat

from modules import shared, modelloader, devices

CLIPG_URL = "https://huggingface.co/AUTOMATIC/stable-diffusion-3-medium-text-encoders/resolve/main/clip_g.safetensors"
CLIPG_CONFIG = {
    "hidden_act": "gelu",
    "hidden_size": 1280,
    "intermediate_size": 5120,
    "num_attention_heads": 20,
    "num_hidden_layers": 32,
}

CLIPL_URL = "https://huggingface.co/AUTOMATIC/stable-diffusion-3-medium-text-encoders/resolve/main/clip_l.safetensors"
CLIPL_CONFIG = {
    "hidden_act": "quick_gelu",
    "hidden_size": 768,
    "intermediate_size": 3072,
    "num_attention_heads": 12,
    "num_hidden_layers": 12,
}

T5_URL = "https://huggingface.co/AUTOMATIC/stable-diffusion-3-medium-text-encoders/resolve/main/t5xxl_fp16.safetensors"
T5_CONFIG = {
    "d_ff": 10240,
    "d_model": 4096,
    "num_heads": 64,
    "num_layers": 24,
    "vocab_size": 32128,
}


class SafetensorsMapping(Mapping):
    def __init__(self, file):
        self.file = file

    def __len__(self):
        return len(self.file.keys())

    def __iter__(self):
        for key in self.file.keys():
            yield key

    def __getitem__(self, key):
        return self.file.get_tensor(key)


class SD3Cond(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.tokenizer = SD3Tokenizer()

        with torch.no_grad():
            self.clip_g = SDXLClipG(CLIPG_CONFIG, device="cpu", dtype=devices.dtype)
            self.clip_l = SDClipModel(layer="hidden", layer_idx=-2, device="cpu", dtype=devices.dtype, layer_norm_hidden_state=False, return_projected_pooled=False, textmodel_json_config=CLIPL_CONFIG)

            if shared.opts.sd3_enable_t5:
                self.t5xxl = T5XXLModel(T5_CONFIG, device="cpu", dtype=devices.dtype)
            else:
                self.t5xxl = None

        self.weights_loaded = False

    def forward(self, prompts: list[str]):
        res = []

        for prompt in prompts:
            tokens = self.tokenizer.tokenize_with_weights(prompt)
            l_out, l_pooled = self.clip_l.encode_token_weights(tokens["l"])
            g_out, g_pooled = self.clip_g.encode_token_weights(tokens["g"])

            if self.t5xxl and shared.opts.sd3_enable_t5:
                t5_out, t5_pooled = self.t5xxl.encode_token_weights(tokens["t5xxl"])
            else:
                t5_out = torch.zeros(l_out.shape[0:2] + (4096,), dtype=l_out.dtype, device=l_out.device)

            lg_out = torch.cat([l_out, g_out], dim=-1)
            lg_out = torch.nn.functional.pad(lg_out, (0, 4096 - lg_out.shape[-1]))
            lgt_out = torch.cat([lg_out, t5_out], dim=-2)
            vector_out = torch.cat((l_pooled, g_pooled), dim=-1)

            res.append({
                'crossattn': lgt_out[0].to(devices.device),
                'vector': vector_out[0].to(devices.device),
            })

        return res

    def load_weights(self):
        if self.weights_loaded:
            return

        clip_path = os.path.join(shared.models_path, "CLIP")

        clip_g_file = modelloader.load_file_from_url(CLIPG_URL, model_dir=clip_path, file_name="clip_g.safetensors")
        with safetensors.safe_open(clip_g_file, framework="pt") as file:
            self.clip_g.transformer.load_state_dict(SafetensorsMapping(file))

        clip_l_file = modelloader.load_file_from_url(CLIPL_URL, model_dir=clip_path, file_name="clip_l.safetensors")
        with safetensors.safe_open(clip_l_file, framework="pt") as file:
            self.clip_l.transformer.load_state_dict(SafetensorsMapping(file), strict=False)

        if self.t5xxl:
            t5_file = modelloader.load_file_from_url(T5_URL, model_dir=clip_path, file_name="t5xxl_fp16.safetensors")
            with safetensors.safe_open(t5_file, framework="pt") as file:
                self.t5xxl.transformer.load_state_dict(SafetensorsMapping(file), strict=False)

        self.weights_loaded = True

    def encode_embedding_init_text(self, init_text, nvpt):
        return torch.tensor([[0]], device=devices.device) # XXX


class SD3Denoiser(k_diffusion.external.DiscreteSchedule):
    def __init__(self, inner_model, sigmas):
        super().__init__(sigmas, quantize=shared.opts.enable_quantization)
        self.inner_model = inner_model

    def forward(self, input, sigma, **kwargs):
        return self.inner_model.apply_model(input, sigma, **kwargs)


class SD3Inferencer(torch.nn.Module):
    def __init__(self, state_dict, shift=3, use_ema=False):
        super().__init__()

        self.shift = shift

        with torch.no_grad():
            self.model = BaseModel(shift=shift, state_dict=state_dict, prefix="model.diffusion_model.", device="cpu", dtype=devices.dtype)
            self.first_stage_model = SDVAE(device="cpu", dtype=devices.dtype_vae)
            self.first_stage_model.dtype = self.model.diffusion_model.dtype

        self.alphas_cumprod = 1 / (self.model.model_sampling.sigmas ** 2 + 1)

        self.cond_stage_model = SD3Cond()
        self.cond_stage_key = 'txt'

        self.parameterization = "eps"
        self.model.conditioning_key = "crossattn"

        self.latent_format = SD3LatentFormat()
        self.latent_channels = 16

    def after_load_weights(self):
        self.cond_stage_model.load_weights()

    def ema_scope(self):
        return contextlib.nullcontext()

    def get_learned_conditioning(self, batch: list[str]):
        with devices.without_autocast():
            return self.cond_stage_model(batch)

    def apply_model(self, x, t, cond):
        return self.model.apply_model(x, t, c_crossattn=cond['crossattn'], y=cond['vector'])

    def decode_first_stage(self, latent):
        latent = self.latent_format.process_out(latent)
        return self.first_stage_model.decode(latent)

    def encode_first_stage(self, image):
        latent = self.first_stage_model.encode(image)
        return self.latent_format.process_in(latent)

    def create_denoiser(self):
        return SD3Denoiser(self, self.model.model_sampling.sigmas)
