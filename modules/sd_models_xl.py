from __future__ import annotations

import torch

import sgm.models.diffusion
import sgm.modules.diffusionmodules.denoiser_scaling
import sgm.modules.diffusionmodules.discretizer
from modules import devices, shared, prompt_parser


def get_learned_conditioning(self: sgm.models.diffusion.DiffusionEngine, batch: prompt_parser.SdConditioning | list[str]):
    for embedder in self.conditioner.embedders:
        embedder.ucg_rate = 0.0

    width = getattr(batch, 'width', 1024)
    height = getattr(batch, 'height', 1024)
    is_negative_prompt = getattr(batch, 'is_negative_prompt', False)
    aesthetic_score = shared.opts.sdxl_refiner_low_aesthetic_score if is_negative_prompt else shared.opts.sdxl_refiner_high_aesthetic_score

    devices_args = dict(device=devices.device, dtype=devices.dtype)

    sdxl_conds = {
        "txt": batch,
        "original_size_as_tuple": torch.tensor([height, width], **devices_args).repeat(len(batch), 1),
        "crop_coords_top_left": torch.tensor([shared.opts.sdxl_crop_top, shared.opts.sdxl_crop_left], **devices_args).repeat(len(batch), 1),
        "target_size_as_tuple": torch.tensor([height, width], **devices_args).repeat(len(batch), 1),
        "aesthetic_score": torch.tensor([aesthetic_score], **devices_args).repeat(len(batch), 1),
    }

    force_zero_negative_prompt = is_negative_prompt and all(x == '' for x in batch)
    c = self.conditioner(sdxl_conds, force_zero_embeddings=['txt'] if force_zero_negative_prompt else [])

    return c


def apply_model(self: sgm.models.diffusion.DiffusionEngine, x, t, cond):
    return self.model(x, t, cond)


def get_first_stage_encoding(self, x):  # SDXL's encode_first_stage does everything so get_first_stage_encoding is just there for compatibility
    return x


sgm.models.diffusion.DiffusionEngine.get_learned_conditioning = get_learned_conditioning
sgm.models.diffusion.DiffusionEngine.apply_model = apply_model
sgm.models.diffusion.DiffusionEngine.get_first_stage_encoding = get_first_stage_encoding


def encode_embedding_init_text(self: sgm.modules.GeneralConditioner, init_text, nvpt):
    res = []

    for embedder in [embedder for embedder in self.embedders if hasattr(embedder, 'encode_embedding_init_text')]:
        encoded = embedder.encode_embedding_init_text(init_text, nvpt)
        res.append(encoded)

    return torch.cat(res, dim=1)


def process_texts(self, texts):
    for embedder in [embedder for embedder in self.embedders if hasattr(embedder, 'process_texts')]:
        return embedder.process_texts(texts)


def get_target_prompt_token_count(self, token_count):
    for embedder in [embedder for embedder in self.embedders if hasattr(embedder, 'get_target_prompt_token_count')]:
        return embedder.get_target_prompt_token_count(token_count)


# those additions to GeneralConditioner make it possible to use it as model.cond_stage_model from SD1.5 in exist
sgm.modules.GeneralConditioner.encode_embedding_init_text = encode_embedding_init_text
sgm.modules.GeneralConditioner.process_texts = process_texts
sgm.modules.GeneralConditioner.get_target_prompt_token_count = get_target_prompt_token_count


def extend_sdxl(model):
    """this adds a bunch of parameters to make SDXL model look a bit more like SD1.5 to the rest of the codebase."""

    dtype = next(model.model.diffusion_model.parameters()).dtype
    model.model.diffusion_model.dtype = dtype
    model.model.conditioning_key = 'crossattn'
    model.cond_stage_key = 'txt'
    # model.cond_stage_model will be set in sd_hijack

    model.parameterization = "v" if isinstance(model.denoiser.scaling, sgm.modules.diffusionmodules.denoiser_scaling.VScaling) else "eps"

    discretization = sgm.modules.diffusionmodules.discretizer.LegacyDDPMDiscretization()
    model.alphas_cumprod = torch.asarray(discretization.alphas_cumprod, device=devices.device, dtype=dtype)

    model.conditioner.wrapped = torch.nn.Module()


sgm.modules.attention.print = lambda *args: None
sgm.modules.diffusionmodules.model.print = lambda *args: None
sgm.modules.diffusionmodules.openaimodel.print = lambda *args: None
sgm.modules.encoders.modules.print = lambda *args: None

# this gets the code to load the vanilla attention that we override
sgm.modules.attention.SDP_IS_AVAILABLE = True
sgm.modules.attention.XFORMERS_IS_AVAILABLE = False
