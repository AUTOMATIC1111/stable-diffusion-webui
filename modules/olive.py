import os
import torch
import diffusers
from typing import Type, Callable, Dict, Any
from transformers.models.clip.modeling_clip import CLIPTextModel, CLIPTextModelWithProjection


class ENVStore:
    __DESERIALIZER: Dict[Type, Callable[[str,], Any]] = {
        bool: lambda x: bool(int(x)),
        int: int,
        str: lambda x: x,
    }
    __SERIALIZER: Dict[Type, Callable[[Any,], str]] = {
        bool: lambda x: str(int(x)),
        int: str,
        str: lambda x: x,
    }

    def __getattr__(self, name: str):
        value = os.environ.get(f"SDNEXT_OLIVE_{name}", None)
        if value is None:
            return
        ty = self.__class__.__annotations__[name]
        deserialize = self.__DESERIALIZER[ty]
        return deserialize(value)

    def __setattr__(self, name: str, value) -> None:
        if name not in self.__class__.__annotations__:
            return
        ty = self.__class__.__annotations__[name]
        serialize = self.__SERIALIZER[ty]
        os.environ[f"SDNEXT_OLIVE_{name}"] = serialize(value)

    def __delattr__(self, name: str) -> None:
        if name not in self.__class__.__annotations__:
            return
        os.environ.pop(f"SDNEXT_OLIVE_{name}")


class OliveOptimizerConfig(ENVStore):
    from_diffusers_cache: bool

    is_sdxl: bool

    vae_id: str
    vae_subfolder: str

    width: int
    height: int
    batch_size: int

    cross_attention_dim: int
    time_ids_size: int


config = OliveOptimizerConfig()


def get_variant():
    from modules.shared import opts
    if opts.diffusers_model_load_variant == 'default':
        from modules import devices
        if devices.dtype == torch.float16:
            return 'fp16'
    elif opts.diffusers_model_load_variant == 'fp32':
        return None
    else:
        return opts.diffusers_model_load_variant


def get_loader_arguments():
    if config.from_diffusers_cache:
        from modules.shared import opts
        return {
            "cache_dir": opts.diffusers_dir,
            "variant": get_variant(),
        }

    return {}


# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------


# Helper latency-only dataloader that creates random tensors with no label
class RandomDataLoader:
    def __init__(self, create_inputs_func, batchsize, torch_dtype):
        self.create_input_func = create_inputs_func
        self.batchsize = batchsize
        self.torch_dtype = torch_dtype

    def __getitem__(self, idx):
        label = None
        return self.create_input_func(self.batchsize, self.torch_dtype), label

# -----------------------------------------------------------------------------
# TEXT ENCODER
# -----------------------------------------------------------------------------


def text_encoder_inputs(_, torch_dtype):
    input_ids = torch.zeros((config.batch_size, 77), dtype=torch_dtype)
    return {
        "input_ids": input_ids,
        "output_hidden_states": True,
    } if config.is_sdxl else input_ids


def text_encoder_load(model_name):
    model = CLIPTextModel.from_pretrained(model_name, subfolder="text_encoder", **get_loader_arguments())
    return model


def text_encoder_conversion_inputs(model):
    return text_encoder_inputs(1, torch.int32)


def text_encoder_data_loader(data_dir, _, *args, **kwargs):
    return RandomDataLoader(text_encoder_inputs, config.batch_size, torch.int32)


# -----------------------------------------------------------------------------
# TEXT ENCODER 2
# -----------------------------------------------------------------------------


def text_encoder_2_inputs(_, torch_dtype):
    return {
        "input_ids": torch.zeros((config.batch_size, 77), dtype=torch_dtype),
        "output_hidden_states": True,
    }


def text_encoder_2_load(model_name):
    model = CLIPTextModelWithProjection.from_pretrained(model_name, subfolder="text_encoder_2", **get_loader_arguments())
    return model


def text_encoder_2_conversion_inputs(model):
    return text_encoder_2_inputs(1, torch.int64)


def text_encoder_2_data_loader(data_dir, _, *args, **kwargs):
    return RandomDataLoader(text_encoder_2_inputs, config.batch_size, torch.int64)


# -----------------------------------------------------------------------------
# UNET
# -----------------------------------------------------------------------------


def unet_inputs(_, torch_dtype, is_conversion_inputs=False):
    if config.is_sdxl:
        inputs = {
            "sample": torch.rand((2 * config.batch_size, 4, config.height // 8, config.width // 8), dtype=torch_dtype),
            "timestep": torch.rand((1,), dtype=torch_dtype),
            "encoder_hidden_states": torch.rand((2 * config.batch_size, 77, config.cross_attention_dim), dtype=torch_dtype),
        }

        if is_conversion_inputs:
            inputs["additional_inputs"] = {
                "added_cond_kwargs": {
                    "text_embeds": torch.rand((2 * config.batch_size, 1280), dtype=torch_dtype),
                    "time_ids": torch.rand((2 * config.batch_size, config.time_ids_size), dtype=torch_dtype),
                }
            }
        else:
            inputs["text_embeds"] = torch.rand((2 * config.batch_size, 1280), dtype=torch_dtype)
            inputs["time_ids"] = torch.rand((2 * config.batch_size, config.time_ids_size), dtype=torch_dtype)
    else:
        inputs = {
            "sample": torch.rand((config.batch_size, 4, config.height // 8, config.width // 8), dtype=torch_dtype),
            "timestep": torch.rand((config.batch_size,), dtype=torch_dtype),
            "encoder_hidden_states": torch.rand((config.batch_size, 77, config.cross_attention_dim), dtype=torch_dtype),
        }

        # use as kwargs since they won't be in the correct position if passed along with the tuple of inputs
        kwargs = {
            "return_dict": False,
        }
        if is_conversion_inputs:
            inputs["additional_inputs"] = {
                **kwargs,
                "added_cond_kwargs": {
                    "text_embeds": torch.rand((1, 1280), dtype=torch_dtype),
                    "time_ids": torch.rand((1, 5), dtype=torch_dtype),
                },
            }
        else:
            inputs.update(kwargs)
            inputs["onnx::Concat_4"] = torch.rand((1, 1280), dtype=torch_dtype)
            inputs["onnx::Shape_5"] = torch.rand((1, 5), dtype=torch_dtype)

    return inputs


def unet_load(model_name):
    model = diffusers.UNet2DConditionModel.from_pretrained(model_name, subfolder="unet", **get_loader_arguments())
    return model


def unet_conversion_inputs(model):
    return tuple(unet_inputs(1, torch.float32, True).values())


def unet_data_loader(data_dir, _, *args, **kwargs):
    return RandomDataLoader(unet_inputs, config.batch_size, torch.float16)


# -----------------------------------------------------------------------------
# VAE ENCODER
# -----------------------------------------------------------------------------


def vae_encoder_inputs(_, torch_dtype):
    return {
        "sample": torch.rand((config.batch_size, 3, config.height, config.width), dtype=torch_dtype),
        "return_dict": False,
    }


def vae_encoder_load(model_name):
    subfolder = "vae_encoder" if os.path.isdir(os.path.join(model_name, "vae_encoder")) else "vae"
    if config.vae_id is not None:
        model_name = config.vae_id
        subfolder = config.vae_subfolder
    model = diffusers.AutoencoderKL.from_pretrained(model_name, subfolder=subfolder, **get_loader_arguments())
    model.forward = lambda sample, return_dict: model.encode(sample, return_dict)[0].sample()
    return model


def vae_encoder_conversion_inputs(model):
    return tuple(vae_encoder_inputs(1, torch.float32).values())


def vae_encoder_data_loader(data_dir, _, *args, **kwargs):
    return RandomDataLoader(vae_encoder_inputs, config.batch_size, torch.float16)


# -----------------------------------------------------------------------------
# VAE DECODER
# -----------------------------------------------------------------------------


def vae_decoder_inputs(_, torch_dtype):
    return {
        "latent_sample": torch.rand((config.batch_size, 4, config.height // 8, config.width // 8), dtype=torch_dtype),
        "return_dict": False,
    }


def vae_decoder_load(model_name):
    subfolder = "vae_decoder" if os.path.isdir(os.path.join(model_name, "vae_decoder")) else "vae"
    if config.vae_id is not None:
        model_name = config.vae_id
        subfolder = config.vae_subfolder
    model = diffusers.AutoencoderKL.from_pretrained(model_name, subfolder=subfolder, **get_loader_arguments())
    model.forward = model.decode
    return model


def vae_decoder_conversion_inputs(model):
    return tuple(vae_decoder_inputs(1, torch.float32).values())


def vae_decoder_data_loader(data_dir, _, *args, **kwargs):
    return RandomDataLoader(vae_decoder_inputs, config.batch_size, torch.float16)
