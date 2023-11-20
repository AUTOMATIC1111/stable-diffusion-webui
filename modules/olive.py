import os
import torch
import diffusers
from transformers.models.clip.modeling_clip import CLIPTextModel, CLIPTextModelWithProjection


is_sdxl = False

width = 512
height = 512
batch_size = 1
hidden_state_size = 768
time_ids_size = 5


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


def text_encoder_inputs(batchsize, torch_dtype):
    input_ids = torch.zeros((batchsize, 77), dtype=torch_dtype)
    return {
        "input_ids": input_ids,
        "output_hidden_states": True,
    } if is_sdxl else input_ids


def text_encoder_load(model_name):
    model = CLIPTextModel.from_pretrained(model_name, subfolder="text_encoder")
    return model


def text_encoder_conversion_inputs(model):
    return text_encoder_inputs(1, torch.int32)


def text_encoder_data_loader(data_dir, batchsize, *args, **kwargs):
    return RandomDataLoader(text_encoder_inputs, batchsize, torch.int32)


# -----------------------------------------------------------------------------
# TEXT ENCODER 2
# -----------------------------------------------------------------------------


def text_encoder_2_inputs(batchsize, torch_dtype):
    return {
        "input_ids": torch.zeros((batchsize, 77), dtype=torch_dtype),
        "output_hidden_states": True,
    }


def text_encoder_2_load(model_name):
    model = CLIPTextModelWithProjection.from_pretrained(model_name, subfolder="text_encoder_2")
    return model


def text_encoder_2_conversion_inputs(model):
    return text_encoder_2_inputs(1, torch.int64)


def text_encoder_2_data_loader(data_dir, batchsize, *args, **kwargs):
    return RandomDataLoader(text_encoder_2_inputs, batchsize, torch.int64)


# -----------------------------------------------------------------------------
# UNET
# -----------------------------------------------------------------------------


def unet_inputs(batchsize, torch_dtype, is_conversion_inputs=False):
    # TODO (pavignol): All the multiplications by 2 here are bacause the XL base has 2 text encoders
    # For refiner, it should be multiplied by 1 (single text encoder)

    if is_sdxl:
        inputs = {
            "sample": torch.rand((2 * batchsize, 4, height // 8, width // 8), dtype=torch_dtype),
            "timestep": torch.rand((1,), dtype=torch_dtype),
            "encoder_hidden_states": torch.rand((2 * batchsize, 77, hidden_state_size), dtype=torch_dtype),
        }

        if is_conversion_inputs:
            inputs["additional_inputs"] = {
                "added_cond_kwargs": {
                    "text_embeds": torch.rand((2 * batchsize, 1280), dtype=torch_dtype),
                    "time_ids": torch.rand((2 * batchsize, time_ids_size), dtype=torch_dtype),
                }
            }
        else:
            inputs["text_embeds"] = torch.rand((2 * batchsize, 1280), dtype=torch_dtype)
            inputs["time_ids"] = torch.rand((2 * batchsize, time_ids_size), dtype=torch_dtype)
    else:
        inputs = {
            "sample": torch.rand((batchsize, 4, height // 8, width // 8), dtype=torch_dtype),
            "timestep": torch.rand((batchsize,), dtype=torch_dtype),
            "encoder_hidden_states": torch.rand((batchsize, 77, hidden_state_size), dtype=torch_dtype),
            "return_dict": False,
        }

        if is_conversion_inputs:
            inputs["additional_inputs"] = {
                "added_cond_kwargs": {
                    "text_embeds": torch.rand((1, 1280), dtype=torch_dtype),
                    "time_ids": torch.rand((1, time_ids_size), dtype=torch_dtype),
                }
            }
        else:
            inputs["onnx::Concat_4"] = torch.rand((1, 1280), dtype=torch_dtype)
            inputs["onnx::Shape_5"] = torch.rand((1, time_ids_size), dtype=torch_dtype)

    return inputs


def unet_load(model_name):
    model = diffusers.UNet2DConditionModel.from_pretrained(model_name, subfolder="unet")
    return model


def unet_conversion_inputs(model):
    return tuple(unet_inputs(1, torch.float32, True).values())


def unet_data_loader(data_dir, batchsize, *args, **kwargs):
    return RandomDataLoader(unet_inputs, batchsize, torch.float16)


# -----------------------------------------------------------------------------
# VAE ENCODER
# -----------------------------------------------------------------------------


def vae_encoder_inputs(batchsize, torch_dtype):
    return {
        "sample": torch.rand((batchsize, 3, height, width), dtype=torch_dtype),
        "return_dict": False,
    }


def vae_encoder_load(model_name):
    source = os.path.join(model_name, "vae")
    if not os.path.isdir(source):
        source += "_encoder"
    model = diffusers.AutoencoderKL.from_pretrained(source)
    model.forward = lambda sample, return_dict: model.encode(sample, return_dict)[0].sample()
    return model


def vae_encoder_conversion_inputs(model):
    return tuple(vae_encoder_inputs(1, torch.float32).values())


def vae_encoder_data_loader(data_dir, batchsize, *args, **kwargs):
    return RandomDataLoader(vae_encoder_inputs, batchsize, torch.float16)


# -----------------------------------------------------------------------------
# VAE DECODER
# -----------------------------------------------------------------------------


def vae_decoder_inputs(batchsize, torch_dtype):
    return {
        "latent_sample": torch.rand((batchsize, 4, height // 8, width // 8), dtype=torch_dtype),
        "return_dict": False,
    }


def vae_decoder_load(model_name):
    source = os.path.join(model_name, "vae")
    if not os.path.isdir(source):
        source += "_decoder"
    model = diffusers.AutoencoderKL.from_pretrained(source)
    model.forward = model.decode
    return model


def vae_decoder_conversion_inputs(model):
    return tuple(vae_decoder_inputs(1, torch.float32).values())


def vae_decoder_data_loader(data_dir, batchsize, *args, **kwargs):
    return RandomDataLoader(vae_decoder_inputs, batchsize, torch.float16)
