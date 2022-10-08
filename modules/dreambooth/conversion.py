# From https://github.com/huggingface/diffusers/blob/main/scripts/convert_original_stable_diffusion_to_diffusers.py

# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Conversion script for the LDM checkpoints. """

import os

import gradio as gr
import torch
from basicsr.utils.download_util import load_file_from_url

import modules.sd_models
from modules import paths
from modules.dreambooth import dreambooth

try:
    from omegaconf import OmegaConf
except ImportError:
    raise ImportError(
        "OmegaConf is required to convert the LDM checkpoints. Please install it with `pip install OmegaConf`."
    )

from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    LDMTextToImagePipeline,
    LMSDiscreteScheduler,
    PNDMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
    DiffusionPipeline
)
from diffusers.pipelines.latent_diffusion.pipeline_latent_diffusion import LDMBertConfig, LDMBertModel
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor, BertTokenizerFast, CLIPTextModel, CLIPTokenizer


def shave_segments(path, n_shave_prefix_segments=1):
    """
    Removes segments. Positive values shave the first segments, negative shave the last segments.
    """
    if n_shave_prefix_segments >= 0:
        return ".".join(path.split(".")[n_shave_prefix_segments:])
    else:
        return ".".join(path.split(".")[:n_shave_prefix_segments])


def renew_resnet_paths(old_list, n_shave_prefix_segments=0):
    """
    Updates paths inside resnets to the new naming scheme (local renaming)
    """
    mapping = []
    for old_item in old_list:
        new_item = old_item.replace("in_layers.0", "norm1")
        new_item = new_item.replace("in_layers.2", "conv1")

        new_item = new_item.replace("out_layers.0", "norm2")
        new_item = new_item.replace("out_layers.3", "conv2")

        new_item = new_item.replace("emb_layers.1", "time_emb_proj")
        new_item = new_item.replace("skip_connection", "conv_shortcut")

        new_item = shave_segments(new_item, n_shave_prefix_segments=n_shave_prefix_segments)

        mapping.append({"old": old_item, "new": new_item})

    return mapping


def renew_vae_resnet_paths(old_list, n_shave_prefix_segments=0):
    """
    Updates paths inside resnets to the new naming scheme (local renaming)
    """
    mapping = []
    for old_item in old_list:
        new_item = old_item

        new_item = new_item.replace("nin_shortcut", "conv_shortcut")
        new_item = shave_segments(new_item, n_shave_prefix_segments=n_shave_prefix_segments)

        mapping.append({"old": old_item, "new": new_item})

    return mapping


def renew_attention_paths(old_list):
    """
    Updates paths inside attentions to the new naming scheme (local renaming)
    """
    mapping = []
    for old_item in old_list:
        new_item = old_item
        mapping.append({"old": old_item, "new": new_item})

    return mapping


def renew_vae_attention_paths(old_list, n_shave_prefix_segments=0):
    """
    Updates paths inside attentions to the new naming scheme (local renaming)
    """
    mapping = []
    for old_item in old_list:
        new_item = old_item

        new_item = new_item.replace("norm.weight", "group_norm.weight")
        new_item = new_item.replace("norm.bias", "group_norm.bias")

        new_item = new_item.replace("q.weight", "query.weight")
        new_item = new_item.replace("q.bias", "query.bias")

        new_item = new_item.replace("k.weight", "key.weight")
        new_item = new_item.replace("k.bias", "key.bias")

        new_item = new_item.replace("v.weight", "value.weight")
        new_item = new_item.replace("v.bias", "value.bias")

        new_item = new_item.replace("proj_out.weight", "proj_attn.weight")
        new_item = new_item.replace("proj_out.bias", "proj_attn.bias")

        new_item = shave_segments(new_item, n_shave_prefix_segments=n_shave_prefix_segments)

        mapping.append({"old": old_item, "new": new_item})

    return mapping


def assign_to_checkpoint(
        paths, checkpoint, old_checkpoint, attention_paths_to_split=None, additional_replacements=None, config=None
):
    """
    This does the final conversion step: take locally converted weights and apply a global renaming
    to them. It splits attention layers, and takes into account additional replacements
    that may arise.

    Assigns the weights to the new checkpoint.
    """
    assert isinstance(paths, list), "Paths should be a list of dicts containing 'old' and 'new' keys."

    # Splits the attention layers into three variables.
    if attention_paths_to_split is not None:
        for path, path_map in attention_paths_to_split.items():
            old_tensor = old_checkpoint[path]
            channels = old_tensor.shape[0] // 3

            target_shape = (-1, channels) if len(old_tensor.shape) == 3 else (-1)

            num_heads = old_tensor.shape[0] // config["num_head_channels"] // 3

            old_tensor = old_tensor.reshape((num_heads, 3 * channels // num_heads) + old_tensor.shape[1:])
            query, key, value = old_tensor.split(channels // num_heads, dim=1)

            checkpoint[path_map["query"]] = query.reshape(target_shape)
            checkpoint[path_map["key"]] = key.reshape(target_shape)
            checkpoint[path_map["value"]] = value.reshape(target_shape)

    for path in paths:
        new_path = path["new"]

        # These have already been assigned
        if attention_paths_to_split is not None and new_path in attention_paths_to_split:
            continue

        # Global renaming happens here
        new_path = new_path.replace("middle_block.0", "mid_block.resnets.0")
        new_path = new_path.replace("middle_block.1", "mid_block.attentions.0")
        new_path = new_path.replace("middle_block.2", "mid_block.resnets.1")

        if additional_replacements is not None:
            for replacement in additional_replacements:
                new_path = new_path.replace(replacement["old"], replacement["new"])

        # proj_attn.weight has to be converted from conv 1D to linear
        if "proj_attn.weight" in new_path:
            checkpoint[new_path] = old_checkpoint[path["old"]][:, :, 0]
        else:
            checkpoint[new_path] = old_checkpoint[path["old"]]


def conv_attn_to_linear(checkpoint):
    keys = list(checkpoint.keys())
    attn_keys = ["query.weight", "key.weight", "value.weight"]
    for key in keys:
        if ".".join(key.split(".")[-2:]) in attn_keys:
            if checkpoint[key].ndim > 2:
                checkpoint[key] = checkpoint[key][:, :, 0, 0]
        elif "proj_attn.weight" in key:
            if checkpoint[key].ndim > 2:
                checkpoint[key] = checkpoint[key][:, :, 0]


def create_unet_diffusers_config(original_config):
    """
    Creates a config for the diffusers based on the config of the LDM model.
    """
    unet_params = original_config.model.params.unet_config.params

    block_out_channels = [unet_params.model_channels * mult for mult in unet_params.channel_mult]

    down_block_types = []
    resolution = 1
    for i in range(len(block_out_channels)):
        block_type = "CrossAttnDownBlock2D" if resolution in unet_params.attention_resolutions else "DownBlock2D"
        down_block_types.append(block_type)
        if i != len(block_out_channels) - 1:
            resolution *= 2

    up_block_types = []
    for i in range(len(block_out_channels)):
        block_type = "CrossAttnUpBlock2D" if resolution in unet_params.attention_resolutions else "UpBlock2D"
        up_block_types.append(block_type)
        resolution //= 2

    config = dict(
        sample_size=unet_params.image_size,
        in_channels=unet_params.in_channels,
        out_channels=unet_params.out_channels,
        down_block_types=tuple(down_block_types),
        up_block_types=tuple(up_block_types),
        block_out_channels=tuple(block_out_channels),
        layers_per_block=unet_params.num_res_blocks,
        cross_attention_dim=unet_params.context_dim,
        attention_head_dim=unet_params.num_heads,
    )

    return config


def create_vae_diffusers_config(original_config):
    """
    Creates a config for the diffusers based on the config of the LDM model.
    """
    vae_params = original_config.model.params.first_stage_config.params.ddconfig
    _ = original_config.model.params.first_stage_config.params.embed_dim

    block_out_channels = [vae_params.ch * mult for mult in vae_params.ch_mult]
    down_block_types = ["DownEncoderBlock2D"] * len(block_out_channels)
    up_block_types = ["UpDecoderBlock2D"] * len(block_out_channels)

    config = dict(
        sample_size=vae_params.resolution,
        in_channels=vae_params.in_channels,
        out_channels=vae_params.out_ch,
        down_block_types=tuple(down_block_types),
        up_block_types=tuple(up_block_types),
        block_out_channels=tuple(block_out_channels),
        latent_channels=vae_params.z_channels,
        layers_per_block=vae_params.num_res_blocks,
    )
    return config


def create_diffusers_schedular(original_config):
    schedular = DDIMScheduler(
        num_train_timesteps=original_config.model.params.timesteps,
        beta_start=original_config.model.params.linear_start,
        beta_end=original_config.model.params.linear_end,
        beta_schedule="scaled_linear",
    )
    return schedular


def create_ldm_bert_config(original_config):
    bert_params = original_config.model.parms.cond_stage_config.params
    config = LDMBertConfig(
        d_model=bert_params.n_embed,
        encoder_layers=bert_params.n_layer,
        encoder_ffn_dim=bert_params.n_embed * 4,
    )
    return config


def convert_ldm_unet_checkpoint(checkpoint, config):
    """
    Takes a state dict and a config, and returns a converted checkpoint.
    """

    # extract state_dict for UNet
    unet_state_dict = {}
    unet_key = "model.diffusion_model."
    keys = list(checkpoint.keys())
    for key in keys:
        if key.startswith(unet_key):
            unet_state_dict[key.replace(unet_key, "")] = checkpoint.pop(key)

    new_checkpoint = {"time_embedding.linear_1.weight": unet_state_dict["time_embed.0.weight"],
                      "time_embedding.linear_1.bias": unet_state_dict["time_embed.0.bias"],
                      "time_embedding.linear_2.weight": unet_state_dict["time_embed.2.weight"],
                      "time_embedding.linear_2.bias": unet_state_dict["time_embed.2.bias"],
                      "conv_in.weight": unet_state_dict["input_blocks.0.0.weight"],
                      "conv_in.bias": unet_state_dict["input_blocks.0.0.bias"],
                      "conv_norm_out.weight": unet_state_dict["out.0.weight"],
                      "conv_norm_out.bias": unet_state_dict["out.0.bias"],
                      "conv_out.weight": unet_state_dict["out.2.weight"],
                      "conv_out.bias": unet_state_dict["out.2.bias"]}

    # Retrieves the keys for the input blocks only
    num_input_blocks = len({".".join(layer.split(".")[:2]) for layer in unet_state_dict if "input_blocks" in layer})
    input_blocks = {
        layer_id: [key for key in unet_state_dict if f"input_blocks.{layer_id}" in key]
        for layer_id in range(num_input_blocks)
    }

    # Retrieves the keys for the middle blocks only
    num_middle_blocks = len({".".join(layer.split(".")[:2]) for layer in unet_state_dict if "middle_block" in layer})
    middle_blocks = {
        layer_id: [key for key in unet_state_dict if f"middle_block.{layer_id}" in key]
        for layer_id in range(num_middle_blocks)
    }

    # Retrieves the keys for the output blocks only
    num_output_blocks = len({".".join(layer.split(".")[:2]) for layer in unet_state_dict if "output_blocks" in layer})
    output_blocks = {
        layer_id: [key for key in unet_state_dict if f"output_blocks.{layer_id}" in key]
        for layer_id in range(num_output_blocks)
    }

    for i in range(1, num_input_blocks):
        block_id = (i - 1) // (config["layers_per_block"] + 1)
        layer_in_block_id = (i - 1) % (config["layers_per_block"] + 1)

        resnets = [
            key for key in input_blocks[i] if f"input_blocks.{i}.0" in key and f"input_blocks.{i}.0.op" not in key
        ]
        attentions = [key for key in input_blocks[i] if f"input_blocks.{i}.1" in key]

        if f"input_blocks.{i}.0.op.weight" in unet_state_dict:
            new_checkpoint[f"down_blocks.{block_id}.downsamplers.0.conv.weight"] = unet_state_dict.pop(
                f"input_blocks.{i}.0.op.weight"
            )
            new_checkpoint[f"down_blocks.{block_id}.downsamplers.0.conv.bias"] = unet_state_dict.pop(
                f"input_blocks.{i}.0.op.bias"
            )

        paths = renew_resnet_paths(resnets)
        meta_path = {"old": f"input_blocks.{i}.0", "new": f"down_blocks.{block_id}.resnets.{layer_in_block_id}"}
        assign_to_checkpoint(
            paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config
        )

        if len(attentions):
            paths = renew_attention_paths(attentions)
            meta_path = {"old": f"input_blocks.{i}.1", "new": f"down_blocks.{block_id}.attentions.{layer_in_block_id}"}
            assign_to_checkpoint(
                paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config
            )

    resnet_0 = middle_blocks[0]
    attentions = middle_blocks[1]
    resnet_1 = middle_blocks[2]

    resnet_0_paths = renew_resnet_paths(resnet_0)
    assign_to_checkpoint(resnet_0_paths, new_checkpoint, unet_state_dict, config=config)

    resnet_1_paths = renew_resnet_paths(resnet_1)
    assign_to_checkpoint(resnet_1_paths, new_checkpoint, unet_state_dict, config=config)

    attentions_paths = renew_attention_paths(attentions)
    meta_path = {"old": "middle_block.1", "new": "mid_block.attentions.0"}
    assign_to_checkpoint(
        attentions_paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config
    )

    for i in range(num_output_blocks):
        block_id = i // (config["layers_per_block"] + 1)
        layer_in_block_id = i % (config["layers_per_block"] + 1)
        output_block_layers = [shave_segments(name, 2) for name in output_blocks[i]]
        output_block_list = {}

        for layer in output_block_layers:
            layer_id, layer_name = layer.split(".")[0], shave_segments(layer, 1)
            if layer_id in output_block_list:
                output_block_list[layer_id].append(layer_name)
            else:
                output_block_list[layer_id] = [layer_name]

        if len(output_block_list) > 1:
            resnets = [key for key in output_blocks[i] if f"output_blocks.{i}.0" in key]
            attentions = [key for key in output_blocks[i] if f"output_blocks.{i}.1" in key]

            paths = renew_resnet_paths(resnets)

            meta_path = {"old": f"output_blocks.{i}.0", "new": f"up_blocks.{block_id}.resnets.{layer_in_block_id}"}
            assign_to_checkpoint(
                paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config
            )

            if ["conv.weight", "conv.bias"] in output_block_list.values():
                index = list(output_block_list.values()).index(["conv.weight", "conv.bias"])
                new_checkpoint[f"up_blocks.{block_id}.upsamplers.0.conv.weight"] = unet_state_dict[
                    f"output_blocks.{i}.{index}.conv.weight"
                ]
                new_checkpoint[f"up_blocks.{block_id}.upsamplers.0.conv.bias"] = unet_state_dict[
                    f"output_blocks.{i}.{index}.conv.bias"
                ]

                # Clear attentions as they have been attributed above.
                if len(attentions) == 2:
                    attentions = []

            if len(attentions):
                paths = renew_attention_paths(attentions)
                meta_path = {
                    "old": f"output_blocks.{i}.1",
                    "new": f"up_blocks.{block_id}.attentions.{layer_in_block_id}",
                }
                assign_to_checkpoint(
                    paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config
                )
        else:
            resnet_0_paths = renew_resnet_paths(output_block_layers, n_shave_prefix_segments=1)
            for path in resnet_0_paths:
                old_path = ".".join(["output_blocks", str(i), path["old"]])
                new_path = ".".join(["up_blocks", str(block_id), "resnets", str(layer_in_block_id), path["new"]])

                new_checkpoint[new_path] = unet_state_dict[old_path]

    return new_checkpoint


def convert_ldm_vae_checkpoint(checkpoint, config):
    # extract state dict for VAE
    vae_state_dict = {}
    vae_key = "first_stage_model."
    keys = list(checkpoint.keys())
    for key in keys:
        if key.startswith(vae_key):
            vae_state_dict[key.replace(vae_key, "")] = checkpoint.get(key)

    new_checkpoint = {"encoder.conv_in.weight": vae_state_dict["encoder.conv_in.weight"],
                      "encoder.conv_in.bias": vae_state_dict["encoder.conv_in.bias"],
                      "encoder.conv_out.weight": vae_state_dict["encoder.conv_out.weight"],
                      "encoder.conv_out.bias": vae_state_dict["encoder.conv_out.bias"],
                      "encoder.conv_norm_out.weight": vae_state_dict["encoder.norm_out.weight"],
                      "encoder.conv_norm_out.bias": vae_state_dict["encoder.norm_out.bias"],
                      "decoder.conv_in.weight": vae_state_dict["decoder.conv_in.weight"],
                      "decoder.conv_in.bias": vae_state_dict["decoder.conv_in.bias"],
                      "decoder.conv_out.weight": vae_state_dict["decoder.conv_out.weight"],
                      "decoder.conv_out.bias": vae_state_dict["decoder.conv_out.bias"],
                      "decoder.conv_norm_out.weight": vae_state_dict["decoder.norm_out.weight"],
                      "decoder.conv_norm_out.bias": vae_state_dict["decoder.norm_out.bias"],
                      "quant_conv.weight": vae_state_dict["quant_conv.weight"],
                      "quant_conv.bias": vae_state_dict["quant_conv.bias"],
                      "post_quant_conv.weight": vae_state_dict["post_quant_conv.weight"],
                      "post_quant_conv.bias": vae_state_dict["post_quant_conv.bias"]}

    # Retrieves the keys for the encoder down blocks only
    num_down_blocks = len({".".join(layer.split(".")[:3]) for layer in vae_state_dict if "encoder.down" in layer})
    down_blocks = {
        layer_id: [key for key in vae_state_dict if f"down.{layer_id}" in key] for layer_id in range(num_down_blocks)
    }

    # Retrieves the keys for the decoder up blocks only
    num_up_blocks = len({".".join(layer.split(".")[:3]) for layer in vae_state_dict if "decoder.up" in layer})
    up_blocks = {
        layer_id: [key for key in vae_state_dict if f"up.{layer_id}" in key] for layer_id in range(num_up_blocks)
    }

    for i in range(num_down_blocks):
        resnets = [key for key in down_blocks[i] if f"down.{i}" in key and f"down.{i}.downsample" not in key]

        if f"encoder.down.{i}.downsample.conv.weight" in vae_state_dict:
            new_checkpoint[f"encoder.down_blocks.{i}.downsamplers.0.conv.weight"] = vae_state_dict.pop(
                f"encoder.down.{i}.downsample.conv.weight"
            )
            new_checkpoint[f"encoder.down_blocks.{i}.downsamplers.0.conv.bias"] = vae_state_dict.pop(
                f"encoder.down.{i}.downsample.conv.bias"
            )

        paths = renew_vae_resnet_paths(resnets)
        meta_path = {"old": f"down.{i}.block", "new": f"down_blocks.{i}.resnets"}
        assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)

    mid_resnets = [key for key in vae_state_dict if "encoder.mid.block" in key]
    num_mid_res_blocks = 2
    for i in range(1, num_mid_res_blocks + 1):
        resnets = [key for key in mid_resnets if f"encoder.mid.block_{i}" in key]

        paths = renew_vae_resnet_paths(resnets)
        meta_path = {"old": f"mid.block_{i}", "new": f"mid_block.resnets.{i - 1}"}
        assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)

    mid_attentions = [key for key in vae_state_dict if "encoder.mid.attn" in key]
    paths = renew_vae_attention_paths(mid_attentions)
    meta_path = {"old": "mid.attn_1", "new": "mid_block.attentions.0"}
    assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)
    conv_attn_to_linear(new_checkpoint)

    for i in range(num_up_blocks):
        block_id = num_up_blocks - 1 - i
        resnets = [
            key for key in up_blocks[block_id] if f"up.{block_id}" in key and f"up.{block_id}.upsample" not in key
        ]

        if f"decoder.up.{block_id}.upsample.conv.weight" in vae_state_dict:
            new_checkpoint[f"decoder.up_blocks.{i}.upsamplers.0.conv.weight"] = vae_state_dict[
                f"decoder.up.{block_id}.upsample.conv.weight"
            ]
            new_checkpoint[f"decoder.up_blocks.{i}.upsamplers.0.conv.bias"] = vae_state_dict[
                f"decoder.up.{block_id}.upsample.conv.bias"
            ]

        paths = renew_vae_resnet_paths(resnets)
        meta_path = {"old": f"up.{block_id}.block", "new": f"up_blocks.{i}.resnets"}
        assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)

    mid_resnets = [key for key in vae_state_dict if "decoder.mid.block" in key]
    num_mid_res_blocks = 2
    for i in range(1, num_mid_res_blocks + 1):
        resnets = [key for key in mid_resnets if f"decoder.mid.block_{i}" in key]

        paths = renew_vae_resnet_paths(resnets)
        meta_path = {"old": f"mid.block_{i}", "new": f"mid_block.resnets.{i - 1}"}
        assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)

    mid_attentions = [key for key in vae_state_dict if "decoder.mid.attn" in key]
    paths = renew_vae_attention_paths(mid_attentions)
    meta_path = {"old": "mid.attn_1", "new": "mid_block.attentions.0"}
    assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)
    conv_attn_to_linear(new_checkpoint)
    return new_checkpoint


def convert_ldm_bert_checkpoint(checkpoint, config):
    def _copy_attn_layer(hf_attn_layer, pt_attn_layer):
        hf_attn_layer.q_proj.weight.data = pt_attn_layer.to_q.weight
        hf_attn_layer.k_proj.weight.data = pt_attn_layer.to_k.weight
        hf_attn_layer.v_proj.weight.data = pt_attn_layer.to_v.weight

        hf_attn_layer.out_proj.weight = pt_attn_layer.to_out.weight
        hf_attn_layer.out_proj.bias = pt_attn_layer.to_out.bias

    def _copy_linear(hf_linear, pt_linear):
        hf_linear.weight = pt_linear.weight
        hf_linear.bias = pt_linear.bias

    def _copy_layer(hf_layer, pt_layer):
        # copy layer norms
        _copy_linear(hf_layer.self_attn_layer_norm, pt_layer[0][0])
        _copy_linear(hf_layer.final_layer_norm, pt_layer[1][0])

        # copy attn
        _copy_attn_layer(hf_layer.self_attn, pt_layer[0][1])

        # copy MLP
        pt_mlp = pt_layer[1][1]
        _copy_linear(hf_layer.fc1, pt_mlp.net[0][0])
        _copy_linear(hf_layer.fc2, pt_mlp.net[2])

    def _copy_layers(hf_layers, pt_layers):
        for i, hf_layer in enumerate(hf_layers):
            if i != 0:
                i += i
            pt_layer = pt_layers[i: i + 2]
            _copy_layer(hf_layer, pt_layer)

    hf_model = LDMBertModel(config).eval()

    # copy  embeds
    hf_model.model.embed_tokens.weight = checkpoint.transformer.token_emb.weight
    hf_model.model.embed_positions.weight.data = checkpoint.transformer.pos_emb.emb.weight

    # copy layer norm
    _copy_linear(hf_model.model.layer_norm, checkpoint.transformer.norm)

    # copy hidden layers
    _copy_layers(hf_model.model.layers, checkpoint.transformer.attn_layers.layers)

    _copy_linear(hf_model.to_logits, checkpoint.transformer.to_logits)

    return hf_model


def convert_ldm_clip_checkpoint(checkpoint):
    text_model = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

    keys = list(checkpoint.keys())

    text_model_dict = {}

    for key in keys:
        if key.startswith("cond_stage_model.transformer"):
            text_model_dict[key[len("cond_stage_model.transformer."):]] = checkpoint[key]

    text_model.load_state_dict(text_model_dict)

    return text_model


def extract_checkpoint(new_model_name: str, checkpoint_path: str, scheduler_type="ddim"):
    # Set up our base directory for the model and sanitize our file name
    new_model_name = "".join(x for x in new_model_name if x.isalnum())
    new_model_dir = create_output_dir(new_model_name)
    # Create folder for the 'extracted' diffusion model.
    out_dir = os.path.join(new_model_dir, "stable-diffusion-v1-4")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Todo: What impact does using the 'og' model config if not using that model as our base?
    original_config_file = load_file_from_url("https://raw.githubusercontent.com/CompVis/stable-diffusion/main"
                                              "/configs/stable-diffusion/v1-inference.yaml", new_model_dir
                                              )

    original_config = OmegaConf.load(original_config_file)
    # Is this right?
    checkpoint_file = modules.sd_models.get_closet_checkpoint_match(checkpoint_path)
    if checkpoint_file is None or not os.path.exists(checkpoint_file[0]):
        print("Unable to find checkpoint file!")
        return None, "Unable to find base checkpoint.", ""
    checkpoint = torch.load(checkpoint_file[0])["state_dict"]
    print(f"Checkpoint loaded from {checkpoint_file}")
    num_train_timesteps = original_config.model.params.timesteps
    beta_start = original_config.model.params.linear_start
    beta_end = original_config.model.params.linear_end
    if scheduler_type == "pndm":
        scheduler = PNDMScheduler(
            beta_end=beta_end,
            beta_schedule="scaled_linear",
            beta_start=beta_start,
            num_train_timesteps=num_train_timesteps,
            skip_prk_steps=True,
        )
    elif scheduler_type == "lms":
        scheduler = LMSDiscreteScheduler(beta_start=beta_start, beta_end=beta_end, beta_schedule="scaled_linear")
    elif scheduler_type == "ddim":
        scheduler = DDIMScheduler(
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )
    else:
        raise ValueError(f"Scheduler of type {scheduler_type} doesn't exist!")
    # Convert the UNet2DConditionModel model.
    unet_config = create_unet_diffusers_config(original_config)
    converted_unet_checkpoint = convert_ldm_unet_checkpoint(checkpoint, unet_config)

    unet = UNet2DConditionModel(**unet_config)
    unet.load_state_dict(converted_unet_checkpoint)

    # Convert the VAE model.
    vae_config = create_vae_diffusers_config(original_config)
    converted_vae_checkpoint = convert_ldm_vae_checkpoint(checkpoint, vae_config)

    vae = AutoencoderKL(**vae_config)
    vae.load_state_dict(converted_vae_checkpoint)

    # Convert the text model.
    text_model_type = original_config.model.params.cond_stage_config.target.split(".")[-1]
    if text_model_type == "FrozenCLIPEmbedder":
        text_model = convert_ldm_clip_checkpoint(checkpoint)
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        safety_checker = StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker")
        feature_extractor = AutoFeatureExtractor.from_pretrained("CompVis/stable-diffusion-safety-checker")
        pipe = StableDiffusionPipeline(
            vae=vae,
            text_encoder=text_model,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )
    else:
        text_config = create_ldm_bert_config(original_config)
        text_model = convert_ldm_bert_checkpoint(checkpoint, text_config)
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        pipe = LDMTextToImagePipeline(vqvae=vae, bert=text_model, tokenizer=tokenizer, unet=unet, scheduler=scheduler)
    pipe.save_pretrained(out_dir)
    if os.path.isfile(original_config_file):
        os.remove(original_config_file)
    dirs = dreambooth.get_db_models()
    return gr.Dropdown.update(choices=sorted(dirs)), f"Created: {new_model_name}", ""


KeyMap = {
    "model.diffusion_model.time_embed.0.weight": "time_embedding.linear_1.weight",
    "model.diffusion_model.time_embed.0.bias": "time_embedding.linear_1.bias",
    "model.diffusion_model.time_embed.2.weight": "time_embedding.linear_2.weight",
    "model.diffusion_model.time_embed.2.bias": "time_embedding.linear_2.bias",
    "model.diffusion_model.input_blocks.0.0.weight": "conv_in.weight",
    "model.diffusion_model.input_blocks.0.0.bias": "conv_in.bias",
    "model.diffusion_model.out.0.weight": "conv_norm_out.weight",
    "model.diffusion_model.out.0.bias": "conv_norm_out.bias",
    "model.diffusion_model.out.2.weight": "conv_out.weight",
    "model.diffusion_model.out.2.bias": "conv_out.bias",
    "model.diffusion_model.input_blocks.1.0.in_layers.0.weight": "down_blocks.0.resnets.0.norm1.weight",
    "model.diffusion_model.input_blocks.1.0.in_layers.0.bias": "down_blocks.0.resnets.0.norm1.bias",
    "model.diffusion_model.input_blocks.1.0.in_layers.2.weight": "down_blocks.0.resnets.0.conv1.weight",
    "model.diffusion_model.input_blocks.1.0.in_layers.2.bias": "down_blocks.0.resnets.0.conv1.bias",
    "model.diffusion_model.input_blocks.1.0.emb_layers.1.weight": "down_blocks.0.resnets.0.time_emb_proj.weight",
    "model.diffusion_model.input_blocks.1.0.emb_layers.1.bias": "down_blocks.0.resnets.0.time_emb_proj.bias",
    "model.diffusion_model.input_blocks.1.0.out_layers.0.weight": "down_blocks.0.resnets.0.norm2.weight",
    "model.diffusion_model.input_blocks.1.0.out_layers.0.bias": "down_blocks.0.resnets.0.norm2.bias",
    "model.diffusion_model.input_blocks.1.0.out_layers.3.weight": "down_blocks.0.resnets.0.conv2.weight",
    "model.diffusion_model.input_blocks.1.0.out_layers.3.bias": "down_blocks.0.resnets.0.conv2.bias",
    "model.diffusion_model.input_blocks.1.1.norm.weight": "down_blocks.0.attentions.0.norm.weight",
    "model.diffusion_model.input_blocks.1.1.norm.bias": "down_blocks.0.attentions.0.norm.bias",
    "model.diffusion_model.input_blocks.1.1.proj_in.weight": "down_blocks.0.attentions.0.proj_in.weight",
    "model.diffusion_model.input_blocks.1.1.proj_in.bias": "down_blocks.0.attentions.0.proj_in.bias",
    "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn1.to_q.weight": "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.weight",
    "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn1.to_k.weight": "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_k.weight",
    "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn1.to_v.weight": "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_v.weight",
    "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn1.to_out.0.weight": "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_out.0.weight",
    "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn1.to_out.0.bias": "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_out.0.bias",
    "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.ff.net.0.proj.weight": "down_blocks.0.attentions.0.transformer_blocks.0.ff.net.0.proj.weight",
    "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.ff.net.0.proj.bias": "down_blocks.0.attentions.0.transformer_blocks.0.ff.net.0.proj.bias",
    "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.ff.net.2.weight": "down_blocks.0.attentions.0.transformer_blocks.0.ff.net.2.weight",
    "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.ff.net.2.bias": "down_blocks.0.attentions.0.transformer_blocks.0.ff.net.2.bias",
    "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_q.weight": "down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_q.weight",
    "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight": "down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_k.weight",
    "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_v.weight": "down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_v.weight",
    "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_out.0.weight": "down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_out.0.weight",
    "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_out.0.bias": "down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_out.0.bias",
    "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.norm1.weight": "down_blocks.0.attentions.0.transformer_blocks.0.norm1.weight",
    "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.norm1.bias": "down_blocks.0.attentions.0.transformer_blocks.0.norm1.bias",
    "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.norm2.weight": "down_blocks.0.attentions.0.transformer_blocks.0.norm2.weight",
    "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.norm2.bias": "down_blocks.0.attentions.0.transformer_blocks.0.norm2.bias",
    "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.norm3.weight": "down_blocks.0.attentions.0.transformer_blocks.0.norm3.weight",
    "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.norm3.bias": "down_blocks.0.attentions.0.transformer_blocks.0.norm3.bias",
    "model.diffusion_model.input_blocks.1.1.proj_out.weight": "down_blocks.0.attentions.0.proj_out.weight",
    "model.diffusion_model.input_blocks.1.1.proj_out.bias": "down_blocks.0.attentions.0.proj_out.bias",
    "model.diffusion_model.input_blocks.2.0.in_layers.0.weight": "down_blocks.0.resnets.1.norm1.weight",
    "model.diffusion_model.input_blocks.2.0.in_layers.0.bias": "down_blocks.0.resnets.1.norm1.bias",
    "model.diffusion_model.input_blocks.2.0.in_layers.2.weight": "down_blocks.0.resnets.1.conv1.weight",
    "model.diffusion_model.input_blocks.2.0.in_layers.2.bias": "down_blocks.0.resnets.1.conv1.bias",
    "model.diffusion_model.input_blocks.2.0.emb_layers.1.weight": "down_blocks.0.resnets.1.time_emb_proj.weight",
    "model.diffusion_model.input_blocks.2.0.emb_layers.1.bias": "down_blocks.0.resnets.1.time_emb_proj.bias",
    "model.diffusion_model.input_blocks.2.0.out_layers.0.weight": "down_blocks.0.resnets.1.norm2.weight",
    "model.diffusion_model.input_blocks.2.0.out_layers.0.bias": "down_blocks.0.resnets.1.norm2.bias",
    "model.diffusion_model.input_blocks.2.0.out_layers.3.weight": "down_blocks.0.resnets.1.conv2.weight",
    "model.diffusion_model.input_blocks.2.0.out_layers.3.bias": "down_blocks.0.resnets.1.conv2.bias",
    "model.diffusion_model.input_blocks.2.1.norm.weight": "down_blocks.0.attentions.1.norm.weight",
    "model.diffusion_model.input_blocks.2.1.norm.bias": "down_blocks.0.attentions.1.norm.bias",
    "model.diffusion_model.input_blocks.2.1.proj_in.weight": "down_blocks.0.attentions.1.proj_in.weight",
    "model.diffusion_model.input_blocks.2.1.proj_in.bias": "down_blocks.0.attentions.1.proj_in.bias",
    "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn1.to_q.weight": "down_blocks.0.attentions.1.transformer_blocks.0.attn1.to_q.weight",
    "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn1.to_k.weight": "down_blocks.0.attentions.1.transformer_blocks.0.attn1.to_k.weight",
    "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn1.to_v.weight": "down_blocks.0.attentions.1.transformer_blocks.0.attn1.to_v.weight",
    "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn1.to_out.0.weight": "down_blocks.0.attentions.1.transformer_blocks.0.attn1.to_out.0.weight",
    "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn1.to_out.0.bias": "down_blocks.0.attentions.1.transformer_blocks.0.attn1.to_out.0.bias",
    "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.ff.net.0.proj.weight": "down_blocks.0.attentions.1.transformer_blocks.0.ff.net.0.proj.weight",
    "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.ff.net.0.proj.bias": "down_blocks.0.attentions.1.transformer_blocks.0.ff.net.0.proj.bias",
    "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.ff.net.2.weight": "down_blocks.0.attentions.1.transformer_blocks.0.ff.net.2.weight",
    "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.ff.net.2.bias": "down_blocks.0.attentions.1.transformer_blocks.0.ff.net.2.bias",
    "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn2.to_q.weight": "down_blocks.0.attentions.1.transformer_blocks.0.attn2.to_q.weight",
    "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn2.to_k.weight": "down_blocks.0.attentions.1.transformer_blocks.0.attn2.to_k.weight",
    "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn2.to_v.weight": "down_blocks.0.attentions.1.transformer_blocks.0.attn2.to_v.weight",
    "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn2.to_out.0.weight": "down_blocks.0.attentions.1.transformer_blocks.0.attn2.to_out.0.weight",
    "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn2.to_out.0.bias": "down_blocks.0.attentions.1.transformer_blocks.0.attn2.to_out.0.bias",
    "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.norm1.weight": "down_blocks.0.attentions.1.transformer_blocks.0.norm1.weight",
    "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.norm1.bias": "down_blocks.0.attentions.1.transformer_blocks.0.norm1.bias",
    "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.norm2.weight": "down_blocks.0.attentions.1.transformer_blocks.0.norm2.weight",
    "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.norm2.bias": "down_blocks.0.attentions.1.transformer_blocks.0.norm2.bias",
    "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.norm3.weight": "down_blocks.0.attentions.1.transformer_blocks.0.norm3.weight",
    "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.norm3.bias": "down_blocks.0.attentions.1.transformer_blocks.0.norm3.bias",
    "model.diffusion_model.input_blocks.2.1.proj_out.weight": "down_blocks.0.attentions.1.proj_out.weight",
    "model.diffusion_model.input_blocks.2.1.proj_out.bias": "down_blocks.0.attentions.1.proj_out.bias",
    "model.diffusion_model.input_blocks.3.0.op.weight": "down_blocks.0.downsamplers.0.conv.weight",
    "model.diffusion_model.input_blocks.3.0.op.bias": "down_blocks.0.downsamplers.0.conv.bias",
    "model.diffusion_model.input_blocks.4.0.in_layers.0.weight": "down_blocks.1.resnets.0.norm1.weight",
    "model.diffusion_model.input_blocks.4.0.in_layers.0.bias": "down_blocks.1.resnets.0.norm1.bias",
    "model.diffusion_model.input_blocks.4.0.in_layers.2.weight": "down_blocks.1.resnets.0.conv1.weight",
    "model.diffusion_model.input_blocks.4.0.in_layers.2.bias": "down_blocks.1.resnets.0.conv1.bias",
    "model.diffusion_model.input_blocks.4.0.emb_layers.1.weight": "down_blocks.1.resnets.0.time_emb_proj.weight",
    "model.diffusion_model.input_blocks.4.0.emb_layers.1.bias": "down_blocks.1.resnets.0.time_emb_proj.bias",
    "model.diffusion_model.input_blocks.4.0.out_layers.0.weight": "down_blocks.1.resnets.0.norm2.weight",
    "model.diffusion_model.input_blocks.4.0.out_layers.0.bias": "down_blocks.1.resnets.0.norm2.bias",
    "model.diffusion_model.input_blocks.4.0.out_layers.3.weight": "down_blocks.1.resnets.0.conv2.weight",
    "model.diffusion_model.input_blocks.4.0.out_layers.3.bias": "down_blocks.1.resnets.0.conv2.bias",
    "model.diffusion_model.input_blocks.4.0.skip_connection.weight": "down_blocks.1.resnets.0.conv_shortcut.weight",
    "model.diffusion_model.input_blocks.4.0.skip_connection.bias": "down_blocks.1.resnets.0.conv_shortcut.bias",
    "model.diffusion_model.input_blocks.4.1.norm.weight": "down_blocks.1.attentions.0.norm.weight",
    "model.diffusion_model.input_blocks.4.1.norm.bias": "down_blocks.1.attentions.0.norm.bias",
    "model.diffusion_model.input_blocks.4.1.proj_in.weight": "down_blocks.1.attentions.0.proj_in.weight",
    "model.diffusion_model.input_blocks.4.1.proj_in.bias": "down_blocks.1.attentions.0.proj_in.bias",
    "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn1.to_q.weight": "down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q.weight",
    "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn1.to_k.weight": "down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_k.weight",
    "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn1.to_v.weight": "down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_v.weight",
    "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn1.to_out.0.weight": "down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_out.0.weight",
    "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn1.to_out.0.bias": "down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_out.0.bias",
    "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.ff.net.0.proj.weight": "down_blocks.1.attentions.0.transformer_blocks.0.ff.net.0.proj.weight",
    "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.ff.net.0.proj.bias": "down_blocks.1.attentions.0.transformer_blocks.0.ff.net.0.proj.bias",
    "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.ff.net.2.weight": "down_blocks.1.attentions.0.transformer_blocks.0.ff.net.2.weight",
    "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.ff.net.2.bias": "down_blocks.1.attentions.0.transformer_blocks.0.ff.net.2.bias",
    "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn2.to_q.weight": "down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_q.weight",
    "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn2.to_k.weight": "down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_k.weight",
    "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn2.to_v.weight": "down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_v.weight",
    "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn2.to_out.0.weight": "down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_out.0.weight",
    "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn2.to_out.0.bias": "down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_out.0.bias",
    "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.norm1.weight": "down_blocks.1.attentions.0.transformer_blocks.0.norm1.weight",
    "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.norm1.bias": "down_blocks.1.attentions.0.transformer_blocks.0.norm1.bias",
    "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.norm2.weight": "down_blocks.1.attentions.0.transformer_blocks.0.norm2.weight",
    "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.norm2.bias": "down_blocks.1.attentions.0.transformer_blocks.0.norm2.bias",
    "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.norm3.weight": "down_blocks.1.attentions.0.transformer_blocks.0.norm3.weight",
    "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.norm3.bias": "down_blocks.1.attentions.0.transformer_blocks.0.norm3.bias",
    "model.diffusion_model.input_blocks.4.1.proj_out.weight": "down_blocks.1.attentions.0.proj_out.weight",
    "model.diffusion_model.input_blocks.4.1.proj_out.bias": "down_blocks.1.attentions.0.proj_out.bias",
    "model.diffusion_model.input_blocks.5.0.in_layers.0.weight": "down_blocks.1.resnets.1.norm1.weight",
    "model.diffusion_model.input_blocks.5.0.in_layers.0.bias": "down_blocks.1.resnets.1.norm1.bias",
    "model.diffusion_model.input_blocks.5.0.in_layers.2.weight": "down_blocks.1.resnets.1.conv1.weight",
    "model.diffusion_model.input_blocks.5.0.in_layers.2.bias": "down_blocks.1.resnets.1.conv1.bias",
    "model.diffusion_model.input_blocks.5.0.emb_layers.1.weight": "down_blocks.1.resnets.1.time_emb_proj.weight",
    "model.diffusion_model.input_blocks.5.0.emb_layers.1.bias": "down_blocks.1.resnets.1.time_emb_proj.bias",
    "model.diffusion_model.input_blocks.5.0.out_layers.0.weight": "down_blocks.1.resnets.1.norm2.weight",
    "model.diffusion_model.input_blocks.5.0.out_layers.0.bias": "down_blocks.1.resnets.1.norm2.bias",
    "model.diffusion_model.input_blocks.5.0.out_layers.3.weight": "down_blocks.1.resnets.1.conv2.weight",
    "model.diffusion_model.input_blocks.5.0.out_layers.3.bias": "down_blocks.1.resnets.1.conv2.bias",
    "model.diffusion_model.input_blocks.5.1.norm.weight": "down_blocks.1.attentions.1.norm.weight",
    "model.diffusion_model.input_blocks.5.1.norm.bias": "down_blocks.1.attentions.1.norm.bias",
    "model.diffusion_model.input_blocks.5.1.proj_in.weight": "down_blocks.1.attentions.1.proj_in.weight",
    "model.diffusion_model.input_blocks.5.1.proj_in.bias": "down_blocks.1.attentions.1.proj_in.bias",
    "model.diffusion_model.input_blocks.5.1.transformer_blocks.0.attn1.to_q.weight": "down_blocks.1.attentions.1.transformer_blocks.0.attn1.to_q.weight",
    "model.diffusion_model.input_blocks.5.1.transformer_blocks.0.attn1.to_k.weight": "down_blocks.1.attentions.1.transformer_blocks.0.attn1.to_k.weight",
    "model.diffusion_model.input_blocks.5.1.transformer_blocks.0.attn1.to_v.weight": "down_blocks.1.attentions.1.transformer_blocks.0.attn1.to_v.weight",
    "model.diffusion_model.input_blocks.5.1.transformer_blocks.0.attn1.to_out.0.weight": "down_blocks.1.attentions.1.transformer_blocks.0.attn1.to_out.0.weight",
    "model.diffusion_model.input_blocks.5.1.transformer_blocks.0.attn1.to_out.0.bias": "down_blocks.1.attentions.1.transformer_blocks.0.attn1.to_out.0.bias",
    "model.diffusion_model.input_blocks.5.1.transformer_blocks.0.ff.net.0.proj.weight": "down_blocks.1.attentions.1.transformer_blocks.0.ff.net.0.proj.weight",
    "model.diffusion_model.input_blocks.5.1.transformer_blocks.0.ff.net.0.proj.bias": "down_blocks.1.attentions.1.transformer_blocks.0.ff.net.0.proj.bias",
    "model.diffusion_model.input_blocks.5.1.transformer_blocks.0.ff.net.2.weight": "down_blocks.1.attentions.1.transformer_blocks.0.ff.net.2.weight",
    "model.diffusion_model.input_blocks.5.1.transformer_blocks.0.ff.net.2.bias": "down_blocks.1.attentions.1.transformer_blocks.0.ff.net.2.bias",
    "model.diffusion_model.input_blocks.5.1.transformer_blocks.0.attn2.to_q.weight": "down_blocks.1.attentions.1.transformer_blocks.0.attn2.to_q.weight",
    "model.diffusion_model.input_blocks.5.1.transformer_blocks.0.attn2.to_k.weight": "down_blocks.1.attentions.1.transformer_blocks.0.attn2.to_k.weight",
    "model.diffusion_model.input_blocks.5.1.transformer_blocks.0.attn2.to_v.weight": "down_blocks.1.attentions.1.transformer_blocks.0.attn2.to_v.weight",
    "model.diffusion_model.input_blocks.5.1.transformer_blocks.0.attn2.to_out.0.weight": "down_blocks.1.attentions.1.transformer_blocks.0.attn2.to_out.0.weight",
    "model.diffusion_model.input_blocks.5.1.transformer_blocks.0.attn2.to_out.0.bias": "down_blocks.1.attentions.1.transformer_blocks.0.attn2.to_out.0.bias",
    "model.diffusion_model.input_blocks.5.1.transformer_blocks.0.norm1.weight": "down_blocks.1.attentions.1.transformer_blocks.0.norm1.weight",
    "model.diffusion_model.input_blocks.5.1.transformer_blocks.0.norm1.bias": "down_blocks.1.attentions.1.transformer_blocks.0.norm1.bias",
    "model.diffusion_model.input_blocks.5.1.transformer_blocks.0.norm2.weight": "down_blocks.1.attentions.1.transformer_blocks.0.norm2.weight",
    "model.diffusion_model.input_blocks.5.1.transformer_blocks.0.norm2.bias": "down_blocks.1.attentions.1.transformer_blocks.0.norm2.bias",
    "model.diffusion_model.input_blocks.5.1.transformer_blocks.0.norm3.weight": "down_blocks.1.attentions.1.transformer_blocks.0.norm3.weight",
    "model.diffusion_model.input_blocks.5.1.transformer_blocks.0.norm3.bias": "down_blocks.1.attentions.1.transformer_blocks.0.norm3.bias",
    "model.diffusion_model.input_blocks.5.1.proj_out.weight": "down_blocks.1.attentions.1.proj_out.weight",
    "model.diffusion_model.input_blocks.5.1.proj_out.bias": "down_blocks.1.attentions.1.proj_out.bias",
    "model.diffusion_model.input_blocks.6.0.op.weight": "down_blocks.1.downsamplers.0.conv.weight",
    "model.diffusion_model.input_blocks.6.0.op.bias": "down_blocks.1.downsamplers.0.conv.bias",
    "model.diffusion_model.input_blocks.7.0.in_layers.0.weight": "down_blocks.2.resnets.0.norm1.weight",
    "model.diffusion_model.input_blocks.7.0.in_layers.0.bias": "down_blocks.2.resnets.0.norm1.bias",
    "model.diffusion_model.input_blocks.7.0.in_layers.2.weight": "down_blocks.2.resnets.0.conv1.weight",
    "model.diffusion_model.input_blocks.7.0.in_layers.2.bias": "down_blocks.2.resnets.0.conv1.bias",
    "model.diffusion_model.input_blocks.7.0.emb_layers.1.weight": "down_blocks.2.resnets.0.time_emb_proj.weight",
    "model.diffusion_model.input_blocks.7.0.emb_layers.1.bias": "down_blocks.2.resnets.0.time_emb_proj.bias",
    "model.diffusion_model.input_blocks.7.0.out_layers.0.weight": "down_blocks.2.resnets.0.norm2.weight",
    "model.diffusion_model.input_blocks.7.0.out_layers.0.bias": "down_blocks.2.resnets.0.norm2.bias",
    "model.diffusion_model.input_blocks.7.0.out_layers.3.weight": "down_blocks.2.resnets.0.conv2.weight",
    "model.diffusion_model.input_blocks.7.0.out_layers.3.bias": "down_blocks.2.resnets.0.conv2.bias",
    "model.diffusion_model.input_blocks.7.0.skip_connection.weight": "down_blocks.2.resnets.0.conv_shortcut.weight",
    "model.diffusion_model.input_blocks.7.0.skip_connection.bias": "down_blocks.2.resnets.0.conv_shortcut.bias",
    "model.diffusion_model.input_blocks.7.1.norm.weight": "down_blocks.2.attentions.0.norm.weight",
    "model.diffusion_model.input_blocks.7.1.norm.bias": "down_blocks.2.attentions.0.norm.bias",
    "model.diffusion_model.input_blocks.7.1.proj_in.weight": "down_blocks.2.attentions.0.proj_in.weight",
    "model.diffusion_model.input_blocks.7.1.proj_in.bias": "down_blocks.2.attentions.0.proj_in.bias",
    "model.diffusion_model.input_blocks.7.1.transformer_blocks.0.attn1.to_q.weight": "down_blocks.2.attentions.0.transformer_blocks.0.attn1.to_q.weight",
    "model.diffusion_model.input_blocks.7.1.transformer_blocks.0.attn1.to_k.weight": "down_blocks.2.attentions.0.transformer_blocks.0.attn1.to_k.weight",
    "model.diffusion_model.input_blocks.7.1.transformer_blocks.0.attn1.to_v.weight": "down_blocks.2.attentions.0.transformer_blocks.0.attn1.to_v.weight",
    "model.diffusion_model.input_blocks.7.1.transformer_blocks.0.attn1.to_out.0.weight": "down_blocks.2.attentions.0.transformer_blocks.0.attn1.to_out.0.weight",
    "model.diffusion_model.input_blocks.7.1.transformer_blocks.0.attn1.to_out.0.bias": "down_blocks.2.attentions.0.transformer_blocks.0.attn1.to_out.0.bias",
    "model.diffusion_model.input_blocks.7.1.transformer_blocks.0.ff.net.0.proj.weight": "down_blocks.2.attentions.0.transformer_blocks.0.ff.net.0.proj.weight",
    "model.diffusion_model.input_blocks.7.1.transformer_blocks.0.ff.net.0.proj.bias": "down_blocks.2.attentions.0.transformer_blocks.0.ff.net.0.proj.bias",
    "model.diffusion_model.input_blocks.7.1.transformer_blocks.0.ff.net.2.weight": "down_blocks.2.attentions.0.transformer_blocks.0.ff.net.2.weight",
    "model.diffusion_model.input_blocks.7.1.transformer_blocks.0.ff.net.2.bias": "down_blocks.2.attentions.0.transformer_blocks.0.ff.net.2.bias",
    "model.diffusion_model.input_blocks.7.1.transformer_blocks.0.attn2.to_q.weight": "down_blocks.2.attentions.0.transformer_blocks.0.attn2.to_q.weight",
    "model.diffusion_model.input_blocks.7.1.transformer_blocks.0.attn2.to_k.weight": "down_blocks.2.attentions.0.transformer_blocks.0.attn2.to_k.weight",
    "model.diffusion_model.input_blocks.7.1.transformer_blocks.0.attn2.to_v.weight": "down_blocks.2.attentions.0.transformer_blocks.0.attn2.to_v.weight",
    "model.diffusion_model.input_blocks.7.1.transformer_blocks.0.attn2.to_out.0.weight": "down_blocks.2.attentions.0.transformer_blocks.0.attn2.to_out.0.weight",
    "model.diffusion_model.input_blocks.7.1.transformer_blocks.0.attn2.to_out.0.bias": "down_blocks.2.attentions.0.transformer_blocks.0.attn2.to_out.0.bias",
    "model.diffusion_model.input_blocks.7.1.transformer_blocks.0.norm1.weight": "down_blocks.2.attentions.0.transformer_blocks.0.norm1.weight",
    "model.diffusion_model.input_blocks.7.1.transformer_blocks.0.norm1.bias": "down_blocks.2.attentions.0.transformer_blocks.0.norm1.bias",
    "model.diffusion_model.input_blocks.7.1.transformer_blocks.0.norm2.weight": "down_blocks.2.attentions.0.transformer_blocks.0.norm2.weight",
    "model.diffusion_model.input_blocks.7.1.transformer_blocks.0.norm2.bias": "down_blocks.2.attentions.0.transformer_blocks.0.norm2.bias",
    "model.diffusion_model.input_blocks.7.1.transformer_blocks.0.norm3.weight": "down_blocks.2.attentions.0.transformer_blocks.0.norm3.weight",
    "model.diffusion_model.input_blocks.7.1.transformer_blocks.0.norm3.bias": "down_blocks.2.attentions.0.transformer_blocks.0.norm3.bias",
    "model.diffusion_model.input_blocks.7.1.proj_out.weight": "down_blocks.2.attentions.0.proj_out.weight",
    "model.diffusion_model.input_blocks.7.1.proj_out.bias": "down_blocks.2.attentions.0.proj_out.bias",
    "model.diffusion_model.input_blocks.8.0.in_layers.0.weight": "down_blocks.2.resnets.1.norm1.weight",
    "model.diffusion_model.input_blocks.8.0.in_layers.0.bias": "down_blocks.2.resnets.1.norm1.bias",
    "model.diffusion_model.input_blocks.8.0.in_layers.2.weight": "down_blocks.2.resnets.1.conv1.weight",
    "model.diffusion_model.input_blocks.8.0.in_layers.2.bias": "down_blocks.2.resnets.1.conv1.bias",
    "model.diffusion_model.input_blocks.8.0.emb_layers.1.weight": "down_blocks.2.resnets.1.time_emb_proj.weight",
    "model.diffusion_model.input_blocks.8.0.emb_layers.1.bias": "down_blocks.2.resnets.1.time_emb_proj.bias",
    "model.diffusion_model.input_blocks.8.0.out_layers.0.weight": "down_blocks.2.resnets.1.norm2.weight",
    "model.diffusion_model.input_blocks.8.0.out_layers.0.bias": "down_blocks.2.resnets.1.norm2.bias",
    "model.diffusion_model.input_blocks.8.0.out_layers.3.weight": "down_blocks.2.resnets.1.conv2.weight",
    "model.diffusion_model.input_blocks.8.0.out_layers.3.bias": "down_blocks.2.resnets.1.conv2.bias",
    "model.diffusion_model.input_blocks.8.1.norm.weight": "down_blocks.2.attentions.1.norm.weight",
    "model.diffusion_model.input_blocks.8.1.norm.bias": "down_blocks.2.attentions.1.norm.bias",
    "model.diffusion_model.input_blocks.8.1.proj_in.weight": "down_blocks.2.attentions.1.proj_in.weight",
    "model.diffusion_model.input_blocks.8.1.proj_in.bias": "down_blocks.2.attentions.1.proj_in.bias",
    "model.diffusion_model.input_blocks.8.1.transformer_blocks.0.attn1.to_q.weight": "down_blocks.2.attentions.1.transformer_blocks.0.attn1.to_q.weight",
    "model.diffusion_model.input_blocks.8.1.transformer_blocks.0.attn1.to_k.weight": "down_blocks.2.attentions.1.transformer_blocks.0.attn1.to_k.weight",
    "model.diffusion_model.input_blocks.8.1.transformer_blocks.0.attn1.to_v.weight": "down_blocks.2.attentions.1.transformer_blocks.0.attn1.to_v.weight",
    "model.diffusion_model.input_blocks.8.1.transformer_blocks.0.attn1.to_out.0.weight": "down_blocks.2.attentions.1.transformer_blocks.0.attn1.to_out.0.weight",
    "model.diffusion_model.input_blocks.8.1.transformer_blocks.0.attn1.to_out.0.bias": "down_blocks.2.attentions.1.transformer_blocks.0.attn1.to_out.0.bias",
    "model.diffusion_model.input_blocks.8.1.transformer_blocks.0.ff.net.0.proj.weight": "down_blocks.2.attentions.1.transformer_blocks.0.ff.net.0.proj.weight",
    "model.diffusion_model.input_blocks.8.1.transformer_blocks.0.ff.net.0.proj.bias": "down_blocks.2.attentions.1.transformer_blocks.0.ff.net.0.proj.bias",
    "model.diffusion_model.input_blocks.8.1.transformer_blocks.0.ff.net.2.weight": "down_blocks.2.attentions.1.transformer_blocks.0.ff.net.2.weight",
    "model.diffusion_model.input_blocks.8.1.transformer_blocks.0.ff.net.2.bias": "down_blocks.2.attentions.1.transformer_blocks.0.ff.net.2.bias",
    "model.diffusion_model.input_blocks.8.1.transformer_blocks.0.attn2.to_q.weight": "down_blocks.2.attentions.1.transformer_blocks.0.attn2.to_q.weight",
    "model.diffusion_model.input_blocks.8.1.transformer_blocks.0.attn2.to_k.weight": "down_blocks.2.attentions.1.transformer_blocks.0.attn2.to_k.weight",
    "model.diffusion_model.input_blocks.8.1.transformer_blocks.0.attn2.to_v.weight": "down_blocks.2.attentions.1.transformer_blocks.0.attn2.to_v.weight",
    "model.diffusion_model.input_blocks.8.1.transformer_blocks.0.attn2.to_out.0.weight": "down_blocks.2.attentions.1.transformer_blocks.0.attn2.to_out.0.weight",
    "model.diffusion_model.input_blocks.8.1.transformer_blocks.0.attn2.to_out.0.bias": "down_blocks.2.attentions.1.transformer_blocks.0.attn2.to_out.0.bias",
    "model.diffusion_model.input_blocks.8.1.transformer_blocks.0.norm1.weight": "down_blocks.2.attentions.1.transformer_blocks.0.norm1.weight",
    "model.diffusion_model.input_blocks.8.1.transformer_blocks.0.norm1.bias": "down_blocks.2.attentions.1.transformer_blocks.0.norm1.bias",
    "model.diffusion_model.input_blocks.8.1.transformer_blocks.0.norm2.weight": "down_blocks.2.attentions.1.transformer_blocks.0.norm2.weight",
    "model.diffusion_model.input_blocks.8.1.transformer_blocks.0.norm2.bias": "down_blocks.2.attentions.1.transformer_blocks.0.norm2.bias",
    "model.diffusion_model.input_blocks.8.1.transformer_blocks.0.norm3.weight": "down_blocks.2.attentions.1.transformer_blocks.0.norm3.weight",
    "model.diffusion_model.input_blocks.8.1.transformer_blocks.0.norm3.bias": "down_blocks.2.attentions.1.transformer_blocks.0.norm3.bias",
    "model.diffusion_model.input_blocks.8.1.proj_out.weight": "down_blocks.2.attentions.1.proj_out.weight",
    "model.diffusion_model.input_blocks.8.1.proj_out.bias": "down_blocks.2.attentions.1.proj_out.bias",
    "model.diffusion_model.input_blocks.9.0.op.weight": "down_blocks.2.downsamplers.0.conv.weight",
    "model.diffusion_model.input_blocks.9.0.op.bias": "down_blocks.2.downsamplers.0.conv.bias",
    "model.diffusion_model.input_blocks.10.0.in_layers.0.weight": "down_blocks.3.resnets.0.norm1.weight",
    "model.diffusion_model.input_blocks.10.0.in_layers.0.bias": "down_blocks.3.resnets.0.norm1.bias",
    "model.diffusion_model.input_blocks.10.0.in_layers.2.weight": "down_blocks.3.resnets.0.conv1.weight",
    "model.diffusion_model.input_blocks.10.0.in_layers.2.bias": "down_blocks.3.resnets.0.conv1.bias",
    "model.diffusion_model.input_blocks.10.0.emb_layers.1.weight": "down_blocks.3.resnets.0.time_emb_proj.weight",
    "model.diffusion_model.input_blocks.10.0.emb_layers.1.bias": "down_blocks.3.resnets.0.time_emb_proj.bias",
    "model.diffusion_model.input_blocks.10.0.out_layers.0.weight": "down_blocks.3.resnets.0.norm2.weight",
    "model.diffusion_model.input_blocks.10.0.out_layers.0.bias": "down_blocks.3.resnets.0.norm2.bias",
    "model.diffusion_model.input_blocks.10.0.out_layers.3.weight": "down_blocks.3.resnets.0.conv2.weight",
    "model.diffusion_model.input_blocks.10.0.out_layers.3.bias": "down_blocks.3.resnets.0.conv2.bias",
    "model.diffusion_model.input_blocks.11.0.in_layers.0.weight": "down_blocks.3.resnets.1.norm1.weight",
    "model.diffusion_model.input_blocks.11.0.in_layers.0.bias": "down_blocks.3.resnets.1.norm1.bias",
    "model.diffusion_model.input_blocks.11.0.in_layers.2.weight": "down_blocks.3.resnets.1.conv1.weight",
    "model.diffusion_model.input_blocks.11.0.in_layers.2.bias": "down_blocks.3.resnets.1.conv1.bias",
    "model.diffusion_model.input_blocks.11.0.emb_layers.1.weight": "down_blocks.3.resnets.1.time_emb_proj.weight",
    "model.diffusion_model.input_blocks.11.0.emb_layers.1.bias": "down_blocks.3.resnets.1.time_emb_proj.bias",
    "model.diffusion_model.input_blocks.11.0.out_layers.0.weight": "down_blocks.3.resnets.1.norm2.weight",
    "model.diffusion_model.input_blocks.11.0.out_layers.0.bias": "down_blocks.3.resnets.1.norm2.bias",
    "model.diffusion_model.input_blocks.11.0.out_layers.3.weight": "down_blocks.3.resnets.1.conv2.weight",
    "model.diffusion_model.input_blocks.11.0.out_layers.3.bias": "down_blocks.3.resnets.1.conv2.bias",
    "model.diffusion_model.middle_block.0.in_layers.0.weight": "mid_block.resnets.0.norm1.weight",
    "model.diffusion_model.middle_block.0.in_layers.0.bias": "mid_block.resnets.0.norm1.bias",
    "model.diffusion_model.middle_block.0.in_layers.2.weight": "mid_block.resnets.0.conv1.weight",
    "model.diffusion_model.middle_block.0.in_layers.2.bias": "mid_block.resnets.0.conv1.bias",
    "model.diffusion_model.middle_block.0.emb_layers.1.weight": "mid_block.resnets.0.time_emb_proj.weight",
    "model.diffusion_model.middle_block.0.emb_layers.1.bias": "mid_block.resnets.0.time_emb_proj.bias",
    "model.diffusion_model.middle_block.0.out_layers.0.weight": "mid_block.resnets.0.norm2.weight",
    "model.diffusion_model.middle_block.0.out_layers.0.bias": "mid_block.resnets.0.norm2.bias",
    "model.diffusion_model.middle_block.0.out_layers.3.weight": "mid_block.resnets.0.conv2.weight",
    "model.diffusion_model.middle_block.0.out_layers.3.bias": "mid_block.resnets.0.conv2.bias",
    "model.diffusion_model.middle_block.2.in_layers.0.weight": "mid_block.resnets.1.norm1.weight",
    "model.diffusion_model.middle_block.2.in_layers.0.bias": "mid_block.resnets.1.norm1.bias",
    "model.diffusion_model.middle_block.2.in_layers.2.weight": "mid_block.resnets.1.conv1.weight",
    "model.diffusion_model.middle_block.2.in_layers.2.bias": "mid_block.resnets.1.conv1.bias",
    "model.diffusion_model.middle_block.2.emb_layers.1.weight": "mid_block.resnets.1.time_emb_proj.weight",
    "model.diffusion_model.middle_block.2.emb_layers.1.bias": "mid_block.resnets.1.time_emb_proj.bias",
    "model.diffusion_model.middle_block.2.out_layers.0.weight": "mid_block.resnets.1.norm2.weight",
    "model.diffusion_model.middle_block.2.out_layers.0.bias": "mid_block.resnets.1.norm2.bias",
    "model.diffusion_model.middle_block.2.out_layers.3.weight": "mid_block.resnets.1.conv2.weight",
    "model.diffusion_model.middle_block.2.out_layers.3.bias": "mid_block.resnets.1.conv2.bias",
    "model.diffusion_model.middle_block.1.norm.weight": "mid_block.attentions.0.norm.weight",
    "model.diffusion_model.middle_block.1.norm.bias": "mid_block.attentions.0.norm.bias",
    "model.diffusion_model.middle_block.1.proj_in.weight": "mid_block.attentions.0.proj_in.weight",
    "model.diffusion_model.middle_block.1.proj_in.bias": "mid_block.attentions.0.proj_in.bias",
    "model.diffusion_model.middle_block.1.transformer_blocks.0.attn1.to_q.weight": "mid_block.attentions.0.transformer_blocks.0.attn1.to_q.weight",
    "model.diffusion_model.middle_block.1.transformer_blocks.0.attn1.to_k.weight": "mid_block.attentions.0.transformer_blocks.0.attn1.to_k.weight",
    "model.diffusion_model.middle_block.1.transformer_blocks.0.attn1.to_v.weight": "mid_block.attentions.0.transformer_blocks.0.attn1.to_v.weight",
    "model.diffusion_model.middle_block.1.transformer_blocks.0.attn1.to_out.0.weight": "mid_block.attentions.0.transformer_blocks.0.attn1.to_out.0.weight",
    "model.diffusion_model.middle_block.1.transformer_blocks.0.attn1.to_out.0.bias": "mid_block.attentions.0.transformer_blocks.0.attn1.to_out.0.bias",
    "model.diffusion_model.middle_block.1.transformer_blocks.0.ff.net.0.proj.weight": "mid_block.attentions.0.transformer_blocks.0.ff.net.0.proj.weight",
    "model.diffusion_model.middle_block.1.transformer_blocks.0.ff.net.0.proj.bias": "mid_block.attentions.0.transformer_blocks.0.ff.net.0.proj.bias",
    "model.diffusion_model.middle_block.1.transformer_blocks.0.ff.net.2.weight": "mid_block.attentions.0.transformer_blocks.0.ff.net.2.weight",
    "model.diffusion_model.middle_block.1.transformer_blocks.0.ff.net.2.bias": "mid_block.attentions.0.transformer_blocks.0.ff.net.2.bias",
    "model.diffusion_model.middle_block.1.transformer_blocks.0.attn2.to_q.weight": "mid_block.attentions.0.transformer_blocks.0.attn2.to_q.weight",
    "model.diffusion_model.middle_block.1.transformer_blocks.0.attn2.to_k.weight": "mid_block.attentions.0.transformer_blocks.0.attn2.to_k.weight",
    "model.diffusion_model.middle_block.1.transformer_blocks.0.attn2.to_v.weight": "mid_block.attentions.0.transformer_blocks.0.attn2.to_v.weight",
    "model.diffusion_model.middle_block.1.transformer_blocks.0.attn2.to_out.0.weight": "mid_block.attentions.0.transformer_blocks.0.attn2.to_out.0.weight",
    "model.diffusion_model.middle_block.1.transformer_blocks.0.attn2.to_out.0.bias": "mid_block.attentions.0.transformer_blocks.0.attn2.to_out.0.bias",
    "model.diffusion_model.middle_block.1.transformer_blocks.0.norm1.weight": "mid_block.attentions.0.transformer_blocks.0.norm1.weight",
    "model.diffusion_model.middle_block.1.transformer_blocks.0.norm1.bias": "mid_block.attentions.0.transformer_blocks.0.norm1.bias",
    "model.diffusion_model.middle_block.1.transformer_blocks.0.norm2.weight": "mid_block.attentions.0.transformer_blocks.0.norm2.weight",
    "model.diffusion_model.middle_block.1.transformer_blocks.0.norm2.bias": "mid_block.attentions.0.transformer_blocks.0.norm2.bias",
    "model.diffusion_model.middle_block.1.transformer_blocks.0.norm3.weight": "mid_block.attentions.0.transformer_blocks.0.norm3.weight",
    "model.diffusion_model.middle_block.1.transformer_blocks.0.norm3.bias": "mid_block.attentions.0.transformer_blocks.0.norm3.bias",
    "model.diffusion_model.middle_block.1.proj_out.weight": "mid_block.attentions.0.proj_out.weight",
    "model.diffusion_model.middle_block.1.proj_out.bias": "mid_block.attentions.0.proj_out.bias",
    "model.diffusion_model.output_blocks.0.0.in_layers.0.weight": "up_blocks.0.resnets.0.norm1.weight",
    "model.diffusion_model.output_blocks.0.0.in_layers.0.bias": "up_blocks.0.resnets.0.norm1.bias",
    "model.diffusion_model.output_blocks.0.0.in_layers.2.weight": "up_blocks.0.resnets.0.conv1.weight",
    "model.diffusion_model.output_blocks.0.0.in_layers.2.bias": "up_blocks.0.resnets.0.conv1.bias",
    "model.diffusion_model.output_blocks.0.0.emb_layers.1.weight": "up_blocks.0.resnets.0.time_emb_proj.weight",
    "model.diffusion_model.output_blocks.0.0.emb_layers.1.bias": "up_blocks.0.resnets.0.time_emb_proj.bias",
    "model.diffusion_model.output_blocks.0.0.out_layers.0.weight": "up_blocks.0.resnets.0.norm2.weight",
    "model.diffusion_model.output_blocks.0.0.out_layers.0.bias": "up_blocks.0.resnets.0.norm2.bias",
    "model.diffusion_model.output_blocks.0.0.out_layers.3.weight": "up_blocks.0.resnets.0.conv2.weight",
    "model.diffusion_model.output_blocks.0.0.out_layers.3.bias": "up_blocks.0.resnets.0.conv2.bias",
    "model.diffusion_model.output_blocks.0.0.skip_connection.weight": "up_blocks.0.resnets.0.conv_shortcut.weight",
    "model.diffusion_model.output_blocks.0.0.skip_connection.bias": "up_blocks.0.resnets.0.conv_shortcut.bias",
    "model.diffusion_model.output_blocks.1.0.in_layers.0.weight": "up_blocks.0.resnets.1.norm1.weight",
    "model.diffusion_model.output_blocks.1.0.in_layers.0.bias": "up_blocks.0.resnets.1.norm1.bias",
    "model.diffusion_model.output_blocks.1.0.in_layers.2.weight": "up_blocks.0.resnets.1.conv1.weight",
    "model.diffusion_model.output_blocks.1.0.in_layers.2.bias": "up_blocks.0.resnets.1.conv1.bias",
    "model.diffusion_model.output_blocks.1.0.emb_layers.1.weight": "up_blocks.0.resnets.1.time_emb_proj.weight",
    "model.diffusion_model.output_blocks.1.0.emb_layers.1.bias": "up_blocks.0.resnets.1.time_emb_proj.bias",
    "model.diffusion_model.output_blocks.1.0.out_layers.0.weight": "up_blocks.0.resnets.1.norm2.weight",
    "model.diffusion_model.output_blocks.1.0.out_layers.0.bias": "up_blocks.0.resnets.1.norm2.bias",
    "model.diffusion_model.output_blocks.1.0.out_layers.3.weight": "up_blocks.0.resnets.1.conv2.weight",
    "model.diffusion_model.output_blocks.1.0.out_layers.3.bias": "up_blocks.0.resnets.1.conv2.bias",
    "model.diffusion_model.output_blocks.1.0.skip_connection.weight": "up_blocks.0.resnets.1.conv_shortcut.weight",
    "model.diffusion_model.output_blocks.1.0.skip_connection.bias": "up_blocks.0.resnets.1.conv_shortcut.bias",
    "model.diffusion_model.output_blocks.2.0.in_layers.0.weight": "up_blocks.0.resnets.2.norm1.weight",
    "model.diffusion_model.output_blocks.2.0.in_layers.0.bias": "up_blocks.0.resnets.2.norm1.bias",
    "model.diffusion_model.output_blocks.2.0.in_layers.2.weight": "up_blocks.0.resnets.2.conv1.weight",
    "model.diffusion_model.output_blocks.2.0.in_layers.2.bias": "up_blocks.0.resnets.2.conv1.bias",
    "model.diffusion_model.output_blocks.2.0.emb_layers.1.weight": "up_blocks.0.resnets.2.time_emb_proj.weight",
    "model.diffusion_model.output_blocks.2.0.emb_layers.1.bias": "up_blocks.0.resnets.2.time_emb_proj.bias",
    "model.diffusion_model.output_blocks.2.0.out_layers.0.weight": "up_blocks.0.resnets.2.norm2.weight",
    "model.diffusion_model.output_blocks.2.0.out_layers.0.bias": "up_blocks.0.resnets.2.norm2.bias",
    "model.diffusion_model.output_blocks.2.0.out_layers.3.weight": "up_blocks.0.resnets.2.conv2.weight",
    "model.diffusion_model.output_blocks.2.0.out_layers.3.bias": "up_blocks.0.resnets.2.conv2.bias",
    "model.diffusion_model.output_blocks.2.0.skip_connection.weight": "up_blocks.0.resnets.2.conv_shortcut.weight",
    "model.diffusion_model.output_blocks.2.0.skip_connection.bias": "up_blocks.0.resnets.2.conv_shortcut.bias",
    "model.diffusion_model.output_blocks.2.1.conv.weight": "up_blocks.0.upsamplers.0.conv.weight",
    "model.diffusion_model.output_blocks.2.1.conv.bias": "up_blocks.0.upsamplers.0.conv.bias",
    "model.diffusion_model.output_blocks.3.0.in_layers.0.weight": "up_blocks.1.resnets.0.norm1.weight",
    "model.diffusion_model.output_blocks.3.0.in_layers.0.bias": "up_blocks.1.resnets.0.norm1.bias",
    "model.diffusion_model.output_blocks.3.0.in_layers.2.weight": "up_blocks.1.resnets.0.conv1.weight",
    "model.diffusion_model.output_blocks.3.0.in_layers.2.bias": "up_blocks.1.resnets.0.conv1.bias",
    "model.diffusion_model.output_blocks.3.0.emb_layers.1.weight": "up_blocks.1.resnets.0.time_emb_proj.weight",
    "model.diffusion_model.output_blocks.3.0.emb_layers.1.bias": "up_blocks.1.resnets.0.time_emb_proj.bias",
    "model.diffusion_model.output_blocks.3.0.out_layers.0.weight": "up_blocks.1.resnets.0.norm2.weight",
    "model.diffusion_model.output_blocks.3.0.out_layers.0.bias": "up_blocks.1.resnets.0.norm2.bias",
    "model.diffusion_model.output_blocks.3.0.out_layers.3.weight": "up_blocks.1.resnets.0.conv2.weight",
    "model.diffusion_model.output_blocks.3.0.out_layers.3.bias": "up_blocks.1.resnets.0.conv2.bias",
    "model.diffusion_model.output_blocks.3.0.skip_connection.weight": "up_blocks.1.resnets.0.conv_shortcut.weight",
    "model.diffusion_model.output_blocks.3.0.skip_connection.bias": "up_blocks.1.resnets.0.conv_shortcut.bias",
    "model.diffusion_model.output_blocks.3.1.norm.weight": "up_blocks.1.attentions.0.norm.weight",
    "model.diffusion_model.output_blocks.3.1.norm.bias": "up_blocks.1.attentions.0.norm.bias",
    "model.diffusion_model.output_blocks.3.1.proj_in.weight": "up_blocks.1.attentions.0.proj_in.weight",
    "model.diffusion_model.output_blocks.3.1.proj_in.bias": "up_blocks.1.attentions.0.proj_in.bias",
    "model.diffusion_model.output_blocks.3.1.transformer_blocks.0.attn1.to_q.weight": "up_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q.weight",
    "model.diffusion_model.output_blocks.3.1.transformer_blocks.0.attn1.to_k.weight": "up_blocks.1.attentions.0.transformer_blocks.0.attn1.to_k.weight",
    "model.diffusion_model.output_blocks.3.1.transformer_blocks.0.attn1.to_v.weight": "up_blocks.1.attentions.0.transformer_blocks.0.attn1.to_v.weight",
    "model.diffusion_model.output_blocks.3.1.transformer_blocks.0.attn1.to_out.0.weight": "up_blocks.1.attentions.0.transformer_blocks.0.attn1.to_out.0.weight",
    "model.diffusion_model.output_blocks.3.1.transformer_blocks.0.attn1.to_out.0.bias": "up_blocks.1.attentions.0.transformer_blocks.0.attn1.to_out.0.bias",
    "model.diffusion_model.output_blocks.3.1.transformer_blocks.0.ff.net.0.proj.weight": "up_blocks.1.attentions.0.transformer_blocks.0.ff.net.0.proj.weight",
    "model.diffusion_model.output_blocks.3.1.transformer_blocks.0.ff.net.0.proj.bias": "up_blocks.1.attentions.0.transformer_blocks.0.ff.net.0.proj.bias",
    "model.diffusion_model.output_blocks.3.1.transformer_blocks.0.ff.net.2.weight": "up_blocks.1.attentions.0.transformer_blocks.0.ff.net.2.weight",
    "model.diffusion_model.output_blocks.3.1.transformer_blocks.0.ff.net.2.bias": "up_blocks.1.attentions.0.transformer_blocks.0.ff.net.2.bias",
    "model.diffusion_model.output_blocks.3.1.transformer_blocks.0.attn2.to_q.weight": "up_blocks.1.attentions.0.transformer_blocks.0.attn2.to_q.weight",
    "model.diffusion_model.output_blocks.3.1.transformer_blocks.0.attn2.to_k.weight": "up_blocks.1.attentions.0.transformer_blocks.0.attn2.to_k.weight",
    "model.diffusion_model.output_blocks.3.1.transformer_blocks.0.attn2.to_v.weight": "up_blocks.1.attentions.0.transformer_blocks.0.attn2.to_v.weight",
    "model.diffusion_model.output_blocks.3.1.transformer_blocks.0.attn2.to_out.0.weight": "up_blocks.1.attentions.0.transformer_blocks.0.attn2.to_out.0.weight",
    "model.diffusion_model.output_blocks.3.1.transformer_blocks.0.attn2.to_out.0.bias": "up_blocks.1.attentions.0.transformer_blocks.0.attn2.to_out.0.bias",
    "model.diffusion_model.output_blocks.3.1.transformer_blocks.0.norm1.weight": "up_blocks.1.attentions.0.transformer_blocks.0.norm1.weight",
    "model.diffusion_model.output_blocks.3.1.transformer_blocks.0.norm1.bias": "up_blocks.1.attentions.0.transformer_blocks.0.norm1.bias",
    "model.diffusion_model.output_blocks.3.1.transformer_blocks.0.norm2.weight": "up_blocks.1.attentions.0.transformer_blocks.0.norm2.weight",
    "model.diffusion_model.output_blocks.3.1.transformer_blocks.0.norm2.bias": "up_blocks.1.attentions.0.transformer_blocks.0.norm2.bias",
    "model.diffusion_model.output_blocks.3.1.transformer_blocks.0.norm3.weight": "up_blocks.1.attentions.0.transformer_blocks.0.norm3.weight",
    "model.diffusion_model.output_blocks.3.1.transformer_blocks.0.norm3.bias": "up_blocks.1.attentions.0.transformer_blocks.0.norm3.bias",
    "model.diffusion_model.output_blocks.3.1.proj_out.weight": "up_blocks.1.attentions.0.proj_out.weight",
    "model.diffusion_model.output_blocks.3.1.proj_out.bias": "up_blocks.1.attentions.0.proj_out.bias",
    "model.diffusion_model.output_blocks.4.0.in_layers.0.weight": "up_blocks.1.resnets.1.norm1.weight",
    "model.diffusion_model.output_blocks.4.0.in_layers.0.bias": "up_blocks.1.resnets.1.norm1.bias",
    "model.diffusion_model.output_blocks.4.0.in_layers.2.weight": "up_blocks.1.resnets.1.conv1.weight",
    "model.diffusion_model.output_blocks.4.0.in_layers.2.bias": "up_blocks.1.resnets.1.conv1.bias",
    "model.diffusion_model.output_blocks.4.0.emb_layers.1.weight": "up_blocks.1.resnets.1.time_emb_proj.weight",
    "model.diffusion_model.output_blocks.4.0.emb_layers.1.bias": "up_blocks.1.resnets.1.time_emb_proj.bias",
    "model.diffusion_model.output_blocks.4.0.out_layers.0.weight": "up_blocks.1.resnets.1.norm2.weight",
    "model.diffusion_model.output_blocks.4.0.out_layers.0.bias": "up_blocks.1.resnets.1.norm2.bias",
    "model.diffusion_model.output_blocks.4.0.out_layers.3.weight": "up_blocks.1.resnets.1.conv2.weight",
    "model.diffusion_model.output_blocks.4.0.out_layers.3.bias": "up_blocks.1.resnets.1.conv2.bias",
    "model.diffusion_model.output_blocks.4.0.skip_connection.weight": "up_blocks.1.resnets.1.conv_shortcut.weight",
    "model.diffusion_model.output_blocks.4.0.skip_connection.bias": "up_blocks.1.resnets.1.conv_shortcut.bias",
    "model.diffusion_model.output_blocks.4.1.norm.weight": "up_blocks.1.attentions.1.norm.weight",
    "model.diffusion_model.output_blocks.4.1.norm.bias": "up_blocks.1.attentions.1.norm.bias",
    "model.diffusion_model.output_blocks.4.1.proj_in.weight": "up_blocks.1.attentions.1.proj_in.weight",
    "model.diffusion_model.output_blocks.4.1.proj_in.bias": "up_blocks.1.attentions.1.proj_in.bias",
    "model.diffusion_model.output_blocks.4.1.transformer_blocks.0.attn1.to_q.weight": "up_blocks.1.attentions.1.transformer_blocks.0.attn1.to_q.weight",
    "model.diffusion_model.output_blocks.4.1.transformer_blocks.0.attn1.to_k.weight": "up_blocks.1.attentions.1.transformer_blocks.0.attn1.to_k.weight",
    "model.diffusion_model.output_blocks.4.1.transformer_blocks.0.attn1.to_v.weight": "up_blocks.1.attentions.1.transformer_blocks.0.attn1.to_v.weight",
    "model.diffusion_model.output_blocks.4.1.transformer_blocks.0.attn1.to_out.0.weight": "up_blocks.1.attentions.1.transformer_blocks.0.attn1.to_out.0.weight",
    "model.diffusion_model.output_blocks.4.1.transformer_blocks.0.attn1.to_out.0.bias": "up_blocks.1.attentions.1.transformer_blocks.0.attn1.to_out.0.bias",
    "model.diffusion_model.output_blocks.4.1.transformer_blocks.0.ff.net.0.proj.weight": "up_blocks.1.attentions.1.transformer_blocks.0.ff.net.0.proj.weight",
    "model.diffusion_model.output_blocks.4.1.transformer_blocks.0.ff.net.0.proj.bias": "up_blocks.1.attentions.1.transformer_blocks.0.ff.net.0.proj.bias",
    "model.diffusion_model.output_blocks.4.1.transformer_blocks.0.ff.net.2.weight": "up_blocks.1.attentions.1.transformer_blocks.0.ff.net.2.weight",
    "model.diffusion_model.output_blocks.4.1.transformer_blocks.0.ff.net.2.bias": "up_blocks.1.attentions.1.transformer_blocks.0.ff.net.2.bias",
    "model.diffusion_model.output_blocks.4.1.transformer_blocks.0.attn2.to_q.weight": "up_blocks.1.attentions.1.transformer_blocks.0.attn2.to_q.weight",
    "model.diffusion_model.output_blocks.4.1.transformer_blocks.0.attn2.to_k.weight": "up_blocks.1.attentions.1.transformer_blocks.0.attn2.to_k.weight",
    "model.diffusion_model.output_blocks.4.1.transformer_blocks.0.attn2.to_v.weight": "up_blocks.1.attentions.1.transformer_blocks.0.attn2.to_v.weight",
    "model.diffusion_model.output_blocks.4.1.transformer_blocks.0.attn2.to_out.0.weight": "up_blocks.1.attentions.1.transformer_blocks.0.attn2.to_out.0.weight",
    "model.diffusion_model.output_blocks.4.1.transformer_blocks.0.attn2.to_out.0.bias": "up_blocks.1.attentions.1.transformer_blocks.0.attn2.to_out.0.bias",
    "model.diffusion_model.output_blocks.4.1.transformer_blocks.0.norm1.weight": "up_blocks.1.attentions.1.transformer_blocks.0.norm1.weight",
    "model.diffusion_model.output_blocks.4.1.transformer_blocks.0.norm1.bias": "up_blocks.1.attentions.1.transformer_blocks.0.norm1.bias",
    "model.diffusion_model.output_blocks.4.1.transformer_blocks.0.norm2.weight": "up_blocks.1.attentions.1.transformer_blocks.0.norm2.weight",
    "model.diffusion_model.output_blocks.4.1.transformer_blocks.0.norm2.bias": "up_blocks.1.attentions.1.transformer_blocks.0.norm2.bias",
    "model.diffusion_model.output_blocks.4.1.transformer_blocks.0.norm3.weight": "up_blocks.1.attentions.1.transformer_blocks.0.norm3.weight",
    "model.diffusion_model.output_blocks.4.1.transformer_blocks.0.norm3.bias": "up_blocks.1.attentions.1.transformer_blocks.0.norm3.bias",
    "model.diffusion_model.output_blocks.4.1.proj_out.weight": "up_blocks.1.attentions.1.proj_out.weight",
    "model.diffusion_model.output_blocks.4.1.proj_out.bias": "up_blocks.1.attentions.1.proj_out.bias",
    "model.diffusion_model.output_blocks.5.0.in_layers.0.weight": "up_blocks.1.resnets.2.norm1.weight",
    "model.diffusion_model.output_blocks.5.0.in_layers.0.bias": "up_blocks.1.resnets.2.norm1.bias",
    "model.diffusion_model.output_blocks.5.0.in_layers.2.weight": "up_blocks.1.resnets.2.conv1.weight",
    "model.diffusion_model.output_blocks.5.0.in_layers.2.bias": "up_blocks.1.resnets.2.conv1.bias",
    "model.diffusion_model.output_blocks.5.0.emb_layers.1.weight": "up_blocks.1.resnets.2.time_emb_proj.weight",
    "model.diffusion_model.output_blocks.5.0.emb_layers.1.bias": "up_blocks.1.resnets.2.time_emb_proj.bias",
    "model.diffusion_model.output_blocks.5.0.out_layers.0.weight": "up_blocks.1.resnets.2.norm2.weight",
    "model.diffusion_model.output_blocks.5.0.out_layers.0.bias": "up_blocks.1.resnets.2.norm2.bias",
    "model.diffusion_model.output_blocks.5.0.out_layers.3.weight": "up_blocks.1.resnets.2.conv2.weight",
    "model.diffusion_model.output_blocks.5.0.out_layers.3.bias": "up_blocks.1.resnets.2.conv2.bias",
    "model.diffusion_model.output_blocks.5.0.skip_connection.weight": "up_blocks.1.resnets.2.conv_shortcut.weight",
    "model.diffusion_model.output_blocks.5.0.skip_connection.bias": "up_blocks.1.resnets.2.conv_shortcut.bias",
    "model.diffusion_model.output_blocks.5.2.conv.weight": "up_blocks.1.upsamplers.0.conv.weight",
    "model.diffusion_model.output_blocks.5.2.conv.bias": "up_blocks.1.upsamplers.0.conv.bias",
    "model.diffusion_model.output_blocks.5.1.norm.weight": "up_blocks.1.attentions.2.norm.weight",
    "model.diffusion_model.output_blocks.5.1.norm.bias": "up_blocks.1.attentions.2.norm.bias",
    "model.diffusion_model.output_blocks.5.1.proj_in.weight": "up_blocks.1.attentions.2.proj_in.weight",
    "model.diffusion_model.output_blocks.5.1.proj_in.bias": "up_blocks.1.attentions.2.proj_in.bias",
    "model.diffusion_model.output_blocks.5.1.transformer_blocks.0.attn1.to_q.weight": "up_blocks.1.attentions.2.transformer_blocks.0.attn1.to_q.weight",
    "model.diffusion_model.output_blocks.5.1.transformer_blocks.0.attn1.to_k.weight": "up_blocks.1.attentions.2.transformer_blocks.0.attn1.to_k.weight",
    "model.diffusion_model.output_blocks.5.1.transformer_blocks.0.attn1.to_v.weight": "up_blocks.1.attentions.2.transformer_blocks.0.attn1.to_v.weight",
    "model.diffusion_model.output_blocks.5.1.transformer_blocks.0.attn1.to_out.0.weight": "up_blocks.1.attentions.2.transformer_blocks.0.attn1.to_out.0.weight",
    "model.diffusion_model.output_blocks.5.1.transformer_blocks.0.attn1.to_out.0.bias": "up_blocks.1.attentions.2.transformer_blocks.0.attn1.to_out.0.bias",
    "model.diffusion_model.output_blocks.5.1.transformer_blocks.0.ff.net.0.proj.weight": "up_blocks.1.attentions.2.transformer_blocks.0.ff.net.0.proj.weight",
    "model.diffusion_model.output_blocks.5.1.transformer_blocks.0.ff.net.0.proj.bias": "up_blocks.1.attentions.2.transformer_blocks.0.ff.net.0.proj.bias",
    "model.diffusion_model.output_blocks.5.1.transformer_blocks.0.ff.net.2.weight": "up_blocks.1.attentions.2.transformer_blocks.0.ff.net.2.weight",
    "model.diffusion_model.output_blocks.5.1.transformer_blocks.0.ff.net.2.bias": "up_blocks.1.attentions.2.transformer_blocks.0.ff.net.2.bias",
    "model.diffusion_model.output_blocks.5.1.transformer_blocks.0.attn2.to_q.weight": "up_blocks.1.attentions.2.transformer_blocks.0.attn2.to_q.weight",
    "model.diffusion_model.output_blocks.5.1.transformer_blocks.0.attn2.to_k.weight": "up_blocks.1.attentions.2.transformer_blocks.0.attn2.to_k.weight",
    "model.diffusion_model.output_blocks.5.1.transformer_blocks.0.attn2.to_v.weight": "up_blocks.1.attentions.2.transformer_blocks.0.attn2.to_v.weight",
    "model.diffusion_model.output_blocks.5.1.transformer_blocks.0.attn2.to_out.0.weight": "up_blocks.1.attentions.2.transformer_blocks.0.attn2.to_out.0.weight",
    "model.diffusion_model.output_blocks.5.1.transformer_blocks.0.attn2.to_out.0.bias": "up_blocks.1.attentions.2.transformer_blocks.0.attn2.to_out.0.bias",
    "model.diffusion_model.output_blocks.5.1.transformer_blocks.0.norm1.weight": "up_blocks.1.attentions.2.transformer_blocks.0.norm1.weight",
    "model.diffusion_model.output_blocks.5.1.transformer_blocks.0.norm1.bias": "up_blocks.1.attentions.2.transformer_blocks.0.norm1.bias",
    "model.diffusion_model.output_blocks.5.1.transformer_blocks.0.norm2.weight": "up_blocks.1.attentions.2.transformer_blocks.0.norm2.weight",
    "model.diffusion_model.output_blocks.5.1.transformer_blocks.0.norm2.bias": "up_blocks.1.attentions.2.transformer_blocks.0.norm2.bias",
    "model.diffusion_model.output_blocks.5.1.transformer_blocks.0.norm3.weight": "up_blocks.1.attentions.2.transformer_blocks.0.norm3.weight",
    "model.diffusion_model.output_blocks.5.1.transformer_blocks.0.norm3.bias": "up_blocks.1.attentions.2.transformer_blocks.0.norm3.bias",
    "model.diffusion_model.output_blocks.5.1.proj_out.weight": "up_blocks.1.attentions.2.proj_out.weight",
    "model.diffusion_model.output_blocks.5.1.proj_out.bias": "up_blocks.1.attentions.2.proj_out.bias",
    "model.diffusion_model.output_blocks.6.0.in_layers.0.weight": "up_blocks.2.resnets.0.norm1.weight",
    "model.diffusion_model.output_blocks.6.0.in_layers.0.bias": "up_blocks.2.resnets.0.norm1.bias",
    "model.diffusion_model.output_blocks.6.0.in_layers.2.weight": "up_blocks.2.resnets.0.conv1.weight",
    "model.diffusion_model.output_blocks.6.0.in_layers.2.bias": "up_blocks.2.resnets.0.conv1.bias",
    "model.diffusion_model.output_blocks.6.0.emb_layers.1.weight": "up_blocks.2.resnets.0.time_emb_proj.weight",
    "model.diffusion_model.output_blocks.6.0.emb_layers.1.bias": "up_blocks.2.resnets.0.time_emb_proj.bias",
    "model.diffusion_model.output_blocks.6.0.out_layers.0.weight": "up_blocks.2.resnets.0.norm2.weight",
    "model.diffusion_model.output_blocks.6.0.out_layers.0.bias": "up_blocks.2.resnets.0.norm2.bias",
    "model.diffusion_model.output_blocks.6.0.out_layers.3.weight": "up_blocks.2.resnets.0.conv2.weight",
    "model.diffusion_model.output_blocks.6.0.out_layers.3.bias": "up_blocks.2.resnets.0.conv2.bias",
    "model.diffusion_model.output_blocks.6.0.skip_connection.weight": "up_blocks.2.resnets.0.conv_shortcut.weight",
    "model.diffusion_model.output_blocks.6.0.skip_connection.bias": "up_blocks.2.resnets.0.conv_shortcut.bias",
    "model.diffusion_model.output_blocks.6.1.norm.weight": "up_blocks.2.attentions.0.norm.weight",
    "model.diffusion_model.output_blocks.6.1.norm.bias": "up_blocks.2.attentions.0.norm.bias",
    "model.diffusion_model.output_blocks.6.1.proj_in.weight": "up_blocks.2.attentions.0.proj_in.weight",
    "model.diffusion_model.output_blocks.6.1.proj_in.bias": "up_blocks.2.attentions.0.proj_in.bias",
    "model.diffusion_model.output_blocks.6.1.transformer_blocks.0.attn1.to_q.weight": "up_blocks.2.attentions.0.transformer_blocks.0.attn1.to_q.weight",
    "model.diffusion_model.output_blocks.6.1.transformer_blocks.0.attn1.to_k.weight": "up_blocks.2.attentions.0.transformer_blocks.0.attn1.to_k.weight",
    "model.diffusion_model.output_blocks.6.1.transformer_blocks.0.attn1.to_v.weight": "up_blocks.2.attentions.0.transformer_blocks.0.attn1.to_v.weight",
    "model.diffusion_model.output_blocks.6.1.transformer_blocks.0.attn1.to_out.0.weight": "up_blocks.2.attentions.0.transformer_blocks.0.attn1.to_out.0.weight",
    "model.diffusion_model.output_blocks.6.1.transformer_blocks.0.attn1.to_out.0.bias": "up_blocks.2.attentions.0.transformer_blocks.0.attn1.to_out.0.bias",
    "model.diffusion_model.output_blocks.6.1.transformer_blocks.0.ff.net.0.proj.weight": "up_blocks.2.attentions.0.transformer_blocks.0.ff.net.0.proj.weight",
    "model.diffusion_model.output_blocks.6.1.transformer_blocks.0.ff.net.0.proj.bias": "up_blocks.2.attentions.0.transformer_blocks.0.ff.net.0.proj.bias",
    "model.diffusion_model.output_blocks.6.1.transformer_blocks.0.ff.net.2.weight": "up_blocks.2.attentions.0.transformer_blocks.0.ff.net.2.weight",
    "model.diffusion_model.output_blocks.6.1.transformer_blocks.0.ff.net.2.bias": "up_blocks.2.attentions.0.transformer_blocks.0.ff.net.2.bias",
    "model.diffusion_model.output_blocks.6.1.transformer_blocks.0.attn2.to_q.weight": "up_blocks.2.attentions.0.transformer_blocks.0.attn2.to_q.weight",
    "model.diffusion_model.output_blocks.6.1.transformer_blocks.0.attn2.to_k.weight": "up_blocks.2.attentions.0.transformer_blocks.0.attn2.to_k.weight",
    "model.diffusion_model.output_blocks.6.1.transformer_blocks.0.attn2.to_v.weight": "up_blocks.2.attentions.0.transformer_blocks.0.attn2.to_v.weight",
    "model.diffusion_model.output_blocks.6.1.transformer_blocks.0.attn2.to_out.0.weight": "up_blocks.2.attentions.0.transformer_blocks.0.attn2.to_out.0.weight",
    "model.diffusion_model.output_blocks.6.1.transformer_blocks.0.attn2.to_out.0.bias": "up_blocks.2.attentions.0.transformer_blocks.0.attn2.to_out.0.bias",
    "model.diffusion_model.output_blocks.6.1.transformer_blocks.0.norm1.weight": "up_blocks.2.attentions.0.transformer_blocks.0.norm1.weight",
    "model.diffusion_model.output_blocks.6.1.transformer_blocks.0.norm1.bias": "up_blocks.2.attentions.0.transformer_blocks.0.norm1.bias",
    "model.diffusion_model.output_blocks.6.1.transformer_blocks.0.norm2.weight": "up_blocks.2.attentions.0.transformer_blocks.0.norm2.weight",
    "model.diffusion_model.output_blocks.6.1.transformer_blocks.0.norm2.bias": "up_blocks.2.attentions.0.transformer_blocks.0.norm2.bias",
    "model.diffusion_model.output_blocks.6.1.transformer_blocks.0.norm3.weight": "up_blocks.2.attentions.0.transformer_blocks.0.norm3.weight",
    "model.diffusion_model.output_blocks.6.1.transformer_blocks.0.norm3.bias": "up_blocks.2.attentions.0.transformer_blocks.0.norm3.bias",
    "model.diffusion_model.output_blocks.6.1.proj_out.weight": "up_blocks.2.attentions.0.proj_out.weight",
    "model.diffusion_model.output_blocks.6.1.proj_out.bias": "up_blocks.2.attentions.0.proj_out.bias",
    "model.diffusion_model.output_blocks.7.0.in_layers.0.weight": "up_blocks.2.resnets.1.norm1.weight",
    "model.diffusion_model.output_blocks.7.0.in_layers.0.bias": "up_blocks.2.resnets.1.norm1.bias",
    "model.diffusion_model.output_blocks.7.0.in_layers.2.weight": "up_blocks.2.resnets.1.conv1.weight",
    "model.diffusion_model.output_blocks.7.0.in_layers.2.bias": "up_blocks.2.resnets.1.conv1.bias",
    "model.diffusion_model.output_blocks.7.0.emb_layers.1.weight": "up_blocks.2.resnets.1.time_emb_proj.weight",
    "model.diffusion_model.output_blocks.7.0.emb_layers.1.bias": "up_blocks.2.resnets.1.time_emb_proj.bias",
    "model.diffusion_model.output_blocks.7.0.out_layers.0.weight": "up_blocks.2.resnets.1.norm2.weight",
    "model.diffusion_model.output_blocks.7.0.out_layers.0.bias": "up_blocks.2.resnets.1.norm2.bias",
    "model.diffusion_model.output_blocks.7.0.out_layers.3.weight": "up_blocks.2.resnets.1.conv2.weight",
    "model.diffusion_model.output_blocks.7.0.out_layers.3.bias": "up_blocks.2.resnets.1.conv2.bias",
    "model.diffusion_model.output_blocks.7.0.skip_connection.weight": "up_blocks.2.resnets.1.conv_shortcut.weight",
    "model.diffusion_model.output_blocks.7.0.skip_connection.bias": "up_blocks.2.resnets.1.conv_shortcut.bias",
    "model.diffusion_model.output_blocks.7.1.norm.weight": "up_blocks.2.attentions.1.norm.weight",
    "model.diffusion_model.output_blocks.7.1.norm.bias": "up_blocks.2.attentions.1.norm.bias",
    "model.diffusion_model.output_blocks.7.1.proj_in.weight": "up_blocks.2.attentions.1.proj_in.weight",
    "model.diffusion_model.output_blocks.7.1.proj_in.bias": "up_blocks.2.attentions.1.proj_in.bias",
    "model.diffusion_model.output_blocks.7.1.transformer_blocks.0.attn1.to_q.weight": "up_blocks.2.attentions.1.transformer_blocks.0.attn1.to_q.weight",
    "model.diffusion_model.output_blocks.7.1.transformer_blocks.0.attn1.to_k.weight": "up_blocks.2.attentions.1.transformer_blocks.0.attn1.to_k.weight",
    "model.diffusion_model.output_blocks.7.1.transformer_blocks.0.attn1.to_v.weight": "up_blocks.2.attentions.1.transformer_blocks.0.attn1.to_v.weight",
    "model.diffusion_model.output_blocks.7.1.transformer_blocks.0.attn1.to_out.0.weight": "up_blocks.2.attentions.1.transformer_blocks.0.attn1.to_out.0.weight",
    "model.diffusion_model.output_blocks.7.1.transformer_blocks.0.attn1.to_out.0.bias": "up_blocks.2.attentions.1.transformer_blocks.0.attn1.to_out.0.bias",
    "model.diffusion_model.output_blocks.7.1.transformer_blocks.0.ff.net.0.proj.weight": "up_blocks.2.attentions.1.transformer_blocks.0.ff.net.0.proj.weight",
    "model.diffusion_model.output_blocks.7.1.transformer_blocks.0.ff.net.0.proj.bias": "up_blocks.2.attentions.1.transformer_blocks.0.ff.net.0.proj.bias",
    "model.diffusion_model.output_blocks.7.1.transformer_blocks.0.ff.net.2.weight": "up_blocks.2.attentions.1.transformer_blocks.0.ff.net.2.weight",
    "model.diffusion_model.output_blocks.7.1.transformer_blocks.0.ff.net.2.bias": "up_blocks.2.attentions.1.transformer_blocks.0.ff.net.2.bias",
    "model.diffusion_model.output_blocks.7.1.transformer_blocks.0.attn2.to_q.weight": "up_blocks.2.attentions.1.transformer_blocks.0.attn2.to_q.weight",
    "model.diffusion_model.output_blocks.7.1.transformer_blocks.0.attn2.to_k.weight": "up_blocks.2.attentions.1.transformer_blocks.0.attn2.to_k.weight",
    "model.diffusion_model.output_blocks.7.1.transformer_blocks.0.attn2.to_v.weight": "up_blocks.2.attentions.1.transformer_blocks.0.attn2.to_v.weight",
    "model.diffusion_model.output_blocks.7.1.transformer_blocks.0.attn2.to_out.0.weight": "up_blocks.2.attentions.1.transformer_blocks.0.attn2.to_out.0.weight",
    "model.diffusion_model.output_blocks.7.1.transformer_blocks.0.attn2.to_out.0.bias": "up_blocks.2.attentions.1.transformer_blocks.0.attn2.to_out.0.bias",
    "model.diffusion_model.output_blocks.7.1.transformer_blocks.0.norm1.weight": "up_blocks.2.attentions.1.transformer_blocks.0.norm1.weight",
    "model.diffusion_model.output_blocks.7.1.transformer_blocks.0.norm1.bias": "up_blocks.2.attentions.1.transformer_blocks.0.norm1.bias",
    "model.diffusion_model.output_blocks.7.1.transformer_blocks.0.norm2.weight": "up_blocks.2.attentions.1.transformer_blocks.0.norm2.weight",
    "model.diffusion_model.output_blocks.7.1.transformer_blocks.0.norm2.bias": "up_blocks.2.attentions.1.transformer_blocks.0.norm2.bias",
    "model.diffusion_model.output_blocks.7.1.transformer_blocks.0.norm3.weight": "up_blocks.2.attentions.1.transformer_blocks.0.norm3.weight",
    "model.diffusion_model.output_blocks.7.1.transformer_blocks.0.norm3.bias": "up_blocks.2.attentions.1.transformer_blocks.0.norm3.bias",
    "model.diffusion_model.output_blocks.7.1.proj_out.weight": "up_blocks.2.attentions.1.proj_out.weight",
    "model.diffusion_model.output_blocks.7.1.proj_out.bias": "up_blocks.2.attentions.1.proj_out.bias",
    "model.diffusion_model.output_blocks.8.0.in_layers.0.weight": "up_blocks.2.resnets.2.norm1.weight",
    "model.diffusion_model.output_blocks.8.0.in_layers.0.bias": "up_blocks.2.resnets.2.norm1.bias",
    "model.diffusion_model.output_blocks.8.0.in_layers.2.weight": "up_blocks.2.resnets.2.conv1.weight",
    "model.diffusion_model.output_blocks.8.0.in_layers.2.bias": "up_blocks.2.resnets.2.conv1.bias",
    "model.diffusion_model.output_blocks.8.0.emb_layers.1.weight": "up_blocks.2.resnets.2.time_emb_proj.weight",
    "model.diffusion_model.output_blocks.8.0.emb_layers.1.bias": "up_blocks.2.resnets.2.time_emb_proj.bias",
    "model.diffusion_model.output_blocks.8.0.out_layers.0.weight": "up_blocks.2.resnets.2.norm2.weight",
    "model.diffusion_model.output_blocks.8.0.out_layers.0.bias": "up_blocks.2.resnets.2.norm2.bias",
    "model.diffusion_model.output_blocks.8.0.out_layers.3.weight": "up_blocks.2.resnets.2.conv2.weight",
    "model.diffusion_model.output_blocks.8.0.out_layers.3.bias": "up_blocks.2.resnets.2.conv2.bias",
    "model.diffusion_model.output_blocks.8.0.skip_connection.weight": "up_blocks.2.resnets.2.conv_shortcut.weight",
    "model.diffusion_model.output_blocks.8.0.skip_connection.bias": "up_blocks.2.resnets.2.conv_shortcut.bias",
    "model.diffusion_model.output_blocks.8.2.conv.weight": "up_blocks.2.upsamplers.0.conv.weight",
    "model.diffusion_model.output_blocks.8.2.conv.bias": "up_blocks.2.upsamplers.0.conv.bias",
    "model.diffusion_model.output_blocks.8.1.norm.weight": "up_blocks.2.attentions.2.norm.weight",
    "model.diffusion_model.output_blocks.8.1.norm.bias": "up_blocks.2.attentions.2.norm.bias",
    "model.diffusion_model.output_blocks.8.1.proj_in.weight": "up_blocks.2.attentions.2.proj_in.weight",
    "model.diffusion_model.output_blocks.8.1.proj_in.bias": "up_blocks.2.attentions.2.proj_in.bias",
    "model.diffusion_model.output_blocks.8.1.transformer_blocks.0.attn1.to_q.weight": "up_blocks.2.attentions.2.transformer_blocks.0.attn1.to_q.weight",
    "model.diffusion_model.output_blocks.8.1.transformer_blocks.0.attn1.to_k.weight": "up_blocks.2.attentions.2.transformer_blocks.0.attn1.to_k.weight",
    "model.diffusion_model.output_blocks.8.1.transformer_blocks.0.attn1.to_v.weight": "up_blocks.2.attentions.2.transformer_blocks.0.attn1.to_v.weight",
    "model.diffusion_model.output_blocks.8.1.transformer_blocks.0.attn1.to_out.0.weight": "up_blocks.2.attentions.2.transformer_blocks.0.attn1.to_out.0.weight",
    "model.diffusion_model.output_blocks.8.1.transformer_blocks.0.attn1.to_out.0.bias": "up_blocks.2.attentions.2.transformer_blocks.0.attn1.to_out.0.bias",
    "model.diffusion_model.output_blocks.8.1.transformer_blocks.0.ff.net.0.proj.weight": "up_blocks.2.attentions.2.transformer_blocks.0.ff.net.0.proj.weight",
    "model.diffusion_model.output_blocks.8.1.transformer_blocks.0.ff.net.0.proj.bias": "up_blocks.2.attentions.2.transformer_blocks.0.ff.net.0.proj.bias",
    "model.diffusion_model.output_blocks.8.1.transformer_blocks.0.ff.net.2.weight": "up_blocks.2.attentions.2.transformer_blocks.0.ff.net.2.weight",
    "model.diffusion_model.output_blocks.8.1.transformer_blocks.0.ff.net.2.bias": "up_blocks.2.attentions.2.transformer_blocks.0.ff.net.2.bias",
    "model.diffusion_model.output_blocks.8.1.transformer_blocks.0.attn2.to_q.weight": "up_blocks.2.attentions.2.transformer_blocks.0.attn2.to_q.weight",
    "model.diffusion_model.output_blocks.8.1.transformer_blocks.0.attn2.to_k.weight": "up_blocks.2.attentions.2.transformer_blocks.0.attn2.to_k.weight",
    "model.diffusion_model.output_blocks.8.1.transformer_blocks.0.attn2.to_v.weight": "up_blocks.2.attentions.2.transformer_blocks.0.attn2.to_v.weight",
    "model.diffusion_model.output_blocks.8.1.transformer_blocks.0.attn2.to_out.0.weight": "up_blocks.2.attentions.2.transformer_blocks.0.attn2.to_out.0.weight",
    "model.diffusion_model.output_blocks.8.1.transformer_blocks.0.attn2.to_out.0.bias": "up_blocks.2.attentions.2.transformer_blocks.0.attn2.to_out.0.bias",
    "model.diffusion_model.output_blocks.8.1.transformer_blocks.0.norm1.weight": "up_blocks.2.attentions.2.transformer_blocks.0.norm1.weight",
    "model.diffusion_model.output_blocks.8.1.transformer_blocks.0.norm1.bias": "up_blocks.2.attentions.2.transformer_blocks.0.norm1.bias",
    "model.diffusion_model.output_blocks.8.1.transformer_blocks.0.norm2.weight": "up_blocks.2.attentions.2.transformer_blocks.0.norm2.weight",
    "model.diffusion_model.output_blocks.8.1.transformer_blocks.0.norm2.bias": "up_blocks.2.attentions.2.transformer_blocks.0.norm2.bias",
    "model.diffusion_model.output_blocks.8.1.transformer_blocks.0.norm3.weight": "up_blocks.2.attentions.2.transformer_blocks.0.norm3.weight",
    "model.diffusion_model.output_blocks.8.1.transformer_blocks.0.norm3.bias": "up_blocks.2.attentions.2.transformer_blocks.0.norm3.bias",
    "model.diffusion_model.output_blocks.8.1.proj_out.weight": "up_blocks.2.attentions.2.proj_out.weight",
    "model.diffusion_model.output_blocks.8.1.proj_out.bias": "up_blocks.2.attentions.2.proj_out.bias",
    "model.diffusion_model.output_blocks.9.0.in_layers.0.weight": "up_blocks.3.resnets.0.norm1.weight",
    "model.diffusion_model.output_blocks.9.0.in_layers.0.bias": "up_blocks.3.resnets.0.norm1.bias",
    "model.diffusion_model.output_blocks.9.0.in_layers.2.weight": "up_blocks.3.resnets.0.conv1.weight",
    "model.diffusion_model.output_blocks.9.0.in_layers.2.bias": "up_blocks.3.resnets.0.conv1.bias",
    "model.diffusion_model.output_blocks.9.0.emb_layers.1.weight": "up_blocks.3.resnets.0.time_emb_proj.weight",
    "model.diffusion_model.output_blocks.9.0.emb_layers.1.bias": "up_blocks.3.resnets.0.time_emb_proj.bias",
    "model.diffusion_model.output_blocks.9.0.out_layers.0.weight": "up_blocks.3.resnets.0.norm2.weight",
    "model.diffusion_model.output_blocks.9.0.out_layers.0.bias": "up_blocks.3.resnets.0.norm2.bias",
    "model.diffusion_model.output_blocks.9.0.out_layers.3.weight": "up_blocks.3.resnets.0.conv2.weight",
    "model.diffusion_model.output_blocks.9.0.out_layers.3.bias": "up_blocks.3.resnets.0.conv2.bias",
    "model.diffusion_model.output_blocks.9.0.skip_connection.weight": "up_blocks.3.resnets.0.conv_shortcut.weight",
    "model.diffusion_model.output_blocks.9.0.skip_connection.bias": "up_blocks.3.resnets.0.conv_shortcut.bias",
    "model.diffusion_model.output_blocks.9.1.norm.weight": "up_blocks.3.attentions.0.norm.weight",
    "model.diffusion_model.output_blocks.9.1.norm.bias": "up_blocks.3.attentions.0.norm.bias",
    "model.diffusion_model.output_blocks.9.1.proj_in.weight": "up_blocks.3.attentions.0.proj_in.weight",
    "model.diffusion_model.output_blocks.9.1.proj_in.bias": "up_blocks.3.attentions.0.proj_in.bias",
    "model.diffusion_model.output_blocks.9.1.transformer_blocks.0.attn1.to_q.weight": "up_blocks.3.attentions.0.transformer_blocks.0.attn1.to_q.weight",
    "model.diffusion_model.output_blocks.9.1.transformer_blocks.0.attn1.to_k.weight": "up_blocks.3.attentions.0.transformer_blocks.0.attn1.to_k.weight",
    "model.diffusion_model.output_blocks.9.1.transformer_blocks.0.attn1.to_v.weight": "up_blocks.3.attentions.0.transformer_blocks.0.attn1.to_v.weight",
    "model.diffusion_model.output_blocks.9.1.transformer_blocks.0.attn1.to_out.0.weight": "up_blocks.3.attentions.0.transformer_blocks.0.attn1.to_out.0.weight",
    "model.diffusion_model.output_blocks.9.1.transformer_blocks.0.attn1.to_out.0.bias": "up_blocks.3.attentions.0.transformer_blocks.0.attn1.to_out.0.bias",
    "model.diffusion_model.output_blocks.9.1.transformer_blocks.0.ff.net.0.proj.weight": "up_blocks.3.attentions.0.transformer_blocks.0.ff.net.0.proj.weight",
    "model.diffusion_model.output_blocks.9.1.transformer_blocks.0.ff.net.0.proj.bias": "up_blocks.3.attentions.0.transformer_blocks.0.ff.net.0.proj.bias",
    "model.diffusion_model.output_blocks.9.1.transformer_blocks.0.ff.net.2.weight": "up_blocks.3.attentions.0.transformer_blocks.0.ff.net.2.weight",
    "model.diffusion_model.output_blocks.9.1.transformer_blocks.0.ff.net.2.bias": "up_blocks.3.attentions.0.transformer_blocks.0.ff.net.2.bias",
    "model.diffusion_model.output_blocks.9.1.transformer_blocks.0.attn2.to_q.weight": "up_blocks.3.attentions.0.transformer_blocks.0.attn2.to_q.weight",
    "model.diffusion_model.output_blocks.9.1.transformer_blocks.0.attn2.to_k.weight": "up_blocks.3.attentions.0.transformer_blocks.0.attn2.to_k.weight",
    "model.diffusion_model.output_blocks.9.1.transformer_blocks.0.attn2.to_v.weight": "up_blocks.3.attentions.0.transformer_blocks.0.attn2.to_v.weight",
    "model.diffusion_model.output_blocks.9.1.transformer_blocks.0.attn2.to_out.0.weight": "up_blocks.3.attentions.0.transformer_blocks.0.attn2.to_out.0.weight",
    "model.diffusion_model.output_blocks.9.1.transformer_blocks.0.attn2.to_out.0.bias": "up_blocks.3.attentions.0.transformer_blocks.0.attn2.to_out.0.bias",
    "model.diffusion_model.output_blocks.9.1.transformer_blocks.0.norm1.weight": "up_blocks.3.attentions.0.transformer_blocks.0.norm1.weight",
    "model.diffusion_model.output_blocks.9.1.transformer_blocks.0.norm1.bias": "up_blocks.3.attentions.0.transformer_blocks.0.norm1.bias",
    "model.diffusion_model.output_blocks.9.1.transformer_blocks.0.norm2.weight": "up_blocks.3.attentions.0.transformer_blocks.0.norm2.weight",
    "model.diffusion_model.output_blocks.9.1.transformer_blocks.0.norm2.bias": "up_blocks.3.attentions.0.transformer_blocks.0.norm2.bias",
    "model.diffusion_model.output_blocks.9.1.transformer_blocks.0.norm3.weight": "up_blocks.3.attentions.0.transformer_blocks.0.norm3.weight",
    "model.diffusion_model.output_blocks.9.1.transformer_blocks.0.norm3.bias": "up_blocks.3.attentions.0.transformer_blocks.0.norm3.bias",
    "model.diffusion_model.output_blocks.9.1.proj_out.weight": "up_blocks.3.attentions.0.proj_out.weight",
    "model.diffusion_model.output_blocks.9.1.proj_out.bias": "up_blocks.3.attentions.0.proj_out.bias",
    "model.diffusion_model.output_blocks.10.0.in_layers.0.weight": "up_blocks.3.resnets.1.norm1.weight",
    "model.diffusion_model.output_blocks.10.0.in_layers.0.bias": "up_blocks.3.resnets.1.norm1.bias",
    "model.diffusion_model.output_blocks.10.0.in_layers.2.weight": "up_blocks.3.resnets.1.conv1.weight",
    "model.diffusion_model.output_blocks.10.0.in_layers.2.bias": "up_blocks.3.resnets.1.conv1.bias",
    "model.diffusion_model.output_blocks.10.0.emb_layers.1.weight": "up_blocks.3.resnets.1.time_emb_proj.weight",
    "model.diffusion_model.output_blocks.10.0.emb_layers.1.bias": "up_blocks.3.resnets.1.time_emb_proj.bias",
    "model.diffusion_model.output_blocks.10.0.out_layers.0.weight": "up_blocks.3.resnets.1.norm2.weight",
    "model.diffusion_model.output_blocks.10.0.out_layers.0.bias": "up_blocks.3.resnets.1.norm2.bias",
    "model.diffusion_model.output_blocks.10.0.out_layers.3.weight": "up_blocks.3.resnets.1.conv2.weight",
    "model.diffusion_model.output_blocks.10.0.out_layers.3.bias": "up_blocks.3.resnets.1.conv2.bias",
    "model.diffusion_model.output_blocks.10.0.skip_connection.weight": "up_blocks.3.resnets.1.conv_shortcut.weight",
    "model.diffusion_model.output_blocks.10.0.skip_connection.bias": "up_blocks.3.resnets.1.conv_shortcut.bias",
    "model.diffusion_model.output_blocks.10.1.norm.weight": "up_blocks.3.attentions.1.norm.weight",
    "model.diffusion_model.output_blocks.10.1.norm.bias": "up_blocks.3.attentions.1.norm.bias",
    "model.diffusion_model.output_blocks.10.1.proj_in.weight": "up_blocks.3.attentions.1.proj_in.weight",
    "model.diffusion_model.output_blocks.10.1.proj_in.bias": "up_blocks.3.attentions.1.proj_in.bias",
    "model.diffusion_model.output_blocks.10.1.transformer_blocks.0.attn1.to_q.weight": "up_blocks.3.attentions.1.transformer_blocks.0.attn1.to_q.weight",
    "model.diffusion_model.output_blocks.10.1.transformer_blocks.0.attn1.to_k.weight": "up_blocks.3.attentions.1.transformer_blocks.0.attn1.to_k.weight",
    "model.diffusion_model.output_blocks.10.1.transformer_blocks.0.attn1.to_v.weight": "up_blocks.3.attentions.1.transformer_blocks.0.attn1.to_v.weight",
    "model.diffusion_model.output_blocks.10.1.transformer_blocks.0.attn1.to_out.0.weight": "up_blocks.3.attentions.1.transformer_blocks.0.attn1.to_out.0.weight",
    "model.diffusion_model.output_blocks.10.1.transformer_blocks.0.attn1.to_out.0.bias": "up_blocks.3.attentions.1.transformer_blocks.0.attn1.to_out.0.bias",
    "model.diffusion_model.output_blocks.10.1.transformer_blocks.0.ff.net.0.proj.weight": "up_blocks.3.attentions.1.transformer_blocks.0.ff.net.0.proj.weight",
    "model.diffusion_model.output_blocks.10.1.transformer_blocks.0.ff.net.0.proj.bias": "up_blocks.3.attentions.1.transformer_blocks.0.ff.net.0.proj.bias",
    "model.diffusion_model.output_blocks.10.1.transformer_blocks.0.ff.net.2.weight": "up_blocks.3.attentions.1.transformer_blocks.0.ff.net.2.weight",
    "model.diffusion_model.output_blocks.10.1.transformer_blocks.0.ff.net.2.bias": "up_blocks.3.attentions.1.transformer_blocks.0.ff.net.2.bias",
    "model.diffusion_model.output_blocks.10.1.transformer_blocks.0.attn2.to_q.weight": "up_blocks.3.attentions.1.transformer_blocks.0.attn2.to_q.weight",
    "model.diffusion_model.output_blocks.10.1.transformer_blocks.0.attn2.to_k.weight": "up_blocks.3.attentions.1.transformer_blocks.0.attn2.to_k.weight",
    "model.diffusion_model.output_blocks.10.1.transformer_blocks.0.attn2.to_v.weight": "up_blocks.3.attentions.1.transformer_blocks.0.attn2.to_v.weight",
    "model.diffusion_model.output_blocks.10.1.transformer_blocks.0.attn2.to_out.0.weight": "up_blocks.3.attentions.1.transformer_blocks.0.attn2.to_out.0.weight",
    "model.diffusion_model.output_blocks.10.1.transformer_blocks.0.attn2.to_out.0.bias": "up_blocks.3.attentions.1.transformer_blocks.0.attn2.to_out.0.bias",
    "model.diffusion_model.output_blocks.10.1.transformer_blocks.0.norm1.weight": "up_blocks.3.attentions.1.transformer_blocks.0.norm1.weight",
    "model.diffusion_model.output_blocks.10.1.transformer_blocks.0.norm1.bias": "up_blocks.3.attentions.1.transformer_blocks.0.norm1.bias",
    "model.diffusion_model.output_blocks.10.1.transformer_blocks.0.norm2.weight": "up_blocks.3.attentions.1.transformer_blocks.0.norm2.weight",
    "model.diffusion_model.output_blocks.10.1.transformer_blocks.0.norm2.bias": "up_blocks.3.attentions.1.transformer_blocks.0.norm2.bias",
    "model.diffusion_model.output_blocks.10.1.transformer_blocks.0.norm3.weight": "up_blocks.3.attentions.1.transformer_blocks.0.norm3.weight",
    "model.diffusion_model.output_blocks.10.1.transformer_blocks.0.norm3.bias": "up_blocks.3.attentions.1.transformer_blocks.0.norm3.bias",
    "model.diffusion_model.output_blocks.10.1.proj_out.weight": "up_blocks.3.attentions.1.proj_out.weight",
    "model.diffusion_model.output_blocks.10.1.proj_out.bias": "up_blocks.3.attentions.1.proj_out.bias",
    "model.diffusion_model.output_blocks.11.0.in_layers.0.weight": "up_blocks.3.resnets.2.norm1.weight",
    "model.diffusion_model.output_blocks.11.0.in_layers.0.bias": "up_blocks.3.resnets.2.norm1.bias",
    "model.diffusion_model.output_blocks.11.0.in_layers.2.weight": "up_blocks.3.resnets.2.conv1.weight",
    "model.diffusion_model.output_blocks.11.0.in_layers.2.bias": "up_blocks.3.resnets.2.conv1.bias",
    "model.diffusion_model.output_blocks.11.0.emb_layers.1.weight": "up_blocks.3.resnets.2.time_emb_proj.weight",
    "model.diffusion_model.output_blocks.11.0.emb_layers.1.bias": "up_blocks.3.resnets.2.time_emb_proj.bias",
    "model.diffusion_model.output_blocks.11.0.out_layers.0.weight": "up_blocks.3.resnets.2.norm2.weight",
    "model.diffusion_model.output_blocks.11.0.out_layers.0.bias": "up_blocks.3.resnets.2.norm2.bias",
    "model.diffusion_model.output_blocks.11.0.out_layers.3.weight": "up_blocks.3.resnets.2.conv2.weight",
    "model.diffusion_model.output_blocks.11.0.out_layers.3.bias": "up_blocks.3.resnets.2.conv2.bias",
    "model.diffusion_model.output_blocks.11.0.skip_connection.weight": "up_blocks.3.resnets.2.conv_shortcut.weight",
    "model.diffusion_model.output_blocks.11.0.skip_connection.bias": "up_blocks.3.resnets.2.conv_shortcut.bias",
    "model.diffusion_model.output_blocks.11.1.norm.weight": "up_blocks.3.attentions.2.norm.weight",
    "model.diffusion_model.output_blocks.11.1.norm.bias": "up_blocks.3.attentions.2.norm.bias",
    "model.diffusion_model.output_blocks.11.1.proj_in.weight": "up_blocks.3.attentions.2.proj_in.weight",
    "model.diffusion_model.output_blocks.11.1.proj_in.bias": "up_blocks.3.attentions.2.proj_in.bias",
    "model.diffusion_model.output_blocks.11.1.transformer_blocks.0.attn1.to_q.weight": "up_blocks.3.attentions.2.transformer_blocks.0.attn1.to_q.weight",
    "model.diffusion_model.output_blocks.11.1.transformer_blocks.0.attn1.to_k.weight": "up_blocks.3.attentions.2.transformer_blocks.0.attn1.to_k.weight",
    "model.diffusion_model.output_blocks.11.1.transformer_blocks.0.attn1.to_v.weight": "up_blocks.3.attentions.2.transformer_blocks.0.attn1.to_v.weight",
    "model.diffusion_model.output_blocks.11.1.transformer_blocks.0.attn1.to_out.0.weight": "up_blocks.3.attentions.2.transformer_blocks.0.attn1.to_out.0.weight",
    "model.diffusion_model.output_blocks.11.1.transformer_blocks.0.attn1.to_out.0.bias": "up_blocks.3.attentions.2.transformer_blocks.0.attn1.to_out.0.bias",
    "model.diffusion_model.output_blocks.11.1.transformer_blocks.0.ff.net.0.proj.weight": "up_blocks.3.attentions.2.transformer_blocks.0.ff.net.0.proj.weight",
    "model.diffusion_model.output_blocks.11.1.transformer_blocks.0.ff.net.0.proj.bias": "up_blocks.3.attentions.2.transformer_blocks.0.ff.net.0.proj.bias",
    "model.diffusion_model.output_blocks.11.1.transformer_blocks.0.ff.net.2.weight": "up_blocks.3.attentions.2.transformer_blocks.0.ff.net.2.weight",
    "model.diffusion_model.output_blocks.11.1.transformer_blocks.0.ff.net.2.bias": "up_blocks.3.attentions.2.transformer_blocks.0.ff.net.2.bias",
    "model.diffusion_model.output_blocks.11.1.transformer_blocks.0.attn2.to_q.weight": "up_blocks.3.attentions.2.transformer_blocks.0.attn2.to_q.weight",
    "model.diffusion_model.output_blocks.11.1.transformer_blocks.0.attn2.to_k.weight": "up_blocks.3.attentions.2.transformer_blocks.0.attn2.to_k.weight",
    "model.diffusion_model.output_blocks.11.1.transformer_blocks.0.attn2.to_v.weight": "up_blocks.3.attentions.2.transformer_blocks.0.attn2.to_v.weight",
    "model.diffusion_model.output_blocks.11.1.transformer_blocks.0.attn2.to_out.0.weight": "up_blocks.3.attentions.2.transformer_blocks.0.attn2.to_out.0.weight",
    "model.diffusion_model.output_blocks.11.1.transformer_blocks.0.attn2.to_out.0.bias": "up_blocks.3.attentions.2.transformer_blocks.0.attn2.to_out.0.bias",
    "model.diffusion_model.output_blocks.11.1.transformer_blocks.0.norm1.weight": "up_blocks.3.attentions.2.transformer_blocks.0.norm1.weight",
    "model.diffusion_model.output_blocks.11.1.transformer_blocks.0.norm1.bias": "up_blocks.3.attentions.2.transformer_blocks.0.norm1.bias",
    "model.diffusion_model.output_blocks.11.1.transformer_blocks.0.norm2.weight": "up_blocks.3.attentions.2.transformer_blocks.0.norm2.weight",
    "model.diffusion_model.output_blocks.11.1.transformer_blocks.0.norm2.bias": "up_blocks.3.attentions.2.transformer_blocks.0.norm2.bias",
    "model.diffusion_model.output_blocks.11.1.transformer_blocks.0.norm3.weight": "up_blocks.3.attentions.2.transformer_blocks.0.norm3.weight",
    "model.diffusion_model.output_blocks.11.1.transformer_blocks.0.norm3.bias": "up_blocks.3.attentions.2.transformer_blocks.0.norm3.bias",
    "model.diffusion_model.output_blocks.11.1.proj_out.weight": "up_blocks.3.attentions.2.proj_out.weight",
    "model.diffusion_model.output_blocks.11.1.proj_out.bias": "up_blocks.3.attentions.2.proj_out.bias"
}


class StableDiffusionPipelineStripped(DiffusionPipeline):
    def __init__(self, unet=None, *args, **kwargs):
        print("got unet", len(args), kwargs.keys())
        self.unet = unet


def convert_diff_to_sd(diffusers_model_path: str, base_ckpt_path: str, output_ckpt_path: str,
                       overwrite=False):
    huggingface_use_auth_token = None
    if not overwrite and os.path.exists(output_ckpt_path):
        print("Specified model already exists and overwrite not set, nothing to do!")
        return

    print(f"loading diff model from {diffusers_model_path!r}")
    try:
        diff_pipe = StableDiffusionPipeline.from_pretrained(diffusers_model_path,
                                                            use_auth_token=huggingface_use_auth_token)
    except Exception as ex:
        if "required positional arguments" in str(ex):
            print("load error, trying loading stripped version", ex)
            diff_pipe = StableDiffusionPipelineStripped.from_pretrained(diffusers_model_path,
                                                                        use_auth_token=huggingface_use_auth_token)
        else:
            raise

    print("loading diff model done!")

    diff_pipe_unet_sd = diff_pipe.unet.state_dict()
    print("diff_pipe_unet_sd done")

    print(f"loading sd ckpt from {base_ckpt_path!r}")
    org_model = torch.load(base_ckpt_path)
    org_sd = org_model["state_dict"]
    print(f"loading sd ckpt done!")

    for ckpt_key, diff_key in KeyMap.items():
        org_sd[ckpt_key] = diff_pipe_unet_sd[diff_key]

    print(f"saving converted unet to {output_ckpt_path!r}")
    torch.save(org_model, output_ckpt_path)
    print("done")


def create_checkpoint(diffusers_model: str, base_ckpt_path: str, output_ckpt_path: str, overwrite=True):
    convert_diff_to_sd(diffusers_model, base_ckpt_path, output_ckpt_path, overwrite=True)


def create_output_dir(new_model):
    print(f"Creating dreambooth model folder: {new_model}")
    model_dir = paths.models_path
    # This is where all dreambooth trainings are saved
    db_dir = os.path.join(model_dir, "dreambooth")
    if not os.path.exists(db_dir):
        os.makedirs(db_dir)
    # This is where our specific model is going to live
    out_dir = os.path.join(db_dir, new_model)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # This is where our training data is stored.
    output_dir = os.path.join(out_dir, "working")
    # I think we can save the pipeline midstream. If not, add an else statement here to delete old working data
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return out_dir
