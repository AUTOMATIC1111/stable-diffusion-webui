# https://gist.github.com/takuma104/4adfb3d968d80bea1d18a30c06439242
# 2nd editing by lllyasviel

import torch


# =================#
# UNet Conversion #
# =================#

unet_conversion_map = [
    # (stable-diffusion, HF Diffusers)
    ("time_embed.0.weight", "time_embedding.linear_1.weight"),
    ("time_embed.0.bias", "time_embedding.linear_1.bias"),
    ("time_embed.2.weight", "time_embedding.linear_2.weight"),
    ("time_embed.2.bias", "time_embedding.linear_2.bias"),
    ("label_emb.0.0.weight", "add_embedding.linear_1.weight"),
    ("label_emb.0.0.bias", "add_embedding.linear_1.bias"),
    ("label_emb.0.2.weight", "add_embedding.linear_2.weight"),
    ("label_emb.0.2.bias", "add_embedding.linear_2.bias"),
    ("input_blocks.0.0.weight", "conv_in.weight"),
    ("input_blocks.0.0.bias", "conv_in.bias"),
    ("middle_block_out.0.weight", "controlnet_mid_block.weight"),
    ("middle_block_out.0.bias", "controlnet_mid_block.bias"),
]

unet_conversion_map_resnet = [
    # (stable-diffusion, HF Diffusers)
    ("in_layers.0", "norm1"),
    ("in_layers.2", "conv1"),
    ("out_layers.0", "norm2"),
    ("out_layers.3", "conv2"),
    ("emb_layers.1", "time_emb_proj"),
    ("skip_connection", "conv_shortcut"),
]

unet_conversion_map_layer = []
# hardcoded number of downblocks and resnets/attentions...
# would need smarter logic for other networks.
for i in range(4):
    # loop over downblocks/upblocks

    for j in range(10):
        # loop over resnets/attentions for downblocks
        hf_down_res_prefix = f"down_blocks.{i}.resnets.{j}."
        sd_down_res_prefix = f"input_blocks.{3*i + j + 1}.0."
        unet_conversion_map_layer.append((sd_down_res_prefix, hf_down_res_prefix))

        hf_down_atn_prefix = f"down_blocks.{i}.attentions.{j}."
        sd_down_atn_prefix = f"input_blocks.{3 * i + j + 1}.1."
        unet_conversion_map_layer.append((sd_down_atn_prefix, hf_down_atn_prefix))

    hf_downsample_prefix = f"down_blocks.{i}.downsamplers.0.conv."
    sd_downsample_prefix = f"input_blocks.{3 * (i + 1)}.0.op."
    unet_conversion_map_layer.append((sd_downsample_prefix, hf_downsample_prefix))


hf_mid_atn_prefix = "mid_block.attentions.0."
sd_mid_atn_prefix = "middle_block.1."
unet_conversion_map_layer.append((sd_mid_atn_prefix, hf_mid_atn_prefix))

for j in range(2):
    hf_mid_res_prefix = f"mid_block.resnets.{j}."
    sd_mid_res_prefix = f"middle_block.{2*j}."
    unet_conversion_map_layer.append((sd_mid_res_prefix, hf_mid_res_prefix))

# controlnet specific

controlnet_cond_embedding_names = ['conv_in'] + [f'blocks.{i}' for i in range(6)] + ['conv_out']
for i, hf_prefix in enumerate(controlnet_cond_embedding_names):
    hf_prefix = f"controlnet_cond_embedding.{hf_prefix}."
    sd_prefix = f"input_hint_block.{i*2}."
    unet_conversion_map_layer.append((sd_prefix, hf_prefix))

for i in range(12):
    hf_prefix = f"controlnet_down_blocks.{i}."
    sd_prefix = f"zero_convs.{i}.0."
    unet_conversion_map_layer.append((sd_prefix, hf_prefix))


def convert_from_diffuser_state_dict(unet_state_dict):
    mapping = {k: k for k in unet_state_dict.keys()}
    for sd_name, hf_name in unet_conversion_map:
        mapping[hf_name] = sd_name
    for k, v in mapping.items():
        if "resnets" in k:
            for sd_part, hf_part in unet_conversion_map_resnet:
                v = v.replace(hf_part, sd_part)
            mapping[k] = v
    for k, v in mapping.items():
        for sd_part, hf_part in unet_conversion_map_layer:
            v = v.replace(hf_part, sd_part)
        mapping[k] = v
    new_state_dict = {v: unet_state_dict[k] for k, v in mapping.items() if k in unet_state_dict}
    return new_state_dict
