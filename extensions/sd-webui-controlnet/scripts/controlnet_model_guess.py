import copy
import os
import torch
from pathlib import Path
from typing import NamedTuple
from modules import devices

from scripts.adapter import PlugableAdapter, Adapter, StyleAdapter, Adapter_light
from scripts.controlnet_lllite import PlugableControlLLLite
from scripts.cldm import PlugableControlModel
from scripts.controlmodel_ipadapter import PlugableIPAdapter
from scripts.logging import logger
from scripts.controlnet_diffusers import convert_from_diffuser_state_dict
from scripts.controlnet_lora import controlnet_lora_hijack, force_load_state_dict
from scripts.enums import ControlModelType


controlnet_default_config = {'adm_in_channels': None,
                             'in_channels': 4,
                             'model_channels': 320,
                             'num_res_blocks': 2,
                             'attention_resolutions': [1, 2, 4],
                             'transformer_depth': [1, 1, 1, 0],
                             'channel_mult': [1, 2, 4, 4],
                             'transformer_depth_middle': 1,
                             'use_linear_in_transformer': False,
                             'context_dim': 768,
                             "num_heads": 8,
                             "global_average_pooling": False}

controlnet_sdxl_config = {'num_classes': 'sequential',
                          'adm_in_channels': 2816,
                          'in_channels': 4,
                          'model_channels': 320,
                          'num_res_blocks': 2,
                          'attention_resolutions': [2, 4],
                          'transformer_depth': [0, 2, 10],
                          'channel_mult': [1, 2, 4],
                          'transformer_depth_middle': 10,
                          'use_linear_in_transformer': True,
                          'context_dim': 2048,
                          "num_head_channels": 64,
                          "global_average_pooling": False}

controlnet_sdxl_mid_config = {'num_classes': 'sequential',
                              'adm_in_channels': 2816,
                              'in_channels': 4,
                              'model_channels': 320,
                              'num_res_blocks': 2,
                              'attention_resolutions': [4],
                              'transformer_depth': [0, 0, 1],
                              'channel_mult': [1, 2, 4],
                              'transformer_depth_middle': 1,
                              'use_linear_in_transformer': True,
                              'context_dim': 2048,
                              "num_head_channels": 64,
                              "global_average_pooling": False}

controlnet_sdxl_small_config = {'num_classes': 'sequential',
                                'adm_in_channels': 2816,
                                'in_channels': 4,
                                'model_channels': 320,
                                'num_res_blocks': 2,
                                'attention_resolutions': [],
                                'transformer_depth': [0, 0, 0],
                                'channel_mult': [1, 2, 4],
                                'transformer_depth_middle': 0,
                                'use_linear_in_transformer': True,
                                "num_head_channels": 64,
                                'context_dim': 1,
                                "global_average_pooling": False}

t2i_adapter_config = {
    'channels': [320, 640, 1280, 1280],
    'nums_rb': 2,
    'ksize': 1,
    'sk': True,
    'cin': 192,
    'use_conv': False
}

t2i_adapter_light_config = {
    'channels': [320, 640, 1280, 1280],
    'nums_rb': 4,
    'cin': 192,
}

t2i_adapter_style_config = {
    'width': 1024,
    'context_dim': 768,
    'num_head': 8,
    'n_layes': 3,
    'num_token': 8,
}


# Stolen from https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/utils.py
def state_dict_key_replace(state_dict, keys_to_replace):
    for x in keys_to_replace:
        if x in state_dict:
            state_dict[keys_to_replace[x]] = state_dict.pop(x)
    return state_dict


# # Stolen from https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/utils.py
def state_dict_prefix_replace(state_dict, replace_prefix):
    for rp in replace_prefix:
        replace = list(map(lambda a: (a, "{}{}".format(replace_prefix[rp], a[len(rp):])), filter(lambda a: a.startswith(rp), state_dict.keys())))
        for x in replace:
            state_dict[x[1]] = state_dict.pop(x[0])
    return state_dict


class ControlModel(NamedTuple):
    model: torch.nn.Module
    type: ControlModelType


def build_model_by_guess(state_dict, unet, model_path: str) -> ControlModel:
    if "lora_controlnet" in state_dict:
        is_sdxl = "input_blocks.11.0.in_layers.0.weight" not in state_dict
        logger.info(f"Using ControlNet lora ({'SDXL' if is_sdxl else 'SD15'})")
        del state_dict['lora_controlnet']
        config = copy.deepcopy(controlnet_sdxl_config if is_sdxl else controlnet_default_config)
        config['global_average_pooling'] = False
        config['hint_channels'] = int(state_dict['input_hint_block.0.weight'].shape[1])
        config['use_fp16'] = devices.dtype_unet == torch.float16
        with controlnet_lora_hijack():
            network = PlugableControlModel(config, state_dict=None)
        force_load_state_dict(network.control_model, state_dict)
        network.is_control_lora = True
        network.to(devices.dtype_unet)
        return ControlModel(network, ControlModelType.ControlLoRA)

    if "controlnet_cond_embedding.conv_in.weight" in state_dict:  # diffusers
        state_dict = convert_from_diffuser_state_dict(state_dict)

    if 'adapter.body.0.resnets.0.block1.weight' in state_dict:  # diffusers
        prefix_replace = {}
        for i in range(4):
            for j in range(2):
                prefix_replace["adapter.body.{}.resnets.{}.".format(i, j)] = "body.{}.".format(i * 2 + j)
            prefix_replace["adapter.body.{}.".format(i)] = "body.{}.".format(i * 2)
        prefix_replace["adapter."] = ""
        state_dict = state_dict_prefix_replace(state_dict, prefix_replace)

    if any('image_proj.' in x for x in state_dict.keys()) and any('ip_adapter.' in x for x in state_dict.keys()):  # safetensor ipadapters
        st_model = {"image_proj": {}, "ip_adapter": {}}
        for key in state_dict.keys():
            if key.startswith("image_proj."):
                st_model["image_proj"][key.replace("image_proj.", "")] = state_dict[key]
            elif key.startswith("ip_adapter."):
                st_model["ip_adapter"][key.replace("ip_adapter.", "")] = state_dict[key]
        # sort keys
        model = {"image_proj": st_model["image_proj"], "ip_adapter": {}}
        sorted_keys = sorted(st_model["ip_adapter"].keys(), key=lambda x: int(x.split(".")[0]))
        for key in sorted_keys:
            model["ip_adapter"][key] = st_model["ip_adapter"][key]
        state_dict = model
        del st_model

    model_has_shuffle_in_filename = 'shuffle' in Path(os.path.abspath(model_path)).stem.lower()
    state_dict = {k.replace("control_model.", ""): v for k, v in state_dict.items()}
    state_dict = {k.replace("adapter.", ""): v for k, v in state_dict.items()}

    if 'input_hint_block.0.weight' in state_dict:
        if 'label_emb.0.0.bias' not in state_dict:
            config = copy.deepcopy(controlnet_default_config)
            logger.info('controlnet_default_config')
            config['global_average_pooling'] = model_has_shuffle_in_filename
            config['hint_channels'] = int(state_dict['input_hint_block.0.weight'].shape[1])
            config['context_dim'] = int(state_dict['input_blocks.5.1.transformer_blocks.0.attn2.to_k.weight'].shape[1])
            for key in state_dict.keys():
                p = state_dict[key]
                if 'proj_in.weight' in key or 'proj_out.weight' in key:
                    if len(p.shape) == 2:
                        p = p[..., None, None]
                state_dict[key] = p
        else:
            has_full_layers = 'input_blocks.8.1.transformer_blocks.9.norm3.weight' in state_dict
            has_mid_layers = 'input_blocks.8.1.transformer_blocks.0.norm3.weight' in state_dict
            if has_full_layers:
                config = copy.deepcopy(controlnet_sdxl_config)
                logger.info('controlnet_sdxl_config')
            elif has_mid_layers:
                config = copy.deepcopy(controlnet_sdxl_mid_config)
                logger.info('controlnet_sdxl_mid_config')
            else:
                config = copy.deepcopy(controlnet_sdxl_small_config)
                logger.info('controlnet_sdxl_small_config')
            config['global_average_pooling'] = False
            config['hint_channels'] = int(state_dict['input_hint_block.0.weight'].shape[1])

        if 'difference' in state_dict and unet is not None:
            unet_state_dict = unet.state_dict()
            unet_state_dict_keys = unet_state_dict.keys()
            final_state_dict = {}
            for key in state_dict.keys():
                p = state_dict[key]
                if key in unet_state_dict_keys:
                    p_new = p + unet_state_dict[key].clone().cpu()
                else:
                    p_new = p
                final_state_dict[key] = p_new
            state_dict = final_state_dict

        config['use_fp16'] = devices.dtype_unet == torch.float16

        network = PlugableControlModel(config, state_dict)
        network.to(devices.dtype_unet)
        if "instant_id" in model_path.lower():
            control_model_type = ControlModelType.InstantID
        else:
            control_model_type = ControlModelType.ControlNet
        return ControlModel(network, control_model_type)

    if 'conv_in.weight' in state_dict:
        logger.info('t2i_adapter_config')
        cin = int(state_dict['conv_in.weight'].shape[1])
        channel = int(state_dict['conv_in.weight'].shape[0])
        ksize = int(state_dict['body.0.block2.weight'].shape[2])
        down_opts = tuple(filter(lambda item: item.endswith("down_opt.op.weight"), state_dict))
        use_conv = len(down_opts) > 0
        is_sdxl = cin == 256 or cin == 768
        adapter = Adapter(
            cin=cin,
            channels=[channel, channel*2, channel*4, channel*4],
            nums_rb=2,
            ksize=ksize,
            sk=True,
            use_conv=use_conv,
            is_sdxl=is_sdxl
        ).cpu()
        adapter.load_state_dict(state_dict, strict=False)
        network = PlugableAdapter(adapter)
        return ControlModel(network, ControlModelType.T2I_Adapter)

    if 'style_embedding' in state_dict:
        config = copy.deepcopy(t2i_adapter_style_config)
        logger.info('t2i_adapter_style_config')
        adapter = StyleAdapter(**config).cpu()
        adapter.load_state_dict(state_dict, strict=False)
        network = PlugableAdapter(adapter)
        return ControlModel(network, ControlModelType.T2I_StyleAdapter)

    if 'body.0.in_conv.weight' in state_dict:
        config = copy.deepcopy(t2i_adapter_light_config)
        logger.info('t2i_adapter_light_config')
        config['cin'] = int(state_dict['body.0.in_conv.weight'].shape[1])
        adapter = Adapter_light(**config).cpu()
        adapter.load_state_dict(state_dict, strict=False)
        network = PlugableAdapter(adapter)
        return ControlModel(network, ControlModelType.T2I_Adapter)

    if 'ip_adapter' in state_dict:
        network = PlugableIPAdapter(state_dict, model_path)
        network.to('cpu')
        return ControlModel(network, ControlModelType.IPAdapter)

    if any('lllite' in k for k in state_dict.keys()):
        network = PlugableControlLLLite(state_dict)
        network.to('cpu')
        return ControlModel(network, ControlModelType.Controlllite)

    raise '[ControlNet Error] Cannot recognize the ControlModel!'
