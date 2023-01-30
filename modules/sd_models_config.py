import re
import os

import torch

from modules import shared, paths, sd_disable_initialization

sd_configs_path = shared.sd_configs_path
sd_repo_configs_path = os.path.join(paths.paths['Stable Diffusion'], "configs", "stable-diffusion")


config_default = shared.sd_default_config
config_sd2 = os.path.join(sd_repo_configs_path, "v2-inference.yaml")
config_sd2v = os.path.join(sd_repo_configs_path, "v2-inference-v.yaml")
config_sd2_inpainting = os.path.join(sd_repo_configs_path, "v2-inpainting-inference.yaml")
config_depth_model = os.path.join(sd_repo_configs_path, "v2-midas-inference.yaml")
config_inpainting = os.path.join(sd_configs_path, "v1-inpainting-inference.yaml")
config_instruct_pix2pix = os.path.join(sd_configs_path, "instruct-pix2pix.yaml")
config_alt_diffusion = os.path.join(sd_configs_path, "alt-diffusion-inference.yaml")


def is_using_v_parameterization_for_sd2(state_dict):
    """
    Detects whether unet in state_dict is using v-parameterization. Returns True if it is. You're welcome.
    """

    import ldm.modules.diffusionmodules.openaimodel
    from modules import devices

    device = devices.cpu

    with sd_disable_initialization.DisableInitialization():
        unet = ldm.modules.diffusionmodules.openaimodel.UNetModel(
            use_checkpoint=True,
            use_fp16=False,
            image_size=32,
            in_channels=4,
            out_channels=4,
            model_channels=320,
            attention_resolutions=[4, 2, 1],
            num_res_blocks=2,
            channel_mult=[1, 2, 4, 4],
            num_head_channels=64,
            use_spatial_transformer=True,
            use_linear_in_transformer=True,
            transformer_depth=1,
            context_dim=1024,
            legacy=False
        )
        unet.eval()

    with torch.no_grad():
        unet_sd = {k.replace("model.diffusion_model.", ""): v for k, v in state_dict.items() if "model.diffusion_model." in k}
        unet.load_state_dict(unet_sd, strict=True)
        unet.to(device=device, dtype=torch.float)

        test_cond = torch.ones((1, 2, 1024), device=device) * 0.5
        x_test = torch.ones((1, 4, 8, 8), device=device) * 0.5

        out = (unet(x_test, torch.asarray([999], device=device), context=test_cond) - x_test).mean().item()

    return out < -1


def guess_model_config_from_state_dict(sd, filename):
    sd2_cond_proj_weight = sd.get('cond_stage_model.model.transformer.resblocks.0.attn.in_proj_weight', None)
    diffusion_model_input = sd.get('model.diffusion_model.input_blocks.0.0.weight', None)

    if sd.get('depth_model.model.pretrained.act_postprocess3.0.project.0.bias', None) is not None:
        return config_depth_model

    if sd2_cond_proj_weight is not None and sd2_cond_proj_weight.shape[1] == 1024:
        if diffusion_model_input.shape[1] == 9:
            return config_sd2_inpainting
        elif is_using_v_parameterization_for_sd2(sd):
            return config_sd2v
        else:
            return config_sd2

    if diffusion_model_input is not None:
        if diffusion_model_input.shape[1] == 9:
            return config_inpainting
        if diffusion_model_input.shape[1] == 8:
            return config_instruct_pix2pix

    if sd.get('cond_stage_model.roberta.embeddings.word_embeddings.weight', None) is not None:
        return config_alt_diffusion

    return config_default


def find_checkpoint_config(state_dict, info):
    if info is None:
        return guess_model_config_from_state_dict(state_dict, "")

    config = find_checkpoint_config_near_filename(info)
    if config is not None:
        return config

    return guess_model_config_from_state_dict(state_dict, info.filename)


def find_checkpoint_config_near_filename(info):
    if info is None:
        return None

    config = os.path.splitext(info.filename)[0] + ".yaml"
    if os.path.exists(config):
        return config

    return None

