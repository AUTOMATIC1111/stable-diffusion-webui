import re
import os

from modules import shared, paths

sd_configs_path = shared.sd_configs_path
sd_repo_configs_path = os.path.join(paths.paths['Stable Diffusion'], "configs", "stable-diffusion")


config_default = shared.sd_default_config
config_sd2 = os.path.join(sd_repo_configs_path, "v2-inference.yaml")
config_sd2v = os.path.join(sd_repo_configs_path, "v2-inference-v.yaml")
config_depth_model = os.path.join(sd_repo_configs_path, "v2-midas-inference.yaml")
config_inpainting = os.path.join(sd_configs_path, "v1-inpainting-inference.yaml")
config_instruct_pix2pix = os.path.join(sd_configs_path, "instruct-pix2pix.yaml")
config_alt_diffusion = os.path.join(sd_configs_path, "alt-diffusion-inference.yaml")

re_parametrization_v = re.compile(r'-v\b')


def guess_model_config_from_state_dict(sd, filename):
    fn = os.path.basename(filename)

    sd2_cond_proj_weight = sd.get('cond_stage_model.model.transformer.resblocks.0.attn.in_proj_weight', None)
    diffusion_model_input = sd.get('model.diffusion_model.input_blocks.0.0.weight', None)

    if sd.get('depth_model.model.pretrained.act_postprocess3.0.project.0.bias', None) is not None:
        return config_depth_model

    if sd2_cond_proj_weight is not None and sd2_cond_proj_weight.shape[1] == 1024:
        if re.search(re_parametrization_v, fn) or "v2-1_768" in fn:
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

