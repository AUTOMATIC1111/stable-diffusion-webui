import os.path


def should_hijack_ip2p(checkpoint_info):
    from modules import sd_models_config

    ckpt_basename = os.path.basename(checkpoint_info.filename).lower()
    cfg_basename = os.path.basename(sd_models_config.find_checkpoint_config_near_filename(checkpoint_info)).lower()

    return "pix2pix" in ckpt_basename and "pix2pix" not in cfg_basename
