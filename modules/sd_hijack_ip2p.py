import collections
import os.path
import sys
import gc
import time

def should_hijack_ip2p(checkpoint_info):
    from modules import sd_models_config

    ckpt_basename = os.path.basename(checkpoint_info.filename).lower()
    cfg_basename = os.path.basename(sd_models_config.find_checkpoint_config_near_filename(checkpoint_info)).lower()

    return "pix2pix" in ckpt_basename and not "pix2pix" in cfg_basename
