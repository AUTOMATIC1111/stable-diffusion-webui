import torch
import os
from collections import namedtuple
from modules import shared, devices
from modules.paths import models_path
import glob

model_dir = "Stable-diffusion"
model_path = os.path.abspath(os.path.join(models_path, model_dir))
vae_dir = "VAE"
vae_path = os.path.abspath(os.path.join(models_path, vae_dir))

vae_ignore_keys = {"model_ema.decay", "model_ema.num_updates"}
default_vae_dict = {"auto": "auto", "None": "None"}
default_vae_list = ["auto", "None"]
default_vae_values = [default_vae_dict[x] for x in default_vae_list]
vae_dict = dict(default_vae_dict)
vae_list = list(default_vae_list)
first_load = True

def get_filename(filepath):
    return os.path.splitext(os.path.basename(filepath))[0]

def refresh_vae_list(vae_path=vae_path, model_path=model_path):
    global vae_dict, vae_list
    res = {}
    candidates = [
        *glob.iglob(os.path.join(model_path, '**/*.vae.pt'), recursive=True),
        *glob.iglob(os.path.join(model_path, '**/*.vae.ckpt'), recursive=True),
        *glob.iglob(os.path.join(vae_path, '**/*.pt'), recursive=True),
        *glob.iglob(os.path.join(vae_path, '**/*.ckpt'), recursive=True)
    ]
    if shared.cmd_opts.vae_path is not None and os.path.isfile(shared.cmd_opts.vae_path):
        candidates.append(shared.cmd_opts.vae_path)
    for filepath in candidates:
        name = get_filename(filepath)
        res[name] = filepath
    vae_list.clear()
    vae_list.extend(default_vae_list)
    vae_list.extend(list(res.keys()))
    vae_dict.clear()
    vae_dict.update(default_vae_dict)
    vae_dict.update(res)
    return vae_list

def load_vae(model, checkpoint_file, vae_file="auto"):
    global first_load, vae_dict, vae_list
    # save_settings = False

    # if vae_file argument is provided, it takes priority
    if vae_file and vae_file not in default_vae_list:
        if not os.path.isfile(vae_file):
            vae_file = "auto"
            # save_settings = True
            print("VAE provided as function argument doesn't exist")
    # for the first load, if vae-path is provided, it takes priority and failure is reported
    if first_load and shared.cmd_opts.vae_path is not None:
        if os.path.isfile(shared.cmd_opts.vae_path):
            vae_file = shared.cmd_opts.vae_path
            # save_settings = True
            # print("Using VAE provided as command line argument")
        else:
            print("VAE provided as command line argument doesn't exist")
    # else, we load from settings
    if vae_file == "auto" and shared.opts.sd_vae is not None:
        # if saved VAE settings isn't recognized, fallback to auto
        vae_file = vae_dict.get(shared.opts.sd_vae, "auto")
        # if VAE selected but not found, fallback to auto
        if vae_file not in default_vae_values and not os.path.isfile(vae_file):
            vae_file = "auto"
            print("Selected VAE doesn't exist")
    # vae-path cmd arg takes priority for auto
    if vae_file == "auto" and shared.cmd_opts.vae_path is not None:
        if os.path.isfile(shared.cmd_opts.vae_path):
            vae_file = shared.cmd_opts.vae_path
            print("Using VAE provided as command line argument")
    # if still not found, try look for ".vae.pt" beside model
    model_path = os.path.splitext(checkpoint_file)[0]
    if vae_file == "auto":
        vae_file_try = model_path + ".vae.pt"
        if os.path.isfile(vae_file_try):
            vae_file = vae_file_try
            print("Using VAE found beside selected model")
    # if still not found, try look for ".vae.ckpt" beside model
    if vae_file == "auto":
        vae_file_try = model_path + ".vae.ckpt"
        if os.path.isfile(vae_file_try):
            vae_file = vae_file_try
            print("Using VAE found beside selected model")
    # No more fallbacks for auto
    if vae_file == "auto":
        vae_file = None
    # Last check, just because
    if vae_file and not os.path.exists(vae_file):
        vae_file = None

    if vae_file:
        print(f"Loading VAE weights from: {vae_file}")
        vae_ckpt = torch.load(vae_file, map_location=shared.weight_load_location)
        vae_dict_1 = {k: v for k, v in vae_ckpt["state_dict"].items() if k[0:4] != "loss" and k not in vae_ignore_keys}
        model.first_stage_model.load_state_dict(vae_dict_1)

    # If vae used is not in dict, update it
    # It will be removed on refresh though
    if vae_file is not None:
        vae_opt = get_filename(vae_file)
        if vae_opt not in vae_dict:
            vae_dict[vae_opt] = vae_file
            vae_list.append(vae_opt)

    """
    # Save current VAE to VAE settings, maybe? will it work?
    if save_settings:
        if vae_file is None:
            vae_opt = "None"

        # shared.opts.sd_vae = vae_opt
    """

    first_load = False
    model.first_stage_model.to(devices.dtype_vae)
