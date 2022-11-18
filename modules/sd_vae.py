import torch
import os
from collections import namedtuple
from modules import shared, devices, script_callbacks
from modules.paths import models_path
from modules.modelloader import model_places
import glob
from copy import deepcopy
from pathlib import Path


model_dir_name = "Stable-diffusion"
model_dir = os.path.abspath(os.path.join(models_path, model_dir_name))
model_dirs = [model_dir]
model_dirs_paths = [Path(x) for x in model_dirs]
vae_dir_name = "VAE"
vae_dir = os.path.abspath(os.path.join(models_path, vae_dir_name))


vae_ignore_keys = {"model_ema.decay", "model_ema.num_updates"}


default_vae_dict = {"auto": "auto", "None": None, None: None}
default_vae_list = ["auto", "None"]


default_vae_values = [default_vae_dict[x] for x in default_vae_list]
vae_dict = dict(default_vae_dict)
vae_list = list(default_vae_list)
first_load = True


base_vae = None
loaded_vae_file = None
checkpoint_info = None


def init():
    global model_dir, model_dirs, model_dirs_paths
    from modules.sd_models import model_path
    model_dir = model_path
    model_dirs = model_places(model_path=model_dir, command_path=shared.cmd_opts.ckpt_dir)
    model_dirs_paths = [Path(x) for x in model_dirs]
    refresh_vae_list()


def get_base_vae(model):
    if base_vae is not None and checkpoint_info == model.sd_checkpoint_info and model:
        return base_vae
    return None


def store_base_vae(model):
    global base_vae, checkpoint_info
    if checkpoint_info != model.sd_checkpoint_info:
        assert not loaded_vae_file, "Trying to store non-base VAE!"
        base_vae = deepcopy(model.first_stage_model.state_dict())
        checkpoint_info = model.sd_checkpoint_info


def delete_base_vae():
    global base_vae, checkpoint_info
    base_vae = None
    checkpoint_info = None


def restore_base_vae(model):
    global loaded_vae_file
    if base_vae is not None and checkpoint_info == model.sd_checkpoint_info:
        print("Restoring base VAE")
        load_vae_dict(model, base_vae)
        loaded_vae_file = None
    delete_base_vae()


def get_filename(filepath):
    if not filepath:
        return "None"
    return os.path.relpath(filepath, models_path)


def search_parent(file):
    if file is not None and os.path.isfile(file):
        path = Path(file)
        for p in model_dirs_paths:
            if p in path.parents:
                path = None
                break
        if path:
            return [
                *glob.iglob(os.path.join(str(path.parent), '**/*.vae.ckpt'), recursive=True),
                *glob.iglob(os.path.join(str(path.parent), '**/*.vae.pt'), recursive=True)
            ]
    return []


def refresh_vae_list(vae_dir=vae_dir, model_dir=model_dir):
    global vae_dict, vae_list
    res = {}
    candidates = []
    for model_dir in model_dirs:
        candidates += [
            *glob.iglob(os.path.join(model_dir, '**/*.vae.ckpt'), recursive=True),
            *glob.iglob(os.path.join(model_dir, '**/*.vae.pt'), recursive=True)
        ]
    candidates += [
        *glob.iglob(os.path.join(vae_dir, '**/*.ckpt'), recursive=True),
        *glob.iglob(os.path.join(vae_dir, '**/*.pt'), recursive=True)
    ]
    if shared.cmd_opts.ckpt is not None:
        candidates += search_parent(shared.cmd_opts.ckpt)
    if shared.cmd_opts.vae_path is not None and os.path.isfile(shared.cmd_opts.vae_path):
        candidates += search_parent(shared.cmd_opts.vae_path)
        candidates.append(shared.cmd_opts.vae_path)
    for filepath in candidates:
        name = get_filename(filepath)
        res[name] = filepath
    vae_list.clear()
    vae_list.extend(default_vae_list)
    vae_list.extend(list(res.keys()))
    vae_dict.clear()
    vae_dict.update(res)
    vae_dict.update(default_vae_dict)
    return vae_list


def get_vae_from_settings(vae_file="auto"):
    # else, we load from settings, if not set to be default
    if vae_file == "auto" and shared.opts.sd_vae is not None:
        # if saved VAE settings isn't recognized, fallback to auto
        vae_file = vae_dict.get(shared.opts.sd_vae, "auto")
        # if VAE selected but not found, fallback to auto
        if vae_file not in default_vae_values and not os.path.isfile(vae_file):
            print(f"Selected VAE doesn't exist: {vae_file}")
            vae_file = "auto"
    return vae_file


def resolve_vae(checkpoint_file=None, vae_file="auto"):
    global first_load, vae_dict, vae_list

    # if vae_file argument is provided, it takes priority, but not saved
    if vae_file and vae_file not in default_vae_list:
        if not os.path.isfile(vae_file):
            print(f"VAE provided as function argument doesn't exist: {vae_file}")
            vae_file = "auto"
    # for the first load, if vae-path is provided, it takes priority, saved, and failure is reported
    if first_load and shared.cmd_opts.vae_path is not None:
        if os.path.isfile(shared.cmd_opts.vae_path):
            vae_file = shared.cmd_opts.vae_path
            shared.opts.data['sd_vae'] = get_filename(vae_file)
        else:
            print(f"VAE provided as command line argument doesn't exist: {vae_file}")
    # fallback to selector in settings, if vae selector not set to act as default fallback
    if not shared.opts.sd_vae_as_default:
        vae_file = get_vae_from_settings(vae_file)
    # vae-path cmd arg takes priority for auto
    if vae_file == "auto" and shared.cmd_opts.vae_path is not None:
        if os.path.isfile(shared.cmd_opts.vae_path):
            vae_file = shared.cmd_opts.vae_path
            print(f"Using VAE provided as command line argument: {vae_file}")
    # if still not found, try look for VAE similar to model
    if vae_file == "auto" and checkpoint_file:
        model_path = os.path.splitext(checkpoint_file)[0]
        trials = [
            model_path + ".vae.pt",
            model_path + ".vae.ckpt"
        ]
        for model_dir_path in model_dirs_paths:
            if model_dir_path in Path(checkpoint_file).parents:
                rel_path = os.path.relpath(model_path, model_dir)
                vae_path = os.path.join(vae_dir, rel_path)
                trials += [
                    vae_path + ".vae.pt",
                    vae_path + ".vae.ckpt",
                    vae_path + ".pt",
                    vae_path + ".ckpt"
                ]
        for vae_file_try in trials:
            if os.path.isfile(vae_file_try):
                vae_file = vae_file_try
                print(f"Using VAE found similar to selected model: {vae_file}")
                break
    # if vae selector were set as default fallback, call here
    if shared.opts.sd_vae_as_default:
        vae_file = get_vae_from_settings(vae_file)

    # No more fallbacks for auto
    if vae_file == "auto":
        vae_file = None
    # Last check, just because
    if vae_file and not os.path.exists(vae_file):
        vae_file = None

    return vae_file


def load_vae(model, vae_file=None):
    global first_load, vae_dict, vae_list, loaded_vae_file
    # save_settings = False

    if vae_file:
        assert os.path.isfile(vae_file), f"VAE file doesn't exist: {vae_file}"
        print(f"Loading VAE weights from: {vae_file}")
        store_base_vae(model)
        vae_ckpt = torch.load(vae_file, map_location=shared.weight_load_location)
        vae_dict_1 = {k: v for k, v in vae_ckpt["state_dict"].items() if k[0:4] != "loss" and k not in vae_ignore_keys}
        load_vae_dict(model, vae_dict_1)

        # If vae used is not in dict, update it
        # It will be removed on refresh though
        vae_opt = get_filename(vae_file)
        if vae_opt not in vae_dict:
            vae_dict[vae_opt] = vae_file
            vae_list.append(vae_opt)
    elif loaded_vae_file:
        restore_base_vae(model)

    loaded_vae_file = vae_file

    first_load = False


# don't call this from outside
def load_vae_dict(model, vae_dict_1):
    model.first_stage_model.load_state_dict(vae_dict_1)
    model.first_stage_model.to(devices.dtype_vae)

def clear_loaded_vae():
    global loaded_vae_file
    loaded_vae_file = None

def reload_vae_weights(sd_model=None, vae_file="auto"):
    from modules import lowvram, devices, sd_hijack

    if not sd_model:
        sd_model = shared.sd_model

    checkpoint_info = sd_model.sd_checkpoint_info
    checkpoint_file = checkpoint_info.filename
    vae_file = resolve_vae(checkpoint_file, vae_file=vae_file)

    if loaded_vae_file == vae_file:
        return

    if shared.cmd_opts.lowvram or shared.cmd_opts.medvram:
        lowvram.send_everything_to_cpu()
    else:
        sd_model.to(devices.cpu)

    sd_hijack.model_hijack.undo_hijack(sd_model)

    load_vae(sd_model, vae_file)

    sd_hijack.model_hijack.hijack(sd_model)
    script_callbacks.model_loaded_callback(sd_model)

    if not shared.cmd_opts.lowvram and not shared.cmd_opts.medvram:
        sd_model.to(devices.device)

    print(f"VAE Weights loaded.")
    return sd_model
