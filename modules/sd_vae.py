import torch
import os
from pathlib import Path
from collections import namedtuple
from modules import shared, devices, script_callbacks
from modules.paths import models_path
import glob


model_dir_name = "Stable-diffusion"
model_dir = os.path.abspath(os.path.join(models_path, model_dir_name))
vae_dir_name = "VAE"
vae_dir = os.path.abspath(os.path.join(models_path, vae_dir_name))
temp_dir = "tmp"
Path(temp_dir).mkdir(parents=True, exist_ok=True)
temp_vae_file = os.path.join(temp_dir, "base.vae.pt")


vae_ignore_keys = {"model_ema.decay", "model_ema.num_updates"}


default_vae_dict = {"auto": "auto", "None": None, None: None}
default_vae_list = ["auto", "None"]


default_vae_values = [default_vae_dict[x] for x in default_vae_list]
vae_dict = dict(default_vae_dict)
vae_list = list(default_vae_list)
vae_auto_label = vae_list[0]
vae_auto_set = {"auto", vae_auto_label}
first_load = True


base_vae = None
loaded_vae_file = None
checkpoint_info = None


def init():
    global caching_mode
    caching_mode = shared.opts.data.get("sd_base_vae_cache", "file")
    refresh_vae_list()


def get_base_vae(model):
    if base_vae is not None and model and checkpoint_info == model.sd_checkpoint_info:
        if caching_mode == "ram" or isinstance(base_vae, dict):
            return base_vae
        if caching_mode == "file" or isinstance(base_vae, str) and os.path.isfile(base_vae):
            print(f"Reading Base VAE weights from: {base_vae}")
            return load_vae_file(base_vae)
    return None


def refresh_caching_mode():
    global caching_mode
    caching_mode = shared.opts.sd_base_vae_cache


def store_base_vae(model, vae_dict_1=None, skip_vae_check=False):
    global checkpoint_info
    refresh_caching_mode()
    if checkpoint_info != model.sd_checkpoint_info:
        if not skip_vae_check:
            assert not loaded_vae_file, "Trying to store non-base VAE!"
        if not vae_dict_1 and caching_mode in {"ram", "file"}:
            vae_dict_1 = model.first_stage_model.state_dict()
        store_base_vae_dict(vae_dict_1)
        checkpoint_info = model.sd_checkpoint_info


def store_base_vae_dict(vae_dict_1):
    global base_vae, caching_mode
    refresh_caching_mode()
    if caching_mode == "ram":
        base_vae = vae_dict_1.copy()
    elif caching_mode == "file":
        base_vae = temp_vae_file
        print(f"Writing Base VAE weights to: {base_vae}")
        torch.save({"state_dict": vae_dict_1}, base_vae)
    elif caching_mode == "none":
        base_vae = None
    else:
        shared.opts.sd_base_vae_cache = "file"
        caching_mode = "file"
        return store_base_vae_dict(vae_dict_1)


def delete_base_vae():
    global base_vae, checkpoint_info
    if base_vae:
        if (caching_mode == "file" or isinstance(base_vae, str)) and os.path.isfile(base_vae):
            os.remove(base_vae)
        base_vae = None
    checkpoint_info = None


def restore_base_vae(model):
    global checkpoint_info
    base_vae = get_base_vae(model)
    if loaded_vae_file:
        if checkpoint_info == model.sd_checkpoint_info:
            if base_vae is not None:
                print("Restoring base VAE")
                load_vae_dict(model, base_vae)
            else:
                print("Reloading checkpoint to restore base VAE")
                from modules import sd_models
                sd_models.load_model_weights(model, model.sd_checkpoint_info, vae_file="None", force=True)
        else:
            raise Exception("Unable to restore base VAE")
        clear_loaded_vae()

    delete_base_vae()


def get_filename(filepath):
    if not filepath:
        return "None"
    return os.path.relpath(filepath, models_path)


def refresh_vae_list(vae_dir=vae_dir, model_dir=model_dir, checkpoint_info=None):
    global vae_dict, vae_list, vae_auto_label, vae_auto_set
    res = {}
    candidates = [
        *glob.iglob(os.path.join(model_dir, '**/*.vae.ckpt'), recursive=True),
        *glob.iglob(os.path.join(model_dir, '**/*.vae.pt'), recursive=True),
        *glob.iglob(os.path.join(vae_dir, '**/*.ckpt'), recursive=True),
        *glob.iglob(os.path.join(vae_dir, '**/*.pt'), recursive=True)
    ]
    if shared.cmd_opts.vae_path is not None and os.path.isfile(shared.cmd_opts.vae_path):
        candidates.append(shared.cmd_opts.vae_path)
    for filepath in candidates:
        name = get_filename(filepath)
        res[name] = filepath

    vae_auto_label = f"auto ({get_filename(resolve_vae(verbose=False, skip_selection=True))})"

    vae_list.clear()
    vae_list.extend([vae_auto_label, *default_vae_list[1:]])
    vae_list.extend(list(res.keys()))

    vae_auto_set.clear()
    vae_auto_set.update({"auto", vae_auto_label})

    vae_dict.clear()
    vae_dict.update(res)
    vae_dict.update(default_vae_dict)
    vae_dict[vae_auto_label] = "auto"

    ret = {"choices": vae_list}

    if shared.opts.data["sd_vae"] not in vae_list:
        shared.opts.data["sd_vae"] = vae_list[0]
        ret["value"] = shared.opts.data["sd_vae"]

    return ret


def get_vae_from_settings(vae_file="auto", verbose=True):
    # else, we load from settings, if not set to be default
    if vae_file in vae_auto_set and shared.opts.sd_vae is not None:
        # if saved VAE settings isn't recognized, fallback to auto
        vae_file = vae_dict.get(shared.opts.sd_vae, vae_auto_label)
        # if VAE selected but not found, fallback to auto
        if vae_file not in default_vae_values and not os.path.isfile(vae_file):
            vae_file = vae_auto_label
            if verbose:
                print("Selected VAE doesn't exist: ", vae_file)
    return vae_file


def resolve_vae(checkpoint_file=None, vae_file="auto", verbose=True, skip_selection=False):
    global first_load, vae_dict, vae_list

    if not checkpoint_file:
        from modules import sd_models
        checkpoint_info = sd_models.select_checkpoint()
        checkpoint_file = checkpoint_info.filename if checkpoint_info else None

    # if vae_file argument is provided, it takes priority, but not saved
    if vae_file and vae_file not in default_vae_list:
        if not os.path.isfile(vae_file):
            vae_file = vae_auto_label
            if verbose:
                print("VAE provided as function argument doesn't exist")
    # for the first load, if vae-path is provided, it takes priority, saved, and failure is reported
    if first_load and shared.cmd_opts.vae_path is not None:
        if os.path.isfile(shared.cmd_opts.vae_path):
            vae_file = shared.cmd_opts.vae_path
            shared.opts.data['sd_vae'] = get_filename(vae_file)
        else:
            if verbose:
                print("VAE provided as command line argument doesn't exist")
    # fallback to selector in settings, if vae selector not set to act as default fallback
    if not shared.opts.sd_vae_as_default and not skip_selection:
        vae_file = get_vae_from_settings(vae_file, verbose=verbose)
    # vae-path cmd arg takes priority for auto
    if vae_file in vae_auto_set and shared.cmd_opts.vae_path is not None:
        if os.path.isfile(shared.cmd_opts.vae_path):
            vae_file = shared.cmd_opts.vae_path
            if verbose:
                print("Using VAE provided as command line argument")
    # if still not found, try look for VAE similar to model
    if vae_file in vae_auto_set and checkpoint_file:
        model_path = os.path.splitext(checkpoint_file)[0]
        rel_path = os.path.relpath(model_path, model_dir)
        vae_path = os.path.join(vae_dir, rel_path)
        trials = [
            model_path + ".vae.pt",
            model_path + ".vae.ckpt",
            vae_path + ".vae.pt",
            vae_path + ".vae.ckpt",
            vae_path + ".pt",
            vae_path + ".ckpt",
        ]
        for vae_file_try in trials:
            if os.path.isfile(vae_file_try):
                vae_file = vae_file_try
                if verbose:
                    print("Using VAE found similar to selected model")
                break
    # if vae selector were set as default fallback, call here
    if shared.opts.sd_vae_as_default and not skip_selection:
        vae_file = get_vae_from_settings(vae_file, verbose=verbose)

    # No more fallbacks for auto
    if vae_file in vae_auto_set:
        vae_file = None
    # Last check, just because
    if vae_file and not os.path.exists(vae_file):
        vae_file = None

    return vae_file


def load_vae(model, vae_file=None):
    global first_load, vae_dict, vae_list, loaded_vae_file, caching_mode

    if vae_file:
        store_base_vae(model)
        print(f"Loading VAE weights from: {vae_file}")
        vae_dict_1 = load_vae_file(vae_file)
        load_vae_dict(model, vae_dict_1)

        # If vae used is not in dict, update it
        # It will be removed on refresh though
        vae_opt = get_filename(vae_file)
        if vae_opt not in vae_dict:
            vae_dict[vae_opt] = vae_file
            vae_list.append(vae_opt)
            # shared.opts.data['sd_vae'] = vae_opt
    elif loaded_vae_file:
        restore_base_vae(model)

    loaded_vae_file = vae_file

    first_load = False


# don't call this from outside
def load_vae_file(vae_file):
    vae_ckpt = torch.load(vae_file, map_location=shared.weight_load_location)
    vae_dict_1 = {k: v for k, v in vae_ckpt["state_dict"].items() if k[0:4] != "loss" and k not in vae_ignore_keys}
    return vae_dict_1


# don't call this from outside
def load_vae_dict(model, vae_dict_1):
    model.first_stage_model.load_state_dict(vae_dict_1)
    model.first_stage_model.to(devices.dtype_vae)


def clear_loaded_vae():
    global loaded_vae_file
    loaded_vae_file = None


def pre_reload(sd_model):
    from modules import lowvram, devices, sd_hijack

    if shared.cmd_opts.lowvram or shared.cmd_opts.medvram:
        lowvram.send_everything_to_cpu()
    else:
        sd_model.to(devices.cpu)

    sd_hijack.model_hijack.undo_hijack(sd_model)


def post_reload(sd_model):
    from modules import devices, sd_hijack

    sd_hijack.model_hijack.hijack(sd_model)
    script_callbacks.model_loaded_callback(sd_model)

    if not shared.cmd_opts.lowvram and not shared.cmd_opts.medvram:
        sd_model.to(devices.device)


def reload_vae_weights(sd_model=None, vae_file="auto"):
    if not sd_model:
        sd_model = shared.sd_model

    vae_file = resolve_vae(vae_file=vae_file)

    if loaded_vae_file == vae_file:
        return

    pre_reload(sd_model)

    load_vae(sd_model, vae_file)

    post_reload(sd_model)

    print(f"VAE Weights loaded.")


def change_vae_cache(sd_model=None):
    global caching_mode

    if not sd_model:
        sd_model = shared.sd_model

    base_vae = get_base_vae(sd_model)
    if base_vae is not None:
        delete_base_vae()
        store_base_vae(sd_model, base_vae, skip_vae_check=True)
    else:
        pre_reload(sd_model)
        
        if loaded_vae_file:
            restore_base_vae(sd_model)
            load_vae(sd_model, resolve_vae())
        else:
            store_base_vae(sd_model)
            
        post_reload(sd_model)
