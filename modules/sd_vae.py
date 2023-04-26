import os
import collections
import glob
from copy import deepcopy
from rich import print # pylint: disable=redefined-builtin
import torch
from modules import paths, shared, devices, script_callbacks, sd_models


vae_ignore_keys = {"model_ema.decay", "model_ema.num_updates"}
vae_dict = {}
base_vae = None
loaded_vae_file = None
checkpoint_info = None
vae_path = os.path.abspath(os.path.join(paths.models_path, 'VAE'))
checkpoints_loaded = collections.OrderedDict()

def get_base_vae(model):
    if base_vae is not None and checkpoint_info == model.sd_checkpoint_info and model:
        return base_vae
    return None


def store_base_vae(model):
    global base_vae, checkpoint_info # pylint: disable=global-statement
    if checkpoint_info != model.sd_checkpoint_info:
        assert not loaded_vae_file, "Trying to store non-base VAE!"
        base_vae = deepcopy(model.first_stage_model.state_dict())
        checkpoint_info = model.sd_checkpoint_info


def delete_base_vae():
    global base_vae, checkpoint_info # pylint: disable=global-statement
    base_vae = None
    checkpoint_info = None


def restore_base_vae(model):
    global loaded_vae_file # pylint: disable=global-statement
    if base_vae is not None and checkpoint_info == model.sd_checkpoint_info:
        print("Restoring base VAE")
        _load_vae_dict(model, base_vae)
        loaded_vae_file = None
    delete_base_vae()


def get_filename(filepath):
    return os.path.basename(filepath)


def refresh_vae_list():
    global vae_path # pylint: disable=global-statement
    vae_path = shared.opts.vae_dir
    vae_dict.clear()

    vae_paths = [
        os.path.join(sd_models.model_path, '**/*.vae.ckpt'),
        os.path.join(sd_models.model_path, '**/*.vae.pt'),
        os.path.join(sd_models.model_path, '**/*.vae.safetensors'),
        os.path.join(shared.opts.vae_dir, '**/*.ckpt'),
        os.path.join(shared.opts.vae_dir, '**/*.pt'),
        os.path.join(shared.opts.vae_dir, '**/*.safetensors'),
    ]
    if shared.opts.ckpt_dir is not None and os.path.isdir(shared.opts.ckpt_dir):
        vae_paths += [
            os.path.join(shared.opts.ckpt_dir, '**/*.vae.ckpt'),
            os.path.join(shared.opts.ckpt_dir, '**/*.vae.pt'),
            os.path.join(shared.opts.ckpt_dir, '**/*.vae.safetensors'),
        ]
    if shared.opts.vae_dir is not None and os.path.isdir(shared.opts.vae_dir):
        vae_paths += [
            os.path.join(shared.opts.vae_dir, '**/*.ckpt'),
            os.path.join(shared.opts.vae_dir, '**/*.pt'),
            os.path.join(shared.opts.vae_dir, '**/*.safetensors'),
        ]
    candidates = []
    for path in vae_paths:
        candidates += glob.iglob(path, recursive=True)

    for filepath in candidates:
        name = get_filename(filepath)
        vae_dict[name] = filepath


def find_vae_near_checkpoint(checkpoint_file):
    checkpoint_path = os.path.splitext(checkpoint_file)[0]
    for vae_location in [checkpoint_path + ".vae.pt", checkpoint_path + ".vae.ckpt", checkpoint_path + ".vae.safetensors"]:
        if os.path.isfile(vae_location):
            return vae_location

    return None


def resolve_vae(checkpoint_file):
    if shared.cmd_opts.vae is not None:
        return shared.cmd_opts.vae, 'forced'

    is_automatic = shared.opts.sd_vae in {"Automatic", "auto"}  # "auto" for people with old config

    vae_near_checkpoint = find_vae_near_checkpoint(checkpoint_file)
    if vae_near_checkpoint is not None and (shared.opts.sd_vae_as_default):
        return vae_near_checkpoint, 'near checkpoint'
    
    if is_automatic:
        for named_vae_location in [os.path.join(vae_path, os.path.splitext(os.path.basename(checkpoint_file))[0] + ".vae.pt"), os.path.join(vae_path, os.path.splitext(os.path.basename(checkpoint_file))[0] + ".vae.ckpt"), os.path.join(vae_path, os.path.splitext(os.path.basename(checkpoint_file))[0] + ".vae.safetensors")]:
            if os.path.isfile(named_vae_location):
                return named_vae_location, 'in VAE dir'

    if shared.opts.sd_vae == "None":
        return None, None

    vae_from_options = vae_dict.get(shared.opts.sd_vae, None)
    if vae_from_options is not None:
        return vae_from_options, 'specified in settings'

    if not is_automatic:
        print(f"VAE not found: {shared.opts.sd_vae}")

    return None, None


def load_vae_dict(filename):
    vae_ckpt = sd_models.read_state_dict(filename)
    vae_dict_1 = {k: v for k, v in vae_ckpt.items() if k[0:4] != "loss" and k not in vae_ignore_keys}
    return vae_dict_1


def load_vae(model, vae_file=None, vae_source="from unknown source"):
    global loaded_vae_file # pylint: disable=global-statement
    # save_settings = False

    cache_enabled = shared.opts.sd_vae_checkpoint_cache > 0

    if vae_file:
        if cache_enabled and vae_file in checkpoints_loaded:
            # use vae checkpoint cache
            print(f"Loading VAE weights {vae_source}: cached {get_filename(vae_file)}")
            store_base_vae(model)
            _load_vae_dict(model, checkpoints_loaded[vae_file])
        else:
            assert os.path.isfile(vae_file), f"VAE {vae_source} doesn't exist: {vae_file}"
            store_base_vae(model)

            vae_dict_1 = load_vae_dict(vae_file)
            _load_vae_dict(model, vae_dict_1)

            if cache_enabled:
                # cache newly loaded vae
                checkpoints_loaded[vae_file] = vae_dict_1.copy()

        # clean up cache if limit is reached
        if cache_enabled:
            while len(checkpoints_loaded) > shared.opts.sd_vae_checkpoint_cache + 1: # we need to count the current model
                checkpoints_loaded.popitem(last=False)  # LRU

        # If vae used is not in dict, update it
        # It will be removed on refresh though
        vae_opt = get_filename(vae_file)
        if vae_opt not in vae_dict:
            vae_dict[vae_opt] = vae_file

    elif loaded_vae_file:
        restore_base_vae(model)

    loaded_vae_file = vae_file


# don't call this from outside
def _load_vae_dict(model, vae_dict_1):
    model.first_stage_model.load_state_dict(vae_dict_1)
    model.first_stage_model.to(devices.dtype_vae)


def clear_loaded_vae():
    global loaded_vae_file # pylint: disable=global-statement
    loaded_vae_file = None


unspecified = object()


def reload_vae_weights(sd_model=None, vae_file=unspecified):
    from modules import lowvram, sd_hijack

    if not sd_model:
        sd_model = shared.sd_model

    global checkpoint_info # pylint: disable=global-statement
    checkpoint_info = sd_model.sd_checkpoint_info
    checkpoint_file = checkpoint_info.filename

    if vae_file == unspecified:
        vae_file, vae_source = resolve_vae(checkpoint_file)
    else:
        vae_source = "from function argument"

    if loaded_vae_file == vae_file:
        return

    if shared.cmd_opts.lowvram or shared.cmd_opts.medvram:
        lowvram.send_everything_to_cpu()
    else:
        sd_model.to(devices.cpu)

    sd_hijack.model_hijack.undo_hijack(sd_model)
    if shared.cmd_opts.rollback_vae and devices.dtype_vae == torch.bfloat16:
        devices.dtype_vae = torch.float16

    load_vae(sd_model, vae_file, vae_source)

    sd_hijack.model_hijack.hijack(sd_model)
    script_callbacks.model_loaded_callback(sd_model)

    if not shared.cmd_opts.lowvram and not shared.cmd_opts.medvram:
        sd_model.to(devices.device)

    print("VAE weights loaded.")
    return sd_model
