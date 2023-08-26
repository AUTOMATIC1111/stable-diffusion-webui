import os
import collections
from dataclasses import dataclass

from modules import paths, shared, devices, script_callbacks, sd_models, extra_networks, lowvram, sd_hijack, hashes

import glob
from copy import deepcopy


clip_path = os.path.abspath(os.path.join(paths.models_path, "clip"))
clip_ignore_keys = {"model_ema.decay", "model_ema.num_updates"}
clip_dict = {}


base_clip = None
loaded_clip_file = None
checkpoint_info = None

checkpoints_loaded = collections.OrderedDict()


def get_loaded_clip_name():
    if loaded_clip_file is None:
        return None

    return os.path.basename(loaded_clip_file)


def get_loaded_clip_hash():
    if loaded_clip_file is None:
        return None

    sha256 = hashes.sha256(loaded_clip_file, 'clip')

    return sha256[0:10] if sha256 else None


def get_base_clip(model):
    if base_clip is not None and checkpoint_info == model.sd_checkpoint_info and model:
        return base_clip
    return None


def store_base_clip(model):
    global base_clip, checkpoint_info
    if checkpoint_info != model.sd_checkpoint_info:
        assert not loaded_clip_file, "Trying to store non-base clip!"
        base_clip = deepcopy(model.cond_stage_model.state_dict())
        checkpoint_info = model.sd_checkpoint_info


def delete_base_clip():
    global base_clip, checkpoint_info
    base_clip = None
    checkpoint_info = None


def restore_base_clip(model):
    global loaded_clip_file
    if base_clip is not None and checkpoint_info == model.sd_checkpoint_info:
        print("Restoring base clip")
        _load_clip_dict(model, base_clip)
        loaded_clip_file = None
    delete_base_clip()


def get_filename(filepath):
    return os.path.basename(filepath)


def refresh_clip_list():
    clip_dict.clear()

    paths = [
        os.path.join(sd_models.model_path, '**/*.clip.ckpt'),
        os.path.join(sd_models.model_path, '**/*.clip.pt'),
        os.path.join(sd_models.model_path, '**/*.clip.safetensors'),
        os.path.join(clip_path, '**/*.ckpt'),
        os.path.join(clip_path, '**/*.pt'),
        os.path.join(clip_path, '**/*.safetensors'),
    ]

    if shared.cmd_opts.ckpt_dir is not None and os.path.isdir(shared.cmd_opts.ckpt_dir):
        paths += [
            os.path.join(shared.cmd_opts.ckpt_dir, '**/*.clip.ckpt'),
            os.path.join(shared.cmd_opts.ckpt_dir, '**/*.clip.pt'),
            os.path.join(shared.cmd_opts.ckpt_dir, '**/*.clip.safetensors'),
        ]

    if shared.cmd_opts.clip_dir is not None and os.path.isdir(shared.cmd_opts.clip_dir):
        paths += [
            os.path.join(shared.cmd_opts.clip_dir, '**/*.ckpt'),
            os.path.join(shared.cmd_opts.clip_dir, '**/*.pt'),
            os.path.join(shared.cmd_opts.clip_dir, '**/*.safetensors'),
        ]

    candidates = []
    for path in paths:
        candidates += glob.iglob(path, recursive=True)

    for filepath in candidates:
        name = get_filename(filepath)
        clip_dict[name] = filepath

    clip_dict.update(dict(sorted(clip_dict.items(), key=lambda item: shared.natural_sort_key(item[0]))))


def find_clip_near_checkpoint(checkpoint_file):
    checkpoint_path = os.path.basename(checkpoint_file).rsplit('.', 1)[0]
    for clip_file in clip_dict.values():
        if os.path.basename(clip_file).startswith(checkpoint_path):
            return clip_file

    return None


@dataclass
class clipResolution:
    clip: str = None
    source: str = None
    resolved: bool = True

    def tuple(self):
        return self.clip, self.source


def is_automatic():
    return shared.opts.sd_clip in {"Automatic", "auto"}  # "auto" for people with old config


def resolve_clip_from_setting() -> clipResolution:
    if shared.opts.sd_clip == "None":
        return clipResolution()

    clip_from_options = clip_dict.get(shared.opts.sd_clip, None)
    if clip_from_options is not None:
        return clipResolution(clip_from_options, 'specified in settings')

    if not is_automatic():
        print(f"Couldn't find clip named {shared.opts.sd_clip}; using None instead")

    return clipResolution(resolved=False)


def resolve_clip_from_user_metadata(checkpoint_file) -> clipResolution:
    metadata = extra_networks.get_user_metadata(checkpoint_file)
    clip_metadata = metadata.get("clip", None)
    if clip_metadata is not None and clip_metadata != "Automatic":
        if clip_metadata == "None":
            return clipResolution()

        clip_from_metadata = clip_dict.get(clip_metadata, None)
        if clip_from_metadata is not None:
            return clipResolution(clip_from_metadata, "from user metadata")

    return clipResolution(resolved=False)


def resolve_clip_near_checkpoint(checkpoint_file) -> clipResolution:
    clip_near_checkpoint = find_clip_near_checkpoint(checkpoint_file)
    if clip_near_checkpoint is not None and (not shared.opts.sd_clip_overrides_per_model_preferences or is_automatic()):
        return clipResolution(clip_near_checkpoint, 'found near the checkpoint')

    return clipResolution(resolved=False)


def resolve_clip(checkpoint_file) -> clipResolution:
    if shared.cmd_opts.clip_path is not None:
        return clipResolution(shared.cmd_opts.clip_path, 'from commandline argument')

    if shared.opts.sd_clip_overrides_per_model_preferences and not is_automatic():
        return resolve_clip_from_setting()

    res = resolve_clip_from_user_metadata(checkpoint_file)
    if res.resolved:
        return res

    res = resolve_clip_near_checkpoint(checkpoint_file)
    if res.resolved:
        return res

    res = resolve_clip_from_setting()

    return res


def load_clip_dict(filename, map_location):
    clip_ckpt = sd_models.read_state_dict(filename, map_location=map_location)
    clip_dict_1 = {k: v for k, v in clip_ckpt.items() if k[0:4] != "loss" and k not in clip_ignore_keys}
    #check key format and fix if transformer. is missing
    for key in list(clip_dict_1.keys()):
        if key.startswith("text_model."):
            new_key = "transformer."+key
            clip_dict_1[new_key] = clip_dict_1.pop(key)
    return clip_dict_1


def load_clip(model, clip_file=None, clip_source="from unknown source"):
    global clip_dict, base_clip, loaded_clip_file
    # save_settings = False

    cache_enabled = shared.opts.sd_clip_checkpoint_cache > 0

    if clip_file:
        if cache_enabled and clip_file in checkpoints_loaded:
            # use clip checkpoint cache
            print(f"Loading clip weights {clip_source}: cached {get_filename(clip_file)}")
            store_base_clip(model)
            _load_clip_dict(model, checkpoints_loaded[clip_file])
        else:
            assert os.path.isfile(clip_file), f"clip {clip_source} doesn't exist: {clip_file}"
            print(f"Loading clip weights {clip_source}: {clip_file}")
            store_base_clip(model)

            clip_dict_1 = load_clip_dict(clip_file, map_location=shared.weight_load_location)
            _load_clip_dict(model, clip_dict_1)

            if cache_enabled:
                # cache newly loaded clip
                checkpoints_loaded[clip_file] = clip_dict_1.copy()

        # clean up cache if limit is reached
        if cache_enabled:
            while len(checkpoints_loaded) > shared.opts.sd_clip_checkpoint_cache + 1: # we need to count the current model
                checkpoints_loaded.popitem(last=False)  # LRU

        # If clip used is not in dict, update it
        # It will be removed on refresh though
        clip_opt = get_filename(clip_file)
        if clip_opt not in clip_dict:
            clip_dict[clip_opt] = clip_file

    elif loaded_clip_file:
        restore_base_clip(model)

    loaded_clip_file = clip_file
    model.base_clip = base_clip
    model.loaded_clip_file = loaded_clip_file


# don't call this from outside
def _load_clip_dict(model, clip_dict_1):
    model.cond_stage_model.load_state_dict(clip_dict_1)
    model.cond_stage_model.to(devices.dtype_clip)


def clear_loaded_clip():
    global loaded_clip_file
    loaded_clip_file = None


unspecified = object()


def reload_clip_weights(sd_model=None, clip_file=unspecified):
    if not sd_model:
        sd_model = shared.sd_model

    checkpoint_info = sd_model.sd_checkpoint_info
    checkpoint_file = checkpoint_info.filename

    if clip_file == unspecified:
        clip_file, clip_source = resolve_clip(checkpoint_file).tuple()
    else:
        clip_source = "from function argument"

    if loaded_clip_file == clip_file:
        return

    if sd_model.lowvram:
        lowvram.send_everything_to_cpu()
    else:
        sd_model.to(devices.cpu)

    sd_hijack.model_hijack.undo_hijack(sd_model)

    load_clip(sd_model, clip_file, clip_source)

    sd_hijack.model_hijack.hijack(sd_model)
    script_callbacks.model_loaded_callback(sd_model)

    if not sd_model.lowvram:
        sd_model.to(devices.device)

    print("clip weights loaded.")
    return sd_model
