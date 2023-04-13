import collections
import os.path
import sys
import gc
import torch
import re
import safetensors.torch
from omegaconf import OmegaConf
from os import mkdir
from urllib import request
import ldm.modules.midas as midas

from ldm.util import instantiate_from_config

from modules import paths, shared, modelloader, devices, script_callbacks, sd_vae, sd_disable_initialization, errors, hashes, sd_models_config
from modules.paths import models_path
from modules.sd_hijack_inpainting import do_inpainting_hijack
from modules.timer import Timer

model_dir = "Stable-diffusion"
model_path = os.path.abspath(os.path.join(paths.models_path, model_dir))

checkpoints_list = {}
checkpoint_alisases = {}
checkpoints_loaded = collections.OrderedDict()


class CheckpointInfo:
    def __init__(self, filename):
        self.filename = filename
        abspath = os.path.abspath(filename)

        if shared.cmd_opts.ckpt_dir is not None and abspath.startswith(shared.cmd_opts.ckpt_dir):
            name = abspath.replace(shared.cmd_opts.ckpt_dir, '')
        elif abspath.startswith(model_path):
            name = abspath.replace(model_path, '')
        else:
            name = os.path.basename(filename)

        if name.startswith("\\") or name.startswith("/"):
            name = name[1:]

        self.name = name
        self.name_for_extra = os.path.splitext(os.path.basename(filename))[0]
        self.model_name = os.path.splitext(name.replace("/", "_").replace("\\", "_"))[0]
        self.hash = model_hash(filename)

        self.sha256 = hashes.sha256_from_cache(self.filename, "checkpoint/" + name)
        self.shorthash = self.sha256[0:10] if self.sha256 else None

        self.title = name if self.shorthash is None else f'{name} [{self.shorthash}]'

        self.ids = [self.hash, self.model_name, self.title, name, f'{name} [{self.hash}]'] + ([self.shorthash, self.sha256, f'{self.name} [{self.shorthash}]'] if self.shorthash else [])

    def register(self):
        checkpoints_list[self.title] = self
        for id in self.ids:
            checkpoint_alisases[id] = self

    def calculate_shorthash(self):
        self.sha256 = hashes.sha256(self.filename, "checkpoint/" + self.name)
        if self.sha256 is None:
            return

        self.shorthash = self.sha256[0:10]

        if self.shorthash not in self.ids:
            self.ids += [self.shorthash, self.sha256, f'{self.name} [{self.shorthash}]']

        checkpoints_list.pop(self.title)
        self.title = f'{self.name} [{self.shorthash}]'
        self.register()

        return self.shorthash


try:
    # this silences the annoying "Some weights of the model checkpoint were not used when initializing..." message at start.

    from transformers import logging, CLIPModel

    logging.set_verbosity_error()
except Exception:
    pass


def setup_model():
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    list_models()
    enable_midas_autodownload()


def checkpoint_tiles():
    def convert(name):
        return int(name) if name.isdigit() else name.lower()

    def alphanumeric_key(key):
        return [convert(c) for c in re.split('([0-9]+)', key)]

    return sorted([x.title for x in checkpoints_list.values()], key=alphanumeric_key)


def list_models():
    checkpoints_list.clear()
    checkpoint_alisases.clear()

    cmd_ckpt = shared.cmd_opts.ckpt
    if shared.cmd_opts.no_download_sd_model or cmd_ckpt != shared.sd_model_file or os.path.exists(cmd_ckpt):
        model_url = None
    else:
        model_url = "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors"

    model_list = modelloader.load_models(model_path=model_path, model_url=model_url, command_path=shared.cmd_opts.ckpt_dir, ext_filter=[".ckpt", ".safetensors"], download_name="v1-5-pruned-emaonly.safetensors", ext_blacklist=[".vae.ckpt", ".vae.safetensors"])

    if os.path.exists(cmd_ckpt):
        checkpoint_info = CheckpointInfo(cmd_ckpt)
        checkpoint_info.register()

        shared.opts.data['sd_model_checkpoint'] = checkpoint_info.title
    elif cmd_ckpt is not None and cmd_ckpt != shared.default_sd_model_file:
        print(f"Checkpoint in --ckpt argument not found (Possible it was moved to {model_path}: {cmd_ckpt}", file=sys.stderr)

    for filename in sorted(model_list, key=str.lower):
        checkpoint_info = CheckpointInfo(filename)
        checkpoint_info.register()


def get_closet_checkpoint_match(search_string):
    checkpoint_info = checkpoint_alisases.get(search_string, None)
    if checkpoint_info is not None:
        return checkpoint_info

    found = sorted([info for info in checkpoints_list.values() if search_string in info.title], key=lambda x: len(x.title))
    if found:
        return found[0]

    return None


def model_hash(filename):
    """old hash that only looks at a small part of the file and is prone to collisions"""

    try:
        with open(filename, "rb") as file:
            import hashlib
            m = hashlib.sha256()

            file.seek(0x100000)
            m.update(file.read(0x10000))
            return m.hexdigest()[0:8]
    except FileNotFoundError:
        return 'NOFILE'


def select_checkpoint():
    model_checkpoint = shared.opts.sd_model_checkpoint
        
    checkpoint_info = checkpoint_alisases.get(model_checkpoint, None)
    if checkpoint_info is not None:
        return checkpoint_info

    if len(checkpoints_list) == 0:
        print("No checkpoints found. When searching for checkpoints, looked at:", file=sys.stderr)
        if shared.cmd_opts.ckpt is not None:
            print(f" - file {os.path.abspath(shared.cmd_opts.ckpt)}", file=sys.stderr)
        print(f" - directory {model_path}", file=sys.stderr)
        if shared.cmd_opts.ckpt_dir is not None:
            print(f" - directory {os.path.abspath(shared.cmd_opts.ckpt_dir)}", file=sys.stderr)
        print("Can't run without a checkpoint. Find and place a .ckpt or .safetensors file into any of those locations. The program will exit.", file=sys.stderr)
        exit(1)

    checkpoint_info = next(iter(checkpoints_list.values()))
    if model_checkpoint is not None:
        print(f"Checkpoint {model_checkpoint} not found; loading fallback {checkpoint_info.title}", file=sys.stderr)

    return checkpoint_info


checkpoint_dict_replacements = {
    'cond_stage_model.transformer.embeddings.': 'cond_stage_model.transformer.text_model.embeddings.',
    'cond_stage_model.transformer.encoder.': 'cond_stage_model.transformer.text_model.encoder.',
    'cond_stage_model.transformer.final_layer_norm.': 'cond_stage_model.transformer.text_model.final_layer_norm.',
}


def transform_checkpoint_dict_key(k):
    for text, replacement in checkpoint_dict_replacements.items():
        if k.startswith(text):
            k = replacement + k[len(text):]

    return k


def get_state_dict_from_checkpoint(pl_sd):
    pl_sd = pl_sd.pop("state_dict", pl_sd)
    pl_sd.pop("state_dict", None)

    sd = {}
    for k, v in pl_sd.items():
        new_key = transform_checkpoint_dict_key(k)

        if new_key is not None:
            sd[new_key] = v

    pl_sd.clear()
    pl_sd.update(sd)

    return pl_sd


def read_metadata_from_safetensors(filename):
    import json

    with open(filename, mode="rb") as file:
        metadata_len = file.read(8)
        metadata_len = int.from_bytes(metadata_len, "little")
        json_start = file.read(2)

        assert metadata_len > 2 and json_start in (b'{"', b"{'"), f"{filename} is not a safetensors file"
        json_data = json_start + file.read(metadata_len-2)
        json_obj = json.loads(json_data)

        res = {}
        for k, v in json_obj.get("__metadata__", {}).items():
            res[k] = v
            if isinstance(v, str) and v[0:1] == '{':
                try:
                    res[k] = json.loads(v)
                except Exception as e:
                    pass

        return res


def read_state_dict(checkpoint_file, print_global_state=False, map_location=None):
    _, extension = os.path.splitext(checkpoint_file)
    if extension.lower() == ".safetensors":
        device = map_location or shared.weight_load_location or devices.get_optimal_device_name()
        pl_sd = safetensors.torch.load_file(checkpoint_file, device=device)
    else:
        pl_sd = torch.load(checkpoint_file, map_location=map_location or shared.weight_load_location)

    if print_global_state and "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")

    sd = get_state_dict_from_checkpoint(pl_sd)
    return sd


def get_checkpoint_state_dict(checkpoint_info: CheckpointInfo, timer):
    sd_model_hash = checkpoint_info.calculate_shorthash()
    timer.record("calculate hash")

    if checkpoint_info in checkpoints_loaded:
        # use checkpoint cache
        print(f"Loading weights [{sd_model_hash}] from cache")
        return checkpoints_loaded[checkpoint_info]

    print(f"Loading weights [{sd_model_hash}] from {checkpoint_info.filename}")
    res = read_state_dict(checkpoint_info.filename)
    timer.record("load weights from disk")

    return res


def load_model_weights(model, checkpoint_info: CheckpointInfo, state_dict, timer):
    sd_model_hash = checkpoint_info.calculate_shorthash()
    timer.record("calculate hash")

    shared.opts.data["sd_model_checkpoint"] = checkpoint_info.title

    if state_dict is None:
        state_dict = get_checkpoint_state_dict(checkpoint_info, timer)

    model.load_state_dict(state_dict, strict=False)
    del state_dict
    timer.record("apply weights to model")

    if shared.opts.sd_checkpoint_cache > 0:
        # cache newly loaded model
        checkpoints_loaded[checkpoint_info] = model.state_dict().copy()

    if shared.cmd_opts.opt_channelslast:
        model.to(memory_format=torch.channels_last)
        timer.record("apply channels_last")

    if not shared.cmd_opts.no_half:
        vae = model.first_stage_model
        depth_model = getattr(model, 'depth_model', None)

        # with --no-half-vae, remove VAE from model when doing half() to prevent its weights from being converted to float16
        if shared.cmd_opts.no_half_vae:
            model.first_stage_model = None
        # with --upcast-sampling, don't convert the depth model weights to float16
        if shared.cmd_opts.upcast_sampling and depth_model:
            model.depth_model = None

        model.half()
        model.first_stage_model = vae
        if depth_model:
            model.depth_model = depth_model

        timer.record("apply half()")

    devices.dtype = torch.float32 if shared.cmd_opts.no_half else torch.float16
    devices.dtype_vae = torch.float32 if shared.cmd_opts.no_half or shared.cmd_opts.no_half_vae else torch.float16
    devices.dtype_unet = model.model.diffusion_model.dtype
    devices.unet_needs_upcast = shared.cmd_opts.upcast_sampling and devices.dtype == torch.float16 and devices.dtype_unet == torch.float16

    model.first_stage_model.to(devices.dtype_vae)
    timer.record("apply dtype to VAE")

    # clean up cache if limit is reached
    while len(checkpoints_loaded) > shared.opts.sd_checkpoint_cache:
        checkpoints_loaded.popitem(last=False)

    model.sd_model_hash = sd_model_hash
    model.sd_model_checkpoint = checkpoint_info.filename
    model.sd_checkpoint_info = checkpoint_info
    shared.opts.data["sd_checkpoint_hash"] = checkpoint_info.sha256

    model.logvar = model.logvar.to(devices.device)  # fix for training

    sd_vae.delete_base_vae()
    sd_vae.clear_loaded_vae()
    vae_file, vae_source = sd_vae.resolve_vae(checkpoint_info.filename)
    sd_vae.load_vae(model, vae_file, vae_source)
    timer.record("load VAE")


def enable_midas_autodownload():
    """
    Gives the ldm.modules.midas.api.load_model function automatic downloading.

    When the 512-depth-ema model, and other future models like it, is loaded,
    it calls midas.api.load_model to load the associated midas depth model.
    This function applies a wrapper to download the model to the correct
    location automatically.
    """

    midas_path = os.path.join(paths.models_path, 'midas')

    # stable-diffusion-stability-ai hard-codes the midas model path to
    # a location that differs from where other scripts using this model look.
    # HACK: Overriding the path here.
    for k, v in midas.api.ISL_PATHS.items():
        file_name = os.path.basename(v)
        midas.api.ISL_PATHS[k] = os.path.join(midas_path, file_name)

    midas_urls = {
        "dpt_large": "https://github.com/intel-isl/DPT/releases/download/1_0/dpt_large-midas-2f21e586.pt",
        "dpt_hybrid": "https://github.com/intel-isl/DPT/releases/download/1_0/dpt_hybrid-midas-501f0c75.pt",
        "midas_v21": "https://github.com/AlexeyAB/MiDaS/releases/download/midas_dpt/midas_v21-f6b98070.pt",
        "midas_v21_small": "https://github.com/AlexeyAB/MiDaS/releases/download/midas_dpt/midas_v21_small-70d6b9c8.pt",
    }

    midas.api.load_model_inner = midas.api.load_model

    def load_model_wrapper(model_type):
        path = midas.api.ISL_PATHS[model_type]
        if not os.path.exists(path):
            if not os.path.exists(midas_path):
                mkdir(midas_path)
    
            print(f"Downloading midas model weights for {model_type} to {path}")
            request.urlretrieve(midas_urls[model_type], path)
            print(f"{model_type} downloaded")

        return midas.api.load_model_inner(model_type)

    midas.api.load_model = load_model_wrapper


def repair_config(sd_config):

    if not hasattr(sd_config.model.params, "use_ema"):
        sd_config.model.params.use_ema = False

    if shared.cmd_opts.no_half:
        sd_config.model.params.unet_config.params.use_fp16 = False
    elif shared.cmd_opts.upcast_sampling:
        sd_config.model.params.unet_config.params.use_fp16 = True

    if getattr(sd_config.model.params.first_stage_config.params.ddconfig, "attn_type", None) == "vanilla-xformers" and not shared.xformers_available:
        sd_config.model.params.first_stage_config.params.ddconfig.attn_type = "vanilla"

    # For UnCLIP-L, override the hardcoded karlo directory
    if hasattr(sd_config.model.params, "noise_aug_config") and hasattr(sd_config.model.params.noise_aug_config.params, "clip_stats_path"):
        karlo_path = os.path.join(paths.models_path, 'karlo')
        sd_config.model.params.noise_aug_config.params.clip_stats_path = sd_config.model.params.noise_aug_config.params.clip_stats_path.replace("checkpoints/karlo_models", karlo_path)


sd1_clip_weight = 'cond_stage_model.transformer.text_model.embeddings.token_embedding.weight'
sd2_clip_weight = 'cond_stage_model.model.transformer.resblocks.0.attn.in_proj_weight'

def load_model(checkpoint_info=None, already_loaded_state_dict=None, time_taken_to_load_state_dict=None):
    from modules import lowvram, sd_hijack
    checkpoint_info = checkpoint_info or select_checkpoint()

    if shared.sd_model:
        sd_hijack.model_hijack.undo_hijack(shared.sd_model)
        shared.sd_model = None
        gc.collect()
        devices.torch_gc()

    do_inpainting_hijack()

    timer = Timer()

    if already_loaded_state_dict is not None:
        state_dict = already_loaded_state_dict
    else:
        state_dict = get_checkpoint_state_dict(checkpoint_info, timer)

    checkpoint_config = sd_models_config.find_checkpoint_config(state_dict, checkpoint_info)
    clip_is_included_into_sd = sd1_clip_weight in state_dict or sd2_clip_weight in state_dict

    timer.record("find config")

    sd_config = OmegaConf.load(checkpoint_config)
    repair_config(sd_config)

    timer.record("load config")

    print(f"Creating model from config: {checkpoint_config}")

    sd_model = None
    try:
        with sd_disable_initialization.DisableInitialization(disable_clip=clip_is_included_into_sd):
            sd_model = instantiate_from_config(sd_config.model)
    except Exception as e:
        pass

    if sd_model is None:
        print('Failed to create model quickly; will retry using slow method.', file=sys.stderr)
        sd_model = instantiate_from_config(sd_config.model)

    sd_model.used_config = checkpoint_config

    timer.record("create model")

    load_model_weights(sd_model, checkpoint_info, state_dict, timer)

    if shared.cmd_opts.lowvram or shared.cmd_opts.medvram:
        lowvram.setup_for_low_vram(sd_model, shared.cmd_opts.medvram)
    else:
        sd_model.to(shared.device)

    timer.record("move model to device")

    sd_hijack.model_hijack.hijack(sd_model)

    timer.record("hijack")

    sd_model.eval()
    shared.sd_model = sd_model

    sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings(force_reload=True)  # Reload embeddings after model load as they may or may not fit the model

    timer.record("load textual inversion embeddings")

    script_callbacks.model_loaded_callback(sd_model)

    timer.record("scripts callbacks")

    print(f"Model loaded in {timer.summary()}.")

    return sd_model


def reload_model_weights(sd_model=None, info=None):
    from modules import lowvram, devices, sd_hijack
    checkpoint_info = info or select_checkpoint()

    if not sd_model:
        sd_model = shared.sd_model

    if sd_model is None:  # previous model load failed
        current_checkpoint_info = None
    else:
        current_checkpoint_info = sd_model.sd_checkpoint_info
        if sd_model.sd_model_checkpoint == checkpoint_info.filename:
            return

        if shared.cmd_opts.lowvram or shared.cmd_opts.medvram:
            lowvram.send_everything_to_cpu()
        else:
            sd_model.to(devices.cpu)

        sd_hijack.model_hijack.undo_hijack(sd_model)

    timer = Timer()

    state_dict = get_checkpoint_state_dict(checkpoint_info, timer)

    checkpoint_config = sd_models_config.find_checkpoint_config(state_dict, checkpoint_info)

    timer.record("find config")

    if sd_model is None or checkpoint_config != sd_model.used_config:
        del sd_model
        checkpoints_loaded.clear()
        load_model(checkpoint_info, already_loaded_state_dict=state_dict)
        return shared.sd_model

    try:
        load_model_weights(sd_model, checkpoint_info, state_dict, timer)
    except Exception as e:
        print("Failed to load checkpoint, restoring previous")
        load_model_weights(sd_model, current_checkpoint_info, None, timer)
        raise
    finally:
        sd_hijack.model_hijack.hijack(sd_model)
        timer.record("hijack")

        script_callbacks.model_loaded_callback(sd_model)
        timer.record("script callbacks")

        if not shared.cmd_opts.lowvram and not shared.cmd_opts.medvram:
            sd_model.to(devices.device)
            timer.record("move model to device")

    print(f"Weights loaded in {timer.summary()}.")

    return sd_model

def unload_model_weights(sd_model=None, info=None):
    from modules import lowvram, devices, sd_hijack
    timer = Timer()

    if shared.sd_model:

        # shared.sd_model.cond_stage_model.to(devices.cpu)
        # shared.sd_model.first_stage_model.to(devices.cpu)
        shared.sd_model.to(devices.cpu)
        sd_hijack.model_hijack.undo_hijack(shared.sd_model)
        shared.sd_model = None
        sd_model = None
        gc.collect()
        devices.torch_gc()
        torch.cuda.empty_cache()

    print(f"Unloaded weights {timer.summary()}.")

    return sd_model