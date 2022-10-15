import collections
import os.path
import sys
from collections import namedtuple
import torch
from omegaconf import OmegaConf

from ldm.util import instantiate_from_config

from modules import shared, modelloader, devices
from modules.paths import models_path

model_dir = "Stable-diffusion"
model_path = os.path.abspath(os.path.join(models_path, model_dir))

CheckpointInfo = namedtuple("CheckpointInfo", ['filename', 'title', 'hash', 'model_name', 'config'])
checkpoints_list = {}
checkpoints_loaded = collections.OrderedDict()

try:
    # this silences the annoying "Some weights of the model checkpoint were not used when initializing..." message at start.

    from transformers import logging

    logging.set_verbosity_error()
except Exception:
    pass


def setup_model():
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    list_models()


def checkpoint_tiles():
    return sorted([x.title for x in checkpoints_list.values()])


def list_models():
    checkpoints_list.clear()
    model_list = modelloader.load_models(model_path=model_path, command_path=shared.cmd_opts.ckpt_dir, ext_filter=[".ckpt"])

    def modeltitle(path, shorthash):
        abspath = os.path.abspath(path)

        if shared.cmd_opts.ckpt_dir is not None and abspath.startswith(shared.cmd_opts.ckpt_dir):
            name = abspath.replace(shared.cmd_opts.ckpt_dir, '')
        elif abspath.startswith(model_path):
            name = abspath.replace(model_path, '')
        else:
            name = os.path.basename(path)

        if name.startswith("\\") or name.startswith("/"):
            name = name[1:]

        shortname = os.path.splitext(name.replace("/", "_").replace("\\", "_"))[0]

        return f'{name} [{shorthash}]', shortname

    cmd_ckpt = shared.cmd_opts.ckpt
    if os.path.exists(cmd_ckpt):
        h = model_hash(cmd_ckpt)
        title, short_model_name = modeltitle(cmd_ckpt, h)
        checkpoints_list[title] = CheckpointInfo(cmd_ckpt, title, h, short_model_name, shared.cmd_opts.config)
        shared.opts.data['sd_model_checkpoint'] = title
    elif cmd_ckpt is not None and cmd_ckpt != shared.default_sd_model_file:
        print(f"Checkpoint in --ckpt argument not found (Possible it was moved to {model_path}: {cmd_ckpt}", file=sys.stderr)
    for filename in model_list:
        h = model_hash(filename)
        title, short_model_name = modeltitle(filename, h)

        basename, _ = os.path.splitext(filename)
        config = basename + ".yaml"
        if not os.path.exists(config):
            config = shared.cmd_opts.config

        checkpoints_list[title] = CheckpointInfo(filename, title, h, short_model_name, config)


def get_closet_checkpoint_match(searchString):
    applicable = sorted([info for info in checkpoints_list.values() if searchString in info.title], key = lambda x:len(x.title))
    if len(applicable) > 0:
        return applicable[0]
    return None


def model_hash(filename):
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
    checkpoint_info = checkpoints_list.get(model_checkpoint, None)
    if checkpoint_info is not None:
        return checkpoint_info

    if len(checkpoints_list) == 0:
        print(f"No checkpoints found. When searching for checkpoints, looked at:", file=sys.stderr)
        if shared.cmd_opts.ckpt is not None:
            print(f" - file {os.path.abspath(shared.cmd_opts.ckpt)}", file=sys.stderr)
        print(f" - directory {model_path}", file=sys.stderr)
        if shared.cmd_opts.ckpt_dir is not None:
            print(f" - directory {os.path.abspath(shared.cmd_opts.ckpt_dir)}", file=sys.stderr)
        print(f"Can't run without a checkpoint. Find and place a .ckpt file into any of those locations. The program will exit.", file=sys.stderr)
        exit(1)

    checkpoint_info = next(iter(checkpoints_list.values()))
    if model_checkpoint is not None:
        print(f"Checkpoint {model_checkpoint} not found; loading fallback {checkpoint_info.title}", file=sys.stderr)

    return checkpoint_info


def get_state_dict_from_checkpoint(pl_sd):
    if "state_dict" in pl_sd:
        return pl_sd["state_dict"]

    return pl_sd


def load_model_weights(model, checkpoint_info):
    checkpoint_file = checkpoint_info.filename
    sd_model_hash = checkpoint_info.hash

    if checkpoint_info not in checkpoints_loaded:
        print(f"Loading weights [{sd_model_hash}] from {checkpoint_file}")

        pl_sd = torch.load(checkpoint_file, map_location=shared.weight_load_location)
        if "global_step" in pl_sd:
            print(f"Global Step: {pl_sd['global_step']}")

        sd = get_state_dict_from_checkpoint(pl_sd)
        model.load_state_dict(sd, strict=False)

        if shared.cmd_opts.opt_channelslast:
            model.to(memory_format=torch.channels_last)

        if not shared.cmd_opts.no_half:
            model.half()

        devices.dtype = torch.float32 if shared.cmd_opts.no_half else torch.float16
        devices.dtype_vae = torch.float32 if shared.cmd_opts.no_half or shared.cmd_opts.no_half_vae else torch.float16

        vae_file = os.path.splitext(checkpoint_file)[0] + ".vae.pt"

        if not os.path.exists(vae_file) and shared.cmd_opts.vae_path is not None:
            vae_file = shared.cmd_opts.vae_path

        if os.path.exists(vae_file):
            print(f"Loading VAE weights from: {vae_file}")
            vae_ckpt = torch.load(vae_file, map_location=shared.weight_load_location)
            vae_dict = {k: v for k, v in vae_ckpt["state_dict"].items() if k[0:4] != "loss"}
            model.first_stage_model.load_state_dict(vae_dict)

        model.first_stage_model.to(devices.dtype_vae)

        checkpoints_loaded[checkpoint_info] = model.state_dict().copy()
        while len(checkpoints_loaded) > shared.opts.sd_checkpoint_cache:
            checkpoints_loaded.popitem(last=False)  # LRU
    else:
        print(f"Loading weights [{sd_model_hash}] from cache")
        checkpoints_loaded.move_to_end(checkpoint_info)
        model.load_state_dict(checkpoints_loaded[checkpoint_info])

    model.sd_model_hash = sd_model_hash
    model.sd_model_checkpoint = checkpoint_file
    model.sd_checkpoint_info = checkpoint_info


def load_model():
    from modules import lowvram, sd_hijack
    checkpoint_info = select_checkpoint()

    if checkpoint_info.config != shared.cmd_opts.config:
        print(f"Loading config from: {checkpoint_info.config}")

    sd_config = OmegaConf.load(checkpoint_info.config)
    sd_model = instantiate_from_config(sd_config.model)
    load_model_weights(sd_model, checkpoint_info)

    if shared.cmd_opts.lowvram or shared.cmd_opts.medvram:
        lowvram.setup_for_low_vram(sd_model, shared.cmd_opts.medvram)
    else:
        sd_model.to(shared.device)

    sd_hijack.model_hijack.hijack(sd_model)

    sd_model.eval()

    print(f"Model loaded.")
    return sd_model


def reload_model_weights(sd_model, info=None):
    from modules import lowvram, devices, sd_hijack
    checkpoint_info = info or select_checkpoint()

    if sd_model.sd_model_checkpoint == checkpoint_info.filename:
        return

    if sd_model.sd_checkpoint_info.config != checkpoint_info.config:
        checkpoints_loaded.clear()
        shared.sd_model = load_model()
        return shared.sd_model

    if shared.cmd_opts.lowvram or shared.cmd_opts.medvram:
        lowvram.send_everything_to_cpu()
    else:
        sd_model.to(devices.cpu)

    sd_hijack.model_hijack.undo_hijack(sd_model)

    load_model_weights(sd_model, checkpoint_info)

    sd_hijack.model_hijack.hijack(sd_model)

    if not shared.cmd_opts.lowvram and not shared.cmd_opts.medvram:
        sd_model.to(devices.device)

    print(f"Weights loaded.")
    return sd_model
