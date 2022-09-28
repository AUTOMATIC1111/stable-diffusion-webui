import glob
import os.path
import sys
from collections import namedtuple
import torch
from omegaconf import OmegaConf


from ldm.util import instantiate_from_config

from modules import shared

CheckpointInfo = namedtuple("CheckpointInfo", ['filename', 'title', 'hash', 'model_name'])
checkpoints_list = {}

try:
    # this silences the annoying "Some weights of the model checkpoint were not used when initializing..." message at start.

    from transformers import logging

    logging.set_verbosity_error()
except Exception:
    pass


def list_models():
    checkpoints_list.clear()

    model_dir = os.path.abspath(shared.cmd_opts.ckpt_dir)

    def modeltitle(path, h):
        abspath = os.path.abspath(path)

        if abspath.startswith(model_dir):
            name = abspath.replace(model_dir, '')
        else:
            name = os.path.basename(path)

        if name.startswith("\\") or name.startswith("/"):
            name = name[1:]

        return f'{name} [{h}]'

    cmd_ckpt = shared.cmd_opts.ckpt
    if os.path.exists(cmd_ckpt):
        h = model_hash(cmd_ckpt)
        title = modeltitle(cmd_ckpt, h)
        model_name = title.rsplit(".",1)[0] # remove extension if present
        checkpoints_list[title] = CheckpointInfo(cmd_ckpt, title, h, model_name)
    elif cmd_ckpt is not None and cmd_ckpt != shared.default_sd_model_file:
        print(f"Checkpoint in --ckpt argument not found: {cmd_ckpt}", file=sys.stderr)

    if os.path.exists(model_dir):
        for filename in glob.glob(model_dir + '/**/*.ckpt', recursive=True):
            h = model_hash(filename)
            title = modeltitle(filename, h)
            model_name = title.rsplit(".",1)[0] # remove extension if present
            checkpoints_list[title] = CheckpointInfo(filename, title, h, model_name)


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
        print(f" - file {os.path.abspath(shared.cmd_opts.ckpt)}", file=sys.stderr)
        print(f" - directory {os.path.abspath(shared.cmd_opts.ckpt_dir)}", file=sys.stderr)
        print(f"Can't run without a checkpoint. Find and place a .ckpt file into any of those locations. The program will exit.", file=sys.stderr)
        exit(1)

    checkpoint_info = next(iter(checkpoints_list.values()))
    if model_checkpoint is not None:
        print(f"Checkpoint {model_checkpoint} not found; loading fallback {checkpoint_info.title}", file=sys.stderr)

    return checkpoint_info


def load_model_weights(model, checkpoint_file, sd_model_hash):
    print(f"Loading weights [{sd_model_hash}] from {checkpoint_file}")

    pl_sd = torch.load(checkpoint_file, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]

    model.load_state_dict(sd, strict=False)

    if shared.cmd_opts.opt_channelslast:
        model.to(memory_format=torch.channels_last)

    if not shared.cmd_opts.no_half:
        model.half()

    model.sd_model_hash = sd_model_hash
    model.sd_model_checkpint = checkpoint_file


def load_model():
    from modules import lowvram, sd_hijack
    checkpoint_info = select_checkpoint()

    sd_config = OmegaConf.load(shared.cmd_opts.config)
    sd_model = instantiate_from_config(sd_config.model)
    load_model_weights(sd_model, checkpoint_info.filename, checkpoint_info.hash)

    if shared.cmd_opts.lowvram or shared.cmd_opts.medvram:
        lowvram.setup_for_low_vram(sd_model, shared.cmd_opts.medvram)
    else:
        sd_model.to(shared.device)

    sd_hijack.model_hijack.hijack(sd_model)

    sd_model.eval()

    print(f"Model loaded.")
    return sd_model


def reload_model_weights(sd_model, info=None):
    from modules import lowvram, devices
    checkpoint_info = info or select_checkpoint()

    if sd_model.sd_model_checkpint == checkpoint_info.filename:
        return

    if shared.cmd_opts.lowvram or shared.cmd_opts.medvram:
        lowvram.send_everything_to_cpu()
    else:
        sd_model.to(devices.cpu)

    load_model_weights(sd_model, checkpoint_info.filename, checkpoint_info.hash)

    if not shared.cmd_opts.lowvram and not shared.cmd_opts.medvram:
        sd_model.to(devices.device)

    print(f"Weights loaded.")
    return sd_model
