# Convert LoRA to different rank approximation (should only be used to go to lower rank)
# This code is based off the extract_lora_from_models.py file which is based on https://github.com/cloneofsimo/lora/blob/develop/lora_diffusion/cli_svd.py
# Thanks to cloneofsimo

import argparse
import math
import os
import torch
from safetensors.torch import load_file, save_file, safe_open
from tqdm import tqdm
from library import train_util, model_util
import numpy as np


def load_state_dict(file_name):
    if model_util.is_safetensors(file_name):
        sd = load_file(file_name)
        with safe_open(file_name, framework="pt") as f:
            metadata = f.metadata()
    else:
        sd = torch.load(file_name, map_location="cpu")
        metadata = None

    return sd, metadata


def save_to_file(file_name, model, metadata):
    if model_util.is_safetensors(file_name):
        save_file(model, file_name, metadata)
    else:
        torch.save(model, file_name)


def split_lora_model(lora_sd, unit):
    max_rank = 0

    # Extract loaded lora dim and alpha
    for key, value in lora_sd.items():
        if "lora_down" in key:
            rank = value.size()[0]
            if rank > max_rank:
                max_rank = rank
    print(f"Max rank: {max_rank}")

    rank = unit
    split_models = []
    new_alpha = None
    while rank < max_rank:
        print(f"Splitting rank {rank}")
        new_sd = {}
        for key, value in lora_sd.items():
            if "lora_down" in key:
                new_sd[key] = value[:rank].contiguous()
            elif "lora_up" in key:
                new_sd[key] = value[:, :rank].contiguous()
            else:
                # なぜかscaleするとおかしくなる……
                # this_rank = lora_sd[key.replace("alpha", "lora_down.weight")].size()[0]
                # scale = math.sqrt(this_rank / rank)  # rank is > unit
                # print(key, value.size(), this_rank, rank, value, scale)
                # new_alpha = value * scale  # always same
                # new_sd[key] = new_alpha
                new_sd[key] = value

        split_models.append((new_sd, rank, new_alpha))
        rank += unit

    return max_rank, split_models


def split(args):
    print("loading Model...")
    lora_sd, metadata = load_state_dict(args.model)

    print("Splitting Model...")
    original_rank, split_models = split_lora_model(lora_sd, args.unit)

    comment = metadata.get("ss_training_comment", "")
    for state_dict, new_rank, new_alpha in split_models:
        # update metadata
        if metadata is None:
            new_metadata = {}
        else:
            new_metadata = metadata.copy()

        new_metadata["ss_training_comment"] = f"split from DyLoRA, rank {original_rank} to {new_rank}; {comment}"
        new_metadata["ss_network_dim"] = str(new_rank)
        # new_metadata["ss_network_alpha"] = str(new_alpha.float().numpy())

        model_hash, legacy_hash = train_util.precalculate_safetensors_hashes(state_dict, metadata)
        metadata["sshs_model_hash"] = model_hash
        metadata["sshs_legacy_hash"] = legacy_hash

        filename, ext = os.path.splitext(args.save_to)
        model_file_name = filename + f"-{new_rank:04d}{ext}"

        print(f"saving model to: {model_file_name}")
        save_to_file(model_file_name, state_dict, new_metadata)


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("--unit", type=int, default=None, help="size of rank to split into / rankを分割するサイズ")
    parser.add_argument(
        "--save_to",
        type=str,
        default=None,
        help="destination base file name: ckpt or safetensors file / 保存先のファイル名のbase、ckptまたはsafetensors",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="DyLoRA model to resize at to new rank: ckpt or safetensors file / 読み込むDyLoRAモデル、ckptまたはsafetensors",
    )

    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    split(args)
