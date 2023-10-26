import math
import argparse
import os
import time
import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm
from sd_scripts.library import sai_model_spec, train_util
import sd_scripts.library.model_util as model_util
import lora


CLAMP_QUANTILE = 0.99


def load_state_dict(file_name, dtype):
    if os.path.splitext(file_name)[1] == ".safetensors":
        sd = load_file(file_name)
        metadata = train_util.load_metadata_from_safetensors(file_name)
    else:
        sd = torch.load(file_name, map_location="cpu")
        metadata = {}

    for key in list(sd.keys()):
        if type(sd[key]) == torch.Tensor:
            sd[key] = sd[key].to(dtype)

    return sd, metadata


def save_to_file(file_name, state_dict, dtype, metadata):
    if dtype is not None:
        for key in list(state_dict.keys()):
            if type(state_dict[key]) == torch.Tensor:
                state_dict[key] = state_dict[key].to(dtype)

    if os.path.splitext(file_name)[1] == ".safetensors":
        save_file(state_dict, file_name, metadata=metadata)
    else:
        torch.save(state_dict, file_name)


def merge_lora_models(models, ratios, new_rank, new_conv_rank, device, merge_dtype):
    print(f"new rank: {new_rank}, new conv rank: {new_conv_rank}")
    merged_sd = {}
    v2 = None
    base_model = None
    for model, ratio in zip(models, ratios):
        print(f"loading: {model}")
        lora_sd, lora_metadata = load_state_dict(model, merge_dtype)

        if lora_metadata is not None:
            if v2 is None:
                v2 = lora_metadata.get(train_util.SS_METADATA_KEY_V2, None)  # return string
            if base_model is None:
                base_model = lora_metadata.get(train_util.SS_METADATA_KEY_BASE_MODEL_VERSION, None)

        # merge
        print(f"merging...")
        for key in tqdm(list(lora_sd.keys())):
            if "lora_down" not in key:
                continue

            lora_module_name = key[: key.rfind(".lora_down")]

            down_weight = lora_sd[key]
            network_dim = down_weight.size()[0]

            up_weight = lora_sd[lora_module_name + ".lora_up.weight"]
            alpha = lora_sd.get(lora_module_name + ".alpha", network_dim)

            in_dim = down_weight.size()[1]
            out_dim = up_weight.size()[0]
            conv2d = len(down_weight.size()) == 4
            kernel_size = None if not conv2d else down_weight.size()[2:4]
            # print(lora_module_name, network_dim, alpha, in_dim, out_dim, kernel_size)

            # make original weight if not exist
            if lora_module_name not in merged_sd:
                weight = torch.zeros((out_dim, in_dim, *kernel_size) if conv2d else (out_dim, in_dim), dtype=merge_dtype)
                if device:
                    weight = weight.to(device)
            else:
                weight = merged_sd[lora_module_name]

            # merge to weight
            if device:
                up_weight = up_weight.to(device)
                down_weight = down_weight.to(device)

            # W <- W + U * D
            scale = alpha / network_dim

            if device:  # and isinstance(scale, torch.Tensor):
                scale = scale.to(device)

            if not conv2d:  # linear
                weight = weight + ratio * (up_weight @ down_weight) * scale
            elif kernel_size == (1, 1):
                weight = (
                    weight
                    + ratio
                    * (up_weight.squeeze(3).squeeze(2) @ down_weight.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
                    * scale
                )
            else:
                conved = torch.nn.functional.conv2d(down_weight.permute(1, 0, 2, 3), up_weight).permute(1, 0, 2, 3)
                weight = weight + ratio * conved * scale

            merged_sd[lora_module_name] = weight

    # extract from merged weights
    print("extract new lora...")
    merged_lora_sd = {}
    with torch.no_grad():
        for lora_module_name, mat in tqdm(list(merged_sd.items())):
            conv2d = len(mat.size()) == 4
            kernel_size = None if not conv2d else mat.size()[2:4]
            conv2d_3x3 = conv2d and kernel_size != (1, 1)
            out_dim, in_dim = mat.size()[0:2]

            if conv2d:
                if conv2d_3x3:
                    mat = mat.flatten(start_dim=1)
                else:
                    mat = mat.squeeze()

            module_new_rank = new_conv_rank if conv2d_3x3 else new_rank
            module_new_rank = min(module_new_rank, in_dim, out_dim)  # LoRA rank cannot exceed the original dim

            U, S, Vh = torch.linalg.svd(mat)

            U = U[:, :module_new_rank]
            S = S[:module_new_rank]
            U = U @ torch.diag(S)

            Vh = Vh[:module_new_rank, :]

            dist = torch.cat([U.flatten(), Vh.flatten()])
            hi_val = torch.quantile(dist, CLAMP_QUANTILE)
            low_val = -hi_val

            U = U.clamp(low_val, hi_val)
            Vh = Vh.clamp(low_val, hi_val)

            if conv2d:
                U = U.reshape(out_dim, module_new_rank, 1, 1)
                Vh = Vh.reshape(module_new_rank, in_dim, kernel_size[0], kernel_size[1])

            up_weight = U
            down_weight = Vh

            merged_lora_sd[lora_module_name + ".lora_up.weight"] = up_weight.to("cpu").contiguous()
            merged_lora_sd[lora_module_name + ".lora_down.weight"] = down_weight.to("cpu").contiguous()
            merged_lora_sd[lora_module_name + ".alpha"] = torch.tensor(module_new_rank)

    # build minimum metadata
    dims = f"{new_rank}"
    alphas = f"{new_rank}"
    if new_conv_rank is not None:
        network_args = {"conv_dim": new_conv_rank, "conv_alpha": new_conv_rank}
    else:
        network_args = None
    metadata = train_util.build_minimum_network_metadata(v2, base_model, "networks.lora", dims, alphas, network_args)

    return merged_lora_sd, metadata, v2 == "True", base_model


def merge(args):
    assert len(args.models) == len(args.ratios), f"number of models must be equal to number of ratios / モデルの数と重みの数は合わせてください"

    def str_to_dtype(p):
        if p == "float":
            return torch.float
        if p == "fp16":
            return torch.float16
        if p == "bf16":
            return torch.bfloat16
        return None

    merge_dtype = str_to_dtype(args.precision)
    save_dtype = str_to_dtype(args.save_precision)
    if save_dtype is None:
        save_dtype = merge_dtype

    new_conv_rank = args.new_conv_rank if args.new_conv_rank is not None else args.new_rank
    state_dict, metadata, v2, base_model = merge_lora_models(
        args.models, args.ratios, args.new_rank, new_conv_rank, args.device, merge_dtype
    )

    print(f"calculating hashes and creating metadata...")

    model_hash, legacy_hash = train_util.precalculate_safetensors_hashes(state_dict, metadata)
    metadata["sshs_model_hash"] = model_hash
    metadata["sshs_legacy_hash"] = legacy_hash

    if not args.no_metadata:
        is_sdxl = base_model is not None and base_model.lower().startswith("sdxl")
        merged_from = sai_model_spec.build_merged_from(args.models)
        title = os.path.splitext(os.path.basename(args.save_to))[0]
        sai_metadata = sai_model_spec.build_metadata(
            state_dict, v2, v2, is_sdxl, True, False, time.time(), title=title, merged_from=merged_from
        )
        if v2:
            # TODO read sai modelspec
            print(
                "Cannot determine if LoRA is for v-prediction, so save metadata as v-prediction / LoRAがv-prediction用か否か不明なため、仮にv-prediction用としてmetadataを保存します"
            )
        metadata.update(sai_metadata)

    print(f"saving model to: {args.save_to}")
    save_to_file(args.save_to, state_dict, save_dtype, metadata)


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_precision",
        type=str,
        default=None,
        choices=[None, "float", "fp16", "bf16"],
        help="precision in saving, same to merging if omitted / 保存時に精度を変更して保存する、省略時はマージ時の精度と同じ",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="float",
        choices=["float", "fp16", "bf16"],
        help="precision in merging (float is recommended) / マージの計算時の精度（floatを推奨）",
    )
    parser.add_argument(
        "--save_to", type=str, default=None, help="destination file name: ckpt or safetensors file / 保存先のファイル名、ckptまたはsafetensors"
    )
    parser.add_argument(
        "--models", type=str, nargs="*", help="LoRA models to merge: ckpt or safetensors file / マージするLoRAモデル、ckptまたはsafetensors"
    )
    parser.add_argument("--ratios", type=float, nargs="*", help="ratios for each model / それぞれのLoRAモデルの比率")
    parser.add_argument("--new_rank", type=int, default=4, help="Specify rank of output LoRA / 出力するLoRAのrank (dim)")
    parser.add_argument(
        "--new_conv_rank",
        type=int,
        default=None,
        help="Specify rank of output LoRA for Conv2d 3x3, None for same as new_rank / 出力するConv2D 3x3 LoRAのrank (dim)、Noneでnew_rankと同じ",
    )
    parser.add_argument("--device", type=str, default=None, help="device to use, cuda for GPU / 計算を行うデバイス、cuda でGPUを使う")
    parser.add_argument(
        "--no_metadata",
        action="store_true",
        help="do not save sai modelspec metadata (minimum ss_metadata for LoRA is saved) / "
        + "sai modelspecのメタデータを保存しない（LoRAの最低限のss_metadataは保存される）",
    )

    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    merge(args)
