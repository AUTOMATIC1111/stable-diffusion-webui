# Convert LoRA to different rank approximation (should only be used to go to lower rank)
# This code is based off the extract_lora_from_models.py file which is based on https://github.com/cloneofsimo/lora/blob/develop/lora_diffusion/cli_svd.py
# Thanks to cloneofsimo

import argparse
import torch
from safetensors.torch import load_file, save_file, safe_open
from tqdm import tqdm
from library import train_util, model_util
import numpy as np

MIN_SV = 1e-6

# Model save and load functions

def load_state_dict(file_name, dtype):
  if model_util.is_safetensors(file_name):
    sd = load_file(file_name)
    with safe_open(file_name, framework="pt") as f:
      metadata = f.metadata()
  else:
    sd = torch.load(file_name, map_location='cpu')
    metadata = None

  for key in list(sd.keys()):
    if type(sd[key]) == torch.Tensor:
      sd[key] = sd[key].to(dtype)

  return sd, metadata


def save_to_file(file_name, model, state_dict, dtype, metadata):
  if dtype is not None:
    for key in list(state_dict.keys()):
      if type(state_dict[key]) == torch.Tensor:
        state_dict[key] = state_dict[key].to(dtype)

  if model_util.is_safetensors(file_name):
    save_file(model, file_name, metadata)
  else:
    torch.save(model, file_name)


# Indexing functions

def index_sv_cumulative(S, target):
  original_sum = float(torch.sum(S))
  cumulative_sums = torch.cumsum(S, dim=0)/original_sum
  index = int(torch.searchsorted(cumulative_sums, target)) + 1
  index = max(1, min(index, len(S)-1))

  return index


def index_sv_fro(S, target):
  S_squared = S.pow(2)
  s_fro_sq = float(torch.sum(S_squared))
  sum_S_squared = torch.cumsum(S_squared, dim=0)/s_fro_sq
  index = int(torch.searchsorted(sum_S_squared, target**2)) + 1
  index = max(1, min(index, len(S)-1))

  return index


def index_sv_ratio(S, target):
  max_sv = S[0]
  min_sv = max_sv/target
  index = int(torch.sum(S > min_sv).item())
  index = max(1, min(index, len(S)-1))

  return index


# Modified from Kohaku-blueleaf's extract/merge functions
def extract_conv(weight, lora_rank, dynamic_method, dynamic_param, device, scale=1):
    out_size, in_size, kernel_size, _ = weight.size()
    U, S, Vh = torch.linalg.svd(weight.reshape(out_size, -1).to(device))
    
    param_dict = rank_resize(S, lora_rank, dynamic_method, dynamic_param, scale)
    lora_rank = param_dict["new_rank"]

    U = U[:, :lora_rank]
    S = S[:lora_rank]
    U = U @ torch.diag(S)
    Vh = Vh[:lora_rank, :]

    param_dict["lora_down"] = Vh.reshape(lora_rank, in_size, kernel_size, kernel_size).cpu()
    param_dict["lora_up"] = U.reshape(out_size, lora_rank, 1, 1).cpu()
    del U, S, Vh, weight
    return param_dict


def extract_linear(weight, lora_rank, dynamic_method, dynamic_param, device, scale=1):
    out_size, in_size = weight.size()
    
    U, S, Vh = torch.linalg.svd(weight.to(device))
    
    param_dict = rank_resize(S, lora_rank, dynamic_method, dynamic_param, scale)
    lora_rank = param_dict["new_rank"]
    
    U = U[:, :lora_rank]
    S = S[:lora_rank]
    U = U @ torch.diag(S)
    Vh = Vh[:lora_rank, :]
    
    param_dict["lora_down"] = Vh.reshape(lora_rank, in_size).cpu()
    param_dict["lora_up"] = U.reshape(out_size, lora_rank).cpu()
    del U, S, Vh, weight
    return param_dict


def merge_conv(lora_down, lora_up, device):
    in_rank, in_size, kernel_size, k_ = lora_down.shape
    out_size, out_rank, _, _ = lora_up.shape
    assert in_rank == out_rank and kernel_size == k_, f"rank {in_rank} {out_rank} or kernel {kernel_size} {k_} mismatch"
    
    lora_down = lora_down.to(device)
    lora_up = lora_up.to(device)

    merged = lora_up.reshape(out_size, -1) @ lora_down.reshape(in_rank, -1)
    weight = merged.reshape(out_size, in_size, kernel_size, kernel_size)
    del lora_up, lora_down
    return weight


def merge_linear(lora_down, lora_up, device):
    in_rank, in_size = lora_down.shape
    out_size, out_rank = lora_up.shape
    assert in_rank == out_rank, f"rank {in_rank} {out_rank} mismatch"
    
    lora_down = lora_down.to(device)
    lora_up = lora_up.to(device)
    
    weight = lora_up @ lora_down
    del lora_up, lora_down
    return weight
  

# Calculate new rank

def rank_resize(S, rank, dynamic_method, dynamic_param, scale=1):
    param_dict = {}

    if dynamic_method=="sv_ratio":
        # Calculate new dim and alpha based off ratio
        new_rank = index_sv_ratio(S, dynamic_param) + 1
        new_alpha = float(scale*new_rank)

    elif dynamic_method=="sv_cumulative":
        # Calculate new dim and alpha based off cumulative sum
        new_rank = index_sv_cumulative(S, dynamic_param) + 1
        new_alpha = float(scale*new_rank)

    elif dynamic_method=="sv_fro":
        # Calculate new dim and alpha based off sqrt sum of squares
        new_rank = index_sv_fro(S, dynamic_param) + 1
        new_alpha = float(scale*new_rank)
    else:
        new_rank = rank
        new_alpha = float(scale*new_rank)

    
    if S[0] <= MIN_SV: # Zero matrix, set dim to 1
        new_rank = 1
        new_alpha = float(scale*new_rank)
    elif new_rank > rank: # cap max rank at rank
        new_rank = rank
        new_alpha = float(scale*new_rank)


    # Calculate resize info
    s_sum = torch.sum(torch.abs(S))
    s_rank = torch.sum(torch.abs(S[:new_rank]))
    
    S_squared = S.pow(2)
    s_fro = torch.sqrt(torch.sum(S_squared))
    s_red_fro = torch.sqrt(torch.sum(S_squared[:new_rank]))
    fro_percent = float(s_red_fro/s_fro)

    param_dict["new_rank"] = new_rank
    param_dict["new_alpha"] = new_alpha
    param_dict["sum_retained"] = (s_rank)/s_sum
    param_dict["fro_retained"] = fro_percent
    param_dict["max_ratio"] = S[0]/S[new_rank - 1]

    return param_dict


def resize_lora_model(lora_sd, new_rank, save_dtype, device, dynamic_method, dynamic_param, verbose):
  network_alpha = None
  network_dim = None
  verbose_str = "\n"
  fro_list = []

  # Extract loaded lora dim and alpha
  for key, value in lora_sd.items():
    if network_alpha is None and 'alpha' in key:
      network_alpha = value
    if network_dim is None and 'lora_down' in key and len(value.size()) == 2:
      network_dim = value.size()[0]
    if network_alpha is not None and network_dim is not None:
      break
    if network_alpha is None:
      network_alpha = network_dim

  scale = network_alpha/network_dim

  if dynamic_method:
    print(f"Dynamically determining new alphas and dims based off {dynamic_method}: {dynamic_param}, max rank is {new_rank}")

  lora_down_weight = None
  lora_up_weight = None

  o_lora_sd = lora_sd.copy()
  block_down_name = None
  block_up_name = None

  with torch.no_grad():
    for key, value in tqdm(lora_sd.items()):
      weight_name = None
      if 'lora_down' in key:
        block_down_name = key.split(".")[0]
        weight_name = key.split(".")[-1]
        lora_down_weight = value
      else:
        continue

      # find corresponding lora_up and alpha
      block_up_name = block_down_name
      lora_up_weight = lora_sd.get(block_up_name + '.lora_up.' + weight_name, None)
      lora_alpha = lora_sd.get(block_down_name + '.alpha', None)

      weights_loaded = (lora_down_weight is not None and lora_up_weight is not None)

      if weights_loaded:

        conv2d = (len(lora_down_weight.size()) == 4)
        if lora_alpha is None:
          scale = 1.0
        else:
          scale = lora_alpha/lora_down_weight.size()[0]

        if conv2d:
          full_weight_matrix = merge_conv(lora_down_weight, lora_up_weight, device)
          param_dict = extract_conv(full_weight_matrix, new_rank, dynamic_method, dynamic_param, device, scale)
        else:
          full_weight_matrix = merge_linear(lora_down_weight, lora_up_weight, device)
          param_dict = extract_linear(full_weight_matrix, new_rank, dynamic_method, dynamic_param, device, scale)

        if verbose:
          max_ratio = param_dict['max_ratio']
          sum_retained = param_dict['sum_retained']
          fro_retained = param_dict['fro_retained']
          if not np.isnan(fro_retained):
            fro_list.append(float(fro_retained))

          verbose_str+=f"{block_down_name:75} | "
          verbose_str+=f"sum(S) retained: {sum_retained:.1%}, fro retained: {fro_retained:.1%}, max(S) ratio: {max_ratio:0.1f}"

        if verbose and dynamic_method:
          verbose_str+=f", dynamic | dim: {param_dict['new_rank']}, alpha: {param_dict['new_alpha']}\n"
        else:
          verbose_str+=f"\n"

        new_alpha = param_dict['new_alpha']
        o_lora_sd[block_down_name + "." + "lora_down.weight"] = param_dict["lora_down"].to(save_dtype).contiguous()
        o_lora_sd[block_up_name + "." + "lora_up.weight"] = param_dict["lora_up"].to(save_dtype).contiguous()
        o_lora_sd[block_up_name + "." "alpha"] = torch.tensor(param_dict['new_alpha']).to(save_dtype)

        block_down_name = None
        block_up_name = None
        lora_down_weight = None
        lora_up_weight = None
        weights_loaded = False
        del param_dict

  if verbose:
    print(verbose_str)

    print(f"Average Frobenius norm retention: {np.mean(fro_list):.2%} | std: {np.std(fro_list):0.3f}")
  print("resizing complete")
  return o_lora_sd, network_dim, new_alpha


def resize(args):

  def str_to_dtype(p):
    if p == 'float':
      return torch.float
    if p == 'fp16':
      return torch.float16
    if p == 'bf16':
      return torch.bfloat16
    return None

  if args.dynamic_method and not args.dynamic_param:
    raise Exception("If using dynamic_method, then dynamic_param is required")

  merge_dtype = str_to_dtype('float')  # matmul method above only seems to work in float32
  save_dtype = str_to_dtype(args.save_precision)
  if save_dtype is None:
    save_dtype = merge_dtype

  print("loading Model...")
  lora_sd, metadata = load_state_dict(args.model, merge_dtype)

  print("Resizing Lora...")
  state_dict, old_dim, new_alpha = resize_lora_model(lora_sd, args.new_rank, save_dtype, args.device, args.dynamic_method, args.dynamic_param, args.verbose)

  # update metadata
  if metadata is None:
    metadata = {}

  comment = metadata.get("ss_training_comment", "")

  if not args.dynamic_method:
    metadata["ss_training_comment"] = f"dimension is resized from {old_dim} to {args.new_rank}; {comment}"
    metadata["ss_network_dim"] = str(args.new_rank)
    metadata["ss_network_alpha"] = str(new_alpha)
  else:
    metadata["ss_training_comment"] = f"Dynamic resize with {args.dynamic_method}: {args.dynamic_param} from {old_dim}; {comment}"
    metadata["ss_network_dim"] = 'Dynamic'
    metadata["ss_network_alpha"] = 'Dynamic'

  model_hash, legacy_hash = train_util.precalculate_safetensors_hashes(state_dict, metadata)
  metadata["sshs_model_hash"] = model_hash
  metadata["sshs_legacy_hash"] = legacy_hash

  print(f"saving model to: {args.save_to}")
  save_to_file(args.save_to, state_dict, state_dict, save_dtype, metadata)


def setup_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser()

  parser.add_argument("--save_precision", type=str, default=None,
                      choices=[None, "float", "fp16", "bf16"], help="precision in saving, float if omitted / 保存時の精度、未指定時はfloat")
  parser.add_argument("--new_rank", type=int, default=4,
                      help="Specify rank of output LoRA / 出力するLoRAのrank (dim)")
  parser.add_argument("--save_to", type=str, default=None,
                      help="destination file name: ckpt or safetensors file / 保存先のファイル名、ckptまたはsafetensors")
  parser.add_argument("--model", type=str, default=None,
                      help="LoRA model to resize at to new rank: ckpt or safetensors file / 読み込むLoRAモデル、ckptまたはsafetensors")
  parser.add_argument("--device", type=str, default=None, help="device to use, cuda for GPU / 計算を行うデバイス、cuda でGPUを使う")
  parser.add_argument("--verbose", action="store_true", 
                      help="Display verbose resizing information / rank変更時の詳細情報を出力する")
  parser.add_argument("--dynamic_method", type=str, default=None, choices=[None, "sv_ratio", "sv_fro", "sv_cumulative"],
                      help="Specify dynamic resizing method, --new_rank is used as a hard limit for max rank")
  parser.add_argument("--dynamic_param", type=float, default=None,
                      help="Specify target for dynamic reduction")
       
  return parser


if __name__ == '__main__':
  parser = setup_parser()

  args = parser.parse_args()
  resize(args)
