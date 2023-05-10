
import math
import argparse
import os
import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm
import library.model_util as model_util
import lora


CLAMP_QUANTILE = 0.99


def load_state_dict(file_name, dtype):
  if os.path.splitext(file_name)[1] == '.safetensors':
    sd = load_file(file_name)
  else:
    sd = torch.load(file_name, map_location='cpu')
  for key in list(sd.keys()):
    if type(sd[key]) == torch.Tensor:
      sd[key] = sd[key].to(dtype)
  return sd


def save_to_file(file_name, state_dict, dtype):
  if dtype is not None:
    for key in list(state_dict.keys()):
      if type(state_dict[key]) == torch.Tensor:
        state_dict[key] = state_dict[key].to(dtype)

  if os.path.splitext(file_name)[1] == '.safetensors':
    save_file(state_dict, file_name)
  else:
    torch.save(state_dict, file_name)


def merge_lora_models(models, ratios, new_rank, new_conv_rank, device, merge_dtype):
  print(f"new rank: {new_rank}, new conv rank: {new_conv_rank}")
  merged_sd = {}
  for model, ratio in zip(models, ratios):
    print(f"loading: {model}")
    lora_sd = load_state_dict(model, merge_dtype)

    # merge
    print(f"merging...")
    for key in tqdm(list(lora_sd.keys())):
      if 'lora_down' not in key:
        continue

      lora_module_name = key[:key.rfind(".lora_down")]

      down_weight = lora_sd[key]
      network_dim = down_weight.size()[0]

      up_weight = lora_sd[lora_module_name + '.lora_up.weight']
      alpha = lora_sd.get(lora_module_name + '.alpha', network_dim)

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
      scale = (alpha / network_dim)

      if device:                      # and isinstance(scale, torch.Tensor):
        scale = scale.to(device)

      if not conv2d:        # linear
        weight = weight + ratio * (up_weight @ down_weight) * scale
      elif kernel_size == (1, 1):
        weight = weight + ratio * (up_weight.squeeze(3).squeeze(2) @ down_weight.squeeze(3).squeeze(2)
                                   ).unsqueeze(2).unsqueeze(3) * scale
      else:
        conved = torch.nn.functional.conv2d(down_weight.permute(1, 0, 2, 3), up_weight).permute(1, 0, 2, 3)
        weight = weight + ratio * conved * scale

      merged_sd[lora_module_name] = weight

  # extract from merged weights
  print("extract new lora...")
  merged_lora_sd = {}
  with torch.no_grad():
    for lora_module_name, mat in tqdm(list(merged_sd.items())):
      conv2d = (len(mat.size()) == 4)
      kernel_size = None if not conv2d else mat.size()[2:4]
      conv2d_3x3 = conv2d and kernel_size != (1, 1)
      out_dim, in_dim = mat.size()[0:2]

      if conv2d:
        if conv2d_3x3:
          mat = mat.flatten(start_dim=1)
        else:
          mat = mat.squeeze()

      module_new_rank = new_conv_rank if conv2d_3x3 else new_rank
      module_new_rank = min(module_new_rank, in_dim, out_dim)                           # LoRA rank cannot exceed the original dim

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

      merged_lora_sd[lora_module_name + '.lora_up.weight'] = up_weight.to("cpu").contiguous()
      merged_lora_sd[lora_module_name + '.lora_down.weight'] = down_weight.to("cpu").contiguous()
      merged_lora_sd[lora_module_name + '.alpha'] = torch.tensor(module_new_rank)

  return merged_lora_sd


def merge(args):
  assert len(args.models) == len(args.ratios), f"number of models must be equal to number of ratios / モデルの数と重みの数は合わせてください"

  def str_to_dtype(p):
    if p == 'float':
      return torch.float
    if p == 'fp16':
      return torch.float16
    if p == 'bf16':
      return torch.bfloat16
    return None

  merge_dtype = str_to_dtype(args.precision)
  save_dtype = str_to_dtype(args.save_precision)
  if save_dtype is None:
    save_dtype = merge_dtype

  new_conv_rank = args.new_conv_rank if args.new_conv_rank is not None else args.new_rank
  state_dict = merge_lora_models(args.models, args.ratios, args.new_rank, new_conv_rank, args.device, merge_dtype)

  print(f"saving model to: {args.save_to}")
  save_to_file(args.save_to, state_dict, save_dtype)


def setup_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser()
  parser.add_argument("--save_precision", type=str, default=None,
                      choices=[None, "float", "fp16", "bf16"], help="precision in saving, same to merging if omitted / 保存時に精度を変更して保存する、省略時はマージ時の精度と同じ")
  parser.add_argument("--precision", type=str, default="float",
                      choices=["float", "fp16", "bf16"], help="precision in merging (float is recommended) / マージの計算時の精度（floatを推奨）")
  parser.add_argument("--save_to", type=str, default=None,
                      help="destination file name: ckpt or safetensors file / 保存先のファイル名、ckptまたはsafetensors")
  parser.add_argument("--models", type=str, nargs='*',
                      help="LoRA models to merge: ckpt or safetensors file / マージするLoRAモデル、ckptまたはsafetensors")
  parser.add_argument("--ratios", type=float, nargs='*',
                      help="ratios for each model / それぞれのLoRAモデルの比率")
  parser.add_argument("--new_rank", type=int, default=4,
                      help="Specify rank of output LoRA / 出力するLoRAのrank (dim)")
  parser.add_argument("--new_conv_rank", type=int, default=None,
                      help="Specify rank of output LoRA for Conv2d 3x3, None for same as new_rank / 出力するConv2D 3x3 LoRAのrank (dim)、Noneでnew_rankと同じ")
  parser.add_argument("--device", type=str, default=None, help="device to use, cuda for GPU / 計算を行うデバイス、cuda でGPUを使う")

  return parser


if __name__ == '__main__':
  parser = setup_parser()

  args = parser.parse_args()
  merge(args)
