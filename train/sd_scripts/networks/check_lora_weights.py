import argparse
import os
import torch
from safetensors.torch import load_file


def main(file):
  print(f"loading: {file}")
  if os.path.splitext(file)[1] == '.safetensors':
    sd = load_file(file)
  else:
    sd = torch.load(file, map_location='cpu')

  values = []

  keys = list(sd.keys())
  for key in keys:
    if 'lora_up' in key or 'lora_down' in key:
      values.append((key, sd[key]))
  print(f"number of LoRA modules: {len(values)}")

  for key, value in values:
    value = value.to(torch.float32)
    print(f"{key},{str(tuple(value.size())).replace(', ', '-')},{torch.mean(torch.abs(value))},{torch.min(torch.abs(value))}")


def setup_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser()
  parser.add_argument("file", type=str, help="model file to check / 重みを確認するモデルファイル")

  return parser


if __name__ == '__main__':
  parser = setup_parser()

  args = parser.parse_args()

  main(args.file)
