from typing import List, NamedTuple, Any
import numpy as np
import cv2
import torch
from safetensors.torch import load_file

from diffusers import UNet2DConditionModel
from diffusers.models.unet_2d_condition import UNet2DConditionOutput

import library.model_util as model_util


class ControlNetInfo(NamedTuple):
  unet: Any
  net: Any
  prep: Any
  weight: float
  ratio: float


class ControlNet(torch.nn.Module):
  def __init__(self) -> None:
    super().__init__()

    # make control model
    self.control_model = torch.nn.Module()

    dims = [320, 320, 320, 320, 640, 640, 640, 1280, 1280, 1280, 1280, 1280]
    zero_convs = torch.nn.ModuleList()
    for i, dim in enumerate(dims):
      sub_list = torch.nn.ModuleList([torch.nn.Conv2d(dim, dim, 1)])
      zero_convs.append(sub_list)
    self.control_model.add_module("zero_convs", zero_convs)

    middle_block_out = torch.nn.Conv2d(1280, 1280, 1)
    self.control_model.add_module("middle_block_out", torch.nn.ModuleList([middle_block_out]))

    dims = [16, 16, 32, 32, 96, 96, 256, 320]
    strides = [1, 1, 2, 1, 2, 1, 2, 1]
    prev_dim = 3
    input_hint_block = torch.nn.Sequential()
    for i, (dim, stride) in enumerate(zip(dims, strides)):
      input_hint_block.append(torch.nn.Conv2d(prev_dim, dim, 3, stride, 1))
      if i < len(dims) - 1:
        input_hint_block.append(torch.nn.SiLU())
      prev_dim = dim
    self.control_model.add_module("input_hint_block", input_hint_block)


def load_control_net(v2, unet, model):
  device = unet.device

  # control sdからキー変換しつつU-Netに対応する部分のみ取り出し、DiffusersのU-Netに読み込む
  # state dictを読み込む
  print(f"ControlNet: loading control SD model : {model}")

  if model_util.is_safetensors(model):
    ctrl_sd_sd = load_file(model)
  else:
    ctrl_sd_sd = torch.load(model, map_location='cpu')
    ctrl_sd_sd = ctrl_sd_sd.pop("state_dict", ctrl_sd_sd)

  # 重みをU-Netに読み込めるようにする。ControlNetはSD版のstate dictなので、それを読み込む
  is_difference = "difference" in ctrl_sd_sd
  print("ControlNet: loading difference")

  # ControlNetには存在しないキーがあるので、まず現在のU-NetでSD版の全keyを作っておく
  # またTransfer Controlの元weightとなる
  ctrl_unet_sd_sd = model_util.convert_unet_state_dict_to_sd(v2, unet.state_dict())

  # 元のU-Netに影響しないようにコピーする。またprefixが付いていないので付ける
  for key in list(ctrl_unet_sd_sd.keys()):
    ctrl_unet_sd_sd["model.diffusion_model." + key] = ctrl_unet_sd_sd.pop(key).clone()

  zero_conv_sd = {}
  for key in list(ctrl_sd_sd.keys()):
    if key.startswith("control_"):
      unet_key = "model.diffusion_" + key[len("control_"):]
      if unet_key not in ctrl_unet_sd_sd:               # zero conv
        zero_conv_sd[key] = ctrl_sd_sd[key]
        continue
      if is_difference:                                 # Transfer Control
        ctrl_unet_sd_sd[unet_key] += ctrl_sd_sd[key].to(device, dtype=unet.dtype)
      else:
        ctrl_unet_sd_sd[unet_key] = ctrl_sd_sd[key].to(device, dtype=unet.dtype)

  unet_config = model_util.create_unet_diffusers_config(v2)
  ctrl_unet_du_sd = model_util.convert_ldm_unet_checkpoint(v2, ctrl_unet_sd_sd, unet_config)    # DiffUsers版ControlNetのstate dict

  # ControlNetのU-Netを作成する
  ctrl_unet = UNet2DConditionModel(**unet_config)
  info = ctrl_unet.load_state_dict(ctrl_unet_du_sd)
  print("ControlNet: loading Control U-Net:", info)

  # U-Net以外のControlNetを作成する
  # TODO support middle only
  ctrl_net = ControlNet()
  info = ctrl_net.load_state_dict(zero_conv_sd)
  print("ControlNet: loading ControlNet:", info)

  ctrl_unet.to(unet.device, dtype=unet.dtype)
  ctrl_net.to(unet.device, dtype=unet.dtype)
  return ctrl_unet, ctrl_net


def load_preprocess(prep_type: str):
  if prep_type is None or prep_type.lower() == "none":
    return None

  if prep_type.startswith("canny"):
    args = prep_type.split("_")
    th1 = int(args[1]) if len(args) >= 2 else 63
    th2 = int(args[2]) if len(args) >= 3 else 191

    def canny(img):
      img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
      return cv2.Canny(img, th1, th2)
    return canny

  print("Unsupported prep type:", prep_type)
  return None


def preprocess_ctrl_net_hint_image(image):
  image = np.array(image).astype(np.float32) / 255.0
  image = image[:, :, ::-1].copy()                         # rgb to bgr
  image = image[None].transpose(0, 3, 1, 2)       # nchw
  image = torch.from_numpy(image)
  return image                              # 0 to 1


def get_guided_hints(control_nets: List[ControlNetInfo], num_latent_input, b_size, hints):
  guided_hints = []
  for i, cnet_info in enumerate(control_nets):
    # hintは 1枚目の画像のcnet1, 1枚目の画像のcnet2, 1枚目の画像のcnet3, 2枚目の画像のcnet1, 2枚目の画像のcnet2 ... と並んでいること
    b_hints = []
    if len(hints) == 1:           # すべて同じ画像をhintとして使う
      hint = hints[0]
      if cnet_info.prep is not None:
        hint = cnet_info.prep(hint)
      hint = preprocess_ctrl_net_hint_image(hint)
      b_hints = [hint for _ in range(b_size)]
    else:
      for bi in range(b_size):
        hint = hints[(bi * len(control_nets) + i) % len(hints)]
        if cnet_info.prep is not None:
          hint = cnet_info.prep(hint)
        hint = preprocess_ctrl_net_hint_image(hint)
        b_hints.append(hint)
    b_hints = torch.cat(b_hints, dim=0)
    b_hints = b_hints.to(cnet_info.unet.device, dtype=cnet_info.unet.dtype)

    guided_hint = cnet_info.net.control_model.input_hint_block(b_hints)
    guided_hints.append(guided_hint)
  return guided_hints


def call_unet_and_control_net(step, num_latent_input, original_unet, control_nets: List[ControlNetInfo], guided_hints, current_ratio, sample, timestep, encoder_hidden_states):
  # ControlNet
  # 複数のControlNetの場合は、出力をマージするのではなく交互に適用する
  cnet_cnt = len(control_nets)
  cnet_idx = step % cnet_cnt
  cnet_info = control_nets[cnet_idx]

  # print(current_ratio, cnet_info.prep, cnet_info.weight, cnet_info.ratio)
  if cnet_info.ratio < current_ratio:
    return original_unet(sample, timestep, encoder_hidden_states)

  guided_hint = guided_hints[cnet_idx]
  guided_hint = guided_hint.repeat((num_latent_input, 1, 1, 1))
  outs = unet_forward(True, cnet_info.net, cnet_info.unet, guided_hint, None, sample, timestep, encoder_hidden_states)
  outs = [o * cnet_info.weight for o in outs]

  # U-Net
  return unet_forward(False, cnet_info.net, original_unet, None, outs, sample, timestep, encoder_hidden_states)


"""
  # これはmergeのバージョン
  # ControlNet
  cnet_outs_list = []
  for i, cnet_info in enumerate(control_nets):
    # print(current_ratio, cnet_info.prep, cnet_info.weight, cnet_info.ratio)
    if cnet_info.ratio < current_ratio:
      continue
    guided_hint = guided_hints[i]
    outs = unet_forward(True, cnet_info.net, cnet_info.unet, guided_hint, None, sample, timestep, encoder_hidden_states)
    for i in range(len(outs)):
      outs[i] *= cnet_info.weight

    cnet_outs_list.append(outs)

  count = len(cnet_outs_list)
  if count == 0:
    return original_unet(sample, timestep, encoder_hidden_states)

  # sum of controlnets
  for i in range(1, count):
    cnet_outs_list[0] += cnet_outs_list[i]

  # U-Net
  return unet_forward(False, cnet_info.net, original_unet, None, cnet_outs_list[0], sample, timestep, encoder_hidden_states)
"""


def unet_forward(is_control_net, control_net: ControlNet, unet: UNet2DConditionModel, guided_hint, ctrl_outs, sample, timestep, encoder_hidden_states):
  # copy from UNet2DConditionModel
  default_overall_up_factor = 2**unet.num_upsamplers

  forward_upsample_size = False
  upsample_size = None

  if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
    print("Forward upsample size to force interpolation output size.")
    forward_upsample_size = True

  # 0. center input if necessary
  if unet.config.center_input_sample:
    sample = 2 * sample - 1.0

  # 1. time
  timesteps = timestep
  if not torch.is_tensor(timesteps):
    # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
    # This would be a good case for the `match` statement (Python 3.10+)
    is_mps = sample.device.type == "mps"
    if isinstance(timestep, float):
      dtype = torch.float32 if is_mps else torch.float64
    else:
      dtype = torch.int32 if is_mps else torch.int64
    timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
  elif len(timesteps.shape) == 0:
    timesteps = timesteps[None].to(sample.device)

  # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
  timesteps = timesteps.expand(sample.shape[0])

  t_emb = unet.time_proj(timesteps)

  # timesteps does not contain any weights and will always return f32 tensors
  # but time_embedding might actually be running in fp16. so we need to cast here.
  # there might be better ways to encapsulate this.
  t_emb = t_emb.to(dtype=unet.dtype)
  emb = unet.time_embedding(t_emb)

  outs = []                     # output of ControlNet
  zc_idx = 0

  # 2. pre-process
  sample = unet.conv_in(sample)
  if is_control_net:
    sample += guided_hint
    outs.append(control_net.control_model.zero_convs[zc_idx][0](sample))  # , emb, encoder_hidden_states))
    zc_idx += 1

  # 3. down
  down_block_res_samples = (sample,)
  for downsample_block in unet.down_blocks:
    if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
      sample, res_samples = downsample_block(
          hidden_states=sample,
          temb=emb,
          encoder_hidden_states=encoder_hidden_states,
      )
    else:
      sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
    if is_control_net:
      for rs in res_samples:
        outs.append(control_net.control_model.zero_convs[zc_idx][0](rs))  # , emb, encoder_hidden_states))
        zc_idx += 1

    down_block_res_samples += res_samples

  # 4. mid
  sample = unet.mid_block(sample, emb, encoder_hidden_states=encoder_hidden_states)
  if is_control_net:
    outs.append(control_net.control_model.middle_block_out[0](sample))
    return outs

  if not is_control_net:
    sample += ctrl_outs.pop()

  # 5. up
  for i, upsample_block in enumerate(unet.up_blocks):
    is_final_block = i == len(unet.up_blocks) - 1

    res_samples = down_block_res_samples[-len(upsample_block.resnets):]
    down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

    if not is_control_net and len(ctrl_outs) > 0:
      res_samples = list(res_samples)
      apply_ctrl_outs = ctrl_outs[-len(res_samples):]
      ctrl_outs = ctrl_outs[:-len(res_samples)]
      for j in range(len(res_samples)):
        res_samples[j] = res_samples[j] + apply_ctrl_outs[j]
      res_samples = tuple(res_samples)

    # if we have not reached the final block and need to forward the
    # upsample size, we do it here
    if not is_final_block and forward_upsample_size:
      upsample_size = down_block_res_samples[-1].shape[2:]

    if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
      sample = upsample_block(
          hidden_states=sample,
          temb=emb,
          res_hidden_states_tuple=res_samples,
          encoder_hidden_states=encoder_hidden_states,
          upsample_size=upsample_size,
      )
    else:
      sample = upsample_block(
          hidden_states=sample, temb=emb, res_hidden_states_tuple=res_samples, upsample_size=upsample_size
      )
  # 6. post-process
  sample = unet.conv_norm_out(sample)
  sample = unet.conv_act(sample)
  sample = unet.conv_out(sample)

  return UNet2DConditionOutput(sample=sample)
