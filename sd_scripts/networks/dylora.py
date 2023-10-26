# some codes are copied from:
# https://github.com/huawei-noah/KD-NLP/blob/main/DyLoRA/

# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
# Changes made to the original code:
# 2022.08.20 - Integrate the DyLoRA layer for the LoRA Linear layer
#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import math
import os
import random
from typing import List, Tuple, Union
import torch
from torch import nn


class DyLoRAModule(torch.nn.Module):
    """
    replaces forward method of the original Linear, instead of replacing the original Linear module.
    """

    # NOTE: support dropout in future
    def __init__(self, lora_name, org_module: torch.nn.Module, multiplier=1.0, lora_dim=4, alpha=1, unit=1):
        super().__init__()
        self.lora_name = lora_name
        self.lora_dim = lora_dim
        self.unit = unit
        assert self.lora_dim % self.unit == 0, "rank must be a multiple of unit"

        if org_module.__class__.__name__ == "Conv2d":
            in_dim = org_module.in_channels
            out_dim = org_module.out_channels
        else:
            in_dim = org_module.in_features
            out_dim = org_module.out_features

        if type(alpha) == torch.Tensor:
            alpha = alpha.detach().float().numpy()  # without casting, bf16 causes error
        alpha = self.lora_dim if alpha is None or alpha == 0 else alpha
        self.scale = alpha / self.lora_dim
        self.register_buffer("alpha", torch.tensor(alpha))  # 定数として扱える

        self.is_conv2d = org_module.__class__.__name__ == "Conv2d"
        self.is_conv2d_3x3 = self.is_conv2d and org_module.kernel_size == (3, 3)

        if self.is_conv2d and self.is_conv2d_3x3:
            kernel_size = org_module.kernel_size
            self.stride = org_module.stride
            self.padding = org_module.padding
            self.lora_A = nn.ParameterList([org_module.weight.new_zeros((1, in_dim, *kernel_size)) for _ in range(self.lora_dim)])
            self.lora_B = nn.ParameterList([org_module.weight.new_zeros((out_dim, 1, 1, 1)) for _ in range(self.lora_dim)])
        else:
            self.lora_A = nn.ParameterList([org_module.weight.new_zeros((1, in_dim)) for _ in range(self.lora_dim)])
            self.lora_B = nn.ParameterList([org_module.weight.new_zeros((out_dim, 1)) for _ in range(self.lora_dim)])

        # same as microsoft's
        for lora in self.lora_A:
            torch.nn.init.kaiming_uniform_(lora, a=math.sqrt(5))
        for lora in self.lora_B:
            torch.nn.init.zeros_(lora)

        self.multiplier = multiplier
        self.org_module = org_module  # remove in applying

    def apply_to(self):
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        del self.org_module

    def forward(self, x):
        result = self.org_forward(x)

        # specify the dynamic rank
        trainable_rank = random.randint(0, self.lora_dim - 1)
        trainable_rank = trainable_rank - trainable_rank % self.unit  # make sure the rank is a multiple of unit

        # 一部のパラメータを固定して、残りのパラメータを学習する
        for i in range(0, trainable_rank):
            self.lora_A[i].requires_grad = False
            self.lora_B[i].requires_grad = False
        for i in range(trainable_rank, trainable_rank + self.unit):
            self.lora_A[i].requires_grad = True
            self.lora_B[i].requires_grad = True
        for i in range(trainable_rank + self.unit, self.lora_dim):
            self.lora_A[i].requires_grad = False
            self.lora_B[i].requires_grad = False

        lora_A = torch.cat(tuple(self.lora_A), dim=0)
        lora_B = torch.cat(tuple(self.lora_B), dim=1)

        # calculate with lora_A and lora_B
        if self.is_conv2d_3x3:
            ab = torch.nn.functional.conv2d(x, lora_A, stride=self.stride, padding=self.padding)
            ab = torch.nn.functional.conv2d(ab, lora_B)
        else:
            ab = x
            if self.is_conv2d:
                ab = ab.reshape(ab.size(0), ab.size(1), -1).transpose(1, 2)  # (N, C, H, W) -> (N, H*W, C)

            ab = torch.nn.functional.linear(ab, lora_A)
            ab = torch.nn.functional.linear(ab, lora_B)

            if self.is_conv2d:
                ab = ab.transpose(1, 2).reshape(ab.size(0), -1, *x.size()[2:])  # (N, H*W, C) -> (N, C, H, W)

        # 最後の項は、低rankをより大きくするためのスケーリング（じゃないかな）
        result = result + ab * self.scale * math.sqrt(self.lora_dim / (trainable_rank + self.unit))

        # NOTE weightに加算してからlinear/conv2dを呼んだほうが速いかも
        return result

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        # state dictを通常のLoRAと同じにする:
        # nn.ParameterListは `.lora_A.0` みたいな名前になるので、forwardと同様にcatして入れ替える
        sd = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)

        lora_A_weight = torch.cat(tuple(self.lora_A), dim=0)
        if self.is_conv2d and not self.is_conv2d_3x3:
            lora_A_weight = lora_A_weight.unsqueeze(-1).unsqueeze(-1)

        lora_B_weight = torch.cat(tuple(self.lora_B), dim=1)
        if self.is_conv2d and not self.is_conv2d_3x3:
            lora_B_weight = lora_B_weight.unsqueeze(-1).unsqueeze(-1)

        sd[self.lora_name + ".lora_down.weight"] = lora_A_weight if keep_vars else lora_A_weight.detach()
        sd[self.lora_name + ".lora_up.weight"] = lora_B_weight if keep_vars else lora_B_weight.detach()

        i = 0
        while True:
            key_a = f"{self.lora_name}.lora_A.{i}"
            key_b = f"{self.lora_name}.lora_B.{i}"
            if key_a in sd:
                sd.pop(key_a)
                sd.pop(key_b)
            else:
                break
            i += 1
        return sd

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        # 通常のLoRAと同じstate dictを読み込めるようにする：この方法はchatGPTに聞いた
        lora_A_weight = state_dict.pop(self.lora_name + ".lora_down.weight", None)
        lora_B_weight = state_dict.pop(self.lora_name + ".lora_up.weight", None)

        if lora_A_weight is None or lora_B_weight is None:
            if strict:
                raise KeyError(f"{self.lora_name}.lora_down/up.weight is not found")
            else:
                return

        if self.is_conv2d and not self.is_conv2d_3x3:
            lora_A_weight = lora_A_weight.squeeze(-1).squeeze(-1)
            lora_B_weight = lora_B_weight.squeeze(-1).squeeze(-1)

        state_dict.update(
            {f"{self.lora_name}.lora_A.{i}": nn.Parameter(lora_A_weight[i].unsqueeze(0)) for i in range(lora_A_weight.size(0))}
        )
        state_dict.update(
            {f"{self.lora_name}.lora_B.{i}": nn.Parameter(lora_B_weight[:, i].unsqueeze(1)) for i in range(lora_B_weight.size(1))}
        )

        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)


def create_network(multiplier, network_dim, network_alpha, vae, text_encoder, unet, **kwargs):
    if network_dim is None:
        network_dim = 4  # default
    if network_alpha is None:
        network_alpha = 1.0

    # extract dim/alpha for conv2d, and block dim
    conv_dim = kwargs.get("conv_dim", None)
    conv_alpha = kwargs.get("conv_alpha", None)
    unit = kwargs.get("unit", None)
    if conv_dim is not None:
        conv_dim = int(conv_dim)
        assert conv_dim == network_dim, "conv_dim must be same as network_dim"
        if conv_alpha is None:
            conv_alpha = 1.0
        else:
            conv_alpha = float(conv_alpha)
    if unit is not None:
        unit = int(unit)
    else:
        unit = 1

    network = DyLoRANetwork(
        text_encoder,
        unet,
        multiplier=multiplier,
        lora_dim=network_dim,
        alpha=network_alpha,
        apply_to_conv=conv_dim is not None,
        unit=unit,
        varbose=True,
    )
    return network


# Create network from weights for inference, weights are not loaded here (because can be merged)
def create_network_from_weights(multiplier, file, vae, text_encoder, unet, weights_sd=None, for_inference=False, **kwargs):
    if weights_sd is None:
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file, safe_open

            weights_sd = load_file(file)
        else:
            weights_sd = torch.load(file, map_location="cpu")

    # get dim/alpha mapping
    modules_dim = {}
    modules_alpha = {}
    for key, value in weights_sd.items():
        if "." not in key:
            continue

        lora_name = key.split(".")[0]
        if "alpha" in key:
            modules_alpha[lora_name] = value
        elif "lora_down" in key:
            dim = value.size()[0]
            modules_dim[lora_name] = dim
            # print(lora_name, value.size(), dim)

    # support old LoRA without alpha
    for key in modules_dim.keys():
        if key not in modules_alpha:
            modules_alpha = modules_dim[key]

    module_class = DyLoRAModule

    network = DyLoRANetwork(
        text_encoder, unet, multiplier=multiplier, modules_dim=modules_dim, modules_alpha=modules_alpha, module_class=module_class
    )
    return network, weights_sd


class DyLoRANetwork(torch.nn.Module):
    UNET_TARGET_REPLACE_MODULE = ["Transformer2DModel"]
    UNET_TARGET_REPLACE_MODULE_CONV2D_3X3 = ["ResnetBlock2D", "Downsample2D", "Upsample2D"]
    TEXT_ENCODER_TARGET_REPLACE_MODULE = ["CLIPAttention", "CLIPMLP"]
    LORA_PREFIX_UNET = "lora_unet"
    LORA_PREFIX_TEXT_ENCODER = "lora_te"

    def __init__(
        self,
        text_encoder,
        unet,
        multiplier=1.0,
        lora_dim=4,
        alpha=1,
        apply_to_conv=False,
        modules_dim=None,
        modules_alpha=None,
        unit=1,
        module_class=DyLoRAModule,
        varbose=False,
    ) -> None:
        super().__init__()
        self.multiplier = multiplier

        self.lora_dim = lora_dim
        self.alpha = alpha
        self.apply_to_conv = apply_to_conv

        if modules_dim is not None:
            print(f"create LoRA network from weights")
        else:
            print(f"create LoRA network. base dim (rank): {lora_dim}, alpha: {alpha}, unit: {unit}")
            if self.apply_to_conv:
                print(f"apply LoRA to Conv2d with kernel size (3,3).")

        # create module instances
        def create_modules(is_unet, root_module: torch.nn.Module, target_replace_modules) -> List[DyLoRAModule]:
            prefix = DyLoRANetwork.LORA_PREFIX_UNET if is_unet else DyLoRANetwork.LORA_PREFIX_TEXT_ENCODER
            loras = []
            for name, module in root_module.named_modules():
                if module.__class__.__name__ in target_replace_modules:
                    for child_name, child_module in module.named_modules():
                        is_linear = child_module.__class__.__name__ == "Linear"
                        is_conv2d = child_module.__class__.__name__ == "Conv2d"
                        is_conv2d_1x1 = is_conv2d and child_module.kernel_size == (1, 1)

                        if is_linear or is_conv2d:
                            lora_name = prefix + "." + name + "." + child_name
                            lora_name = lora_name.replace(".", "_")

                            dim = None
                            alpha = None
                            if modules_dim is not None:
                                if lora_name in modules_dim:
                                    dim = modules_dim[lora_name]
                                    alpha = modules_alpha[lora_name]
                            else:
                                if is_linear or is_conv2d_1x1 or apply_to_conv:
                                    dim = self.lora_dim
                                    alpha = self.alpha

                            if dim is None or dim == 0:
                                continue

                            # dropout and fan_in_fan_out is default
                            lora = module_class(lora_name, child_module, self.multiplier, dim, alpha, unit)
                            loras.append(lora)
            return loras

        self.text_encoder_loras = create_modules(False, text_encoder, DyLoRANetwork.TEXT_ENCODER_TARGET_REPLACE_MODULE)
        print(f"create LoRA for Text Encoder: {len(self.text_encoder_loras)} modules.")

        # extend U-Net target modules if conv2d 3x3 is enabled, or load from weights
        target_modules = DyLoRANetwork.UNET_TARGET_REPLACE_MODULE
        if modules_dim is not None or self.apply_to_conv:
            target_modules += DyLoRANetwork.UNET_TARGET_REPLACE_MODULE_CONV2D_3X3

        self.unet_loras = create_modules(True, unet, target_modules)
        print(f"create LoRA for U-Net: {len(self.unet_loras)} modules.")

    def set_multiplier(self, multiplier):
        self.multiplier = multiplier
        for lora in self.text_encoder_loras + self.unet_loras:
            lora.multiplier = self.multiplier

    def load_weights(self, file):
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file

            weights_sd = load_file(file)
        else:
            weights_sd = torch.load(file, map_location="cpu")

        info = self.load_state_dict(weights_sd, False)
        return info

    def apply_to(self, text_encoder, unet, apply_text_encoder=True, apply_unet=True):
        if apply_text_encoder:
            print("enable LoRA for text encoder")
        else:
            self.text_encoder_loras = []

        if apply_unet:
            print("enable LoRA for U-Net")
        else:
            self.unet_loras = []

        for lora in self.text_encoder_loras + self.unet_loras:
            lora.apply_to()
            self.add_module(lora.lora_name, lora)

    """
    def merge_to(self, text_encoder, unet, weights_sd, dtype, device):
        apply_text_encoder = apply_unet = False
        for key in weights_sd.keys():
            if key.startswith(DyLoRANetwork.LORA_PREFIX_TEXT_ENCODER):
                apply_text_encoder = True
            elif key.startswith(DyLoRANetwork.LORA_PREFIX_UNET):
                apply_unet = True

        if apply_text_encoder:
            print("enable LoRA for text encoder")
        else:
            self.text_encoder_loras = []

        if apply_unet:
            print("enable LoRA for U-Net")
        else:
            self.unet_loras = []

        for lora in self.text_encoder_loras + self.unet_loras:
            sd_for_lora = {}
            for key in weights_sd.keys():
                if key.startswith(lora.lora_name):
                    sd_for_lora[key[len(lora.lora_name) + 1 :]] = weights_sd[key]
            lora.merge_to(sd_for_lora, dtype, device)

        print(f"weights are merged")
    """

    def prepare_optimizer_params(self, text_encoder_lr, unet_lr, default_lr):
        self.requires_grad_(True)
        all_params = []

        def enumerate_params(loras):
            params = []
            for lora in loras:
                params.extend(lora.parameters())
            return params

        if self.text_encoder_loras:
            param_data = {"params": enumerate_params(self.text_encoder_loras)}
            if text_encoder_lr is not None:
                param_data["lr"] = text_encoder_lr
            all_params.append(param_data)

        if self.unet_loras:
            param_data = {"params": enumerate_params(self.unet_loras)}
            if unet_lr is not None:
                param_data["lr"] = unet_lr
            all_params.append(param_data)

        return all_params

    def enable_gradient_checkpointing(self):
        # not supported
        pass

    def prepare_grad_etc(self, text_encoder, unet):
        self.requires_grad_(True)

    def on_epoch_start(self, text_encoder, unet):
        self.train()

    def get_trainable_params(self):
        return self.parameters()

    def save_weights(self, file, dtype, metadata):
        if metadata is not None and len(metadata) == 0:
            metadata = None

        state_dict = self.state_dict()

        if dtype is not None:
            for key in list(state_dict.keys()):
                v = state_dict[key]
                v = v.detach().clone().to("cpu").to(dtype)
                state_dict[key] = v

        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import save_file
            from library import train_util

            # Precalculate model hashes to save time on indexing
            if metadata is None:
                metadata = {}
            model_hash, legacy_hash = train_util.precalculate_safetensors_hashes(state_dict, metadata)
            metadata["sshs_model_hash"] = model_hash
            metadata["sshs_legacy_hash"] = legacy_hash

            save_file(state_dict, file, metadata)
        else:
            torch.save(state_dict, file)

    # mask is a tensor with values from 0 to 1
    def set_region(self, sub_prompt_index, is_last_network, mask):
        pass

    def set_current_generation(self, batch_size, num_sub_prompts, width, height, shared):
        pass
