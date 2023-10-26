# Diffusersで動くLoRA。このファイル単独で完結する。
# LoRA module for Diffusers. This file works independently.

import bisect
import math
import random
from typing import Any, Dict, List, Mapping, Optional, Union
from diffusers import UNet2DConditionModel
import numpy as np
from tqdm import tqdm
from transformers import CLIPTextModel
import torch


def make_unet_conversion_map() -> Dict[str, str]:
    unet_conversion_map_layer = []

    for i in range(3):  # num_blocks is 3 in sdxl
        # loop over downblocks/upblocks
        for j in range(2):
            # loop over resnets/attentions for downblocks
            hf_down_res_prefix = f"down_blocks.{i}.resnets.{j}."
            sd_down_res_prefix = f"input_blocks.{3*i + j + 1}.0."
            unet_conversion_map_layer.append((sd_down_res_prefix, hf_down_res_prefix))

            if i < 3:
                # no attention layers in down_blocks.3
                hf_down_atn_prefix = f"down_blocks.{i}.attentions.{j}."
                sd_down_atn_prefix = f"input_blocks.{3*i + j + 1}.1."
                unet_conversion_map_layer.append((sd_down_atn_prefix, hf_down_atn_prefix))

        for j in range(3):
            # loop over resnets/attentions for upblocks
            hf_up_res_prefix = f"up_blocks.{i}.resnets.{j}."
            sd_up_res_prefix = f"output_blocks.{3*i + j}.0."
            unet_conversion_map_layer.append((sd_up_res_prefix, hf_up_res_prefix))

            # if i > 0: commentout for sdxl
            # no attention layers in up_blocks.0
            hf_up_atn_prefix = f"up_blocks.{i}.attentions.{j}."
            sd_up_atn_prefix = f"output_blocks.{3*i + j}.1."
            unet_conversion_map_layer.append((sd_up_atn_prefix, hf_up_atn_prefix))

        if i < 3:
            # no downsample in down_blocks.3
            hf_downsample_prefix = f"down_blocks.{i}.downsamplers.0.conv."
            sd_downsample_prefix = f"input_blocks.{3*(i+1)}.0.op."
            unet_conversion_map_layer.append((sd_downsample_prefix, hf_downsample_prefix))

            # no upsample in up_blocks.3
            hf_upsample_prefix = f"up_blocks.{i}.upsamplers.0."
            sd_upsample_prefix = f"output_blocks.{3*i + 2}.{2}."  # change for sdxl
            unet_conversion_map_layer.append((sd_upsample_prefix, hf_upsample_prefix))

    hf_mid_atn_prefix = "mid_block.attentions.0."
    sd_mid_atn_prefix = "middle_block.1."
    unet_conversion_map_layer.append((sd_mid_atn_prefix, hf_mid_atn_prefix))

    for j in range(2):
        hf_mid_res_prefix = f"mid_block.resnets.{j}."
        sd_mid_res_prefix = f"middle_block.{2*j}."
        unet_conversion_map_layer.append((sd_mid_res_prefix, hf_mid_res_prefix))

    unet_conversion_map_resnet = [
        # (stable-diffusion, HF Diffusers)
        ("in_layers.0.", "norm1."),
        ("in_layers.2.", "conv1."),
        ("out_layers.0.", "norm2."),
        ("out_layers.3.", "conv2."),
        ("emb_layers.1.", "time_emb_proj."),
        ("skip_connection.", "conv_shortcut."),
    ]

    unet_conversion_map = []
    for sd, hf in unet_conversion_map_layer:
        if "resnets" in hf:
            for sd_res, hf_res in unet_conversion_map_resnet:
                unet_conversion_map.append((sd + sd_res, hf + hf_res))
        else:
            unet_conversion_map.append((sd, hf))

    for j in range(2):
        hf_time_embed_prefix = f"time_embedding.linear_{j+1}."
        sd_time_embed_prefix = f"time_embed.{j*2}."
        unet_conversion_map.append((sd_time_embed_prefix, hf_time_embed_prefix))

    for j in range(2):
        hf_label_embed_prefix = f"add_embedding.linear_{j+1}."
        sd_label_embed_prefix = f"label_emb.0.{j*2}."
        unet_conversion_map.append((sd_label_embed_prefix, hf_label_embed_prefix))

    unet_conversion_map.append(("input_blocks.0.0.", "conv_in."))
    unet_conversion_map.append(("out.0.", "conv_norm_out."))
    unet_conversion_map.append(("out.2.", "conv_out."))

    sd_hf_conversion_map = {sd.replace(".", "_")[:-1]: hf.replace(".", "_")[:-1] for sd, hf in unet_conversion_map}
    return sd_hf_conversion_map


UNET_CONVERSION_MAP = make_unet_conversion_map()


class LoRAModule(torch.nn.Module):
    """
    replaces forward method of the original Linear, instead of replacing the original Linear module.
    """

    def __init__(
        self,
        lora_name,
        org_module: torch.nn.Module,
        multiplier=1.0,
        lora_dim=4,
        alpha=1,
    ):
        """if alpha == 0 or None, alpha is rank (no scaling)."""
        super().__init__()
        self.lora_name = lora_name

        if org_module.__class__.__name__ == "Conv2d" or org_module.__class__.__name__ == "LoRACompatibleConv":
            in_dim = org_module.in_channels
            out_dim = org_module.out_channels
        else:
            in_dim = org_module.in_features
            out_dim = org_module.out_features

        self.lora_dim = lora_dim

        if org_module.__class__.__name__ == "Conv2d" or org_module.__class__.__name__ == "LoRACompatibleConv":
            kernel_size = org_module.kernel_size
            stride = org_module.stride
            padding = org_module.padding
            self.lora_down = torch.nn.Conv2d(in_dim, self.lora_dim, kernel_size, stride, padding, bias=False)
            self.lora_up = torch.nn.Conv2d(self.lora_dim, out_dim, (1, 1), (1, 1), bias=False)
        else:
            self.lora_down = torch.nn.Linear(in_dim, self.lora_dim, bias=False)
            self.lora_up = torch.nn.Linear(self.lora_dim, out_dim, bias=False)

        if type(alpha) == torch.Tensor:
            alpha = alpha.detach().float().numpy()  # without casting, bf16 causes error
        alpha = self.lora_dim if alpha is None or alpha == 0 else alpha
        self.scale = alpha / self.lora_dim
        self.register_buffer("alpha", torch.tensor(alpha))  # 勾配計算に含めない / not included in gradient calculation

        # same as microsoft's
        torch.nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_up.weight)

        self.multiplier = multiplier
        self.org_module = [org_module]
        self.enabled = True
        self.network: LoRANetwork = None
        self.org_forward = None

    # override org_module's forward method
    def apply_to(self, multiplier=None):
        if multiplier is not None:
            self.multiplier = multiplier
        if self.org_forward is None:
            self.org_forward = self.org_module[0].forward
            self.org_module[0].forward = self.forward

    # restore org_module's forward method
    def unapply_to(self):
        if self.org_forward is not None:
            self.org_module[0].forward = self.org_forward

    # forward with lora
    # scale is used LoRACompatibleConv, but we ignore it because we have multiplier
    def forward(self, x, scale=1.0):
        if not self.enabled:
            return self.org_forward(x)
        return self.org_forward(x) + self.lora_up(self.lora_down(x)) * self.multiplier * self.scale

    def set_network(self, network):
        self.network = network

    # merge lora weight to org weight
    def merge_to(self, multiplier=1.0):
        # get lora weight
        lora_weight = self.get_weight(multiplier)

        # get org weight
        org_sd = self.org_module[0].state_dict()
        org_weight = org_sd["weight"]
        weight = org_weight + lora_weight.to(org_weight.device, dtype=org_weight.dtype)

        # set weight to org_module
        org_sd["weight"] = weight
        self.org_module[0].load_state_dict(org_sd)

    # restore org weight from lora weight
    def restore_from(self, multiplier=1.0):
        # get lora weight
        lora_weight = self.get_weight(multiplier)

        # get org weight
        org_sd = self.org_module[0].state_dict()
        org_weight = org_sd["weight"]
        weight = org_weight - lora_weight.to(org_weight.device, dtype=org_weight.dtype)

        # set weight to org_module
        org_sd["weight"] = weight
        self.org_module[0].load_state_dict(org_sd)

    # return lora weight
    def get_weight(self, multiplier=None):
        if multiplier is None:
            multiplier = self.multiplier

        # get up/down weight from module
        up_weight = self.lora_up.weight.to(torch.float)
        down_weight = self.lora_down.weight.to(torch.float)

        # pre-calculated weight
        if len(down_weight.size()) == 2:
            # linear
            weight = self.multiplier * (up_weight @ down_weight) * self.scale
        elif down_weight.size()[2:4] == (1, 1):
            # conv2d 1x1
            weight = (
                self.multiplier
                * (up_weight.squeeze(3).squeeze(2) @ down_weight.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
                * self.scale
            )
        else:
            # conv2d 3x3
            conved = torch.nn.functional.conv2d(down_weight.permute(1, 0, 2, 3), up_weight).permute(1, 0, 2, 3)
            weight = self.multiplier * conved * self.scale

        return weight


# Create network from weights for inference, weights are not loaded here
def create_network_from_weights(
    text_encoder: Union[CLIPTextModel, List[CLIPTextModel]], unet: UNet2DConditionModel, weights_sd: Dict, multiplier: float = 1.0
):
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
            modules_alpha[key] = modules_dim[key]

    return LoRANetwork(text_encoder, unet, multiplier=multiplier, modules_dim=modules_dim, modules_alpha=modules_alpha)


def merge_lora_weights(pipe, weights_sd: Dict, multiplier: float = 1.0):
    text_encoders = [pipe.text_encoder, pipe.text_encoder_2] if hasattr(pipe, "text_encoder_2") else [pipe.text_encoder]
    unet = pipe.unet

    lora_network = create_network_from_weights(text_encoders, unet, weights_sd, multiplier=multiplier)
    lora_network.load_state_dict(weights_sd)
    lora_network.merge_to(multiplier=multiplier)


# block weightや学習に対応しない簡易版 / simple version without block weight and training
class LoRANetwork(torch.nn.Module):
    UNET_TARGET_REPLACE_MODULE = ["Transformer2DModel"]
    UNET_TARGET_REPLACE_MODULE_CONV2D_3X3 = ["ResnetBlock2D", "Downsample2D", "Upsample2D"]
    TEXT_ENCODER_TARGET_REPLACE_MODULE = ["CLIPAttention", "CLIPMLP"]
    LORA_PREFIX_UNET = "lora_unet"
    LORA_PREFIX_TEXT_ENCODER = "lora_te"

    # SDXL: must starts with LORA_PREFIX_TEXT_ENCODER
    LORA_PREFIX_TEXT_ENCODER1 = "lora_te1"
    LORA_PREFIX_TEXT_ENCODER2 = "lora_te2"

    def __init__(
        self,
        text_encoder: Union[List[CLIPTextModel], CLIPTextModel],
        unet: UNet2DConditionModel,
        multiplier: float = 1.0,
        modules_dim: Optional[Dict[str, int]] = None,
        modules_alpha: Optional[Dict[str, int]] = None,
        varbose: Optional[bool] = False,
    ) -> None:
        super().__init__()
        self.multiplier = multiplier

        print(f"create LoRA network from weights")

        # convert SDXL Stability AI's U-Net modules to Diffusers
        converted = self.convert_unet_modules(modules_dim, modules_alpha)
        if converted:
            print(f"converted {converted} Stability AI's U-Net LoRA modules to Diffusers (SDXL)")

        # create module instances
        def create_modules(
            is_unet: bool,
            text_encoder_idx: Optional[int],  # None, 1, 2
            root_module: torch.nn.Module,
            target_replace_modules: List[torch.nn.Module],
        ) -> List[LoRAModule]:
            prefix = (
                self.LORA_PREFIX_UNET
                if is_unet
                else (
                    self.LORA_PREFIX_TEXT_ENCODER
                    if text_encoder_idx is None
                    else (self.LORA_PREFIX_TEXT_ENCODER1 if text_encoder_idx == 1 else self.LORA_PREFIX_TEXT_ENCODER2)
                )
            )
            loras = []
            skipped = []
            for name, module in root_module.named_modules():
                if module.__class__.__name__ in target_replace_modules:
                    for child_name, child_module in module.named_modules():
                        is_linear = (
                            child_module.__class__.__name__ == "Linear" or child_module.__class__.__name__ == "LoRACompatibleLinear"
                        )
                        is_conv2d = (
                            child_module.__class__.__name__ == "Conv2d" or child_module.__class__.__name__ == "LoRACompatibleConv"
                        )

                        if is_linear or is_conv2d:
                            lora_name = prefix + "." + name + "." + child_name
                            lora_name = lora_name.replace(".", "_")

                            if lora_name not in modules_dim:
                                # print(f"skipped {lora_name} (not found in modules_dim)")
                                skipped.append(lora_name)
                                continue

                            dim = modules_dim[lora_name]
                            alpha = modules_alpha[lora_name]
                            lora = LoRAModule(
                                lora_name,
                                child_module,
                                self.multiplier,
                                dim,
                                alpha,
                            )
                            loras.append(lora)
            return loras, skipped

        text_encoders = text_encoder if type(text_encoder) == list else [text_encoder]

        # create LoRA for text encoder
        # 毎回すべてのモジュールを作るのは無駄なので要検討 / it is wasteful to create all modules every time, need to consider
        self.text_encoder_loras: List[LoRAModule] = []
        skipped_te = []
        for i, text_encoder in enumerate(text_encoders):
            if len(text_encoders) > 1:
                index = i + 1
            else:
                index = None

            text_encoder_loras, skipped = create_modules(False, index, text_encoder, LoRANetwork.TEXT_ENCODER_TARGET_REPLACE_MODULE)
            self.text_encoder_loras.extend(text_encoder_loras)
            skipped_te += skipped
        print(f"create LoRA for Text Encoder: {len(self.text_encoder_loras)} modules.")
        if len(skipped_te) > 0:
            print(f"skipped {len(skipped_te)} modules because of missing weight for text encoder.")

        # extend U-Net target modules to include Conv2d 3x3
        target_modules = LoRANetwork.UNET_TARGET_REPLACE_MODULE + LoRANetwork.UNET_TARGET_REPLACE_MODULE_CONV2D_3X3

        self.unet_loras: List[LoRAModule]
        self.unet_loras, skipped_un = create_modules(True, None, unet, target_modules)
        print(f"create LoRA for U-Net: {len(self.unet_loras)} modules.")
        if len(skipped_un) > 0:
            print(f"skipped {len(skipped_un)} modules because of missing weight for U-Net.")

        # assertion
        names = set()
        for lora in self.text_encoder_loras + self.unet_loras:
            names.add(lora.lora_name)
        for lora_name in modules_dim.keys():
            assert lora_name in names, f"{lora_name} is not found in created LoRA modules."

        # make to work load_state_dict
        for lora in self.text_encoder_loras + self.unet_loras:
            self.add_module(lora.lora_name, lora)

    # SDXL: convert SDXL Stability AI's U-Net modules to Diffusers
    def convert_unet_modules(self, modules_dim, modules_alpha):
        converted_count = 0
        not_converted_count = 0

        map_keys = list(UNET_CONVERSION_MAP.keys())
        map_keys.sort()

        for key in list(modules_dim.keys()):
            if key.startswith(LoRANetwork.LORA_PREFIX_UNET + "_"):
                search_key = key.replace(LoRANetwork.LORA_PREFIX_UNET + "_", "")
                position = bisect.bisect_right(map_keys, search_key)
                map_key = map_keys[position - 1]
                if search_key.startswith(map_key):
                    new_key = key.replace(map_key, UNET_CONVERSION_MAP[map_key])
                    modules_dim[new_key] = modules_dim[key]
                    modules_alpha[new_key] = modules_alpha[key]
                    del modules_dim[key]
                    del modules_alpha[key]
                    converted_count += 1
                else:
                    not_converted_count += 1
        assert (
            converted_count == 0 or not_converted_count == 0
        ), f"some modules are not converted: {converted_count} converted, {not_converted_count} not converted"
        return converted_count

    def set_multiplier(self, multiplier):
        self.multiplier = multiplier
        for lora in self.text_encoder_loras + self.unet_loras:
            lora.multiplier = self.multiplier

    def apply_to(self, multiplier=1.0, apply_text_encoder=True, apply_unet=True):
        if apply_text_encoder:
            print("enable LoRA for text encoder")
            for lora in self.text_encoder_loras:
                lora.apply_to(multiplier)
        if apply_unet:
            print("enable LoRA for U-Net")
            for lora in self.unet_loras:
                lora.apply_to(multiplier)

    def unapply_to(self):
        for lora in self.text_encoder_loras + self.unet_loras:
            lora.unapply_to()

    def merge_to(self, multiplier=1.0):
        print("merge LoRA weights to original weights")
        for lora in tqdm(self.text_encoder_loras + self.unet_loras):
            lora.merge_to(multiplier)
        print(f"weights are merged")

    def restore_from(self, multiplier=1.0):
        print("restore LoRA weights from original weights")
        for lora in tqdm(self.text_encoder_loras + self.unet_loras):
            lora.restore_from(multiplier)
        print(f"weights are restored")

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        # convert SDXL Stability AI's state dict to Diffusers' based state dict
        map_keys = list(UNET_CONVERSION_MAP.keys())  # prefix of U-Net modules
        map_keys.sort()
        for key in list(state_dict.keys()):
            if key.startswith(LoRANetwork.LORA_PREFIX_UNET + "_"):
                search_key = key.replace(LoRANetwork.LORA_PREFIX_UNET + "_", "")
                position = bisect.bisect_right(map_keys, search_key)
                map_key = map_keys[position - 1]
                if search_key.startswith(map_key):
                    new_key = key.replace(map_key, UNET_CONVERSION_MAP[map_key])
                    state_dict[new_key] = state_dict[key]
                    del state_dict[key]

        # in case of V2, some weights have different shape, so we need to convert them
        # because V2 LoRA is based on U-Net created by use_linear_projection=False
        my_state_dict = self.state_dict()
        for key in state_dict.keys():
            if state_dict[key].size() != my_state_dict[key].size():
                # print(f"convert {key} from {state_dict[key].size()} to {my_state_dict[key].size()}")
                state_dict[key] = state_dict[key].view(my_state_dict[key].size())

        return super().load_state_dict(state_dict, strict)


if __name__ == "__main__":
    # sample code to use LoRANetwork
    import os
    import argparse
    from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default=None, help="model id for huggingface")
    parser.add_argument("--lora_weights", type=str, default=None, help="path to LoRA weights")
    parser.add_argument("--sdxl", action="store_true", help="use SDXL model")
    parser.add_argument("--prompt", type=str, default="A photo of cat", help="prompt text")
    parser.add_argument("--negative_prompt", type=str, default="", help="negative prompt text")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    args = parser.parse_args()

    image_prefix = args.model_id.replace("/", "_") + "_"

    # load Diffusers model
    print(f"load model from {args.model_id}")
    pipe: Union[StableDiffusionPipeline, StableDiffusionXLPipeline]
    if args.sdxl:
        # use_safetensors=True does not work with 0.18.2
        pipe = StableDiffusionXLPipeline.from_pretrained(args.model_id, variant="fp16", torch_dtype=torch.float16)
    else:
        pipe = StableDiffusionPipeline.from_pretrained(args.model_id, variant="fp16", torch_dtype=torch.float16)
    pipe.to(device)
    pipe.set_use_memory_efficient_attention_xformers(True)

    text_encoders = [pipe.text_encoder, pipe.text_encoder_2] if args.sdxl else [pipe.text_encoder]

    # load LoRA weights
    print(f"load LoRA weights from {args.lora_weights}")
    if os.path.splitext(args.lora_weights)[1] == ".safetensors":
        from safetensors.torch import load_file

        lora_sd = load_file(args.lora_weights)
    else:
        lora_sd = torch.load(args.lora_weights)

    # create by LoRA weights and load weights
    print(f"create LoRA network")
    lora_network: LoRANetwork = create_network_from_weights(text_encoders, pipe.unet, lora_sd, multiplier=1.0)

    print(f"load LoRA network weights")
    lora_network.load_state_dict(lora_sd)

    lora_network.to(device, dtype=pipe.unet.dtype)  # required to apply_to. merge_to works without this

    # 必要があれば、元のモデルの重みをバックアップしておく
    # back-up unet/text encoder weights if necessary
    def detach_and_move_to_cpu(state_dict):
        for k, v in state_dict.items():
            state_dict[k] = v.detach().cpu()
        return state_dict

    org_unet_sd = pipe.unet.state_dict()
    detach_and_move_to_cpu(org_unet_sd)

    org_text_encoder_sd = pipe.text_encoder.state_dict()
    detach_and_move_to_cpu(org_text_encoder_sd)

    if args.sdxl:
        org_text_encoder_2_sd = pipe.text_encoder_2.state_dict()
        detach_and_move_to_cpu(org_text_encoder_2_sd)

    def seed_everything(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

    # create image with original weights
    print(f"create image with original weights")
    seed_everything(args.seed)
    image = pipe(args.prompt, negative_prompt=args.negative_prompt).images[0]
    image.save(image_prefix + "original.png")

    # apply LoRA network to the model: slower than merge_to, but can be reverted easily
    print(f"apply LoRA network to the model")
    lora_network.apply_to(multiplier=1.0)

    print(f"create image with applied LoRA")
    seed_everything(args.seed)
    image = pipe(args.prompt, negative_prompt=args.negative_prompt).images[0]
    image.save(image_prefix + "applied_lora.png")

    # unapply LoRA network to the model
    print(f"unapply LoRA network to the model")
    lora_network.unapply_to()

    print(f"create image with unapplied LoRA")
    seed_everything(args.seed)
    image = pipe(args.prompt, negative_prompt=args.negative_prompt).images[0]
    image.save(image_prefix + "unapplied_lora.png")

    # merge LoRA network to the model: faster than apply_to, but requires back-up of original weights (or unmerge_to)
    print(f"merge LoRA network to the model")
    lora_network.merge_to(multiplier=1.0)

    print(f"create image with LoRA")
    seed_everything(args.seed)
    image = pipe(args.prompt, negative_prompt=args.negative_prompt).images[0]
    image.save(image_prefix + "merged_lora.png")

    # restore (unmerge) LoRA weights: numerically unstable
    # マージされた重みを元に戻す。計算誤差のため、元の重みと完全に一致しないことがあるかもしれない
    # 保存したstate_dictから元の重みを復元するのが確実
    print(f"restore (unmerge) LoRA weights")
    lora_network.restore_from(multiplier=1.0)

    print(f"create image without LoRA")
    seed_everything(args.seed)
    image = pipe(args.prompt, negative_prompt=args.negative_prompt).images[0]
    image.save(image_prefix + "unmerged_lora.png")

    # restore original weights
    print(f"restore original weights")
    pipe.unet.load_state_dict(org_unet_sd)
    pipe.text_encoder.load_state_dict(org_text_encoder_sd)
    if args.sdxl:
        pipe.text_encoder_2.load_state_dict(org_text_encoder_2_sd)

    print(f"create image with restored original weights")
    seed_everything(args.seed)
    image = pipe(args.prompt, negative_prompt=args.negative_prompt).images[0]
    image.save(image_prefix + "restore_original.png")

    # use convenience function to merge LoRA weights
    print(f"merge LoRA weights with convenience function")
    merge_lora_weights(pipe, lora_sd, multiplier=1.0)

    print(f"create image with merged LoRA weights")
    seed_everything(args.seed)
    image = pipe(args.prompt, negative_prompt=args.negative_prompt).images[0]
    image.save(image_prefix + "convenience_merged_lora.png")
