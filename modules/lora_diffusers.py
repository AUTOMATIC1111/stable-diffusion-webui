import diffusers
import diffusers.models.lora as diffusers_lora
# from modules import shared
import modules.shared as shared


lora_state = { # TODO Lora state for Diffusers
    'multiplier': [],
    'active': False,
    'loaded': 0,
    'all_loras': []
}
def unload_diffusers_lora():
    try:
        pipe = shared.sd_model
        if shared.opts.diffusers_lora_loader == "diffusers default":
            pipe.unload_lora_weights()
            pipe._remove_text_encoder_monkey_patch() # pylint: disable=W0212
            proc_cls_name = next(iter(pipe.unet.attn_processors.values())).__class__.__name__
            non_lora_proc_cls = getattr(diffusers.models.attention_processor, proc_cls_name)#[len("LORA"):])
            pipe.unet.set_attn_processor(non_lora_proc_cls())
            # shared.log.debug('Diffusers LoRA unloaded')
        else:
            lora_state['all_loras'].reverse()
            lora_state['multiplier'].reverse()
            for i, lora_network in enumerate(lora_state['all_loras']):
                if shared.opts.diffusers_lora_loader == "merge and apply":
                    lora_network.restore_from(multiplier=lora_state['multiplier'][i])
                if shared.opts.diffusers_lora_loader == "sequential apply":
                    lora_network.unapply_to()
        lora_state['active'] = False
        lora_state['loaded'] = 0
        lora_state['all_loras'] = []
        lora_state['multiplier'] = []

    except Exception as e:
        shared.log.error(f"Diffusers LoRA unloading failed: {e}")


def load_diffusers_lora(name, lora, strength = 1.0):
    try:
        pipe = shared.sd_model
        lora_state['active'] = True
        lora_state['loaded'] += 1
        lora_state['multiplier'].append(strength)
        if shared.opts.diffusers_lora_loader == "diffusers default":
            pipe.load_lora_weights(lora.filename, cache_dir=shared.opts.diffusers_dir, local_files_only=True, lora_scale=strength)
        else:
            from safetensors.torch import load_file
            lora_sd = load_file(lora.filename)
            if "XL" in pipe.__class__.__name__:
                text_encoders = [pipe.text_encoder, pipe.text_encoder_2]
            else:
                text_encoders = pipe.text_encoder
            lora_network: LoRANetwork = create_network_from_weights(text_encoders, pipe.unet, lora_sd, multiplier=strength)
            lora_network.load_state_dict(lora_sd)
            if shared.opts.diffusers_lora_loader == "merge and apply":
                lora_network.merge_to(multiplier=strength)
            if shared.opts.diffusers_lora_loader == "sequential apply":
                lora_network.to(shared.device, dtype=pipe.unet.dtype)
                lora_network.apply_to(multiplier=strength)
            lora_state['all_loras'].append(lora_network)
        shared.log.info(f"LoRA loaded: {name} strength={strength} loader={shared.opts.diffusers_lora_loader}")
    except Exception as e:
        shared.log.error(f"LoRA loading failed: {name} {e}")


# Diffusersで動くLoRA。このファイル単独で完結する。
# LoRA module for Diffusers. This file works independently.
import bisect
import math
from typing import Any, Dict, List, Mapping, Optional, Union
from diffusers import UNet2DConditionModel
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

        if isinstance(org_module, diffusers_lora.LoRACompatibleConv): #Modified to support Diffusers>=0.19.2
            in_dim = org_module.in_channels
            out_dim = org_module.out_channels
        else:
            in_dim = org_module.in_features
            out_dim = org_module.out_features

        self.lora_dim = lora_dim

        if isinstance(org_module, diffusers_lora.LoRACompatibleConv): #Modified to support Diffusers>=0.19.2
            kernel_size = org_module.kernel_size
            stride = org_module.stride
            padding = org_module.padding
            self.lora_down = torch.nn.Conv2d(in_dim, self.lora_dim, kernel_size, stride, padding, bias=False)
            self.lora_up = torch.nn.Conv2d(self.lora_dim, out_dim, (1, 1), (1, 1), bias=False)
        else:
            self.lora_down = torch.nn.Linear(in_dim, self.lora_dim, bias=False)
            self.lora_up = torch.nn.Linear(self.lora_dim, out_dim, bias=False)

        if isinstance(alpha, torch.Tensor):
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
    def forward(self, x):
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
class LoRANetwork(torch.nn.Module): # pylint: disable=abstract-method
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
        varbose: Optional[bool] = False, # pylint: disable=unused-argument
    ) -> None:
        super().__init__()
        self.multiplier = multiplier

        # shared.log.debug("create LoRA network from weights")

        # convert SDXL Stability AI's U-Net modules to Diffusers
        converted = self.convert_unet_modules(modules_dim, modules_alpha)
        if converted:
            shared.log.debug(f"LoRA convert: modules={converted} SDXL SAI/SGM to Diffusers")

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
                        is_linear = isinstance(child_module, (torch.nn.Linear, diffusers_lora.LoRACompatibleLinear))  #Modified to support Diffusers>=0.19.2
                        is_conv2d = isinstance(child_module, (torch.nn.Conv2d, diffusers_lora.LoRACompatibleConv))  #Modified to support Diffusers>=0.19.2

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

        # extend U-Net target modules to include Conv2d 3x3
        target_modules = LoRANetwork.UNET_TARGET_REPLACE_MODULE + LoRANetwork.UNET_TARGET_REPLACE_MODULE_CONV2D_3X3

        self.unet_loras: List[LoRAModule]
        self.unet_loras, skipped_un = create_modules(True, None, unet, target_modules)
        shared.log.debug(f"LoRA modules loaded/skipped: te={len(self.text_encoder_loras)}/{len(skipped_te)} unet={len(self.unet_loras)}/skip={len(skipped_un)}")

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
            # shared.log.debug("LoRA apply for text encoder")
            for lora in self.text_encoder_loras:
                lora.apply_to(multiplier)
        if apply_unet:
            # shared.log.debug("LoRA apply for U-Net")
            for lora in self.unet_loras:
                lora.apply_to(multiplier)

    def unapply_to(self):
        for lora in self.text_encoder_loras + self.unet_loras:
            lora.unapply_to()

    def merge_to(self, multiplier=1.0):
        # shared.log.debug("LoRA merge weights for text encoder")
        for lora in tqdm(self.text_encoder_loras + self.unet_loras):
            lora.merge_to(multiplier)

    def restore_from(self, multiplier=1.0):
        # shared.log.debug("LoRA restore weights")
        for lora in tqdm(self.text_encoder_loras + self.unet_loras):
            lora.restore_from(multiplier)

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
