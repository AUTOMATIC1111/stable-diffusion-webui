from __future__ import annotations
import os
from collections import namedtuple
import enum

import torch.nn as nn
import torch.nn.functional as F

from modules import sd_models, cache, errors, hashes, shared
import modules.models.sd3.mmdit

NetworkWeights = namedtuple('NetworkWeights', ['network_key', 'sd_key', 'w', 'sd_module'])

metadata_tags_order = {"ss_sd_model_name": 1, "ss_resolution": 2, "ss_clip_skip": 3, "ss_num_train_images": 10, "ss_tag_frequency": 20}


class SdVersion(enum.Enum):
    Unknown = 1
    SD1 = 2
    SD2 = 3
    SDXL = 4


class NetworkOnDisk:
    def __init__(self, name, filename):
        self.name = name
        self.filename = filename
        self.metadata = {}
        self.is_safetensors = os.path.splitext(filename)[1].lower() == ".safetensors"

        def read_metadata():
            metadata = sd_models.read_metadata_from_safetensors(filename)

            return metadata

        if self.is_safetensors:
            try:
                self.metadata = cache.cached_data_for_file('safetensors-metadata', "lora/" + self.name, filename, read_metadata)
            except Exception as e:
                errors.display(e, f"reading lora {filename}")

        if self.metadata:
            m = {}
            for k, v in sorted(self.metadata.items(), key=lambda x: metadata_tags_order.get(x[0], 999)):
                m[k] = v

            self.metadata = m

        self.alias = self.metadata.get('ss_output_name', self.name)

        self.hash = None
        self.shorthash = None
        self.set_hash(
            self.metadata.get('sshs_model_hash') or
            hashes.sha256_from_cache(self.filename, "lora/" + self.name, use_addnet_hash=self.is_safetensors) or
            ''
        )

        self.sd_version = self.detect_version()

    def detect_version(self):
        if str(self.metadata.get('ss_base_model_version', "")).startswith("sdxl_"):
            return SdVersion.SDXL
        elif str(self.metadata.get('ss_v2', "")) == "True":
            return SdVersion.SD2
        elif len(self.metadata):
            return SdVersion.SD1

        return SdVersion.Unknown

    def set_hash(self, v):
        self.hash = v
        self.shorthash = self.hash[0:12]

        if self.shorthash:
            import networks
            networks.available_network_hash_lookup[self.shorthash] = self

    def read_hash(self):
        if not self.hash:
            self.set_hash(hashes.sha256(self.filename, "lora/" + self.name, use_addnet_hash=self.is_safetensors) or '')

    def get_alias(self):
        import networks
        if shared.opts.lora_preferred_name == "Filename" or self.alias.lower() in networks.forbidden_network_aliases:
            return self.name
        else:
            return self.alias


class Network:  # LoraModule
    def __init__(self, name, network_on_disk: NetworkOnDisk):
        self.name = name
        self.network_on_disk = network_on_disk
        self.te_multiplier = 1.0
        self.unet_multiplier = 1.0
        self.dyn_dim = None
        self.modules = {}
        self.bundle_embeddings = {}
        self.mtime = None

        self.mentioned_name = None
        """the text that was used to add the network to prompt - can be either name or an alias"""


class ModuleType:
    def create_module(self, net: Network, weights: NetworkWeights) -> Network | None:
        return None


class NetworkModule:
    def __init__(self, net: Network, weights: NetworkWeights):
        self.network = net
        self.network_key = weights.network_key
        self.sd_key = weights.sd_key
        self.sd_module = weights.sd_module

        if isinstance(self.sd_module, modules.models.sd3.mmdit.QkvLinear):
            s = self.sd_module.weight.shape
            self.shape = (s[0] // 3, s[1])
        elif hasattr(self.sd_module, 'weight'):
            self.shape = self.sd_module.weight.shape
        elif isinstance(self.sd_module, nn.MultiheadAttention):
            # For now, only self-attn use Pytorch's MHA
            # So assume all qkvo proj have same shape
            self.shape = self.sd_module.out_proj.weight.shape
        else:
            self.shape = None

        self.ops = None
        self.extra_kwargs = {}
        if isinstance(self.sd_module, nn.Conv2d):
            self.ops = F.conv2d
            self.extra_kwargs = {
                'stride': self.sd_module.stride,
                'padding': self.sd_module.padding
            }
        elif isinstance(self.sd_module, nn.Linear):
            self.ops = F.linear
        elif isinstance(self.sd_module, nn.LayerNorm):
            self.ops = F.layer_norm
            self.extra_kwargs = {
                'normalized_shape': self.sd_module.normalized_shape,
                'eps': self.sd_module.eps
            }
        elif isinstance(self.sd_module, nn.GroupNorm):
            self.ops = F.group_norm
            self.extra_kwargs = {
                'num_groups': self.sd_module.num_groups,
                'eps': self.sd_module.eps
            }

        self.dim = None
        self.bias = weights.w.get("bias")
        self.alpha = weights.w["alpha"].item() if "alpha" in weights.w else None
        self.scale = weights.w["scale"].item() if "scale" in weights.w else None

        self.dora_scale = weights.w.get("dora_scale", None)
        self.dora_norm_dims = len(self.shape) - 1

    def multiplier(self):
        if 'transformer' in self.sd_key[:20]:
            return self.network.te_multiplier
        else:
            return self.network.unet_multiplier

    def calc_scale(self):
        if self.scale is not None:
            return self.scale
        if self.dim is not None and self.alpha is not None:
            return self.alpha / self.dim

        return 1.0

    def apply_weight_decompose(self, updown, orig_weight):
        # Match the device/dtype
        orig_weight = orig_weight.to(updown.dtype)
        dora_scale = self.dora_scale.to(device=orig_weight.device, dtype=updown.dtype)
        updown = updown.to(orig_weight.device)

        merged_scale1 = updown + orig_weight
        merged_scale1_norm = (
            merged_scale1.transpose(0, 1)
            .reshape(merged_scale1.shape[1], -1)
            .norm(dim=1, keepdim=True)
            .reshape(merged_scale1.shape[1], *[1] * self.dora_norm_dims)
            .transpose(0, 1)
        )

        dora_merged = (
            merged_scale1 * (dora_scale / merged_scale1_norm)
        )
        final_updown = dora_merged - orig_weight
        return final_updown

    def finalize_updown(self, updown, orig_weight, output_shape, ex_bias=None):
        if self.bias is not None:
            updown = updown.reshape(self.bias.shape)
            updown += self.bias.to(orig_weight.device, dtype=updown.dtype)
            updown = updown.reshape(output_shape)

        if len(output_shape) == 4:
            updown = updown.reshape(output_shape)

        if orig_weight.size().numel() == updown.size().numel():
            updown = updown.reshape(orig_weight.shape)

        if ex_bias is not None:
            ex_bias = ex_bias * self.multiplier()

        updown = updown * self.calc_scale()

        if self.dora_scale is not None:
            updown = self.apply_weight_decompose(updown, orig_weight)

        return updown * self.multiplier(), ex_bias

    def calc_updown(self, target):
        raise NotImplementedError()

    def forward(self, x, y):
        """A general forward implementation for all modules"""
        if self.ops is None:
            raise NotImplementedError()
        else:
            updown, ex_bias = self.calc_updown(self.sd_module.weight)
            return y + self.ops(x, weight=updown, bias=ex_bias, **self.extra_kwargs)

