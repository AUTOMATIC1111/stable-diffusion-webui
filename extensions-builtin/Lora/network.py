from __future__ import annotations
import os
from collections import namedtuple
import enum

from modules import sd_models, hashes, shared

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

        if self.is_safetensors:
            self.metadata = sd_models.read_metadata_from_safetensors(filename)
        if self.metadata:
            m = {}
            for k, v in sorted(self.metadata.items(), key=lambda x: metadata_tags_order.get(x[0], 999)):
                m[k] = v
            self.metadata = m
        self.alias = self.metadata.get('ss_output_name', self.name)
        self.hash = None
        self.shorthash = None
        self.set_hash(self.metadata.get('sshs_model_hash') or hashes.sha256_from_cache(self.filename, "lora/" + self.name, use_addnet_hash=self.is_safetensors) or '')
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

    def read_hash(self):
        if not self.hash:
            self.set_hash(hashes.sha256(self.filename, "lora/" + self.name, use_addnet_hash=self.is_safetensors) or '')

    def get_alias(self):
        import networks
        return self.name if shared.opts.lora_preferred_name == "filename" or self.alias.lower() in networks.forbidden_network_aliases else self.alias


class Network:  # LoraModule
    def __init__(self, name, network_on_disk: NetworkOnDisk):
        self.name = name
        self.network_on_disk = network_on_disk
        self.te_multiplier = 1.0
        self.unet_multiplier = [1.0] * 3
        self.dyn_dim = None
        self.modules = {}
        self.mtime = None
        self.mentioned_name = None
        """the text that was used to add the network to prompt - can be either name or an alias"""


class ModuleType:
    def create_module(self, net: Network, weights: NetworkWeights) -> Network | None: # pylint: disable=W0613
        return None


class NetworkModule:
    def __init__(self, net: Network, weights: NetworkWeights):
        self.network = net
        self.network_key = weights.network_key
        self.sd_key = weights.sd_key
        self.sd_module = weights.sd_module
        if hasattr(self.sd_module, 'weight'):
            self.shape = self.sd_module.weight.shape
        self.dim = None
        self.bias = weights.w.get("bias")
        self.alpha = weights.w["alpha"].item() if "alpha" in weights.w else None
        self.scale = weights.w["scale"].item() if "scale" in weights.w else None

    def multiplier(self):
        if 'transformer' in self.sd_key[:20]:
            return self.network.te_multiplier
        if "down_blocks" in self.sd_key:
            return self.network.unet_multiplier[0]
        if "mid_block" in self.sd_key:
            return self.network.unet_multiplier[1]
        if "up_blocks" in self.sd_key:
            return self.network.unet_multiplier[2]
        else:
            return self.network.unet_multiplier[0]

    def calc_scale(self):
        if self.scale is not None:
            return self.scale
        if self.dim is not None and self.alpha is not None:
            return self.alpha / self.dim
        return 1.0

    def finalize_updown(self, updown, orig_weight, output_shape, ex_bias=None):
        if self.bias is not None:
            updown = updown.reshape(self.bias.shape)
            updown += self.bias.to(orig_weight.device, dtype=orig_weight.dtype)
            updown = updown.reshape(output_shape)
        if len(output_shape) == 4:
            updown = updown.reshape(output_shape)
        if orig_weight.size().numel() == updown.size().numel():
            updown = updown.reshape(orig_weight.shape)
        if ex_bias is not None:
            ex_bias = ex_bias * self.multiplier()
        return updown * self.calc_scale() * self.multiplier(), ex_bias

    def calc_updown(self, target):
        raise NotImplementedError()

    def forward(self, x, y):
        raise NotImplementedError()
