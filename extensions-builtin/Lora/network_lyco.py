import torch

import lyco_helpers
import network
from modules import devices


class NetworkModuleLyco(network.NetworkModule):
    def __init__(self, net: network.Network, weights: network.NetworkWeights):
        super().__init__(net, weights)

        if hasattr(self.sd_module, 'weight'):
            self.shape = self.sd_module.weight.shape

        self.dim = None
        self.bias = weights.w.get("bias")
        self.alpha = weights.w["alpha"].item() if "alpha" in weights.w else None
        self.scale = weights.w["scale"].item() if "scale" in weights.w else None

    def finalize_updown(self, updown, orig_weight, output_shape):
        if self.bias is not None:
            updown = updown.reshape(self.bias.shape)
            updown += self.bias.to(orig_weight.device, dtype=orig_weight.dtype)
            updown = updown.reshape(output_shape)

        if len(output_shape) == 4:
            updown = updown.reshape(output_shape)

        if orig_weight.size().numel() == updown.size().numel():
            updown = updown.reshape(orig_weight.shape)

        scale = (
            self.scale if self.scale is not None
            else self.alpha / self.dim if self.dim is not None and self.alpha is not None
            else 1.0
        )

        return updown * scale * self.network.multiplier

