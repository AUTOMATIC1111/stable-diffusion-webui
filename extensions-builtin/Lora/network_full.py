import network


class ModuleTypeFull(network.ModuleType):
    def create_module(self, net: network.Network, weights: network.NetworkWeights):
        if all(x in weights.w for x in ["diff"]):
            return NetworkModuleFull(net, weights)

        return None


class NetworkModuleFull(network.NetworkModule):
    def __init__(self,  net: network.Network, weights: network.NetworkWeights):
        super().__init__(net, weights)

        self.weight = weights.w.get("diff")
        self.ex_bias = weights.w.get("diff_b")

    def calc_updown(self, orig_weight):
        output_shape = self.weight.shape
        updown = self.weight.to(orig_weight.device, dtype=orig_weight.dtype)
        if self.ex_bias is not None:
            ex_bias = self.ex_bias.to(orig_weight.device, dtype=orig_weight.dtype)
        else:
            ex_bias = None

        return self.finalize_updown(updown, orig_weight, output_shape, ex_bias)
