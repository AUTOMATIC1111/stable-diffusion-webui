import lyco_helpers
import network


class ModuleTypeHada(network.ModuleType):
    def create_module(self, net: network.Network, weights: network.NetworkWeights):
        if all(x in weights.w for x in ["hada_w1_a", "hada_w1_b", "hada_w2_a", "hada_w2_b"]):
            return NetworkModuleHada(net, weights)

        return None


class NetworkModuleHada(network.NetworkModule):
    def __init__(self,  net: network.Network, weights: network.NetworkWeights):
        super().__init__(net, weights)

        if hasattr(self.sd_module, 'weight'):
            self.shape = self.sd_module.weight.shape

        self.w1a = weights.w["hada_w1_a"]
        self.w1b = weights.w["hada_w1_b"]
        self.dim = self.w1b.shape[0]
        self.w2a = weights.w["hada_w2_a"]
        self.w2b = weights.w["hada_w2_b"]

        self.t1 = weights.w.get("hada_t1")
        self.t2 = weights.w.get("hada_t2")

    def calc_updown(self, orig_weight):
        w1a = self.w1a.to(orig_weight.device, dtype=orig_weight.dtype)
        w1b = self.w1b.to(orig_weight.device, dtype=orig_weight.dtype)
        w2a = self.w2a.to(orig_weight.device, dtype=orig_weight.dtype)
        w2b = self.w2b.to(orig_weight.device, dtype=orig_weight.dtype)

        output_shape = [w1a.size(0), w1b.size(1)]

        if self.t1 is not None:
            output_shape = [w1a.size(1), w1b.size(1)]
            t1 = self.t1.to(orig_weight.device, dtype=orig_weight.dtype)
            updown1 = lyco_helpers.make_weight_cp(t1, w1a, w1b)
            output_shape += t1.shape[2:]
        else:
            if len(w1b.shape) == 4:
                output_shape += w1b.shape[2:]
            updown1 = lyco_helpers.rebuild_conventional(w1a, w1b, output_shape)

        if self.t2 is not None:
            t2 = self.t2.to(orig_weight.device, dtype=orig_weight.dtype)
            updown2 = lyco_helpers.make_weight_cp(t2, w2a, w2b)
        else:
            updown2 = lyco_helpers.rebuild_conventional(w2a, w2b, output_shape)

        updown = updown1 * updown2

        return self.finalize_updown(updown, orig_weight, output_shape)
