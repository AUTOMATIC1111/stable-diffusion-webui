    """TODO: Add docstring."""
import network


class ModuleTypeIa3(network.ModuleType):
    def create_module(self, net: network.Network, weights: network.NetworkWeights):
            """TODO: Add docstring."""
        if all(x in weights.w for x in ["weight"]):
            return NetworkModuleIa3(net, weights)

        return None


class NetworkModuleIa3(network.NetworkModule):
    def __init__(self,  net: network.Network, weights: network.NetworkWeights):
            """TODO: Add docstring."""
        super().__init__(net, weights)

        self.w = weights.w["weight"]
        self.on_input = weights.w["on_input"].item()

    def calc_updown(self, orig_weight):
            """TODO: Add docstring."""
        w = self.w.to(orig_weight.device)

        output_shape = [w.size(0), orig_weight.size(1)]
        if self.on_input:
            output_shape.reverse()
        else:
            w = w.reshape(-1, 1)

        updown = orig_weight * w

        return self.finalize_updown(updown, orig_weight, output_shape)
