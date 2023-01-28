from modules import extra_networks
from modules.hypernetworks import hypernetwork


class ExtraNetworkHypernet(extra_networks.ExtraNetwork):
    def __init__(self):
        super().__init__('hypernet')

    def activate(self, p, params_list):
        names = []
        multipliers = []
        for params in params_list:
            assert len(params.items) > 0

            names.append(params.items[0])
            multipliers.append(float(params.items[1]) if len(params.items) > 1 else 1.0)

        hypernetwork.load_hypernetworks(names, multipliers)

    def deactivate(self, p):
        pass
