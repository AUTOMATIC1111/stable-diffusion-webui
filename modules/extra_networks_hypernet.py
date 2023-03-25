from modules import extra_networks, shared, extra_networks, hashes
from modules.hypernetworks import hypernetwork


class ExtraNetworkHypernet(extra_networks.ExtraNetwork):
    def __init__(self):
        super().__init__('hypernet')

    def activate(self, p, params_list):
        additional = shared.opts.sd_hypernetwork

        if additional != "" and additional in shared.hypernetworks and len([x for x in params_list if x.items[0] == additional]) == 0:
            p.all_prompts = [x + f"<hypernet:{additional}:{shared.opts.extra_networks_default_multiplier}>" for x in p.all_prompts]
            params_list.append(extra_networks.ExtraNetworkParams(items=[additional, shared.opts.extra_networks_default_multiplier]))

        names = []
        multipliers = []
        for params in params_list:
            assert len(params.items) > 0

            names.append(params.items[0])
            multipliers.append(float(params.items[1]) if len(params.items) > 1 else 1.0)

        hypernetwork.load_hypernetworks(names, multipliers)

    def deactivate(self, p):
        pass

    def infotext_params(self, p, params_list):
        hypernet_hashes = []

        for params in params_list:
            assert len(params.items) > 0
            name = params.items[0]

            filename = shared.hypernetworks.get(name, None)
            if filename:
                sha256 = hashes.sha256(filename, f'hypernet/{name}')
                if sha256:
                    shorthash = sha256[0:10]
                    hypernet_hashes.append(f"{name}:{shorthash}")

        return {"Hypernet hashes": ", ".join(hypernet_hashes)}
