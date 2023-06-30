from modules import extra_networks, shared
from modules.hypernetworks import hypernetwork


class ExtraNetworkHypernet(extra_networks.ExtraNetwork):
    def __init__(self):
        super().__init__('hypernet')

    def activate(self, p, params_list):
        additional = shared.opts.sd_hypernetwork

        if additional != "None" and additional in shared.hypernetworks and not any(x for x in params_list if x.items[0] == additional):
            hypernet_prompt_text = f"<hypernet:{additional}:{shared.opts.extra_networks_default_multiplier}>"
            p.all_prompts = [f"{prompt}{hypernet_prompt_text}" for prompt in p.all_prompts]
            params_list.append(extra_networks.ExtraNetworkParams(items=[additional, shared.opts.extra_networks_default_multiplier]))

        names = []
        multipliers = []
        for params in params_list:
            assert params.items

            names.append(params.items[0])
            multipliers.append(float(params.items[1]) if len(params.items) > 1 else 1.0)

        hypernetwork.load_hypernetworks(names, multipliers)

    def deactivate(self, p):
        pass
