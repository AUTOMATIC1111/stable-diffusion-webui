from modules import extra_networks, shared
import lora


class ExtraNetworkLora(extra_networks.ExtraNetwork):
    def __init__(self):
        super().__init__('lora')

    def activate(self, p, params_list):
        additional = shared.opts.sd_lora

        if additional != "None" and additional in lora.available_loras and len([x for x in params_list if x.items[0] == additional]) == 0:
            p.all_prompts = [x + f"<lora:{additional}:{shared.opts.extra_networks_default_multiplier}>" for x in p.all_prompts]
            params_list.append(extra_networks.ExtraNetworkParams(items=[additional, shared.opts.extra_networks_default_multiplier]))

        names = []
        multipliers = []
        for params in params_list:
            assert len(params.items) > 0

            names.append(params.items[0])
            multipliers.append(float(params.items[1]) if len(params.items) > 1 else 1.0)

        lora.load_loras(names, multipliers)

        if shared.opts.lora_add_hashes_to_infotext:
            lora_hashes = []
            for item in lora.loaded_loras:
                shorthash = item.lora_on_disk.shorthash
                if not shorthash:
                    continue

                alias = item.mentioned_name
                if not alias:
                    continue

                alias = alias.replace(":", "").replace(",", "")

                lora_hashes.append(f"{alias}: {shorthash}")

            if lora_hashes:
                p.extra_generation_params["Lora hashes"] = ", ".join(lora_hashes)

    def deactivate(self, p):
        pass
