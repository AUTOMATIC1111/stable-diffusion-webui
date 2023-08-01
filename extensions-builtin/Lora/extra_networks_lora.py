from modules import extra_networks, shared
import networks


class ExtraNetworkLora(extra_networks.ExtraNetwork):
    def __init__(self):
        super().__init__('lora')

    def activate(self, p, params_list):
        additional = shared.opts.sd_lora

        if additional != "None" and additional in networks.available_networks and not any(x for x in params_list if x.items[0] == additional):
            p.all_prompts = [x + f"<lora:{additional}:{shared.opts.extra_networks_default_multiplier}>" for x in p.all_prompts]
            params_list.append(extra_networks.ExtraNetworkParams(items=[additional, shared.opts.extra_networks_default_multiplier]))

        names = []
        te_multipliers = []
        unet_multipliers = []
        dyn_dims = []
        for params in params_list:
            assert params.items

            names.append(params.positional[0])

            te_multiplier = float(params.positional[1]) if len(params.positional) > 1 else 1.0
            te_multiplier = float(params.named.get("te", te_multiplier))

            unet_multiplier = float(params.positional[2]) if len(params.positional) > 2 else te_multiplier
            unet_multiplier = float(params.named.get("unet", unet_multiplier))

            dyn_dim = int(params.positional[3]) if len(params.positional) > 3 else None
            dyn_dim = int(params.named["dyn"]) if "dyn" in params.named else dyn_dim

            te_multipliers.append(te_multiplier)
            unet_multipliers.append(unet_multiplier)
            dyn_dims.append(dyn_dim)

        networks.load_networks(names, te_multipliers, unet_multipliers, dyn_dims)

        if shared.opts.lora_add_hashes_to_infotext:
            network_hashes = []
            for item in networks.loaded_networks:
                shorthash = item.network_on_disk.shorthash
                if not shorthash:
                    continue

                alias = item.mentioned_name
                if not alias:
                    continue

                alias = alias.replace(":", "").replace(",", "")

                network_hashes.append(f"{alias}: {shorthash}")

            if network_hashes:
                p.extra_generation_params["Lora hashes"] = ", ".join(network_hashes)

    def deactivate(self, p):
        pass
