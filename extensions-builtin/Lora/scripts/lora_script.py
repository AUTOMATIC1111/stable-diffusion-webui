import re
import networks
import lora # noqa:F401 # pylint: disable=unused-import
from network import NetworkOnDisk
from ui_extra_networks_lora import ExtraNetworksPageLora
from extra_networks_lora import ExtraNetworkLora
from modules import script_callbacks, ui_extra_networks, extra_networks


re_lora = re.compile("<lora:([^:]+):")


def before_ui():
    ui_extra_networks.register_page(ExtraNetworksPageLora())
    networks.extra_network_lora = ExtraNetworkLora()
    extra_networks.register_extra_network(networks.extra_network_lora)


def create_lora_json(obj: NetworkOnDisk):
    return {
        "name": obj.name,
        "alias": obj.alias,
        "path": obj.filename,
        "metadata": obj.metadata,
    }


def api_networks(_, app):
    @app.get("/sdapi/v1/loras")
    async def get_loras():
        return [create_lora_json(obj) for obj in networks.available_networks.values()]

    @app.post("/sdapi/v1/refresh-loras")
    async def refresh_loras():
        return networks.list_available_networks()


def infotext_pasted(infotext, d): # pylint: disable=unused-argument
    hashes = d.get("Lora hashes", None)
    if hashes is None:
        return

    def network_replacement(m):
        alias = m.group(1)
        shorthash = hashes.get(alias)
        if shorthash is None:
            return m.group(0)
        network_on_disk = networks.available_network_hash_lookup.get(shorthash)
        if network_on_disk is None:
            return m.group(0)
        return f'<lora:{network_on_disk.get_alias()}:'

    hashes = [x.strip().split(':', 1) for x in hashes.split(",")]
    hashes = {x[0].strip().replace(",", ""): x[1].strip() for x in hashes}
    d["Prompt"] = re.sub(re_lora, network_replacement, d["Prompt"])


script_callbacks.on_app_started(api_networks)
script_callbacks.on_before_ui(before_ui)
script_callbacks.on_model_loaded(networks.assign_network_names_to_compvis_modules)
script_callbacks.on_infotext_pasted(networks.infotext_pasted)
script_callbacks.on_infotext_pasted(infotext_pasted)
