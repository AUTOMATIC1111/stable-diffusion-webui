import re

import torch
import gradio as gr
from fastapi import FastAPI

import network
import networks
import lora  # noqa:F401
import extra_networks_lora
import ui_extra_networks_lora
from modules import script_callbacks, ui_extra_networks, extra_networks, shared

def unload():
    torch.nn.Linear.forward = torch.nn.Linear_forward_before_network
    torch.nn.Linear._load_from_state_dict = torch.nn.Linear_load_state_dict_before_network
    torch.nn.Conv2d.forward = torch.nn.Conv2d_forward_before_network
    torch.nn.Conv2d._load_from_state_dict = torch.nn.Conv2d_load_state_dict_before_network
    torch.nn.MultiheadAttention.forward = torch.nn.MultiheadAttention_forward_before_network
    torch.nn.MultiheadAttention._load_from_state_dict = torch.nn.MultiheadAttention_load_state_dict_before_network


def before_ui():
    ui_extra_networks.register_page(ui_extra_networks_lora.ExtraNetworksPageLora())

    extra_network = extra_networks_lora.ExtraNetworkLora()
    extra_networks.register_extra_network(extra_network)
    extra_networks.register_extra_network_alias(extra_network, "lyco")


if not hasattr(torch.nn, 'Linear_forward_before_network'):
    torch.nn.Linear_forward_before_network = torch.nn.Linear.forward

if not hasattr(torch.nn, 'Linear_load_state_dict_before_network'):
    torch.nn.Linear_load_state_dict_before_network = torch.nn.Linear._load_from_state_dict

if not hasattr(torch.nn, 'Conv2d_forward_before_network'):
    torch.nn.Conv2d_forward_before_network = torch.nn.Conv2d.forward

if not hasattr(torch.nn, 'Conv2d_load_state_dict_before_network'):
    torch.nn.Conv2d_load_state_dict_before_network = torch.nn.Conv2d._load_from_state_dict

if not hasattr(torch.nn, 'MultiheadAttention_forward_before_network'):
    torch.nn.MultiheadAttention_forward_before_network = torch.nn.MultiheadAttention.forward

if not hasattr(torch.nn, 'MultiheadAttention_load_state_dict_before_network'):
    torch.nn.MultiheadAttention_load_state_dict_before_network = torch.nn.MultiheadAttention._load_from_state_dict

torch.nn.Linear.forward = networks.network_Linear_forward
torch.nn.Linear._load_from_state_dict = networks.network_Linear_load_state_dict
torch.nn.Conv2d.forward = networks.network_Conv2d_forward
torch.nn.Conv2d._load_from_state_dict = networks.network_Conv2d_load_state_dict
torch.nn.MultiheadAttention.forward = networks.network_MultiheadAttention_forward
torch.nn.MultiheadAttention._load_from_state_dict = networks.network_MultiheadAttention_load_state_dict

script_callbacks.on_model_loaded(networks.assign_network_names_to_compvis_modules)
script_callbacks.on_script_unloaded(unload)
script_callbacks.on_before_ui(before_ui)
script_callbacks.on_infotext_pasted(networks.infotext_pasted)


shared.options_templates.update(shared.options_section(('extra_networks', "Extra Networks"), {
    "sd_lora": shared.OptionInfo("None", "Add network to prompt", gr.Dropdown, lambda: {"choices": ["None", *networks.available_networks]}, refresh=networks.list_available_networks),
    "lora_preferred_name": shared.OptionInfo("Alias from file", "When adding to prompt, refer to Lora by", gr.Radio, {"choices": ["Alias from file", "Filename"]}),
    "lora_add_hashes_to_infotext": shared.OptionInfo(True, "Add Lora hashes to infotext"),
    "lora_show_all": shared.OptionInfo(False, "Always show all networks on the Lora page").info("otherwise, those detected as for incompatible version of Stable Diffusion will be hidden"),
    "lora_hide_unknown_for_versions": shared.OptionInfo([], "Hide networks of unknown versions for model versions", gr.CheckboxGroup, {"choices": ["SD1", "SD2", "SDXL"]}),
}))


shared.options_templates.update(shared.options_section(('compatibility', "Compatibility"), {
    "lora_functional": shared.OptionInfo(False, "Lora/Networks: use old method that takes longer when you have multiple Loras active and produces same results as kohya-ss/sd-webui-additional-networks extension"),
}))


def create_lora_json(obj: network.NetworkOnDisk):
    return {
        "name": obj.name,
        "alias": obj.alias,
        "path": obj.filename,
        "metadata": obj.metadata,
    }


def api_networks(_: gr.Blocks, app: FastAPI):
    @app.get("/sdapi/v1/loras")
    async def get_loras():
        return [create_lora_json(obj) for obj in networks.available_networks.values()]

    @app.post("/sdapi/v1/refresh-loras")
    async def refresh_loras():
        return networks.list_available_networks()


script_callbacks.on_app_started(api_networks)

re_lora = re.compile("<lora:([^:]+):")


def infotext_pasted(infotext, d):
    hashes = d.get("Lora hashes")
    if not hashes:
        return

    hashes = [x.strip().split(':', 1) for x in hashes.split(",")]
    hashes = {x[0].strip().replace(",", ""): x[1].strip() for x in hashes}

    def network_replacement(m):
        alias = m.group(1)
        shorthash = hashes.get(alias)
        if shorthash is None:
            return m.group(0)

        network_on_disk = networks.available_network_hash_lookup.get(shorthash)
        if network_on_disk is None:
            return m.group(0)

        return f'<lora:{network_on_disk.get_alias()}:'

    d["Prompt"] = re.sub(re_lora, network_replacement, d["Prompt"])


script_callbacks.on_infotext_pasted(infotext_pasted)
