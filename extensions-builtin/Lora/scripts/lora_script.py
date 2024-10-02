import re
import torch

import gradio as gr
from fastapi import FastAPI

import network
import networks
import lora  # noqa:F401
import lora_patches
import extra_networks_lora
import ui_extra_networks_lora
from modules import script_callbacks, ui_extra_networks, extra_networks, shared, scripts, devices


def unload():
    networks.originals.undo()


def before_ui():
    ui_extra_networks.register_page(ui_extra_networks_lora.ExtraNetworksPageLora())

    networks.extra_network_lora = extra_networks_lora.ExtraNetworkLora()
    extra_networks.register_extra_network(networks.extra_network_lora)
    extra_networks.register_extra_network_alias(networks.extra_network_lora, "lyco")


networks.originals = lora_patches.LoraPatches()

script_callbacks.on_model_loaded(networks.assign_network_names_to_compvis_modules)
script_callbacks.on_script_unloaded(unload)
script_callbacks.on_before_ui(before_ui)
script_callbacks.on_infotext_pasted(networks.infotext_pasted)


shared.options_templates.update(shared.options_section(('extra_networks', "Extra Networks"), {
    "sd_lora": shared.OptionInfo("None", "Add network to prompt", gr.Dropdown, lambda: {"choices": ["None", *networks.available_networks]}, refresh=networks.list_available_networks),
    "lora_preferred_name": shared.OptionInfo("Alias from file", "When adding to prompt, refer to Lora by", gr.Radio, {"choices": ["Alias from file", "Filename"]}),
    "lora_add_hashes_to_infotext": shared.OptionInfo(True, "Add Lora hashes to infotext"),
    "lora_bundled_ti_to_infotext": shared.OptionInfo(True, "Add Lora name as TI hashes for bundled Textual Inversion").info('"Add Textual Inversion hashes to infotext" needs to be enabled'),
    "lora_show_all": shared.OptionInfo(False, "Always show all networks on the Lora page").info("otherwise, those detected as for incompatible version of Stable Diffusion will be hidden"),
    "lora_hide_unknown_for_versions": shared.OptionInfo([], "Hide networks of unknown versions for model versions", gr.CheckboxGroup, {"choices": ["SD1", "SD2", "SDXL"]}),
    "lora_in_memory_limit": shared.OptionInfo(0, "Number of Lora networks to keep cached in memory", gr.Number, {"precision": 0}),
    "lora_not_found_warning_console": shared.OptionInfo(False, "Lora not found warning in console"),
    "lora_not_found_gradio_warning": shared.OptionInfo(False, "Lora not found warning popup in webui"),
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


class ScriptLora(scripts.Script):
    name = "Lora"

    def title(self):
        return self.name

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def after_extra_networks_activate(self, p, *args, **kwargs):
        # check modules and setup org_dtype
        modules = []
        if shared.sd_model.is_sdxl:
            for _i, embedder in enumerate(shared.sd_model.conditioner.embedders):
                if not hasattr(embedder, 'wrapped'):
                    continue

                for _name, module in embedder.wrapped.named_modules():
                    if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear, torch.nn.GroupNorm, torch.nn.LayerNorm, torch.nn.MultiheadAttention)):
                        if hasattr(module, 'weight'):
                            modules.append(module)
                        elif isinstance(module, torch.nn.MultiheadAttention):
                            modules.append(module)

        else:
            cond_stage_model = getattr(shared.sd_model.cond_stage_model, 'wrapped', shared.sd_model.cond_stage_model)

            for _name, module in cond_stage_model.named_modules():
                if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear, torch.nn.GroupNorm, torch.nn.LayerNorm, torch.nn.MultiheadAttention)):
                    if hasattr(module, 'weight'):
                        modules.append(module)
                    elif isinstance(module, torch.nn.MultiheadAttention):
                        modules.append(module)

        for _name, module in shared.sd_model.model.named_modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear, torch.nn.GroupNorm, torch.nn.LayerNorm, torch.nn.MultiheadAttention)):
                if hasattr(module, 'weight'):
                    modules.append(module)
                elif isinstance(module, torch.nn.MultiheadAttention):
                    modules.append(module)

        print("Total lora modules after_extra_networks_activate() =", len(modules))

        target_dtype = devices.dtype_inference
        for module in modules:
            if isinstance(module, torch.nn.MultiheadAttention):
                org_dtype = torch.float32
            else:
                org_dtype = None
                for _name, param in module.named_parameters():
                    if param.dtype != target_dtype:
                        org_dtype = param.dtype
                        break

            # set org_dtype
            module.org_dtype = org_dtype


script_callbacks.on_infotext_pasted(infotext_pasted)

shared.opts.onchange("lora_in_memory_limit", networks.purge_networks_from_memory)
