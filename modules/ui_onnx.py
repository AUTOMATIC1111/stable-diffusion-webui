import os
import json
from typing import Dict, List, Union
import gradio as gr


def get_recursively(d: Union[Dict, List], *args):
    if len(args) == 0:
        return d
    return get_recursively(d.get(args[0]), *args[1:])


def create_ui():
    from modules.ui_components import DropdownMulti
    from modules.shared import log, opts, cmd_opts
    from modules.paths import sd_configs_path
    from modules.onnx_ep import ExecutionProvider, install_execution_provider
    from modules.olive import config as olive_config

    with gr.Blocks(analytics_enabled=False) as ui:
        with gr.Row():
            with gr.Tabs(elem_id="tabs_onnx"):
                with gr.TabItem("Manage execution providers", id="onnxep"):
                    gr.Markdown("Uninstall existing execution provider and install another one.")

                    choices = []

                    for ep in ExecutionProvider:
                        choices.append(ep)

                    ep_default = None
                    if cmd_opts.use_directml:
                        ep_default = ExecutionProvider.DirectML
                    elif cmd_opts.use_cuda:
                        ep_default = ExecutionProvider.CUDA
                    elif cmd_opts.use_rocm:
                        ep_default = ExecutionProvider.ROCm
                    elif cmd_opts.use_openvino:
                        ep_default = ExecutionProvider.OpenVINO

                    ep_checkbox = gr.Radio(label="Execution provider", value=ep_default, choices=choices)
                    ep_install = gr.Button(value="Install")
                    gr.Markdown("**Warning! If you are trying to reinstall, it may not work due to permission issue.**")

                    ep_install.click(fn=install_execution_provider, inputs=ep_checkbox)

                with gr.TabItem("Override VAE", id="force_vae"):
                    gr.Markdown("Ignore baked-in vae and replace it with what you want.")

                    onnx_vae_id = gr.Textbox(label="Huggingface VAE ID", info="Leave empty for default (baked-in vae).", value="")
                    onnx_vae_subfolder = gr.Textbox(label="VAE subfolder", info="Leave empty for root. Default: vae", value="vae")
                    onnx_vae_apply_button = gr.Button(value="Apply")

                    def onnx_vae_apply(id: str, subfolder: str):
                        olive_config.vae_id = id
                        olive_config.vae_subfolder = subfolder
                        if id == "":
                            log.info("ONNX: VAE override unset.")
                            del olive_config.vae_id
                            olive_config.vae_subfolder = "vae"
                        else:
                            log.info(f"ONNX: VAE override set: id={id}, subfolder={subfolder}")

                    onnx_vae_apply_button.click(fn=onnx_vae_apply, inputs=[onnx_vae_id, onnx_vae_subfolder,])

            if opts.cuda_compile_backend == "olive-ai":
                from olive.passes import REGISTRY

                with gr.Tabs(elem_id="tabs_olive"):
                    with gr.TabItem("Customize pass flow", id="pass_flow"):
                        with gr.Tabs(elem_id="tabs_model_type"):
                            with gr.TabItem("Stable Diffusion", id="sd"):
                                sd_config_path = os.path.join(sd_configs_path, "olive", "sd")
                                sd_submodels = os.listdir(sd_config_path)
                                sd_configs: Dict[str, Dict] = {}

                                with gr.Tabs(elem_id="tabs_sd_submodel"):
                                    def sd_create_change_listener(*args):
                                        def listener(v: Dict):
                                            get_recursively(sd_configs, *args[:-1])[args[-1]] = v
                                        return listener

                                    for submodel in sd_submodels:
                                        config: Dict = None

                                        with open(os.path.join(sd_config_path, submodel), "r") as file:
                                            config = json.load(file)
                                        sd_configs[submodel] = config

                                        submodel_name = submodel[:-5]
                                        with gr.TabItem(submodel_name, id=f"sd_{submodel_name}"):
                                            pass_flows = DropdownMulti(label="Pass flow", value=sd_configs[submodel]["pass_flows"][0], choices=sd_configs[submodel]["passes"].keys())
                                            pass_flows.change(fn=sd_create_change_listener(submodel, "pass_flows", 0), inputs=pass_flows)

                                            with gr.Tabs(elem_id=f"tabs_sd_{submodel_name}_pass"):
                                                for k in sd_configs[submodel]["passes"]:
                                                    with gr.TabItem(k, id=f"sd_{submodel_name}_pass_{k}"):
                                                        pass_type = gr.Dropdown(label="Type", value=sd_configs[submodel]["passes"][k]["type"], choices=(x.__name__ for x in tuple(REGISTRY.values())))

                                                        pass_type.change(fn=sd_create_change_listener(submodel, "passes", k, "type"), inputs=pass_type)

                                def sd_save():
                                    for k, v in sd_configs.items():
                                        with open(os.path.join(sd_config_path, k), "w") as file:
                                            json.dump(v, file)
                                    log.info("Olive: config for SD was saved.")

                                sd_save_button = gr.Button(value="Save")
                                sd_save_button.click(fn=sd_save)

                            with gr.TabItem("Stable Diffusion XL", id="sdxl"):
                                sdxl_config_path = os.path.join(sd_configs_path, "olive", "sdxl")
                                sdxl_submodels = os.listdir(sdxl_config_path)
                                sdxl_configs: Dict[str, Dict] = {}

                                with gr.Tabs(elem_id="tabs_sdxl_submodel"):
                                    def sdxl_create_change_listener(*args):
                                        def listener(v: Dict):
                                            get_recursively(sdxl_configs, *args[:-1])[args[-1]] = v
                                        return listener

                                    for submodel in sdxl_submodels:
                                        config: Dict = None

                                        with open(os.path.join(sdxl_config_path, submodel), "r") as file:
                                            config = json.load(file)
                                        sdxl_configs[submodel] = config

                                        submodel_name = submodel[:-5]
                                        with gr.TabItem(submodel_name, id=f"sdxl_{submodel_name}"):
                                            pass_flows = DropdownMulti(label="Pass flow", value=sdxl_configs[submodel]["pass_flows"][0], choices=sdxl_configs[submodel]["passes"].keys())
                                            pass_flows.change(fn=sdxl_create_change_listener(submodel, "pass_flows", 0), inputs=pass_flows)

                                            with gr.Tabs(elem_id=f"tabs_sdxl_{submodel_name}_pass"):
                                                for k in sdxl_configs[submodel]["passes"]:
                                                    with gr.TabItem(k, id=f"sdxl_{submodel_name}_pass_{k}"):
                                                        pass_type = gr.Dropdown(label="Type", value=sdxl_configs[submodel]["passes"][k]["type"], choices=(x.__name__ for x in tuple(REGISTRY.values())))

                                                        pass_type.change(fn=sdxl_create_change_listener(submodel, "passes", k, "type"), inputs=pass_type)

                                def sdxl_save():
                                    for k, v in sdxl_configs.items():
                                        with open(os.path.join(sdxl_config_path, k), "w") as file:
                                            json.dump(v, file)
                                    log.info("Olive: config for SDXL was saved.")

                                sdxl_save_button = gr.Button(value="Save")
                                sdxl_save_button.click(fn=sdxl_save)
    return ui
