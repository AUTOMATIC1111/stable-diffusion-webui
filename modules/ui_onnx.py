import os
import json
import shutil
from typing import Dict, List, Union
import gradio as gr


def get_recursively(d: Union[Dict, List], *args):
    if len(args) == 0:
        return d
    return get_recursively(d.get(args[0]), *args[1:])


def create_ui():
    from modules.ui_common import create_refresh_button
    from modules.ui_components import DropdownMulti
    from modules.shared import log, opts, cmd_opts, refresh_checkpoints
    from modules.sd_models import checkpoint_tiles, get_closet_checkpoint_match
    from modules.paths import sd_configs_path
    from modules.onnx_ep import ExecutionProvider, install_execution_provider
    from modules.onnx_utils import check_diffusers_cache
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
                import olive.passes as olive_passes
                from olive.hardware.accelerator import AcceleratorSpec, Device

                accelerator = AcceleratorSpec(accelerator_type=Device.GPU, execution_provider=opts.onnx_execution_provider)

                with gr.Tabs(elem_id="tabs_olive"):
                    with gr.TabItem("Manage cache", id="manage_cache"):
                        cache_state_dirname = gr.Textbox(value=None, visible=False)

                        with gr.Row():
                            model_dropdown = gr.Dropdown(label="Model", value="Please select model", choices=checkpoint_tiles())
                            create_refresh_button(model_dropdown, refresh_checkpoints, {}, "onnx_cache_refresh_diffusers_model")

                        with gr.Row():
                            def remove_cache_onnx_converted(dirname: str):
                                shutil.rmtree(os.path.join(opts.onnx_cached_models_path, dirname))
                                log.info(f"ONNX converted cache of '{dirname}' is removed.")

                            cache_onnx_converted = gr.Markdown("Please select model")
                            cache_remove_onnx_converted = gr.Button(value="Remove cache", visible=False)
                            cache_remove_onnx_converted.click(fn=remove_cache_onnx_converted, inputs=[cache_state_dirname,])

                        with gr.Column():
                            cache_optimized_selected = gr.Textbox(value=None, visible=False)

                            def select_cache_optimized(evt: gr.SelectData, data):
                                return ",".join(data[evt.index[0]])

                            def remove_cache_optimized(dirname: str, s: str):
                                if s == "":
                                    return
                                size = s.split(",")
                                shutil.rmtree(os.path.join(opts.onnx_cached_models_path, f"{dirname}-{size[0]}w-{size[1]}h"))
                                log.info(f"Olive processed cache of '{dirname}' is removed: width={size[0]}, height={size[1]}")

                            with gr.Row():
                                cache_list_optimized_headers = ["height", "width"]
                                cache_list_optimized_types = ["str", "str"]
                                cache_list_optimized = gr.Dataframe(None, label="Optimized caches", show_label=True, overflow_row_behaviour='paginate', interactive=False, max_rows=10, headers=cache_list_optimized_headers, datatype=cache_list_optimized_types, type="array")
                                cache_list_optimized.select(fn=select_cache_optimized, inputs=[cache_list_optimized,], outputs=[cache_optimized_selected,])

                            cache_remove_optimized = gr.Button(value="Remove selected cache", visible=False)
                            cache_remove_optimized.click(fn=remove_cache_optimized, inputs=[cache_state_dirname, cache_optimized_selected,])

                        def cache_update_menus(query: str):
                            checkpoint_info = get_closet_checkpoint_match(query)
                            if checkpoint_info is None:
                                log.error(f"Could not find checkpoint object for '{query}'.")
                                return
                            model_name = os.path.basename(os.path.dirname(os.path.dirname(checkpoint_info.path)) if check_diffusers_cache(checkpoint_info.path) else checkpoint_info.path)
                            caches = os.listdir(opts.onnx_cached_models_path)
                            onnx_converted = False
                            optimized_sizes = []
                            for cache in caches:
                                if cache == model_name:
                                    onnx_converted = True
                                elif model_name in cache:
                                    try:
                                        splitted = cache.split("-")
                                        height = splitted[-1][:-1]
                                        width = splitted[-2][:-1]
                                        optimized_sizes.append((width, height,))
                                    except Exception:
                                        pass
                            return (
                                model_name,
                                cache_onnx_converted.update(value="ONNX model cache of this model exists." if onnx_converted else "ONNX model cache of this model does not exist."),
                                cache_remove_onnx_converted.update(visible=onnx_converted),
                                None if len(optimized_sizes) == 0 else optimized_sizes,
                                cache_remove_optimized.update(visible=True),
                            )

                        model_dropdown.change(fn=cache_update_menus, inputs=[model_dropdown,], outputs=[
                            cache_state_dirname,
                            cache_onnx_converted, cache_remove_onnx_converted,
                            cache_list_optimized, cache_remove_optimized,
                        ])

                    with gr.TabItem("Customize pass flow", id="pass_flow"):
                        with gr.Tabs(elem_id="tabs_model_type"):
                            with gr.TabItem("Stable Diffusion", id="sd"):
                                sd_config_path = os.path.join(sd_configs_path, "olive", "sd")
                                sd_submodels = os.listdir(sd_config_path)
                                sd_configs: Dict[str, Dict[str, Dict[str, Dict]]] = {}
                                sd_pass_config_components: Dict[str, Dict[str, Dict]] = {}

                                with gr.Tabs(elem_id="tabs_sd_submodel"):
                                    def sd_create_change_listener(*args):
                                        def listener(v: Dict):
                                            get_recursively(sd_configs, *args[:-1])[args[-1]] = v
                                        return listener

                                    for submodel in sd_submodels:
                                        config: Dict = None

                                        sd_pass_config_components[submodel] = {}

                                        with open(os.path.join(sd_config_path, submodel), "r") as file:
                                            config = json.load(file)
                                        sd_configs[submodel] = config

                                        submodel_name = submodel[:-5]
                                        with gr.TabItem(submodel_name, id=f"sd_{submodel_name}"):
                                            pass_flows = DropdownMulti(label="Pass flow", value=sd_configs[submodel]["pass_flows"][0], choices=sd_configs[submodel]["passes"].keys())
                                            pass_flows.change(fn=sd_create_change_listener(submodel, "pass_flows", 0), inputs=pass_flows)

                                            with gr.Tabs(elem_id=f"tabs_sd_{submodel_name}_pass"):
                                                for pass_name in sd_configs[submodel]["passes"]:
                                                    sd_pass_config_components[submodel][pass_name] = {}

                                                    with gr.TabItem(pass_name, id=f"sd_{submodel_name}_pass_{pass_name}"):
                                                        config_dict = sd_configs[submodel]["passes"][pass_name]

                                                        pass_type = gr.Dropdown(label="Type", value=config_dict["type"], choices=(x.__name__ for x in tuple(olive_passes.REGISTRY.values())))


                                                        def create_pass_config_change_listener(submodel, pass_name, config_key):
                                                            def listener(value):
                                                                sd_configs[submodel]["passes"][pass_name]["config"][config_key] = value
                                                            return listener


                                                        for config_key, v in getattr(olive_passes, config_dict["type"], olive_passes.Pass)._default_config(accelerator).items():
                                                            component = None

                                                            if v.type_ == bool:
                                                                component = gr.Checkbox
                                                            elif v.type_ == str:
                                                                component = gr.Textbox
                                                            elif v.type_ == int:
                                                                component = gr.Number

                                                            if component is not None:
                                                                component = component(value=config_dict["config"][config_key] if config_key in config_dict["config"] else v.default_value, label=config_key)
                                                                sd_pass_config_components[submodel][pass_name][config_key] = component
                                                                component.change(fn=create_pass_config_change_listener(submodel, pass_name, config_key), inputs=component)

                                                        pass_type.change(fn=sd_create_change_listener(submodel, "passes", config_key, "type"), inputs=pass_type)

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
                                sdxl_configs: Dict[str, Dict[str, Dict[str, Dict]]] = {}
                                sdxl_pass_config_components: Dict[str, Dict[str, Dict]] = {}

                                with gr.Tabs(elem_id="tabs_sdxl_submodel"):
                                    def sdxl_create_change_listener(*args):
                                        def listener(v: Dict):
                                            get_recursively(sdxl_configs, *args[:-1])[args[-1]] = v
                                        return listener

                                    for submodel in sdxl_submodels:
                                        config: Dict = None

                                        sdxl_pass_config_components[submodel] = {}

                                        with open(os.path.join(sdxl_config_path, submodel), "r") as file:
                                            config = json.load(file)
                                        sdxl_configs[submodel] = config

                                        submodel_name = submodel[:-5]
                                        with gr.TabItem(submodel_name, id=f"sdxl_{submodel_name}"):
                                            pass_flows = DropdownMulti(label="Pass flow", value=sdxl_configs[submodel]["pass_flows"][0], choices=sdxl_configs[submodel]["passes"].keys())
                                            pass_flows.change(fn=sdxl_create_change_listener(submodel, "pass_flows", 0), inputs=pass_flows)

                                            with gr.Tabs(elem_id=f"tabs_sdxl_{submodel_name}_pass"):
                                                for pass_name in sdxl_configs[submodel]["passes"]:
                                                    sdxl_pass_config_components[submodel][pass_name] = {}

                                                    with gr.TabItem(pass_name, id=f"sdxl_{submodel_name}_pass_{pass_name}"):
                                                        config_dict = sdxl_configs[submodel]["passes"][pass_name]

                                                        pass_type = gr.Dropdown(label="Type", value=sdxl_configs[submodel]["passes"][pass_name]["type"], choices=(x.__name__ for x in tuple(olive_passes.REGISTRY.values())))


                                                        def create_pass_config_change_listener(submodel, pass_name, config_key):
                                                            def listener(value):
                                                                sdxl_configs[submodel]["passes"][pass_name]["config"][config_key] = value
                                                            return listener


                                                        for config_key, v in getattr(olive_passes, config_dict["type"], olive_passes.Pass)._default_config(accelerator).items():
                                                            component = None

                                                            if v.type_ == bool:
                                                                component = gr.Checkbox
                                                            elif v.type_ == str:
                                                                component = gr.Textbox
                                                            elif v.type_ == int:
                                                                component = gr.Number

                                                            if component is not None:
                                                                component = component(value=config_dict["config"][config_key] if config_key in config_dict["config"] else v.default_value, label=config_key)
                                                                sdxl_pass_config_components[submodel][pass_name][config_key] = component
                                                                component.change(fn=create_pass_config_change_listener(submodel, pass_name, config_key), inputs=component)

                                                        pass_type.change(fn=sdxl_create_change_listener(submodel, "passes", pass_name, "type"), inputs=pass_type)

                                def sdxl_save():
                                    for k, v in sdxl_configs.items():
                                        with open(os.path.join(sdxl_config_path, k), "w") as file:
                                            json.dump(v, file)
                                    log.info("Olive: config for SDXL was saved.")

                                sdxl_save_button = gr.Button(value="Save")
                                sdxl_save_button.click(fn=sdxl_save)
    return ui
