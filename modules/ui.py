import os
import mimetypes
import gradio as gr
import gradio.routes
import gradio.utils
from modules.call_queue import wrap_gradio_call
from modules import timer, gr_hijack, shared, theme, sd_models, script_callbacks, modelloader, ui_common, ui_loadsave, ui_symbols, ui_javascript, generation_parameters_copypaste, call_queue
from modules.paths import script_path, data_path # pylint: disable=unused-import
from modules.dml import directml_override_opts
from modules.onnx_impl import install_olive
import modules.scripts
import modules.errors


modules.errors.install()
mimetypes.init()
mimetypes.add_type('application/javascript', '.js')
mimetypes.add_type('image/webp', '.webp')
log = shared.log
opts = shared.opts
cmd_opts = shared.cmd_opts
ui_system_tabs = None
switch_values_symbol = ui_symbols.switch
detect_image_size_symbol = ui_symbols.detect
paste_symbol = ui_symbols.paste
clear_prompt_symbol = ui_symbols.clear
restore_progress_symbol = ui_symbols.apply
folder_symbol = ui_symbols.folder
extra_networks_symbol = ui_symbols.networks
apply_style_symbol = ui_symbols.apply
save_style_symbol = ui_symbols.save
paste_function = None
wrap_queued_call = call_queue.wrap_queued_call
gr_hijack.init()


if not cmd_opts.share and not cmd_opts.listen:
    # fix gradio phoning home
    gradio.utils.version_check = lambda: None
    gradio.utils.get_local_ip_address = lambda: '127.0.0.1'


def gr_show(visible=True):
    return {"visible": visible, "__type__": "update"}


def create_output_panel(tabname, outdir): # pylint: disable=unused-argument # outdir is used by extensions
    a, b, c, _d, e = ui_common.create_output_panel(tabname)
    return a, b, c, e


def plaintext_to_html(text): # may be referenced by extensions
    return ui_common.plaintext_to_html(text)


def infotext_to_html(text): # may be referenced by extensions
    return ui_common.infotext_to_html(text)


def send_gradio_gallery_to_image(x):
    if len(x) == 0:
        return None
    return generation_parameters_copypaste.image_from_url_text(x[0])


def create_refresh_button(refresh_component, refresh_method, refreshed_args, elem_id):
    return ui_common.create_refresh_button(refresh_component, refresh_method, refreshed_args, elem_id)


def connect_clear_prompt(button): # pylint: disable=unused-argument
    pass


def setup_progressbar(*args, **kwargs): # pylint: disable=unused-argument
    pass


def apply_setting(key, value):
    if value is None:
        return gr.update()
    if shared.cmd_opts.freeze:
        return gr.update()
    # dont allow model to be swapped when model hash exists in prompt
    if key == "sd_model_checkpoint" and opts.disable_weights_auto_swap:
        return gr.update()
    if key == "sd_model_checkpoint":
        ckpt_info = sd_models.get_closet_checkpoint_match(value)
        if ckpt_info is not None:
            value = ckpt_info.title
        else:
            return gr.update()
    comp_args = opts.data_labels[key].component_args
    if comp_args and isinstance(comp_args, dict) and comp_args.get('visible') is False:
        return gr.update()
    valtype = type(opts.data_labels[key].default)
    oldval = opts.data.get(key, None)
    opts.data[key] = valtype(value) if valtype != type(None) else value
    if oldval != value and opts.data_labels[key].onchange is not None:
        opts.data_labels[key].onchange()
    opts.save(shared.config_filename)
    return getattr(opts, key)


def get_value_for_setting(key):
    value = getattr(opts, key)
    info = opts.data_labels[key]
    args = info.component_args() if callable(info.component_args) else info.component_args or {}
    args = {k: v for k, v in args.items() if k not in {'precision', 'multiselect'}}
    return gr.update(value=value, **args)


def ordered_ui_categories():
    return ['dimensions', 'sampler', 'seed', 'denoising', 'cfg', 'checkboxes', 'accordions', 'override_settings', 'scripts'] # a1111 compatibility item, not implemented


def create_ui(startup_timer = None):
    if startup_timer is None:
        timer.startup = timer.Timer()
    ui_javascript.reload_javascript()
    generation_parameters_copypaste.reset()

    with gr.Blocks(analytics_enabled=False) as txt2img_interface:
        from modules import ui_txt2img
        ui_txt2img.create_ui()
        timer.startup.record("ui-txt2img")

    with gr.Blocks(analytics_enabled=False) as img2img_interface:
        from modules import ui_img2img
        ui_img2img.create_ui()
        timer.startup.record("ui-img2img")

    modules.scripts.scripts_current = None

    with gr.Blocks(analytics_enabled=False) as control_interface:
        if shared.backend == shared.Backend.DIFFUSERS:
            from modules import ui_control
            ui_control.create_ui()
            timer.startup.record("ui-control")

    with gr.Blocks(analytics_enabled=False) as extras_interface:
        from modules import ui_postprocessing
        ui_postprocessing.create_ui()
        timer.startup.record("ui-extras")

    with gr.Blocks(analytics_enabled=False) as train_interface:
        from modules import ui_train
        ui_train.create_ui()
        timer.startup.record("ui-train")

    with gr.Blocks(analytics_enabled=False) as models_interface:
        from modules import ui_models
        ui_models.create_ui()
        timer.startup.record("ui-models")

    with gr.Blocks(analytics_enabled=False) as interrogate_interface:
        from modules import ui_interrogate
        ui_interrogate.create_ui()
        timer.startup.record("ui-interrogate")


    def create_setting_component(key, is_quicksettings=False):
        def fun():
            return opts.data[key] if key in opts.data else opts.data_labels[key].default

        info = opts.data_labels[key]
        t = type(info.default)
        args = (info.component_args() if callable(info.component_args) else info.component_args) or {}
        if info.component is not None:
            comp = info.component
        elif t == str:
            comp = gr.Textbox
        elif t == int:
            comp = gr.Number
        elif t == bool:
            comp = gr.Checkbox
        else:
            raise ValueError(f'bad options item type: {t} for key {key}')
        elem_id = f"setting_{key}"

        if not is_quicksettings:
            dirtyable_setting = gr.Group(elem_classes="dirtyable", visible=args.get("visible", True))
            dirtyable_setting.__enter__()
            dirty_indicator = gr.Button("", elem_classes="modification-indicator", elem_id="modification_indicator_" + key)

        if info.refresh is not None:
            if is_quicksettings:
                res = comp(label=info.label, value=fun(), elem_id=elem_id, **args)
                ui_common.create_refresh_button(res, info.refresh, info.component_args, f"refresh_{key}")
            else:
                with gr.Row():
                    res = comp(label=info.label, value=fun(), elem_id=elem_id, **args)
                    ui_common.create_refresh_button(res, info.refresh, info.component_args, f"refresh_{key}")
        elif info.folder is not None:
            with gr.Row():
                res = comp(label=info.label, value=fun(), elem_id=elem_id, elem_classes="folder-selector", **args)
                # ui_common.create_browse_button(res, f"folder_{key}")
        else:
            try:
                res = comp(label=info.label, value=fun(), elem_id=elem_id, **args)
            except Exception as e:
                log.error(f'Error creating setting: {key} {e}')
                res = None

        if res is not None and not is_quicksettings:
            res.change(fn=None, inputs=res, _js=f'(val) => markIfModified("{key}", val)')
            dirty_indicator.click(fn=lambda: getattr(opts, key), outputs=res, show_progress=False)
            dirtyable_setting.__exit__()

        return res

    def create_dirty_indicator(key, keys_to_reset, **kwargs):
        def get_opt_values():
            return [getattr(opts, _key) for _key in keys_to_reset]

        elements_to_reset = [component_dict[_key] for _key in keys_to_reset if component_dict[_key] is not None]
        indicator = gr.Button("", elem_classes="modification-indicator", elem_id=f"modification_indicator_{key}", **kwargs)
        indicator.click(fn=get_opt_values, outputs=elements_to_reset, show_progress=False)
        return indicator

    loadsave = ui_loadsave.UiLoadsave(cmd_opts.ui_config)
    components = []
    component_dict = {}
    shared.settings_components = component_dict

    script_callbacks.ui_settings_callback()
    opts.reorder()

    def run_settings(*args):
        changed = []
        for key, value, comp in zip(opts.data_labels.keys(), args, components):
            if comp == dummy_component or value=='dummy':
                continue
            if not opts.same_type(value, opts.data_labels[key].default):
                log.error(f'Setting bad value: {key}={value} expecting={type(opts.data_labels[key].default).__name__}')
                continue
            if opts.set(key, value):
                changed.append(key)
        if shared.opts.cuda_compile_backend == "olive-ai":
            install_olive()
        if cmd_opts.use_directml:
            directml_override_opts()
        if cmd_opts.use_openvino:
            if not shared.opts.cuda_compile:
                shared.log.warning("OpenVINO: Enabling Torch Compile")
                shared.opts.cuda_compile = True
            if shared.opts.cuda_compile_backend != "openvino_fx":
                shared.log.warning("OpenVINO: Setting Torch Compiler backend to OpenVINO FX")
                shared.opts.cuda_compile_backend = "openvino_fx"
            if shared.opts.sd_backend != "diffusers":
                shared.log.warning("OpenVINO: Setting backend to Diffusers")
                shared.opts.sd_backend = "diffusers"
        try:
            if len(changed) > 0:
                opts.save(shared.config_filename)
                log.info(f'Settings: changed={len(changed)} {changed}')
        except RuntimeError:
            log.error(f'Settings failed: change={len(changed)} {changed}')
            return opts.dumpjson(), f'{len(changed)} Settings changed without save: {", ".join(changed)}'
        return opts.dumpjson(), f'{len(changed)} Settings changed{": " if len(changed) > 0 else ""}{", ".join(changed)}'

    def run_settings_single(value, key):
        if not opts.same_type(value, opts.data_labels[key].default):
            return gr.update(visible=True), opts.dumpjson()
        if not opts.set(key, value):
            return gr.update(value=getattr(opts, key)), opts.dumpjson()
        if key == "cuda_compile_backend" and value == "olive-ai":
            install_olive()
        if cmd_opts.use_directml:
            directml_override_opts()
        opts.save(shared.config_filename)
        log.debug(f'Setting changed: key={key}, value={value}')
        return get_value_for_setting(key), opts.dumpjson()

    with gr.Blocks(analytics_enabled=False) as settings_interface:
        with gr.Row(elem_id="system_row"):
            restart_submit = gr.Button(value="Restart server", variant='primary', elem_id="restart_submit")
            shutdown_submit = gr.Button(value="Shutdown server", variant='primary', elem_id="shutdown_submit")
            unload_sd_model = gr.Button(value='Unload checkpoint', variant='primary', elem_id="sett_unload_sd_model")
            reload_sd_model = gr.Button(value='Reload checkpoint', variant='primary', elem_id="sett_reload_sd_model")

        with gr.Tabs(elem_id="system") as system_tabs:
            global ui_system_tabs # pylint: disable=global-statement
            ui_system_tabs = system_tabs
            with gr.TabItem("Settings", id="system_settings", elem_id="tab_settings"):
                with gr.Row(elem_id="settings_row"):
                    settings_submit = gr.Button(value="Apply settings", variant='primary', elem_id="settings_submit")
                    preview_theme = gr.Button(value="Preview theme", variant='primary', elem_id="settings_preview_theme")
                    defaults_submit = gr.Button(value="Restore defaults", variant='primary', elem_id="defaults_submit")
                with gr.Row():
                    _settings_search = gr.Text(label="Search", elem_id="settings_search")

                result = gr.HTML(elem_id="settings_result")
                quicksettings_names = opts.quicksettings_list
                quicksettings_names = {x: i for i, x in enumerate(quicksettings_names) if x != 'quicksettings'}
                quicksettings_list = []

                previous_section = []
                tab_item_keys = []
                current_tab = None
                current_row = None
                dummy_component = gr.Textbox(visible=False, value='dummy')
                with gr.Tabs(elem_id="settings"):
                    for i, (k, item) in enumerate(opts.data_labels.items()):
                        section_must_be_skipped = item.section[0] is None
                        if previous_section != item.section and not section_must_be_skipped:
                            elem_id, text = item.section
                            if current_tab is not None and len(previous_section) > 0:
                                create_dirty_indicator(previous_section[0], tab_item_keys)
                                tab_item_keys = []
                                current_row.__exit__()
                                current_tab.__exit__()
                            current_tab = gr.TabItem(elem_id=f"settings_{elem_id}", label=text)
                            current_tab.__enter__()
                            current_row = gr.Column(variant='compact')
                            current_row.__enter__()
                            previous_section = item.section
                        if k in quicksettings_names and not shared.cmd_opts.freeze:
                            quicksettings_list.append((i, k, item))
                            components.append(dummy_component)
                        elif section_must_be_skipped:
                            components.append(dummy_component)
                        else:
                            component = create_setting_component(k)
                            component_dict[k] = component
                            tab_item_keys.append(k)
                            components.append(component)
                    if current_tab is not None and len(previous_section) > 0:
                        create_dirty_indicator(previous_section[0], tab_item_keys)
                        tab_item_keys = []
                        current_row.__exit__()
                        current_tab.__exit__()

                    request_notifications = gr.Button(value='Request browser notifications', elem_id="request_notifications", visible=False)
                    with gr.TabItem("Show all pages", elem_id="settings_show_all_pages"):
                        create_dirty_indicator("show_all_pages", [], interactive=False)

            with gr.TabItem("Update", id="system_update", elem_id="tab_update"):
                from modules import update
                update.create_ui()

            with gr.TabItem("User interface", id="system_config", elem_id="tab_config"):
                loadsave.create_ui()
                create_dirty_indicator("tab_defaults", [], interactive=False)

            with gr.TabItem("Change log", id="change_log", elem_id="system_tab_changelog"):
                with open('CHANGELOG.md', 'r', encoding='utf-8') as f:
                    md = f.read()
                gr.Markdown(md)

            with gr.TabItem("Licenses", id="system_licenses", elem_id="system_tab_licenses"):
                gr.HTML(shared.html("licenses.html"), elem_id="licenses", elem_classes="licenses")
                create_dirty_indicator("tab_licenses", [], interactive=False)

        def unload_sd_weights():
            modules.sd_models.unload_model_weights(op='model')
            modules.sd_models.unload_model_weights(op='refiner')

        def reload_sd_weights():
            modules.sd_models.reload_model_weights()

        unload_sd_model.click(fn=unload_sd_weights, inputs=[], outputs=[])
        reload_sd_model.click(fn=reload_sd_weights, inputs=[], outputs=[])
        request_notifications.click(fn=lambda: None, inputs=[], outputs=[], _js='function(){}')
        preview_theme.click(fn=None, _js='previewTheme', inputs=[], outputs=[])

    timer.startup.record("ui-settings")

    interfaces = []
    interfaces += [(txt2img_interface, "Text", "txt2img")]
    interfaces += [(img2img_interface, "Image", "img2img")]
    interfaces += [(control_interface, "Control", "control")] if control_interface is not None else []
    interfaces += [(extras_interface, "Process", "process")]
    interfaces += [(interrogate_interface, "Interrogate", "interrogate")]
    interfaces += [(train_interface, "Train", "train")]
    interfaces += [(models_interface, "Models", "models")]
    if shared.opts.onnx_show_menu:
        with gr.Blocks(analytics_enabled=False) as onnx_interface:
            if shared.backend == shared.Backend.DIFFUSERS:
                from modules.onnx_impl import ui as ui_onnx
                ui_onnx.create_ui()
                timer.startup.record("ui-onnx")
        interfaces += [(onnx_interface, "ONNX", "onnx")]
    interfaces += script_callbacks.ui_tabs_callback()
    interfaces += [(settings_interface, "System", "system")]

    from modules import ui_extensions
    extensions_interface = ui_extensions.create_ui()
    interfaces += [(extensions_interface, "Extensions", "extensions")]
    timer.startup.record("ui-extensions")

    shared.tab_names = []
    for _interface, label, _ifid in interfaces:
        shared.tab_names.append(label)

    with gr.Blocks(theme=theme.gradio_theme, analytics_enabled=False, title="SD.Next") as ui_app:
        with gr.Row(elem_id="quicksettings", variant="compact"):
            for _i, k, _item in sorted(quicksettings_list, key=lambda x: quicksettings_names.get(x[1], x[0])):
                component = create_setting_component(k, is_quicksettings=True)
                component_dict[k] = component

        generation_parameters_copypaste.connect_paste_params_buttons()

        with gr.Tabs(elem_id="tabs") as tabs:
            for interface, label, ifid in interfaces:
                if interface is None:
                    continue
                # if label in shared.opts.hidden_tabs or label == '':
                #    continue
                with gr.TabItem(label, id=ifid, elem_id=f"tab_{ifid}"):
                    # log.debug(f'UI render: id={ifid}')
                    interface.render()
            for interface, _label, ifid in interfaces:
                if interface is None:
                    continue
                if ifid in ["extensions", "system"]:
                    continue
                loadsave.add_block(interface, ifid)
            loadsave.add_component(f"webui/Tabs@{tabs.elem_id}", tabs)
            loadsave.setup_ui()

        if opts.notification_audio_enable and os.path.exists(os.path.join(script_path, opts.notification_audio_path)):
            gr.Audio(interactive=False, value=os.path.join(script_path, opts.notification_audio_path), elem_id="audio_notification", visible=False)

        text_settings = gr.Textbox(elem_id="settings_json", value=lambda: opts.dumpjson(), visible=False)
        components = [c for c in components if c is not None]
        settings_submit.click(
            fn=wrap_gradio_call(run_settings, extra_outputs=[gr.update()]),
            inputs=components,
            outputs=[text_settings, result],
        )
        defaults_submit.click(fn=lambda: shared.restore_defaults(restart=True), _js="restartReload")
        restart_submit.click(fn=lambda: shared.restart_server(restart=True), _js="restartReload")
        shutdown_submit.click(fn=lambda: shared.restart_server(restart=False), _js="restartReload")

        for _i, k, _item in quicksettings_list:
            component = component_dict[k]
            info = opts.data_labels[k]
            change_handler = component.release if hasattr(component, 'release') else component.change
            change_handler(
                fn=lambda value, k=k: run_settings_single(value, key=k),
                inputs=[component],
                outputs=[component, text_settings],
                show_progress=info.refresh is not None,
            )

        dummy_component = gr.Textbox(visible=False, value='dummy')
        button_set_checkpoint = gr.Button('Change model', elem_id='change_checkpoint', visible=False)
        button_set_checkpoint.click(
            fn=lambda value, _: run_settings_single(value, key='sd_model_checkpoint'),
            _js="function(v){ var res = desiredCheckpointName; desiredCheckpointName = ''; return [res || v, null]; }",
            inputs=[component_dict['sd_model_checkpoint'], dummy_component],
            outputs=[component_dict['sd_model_checkpoint'], text_settings],
        )
        button_set_refiner = gr.Button('Change refiner', elem_id='change_refiner', visible=False)
        button_set_refiner.click(
            fn=lambda value, _: run_settings_single(value, key='sd_model_checkpoint'),
            _js="function(v){ var res = desiredCheckpointName; desiredCheckpointName = ''; return [res || v, null]; }",
            inputs=[component_dict['sd_model_refiner'], dummy_component],
            outputs=[component_dict['sd_model_refiner'], text_settings],
        )
        button_set_vae = gr.Button('Change VAE', elem_id='change_vae', visible=False)
        button_set_vae.click(
            fn=lambda value, _: run_settings_single(value, key='sd_vae'),
            _js="function(v){ var res = desiredVAEName; desiredVAEName = ''; return [res || v, null]; }",
            inputs=[component_dict['sd_vae'], dummy_component],
            outputs=[component_dict['sd_vae'], text_settings],
        )

        def reference_submit(model):
            if '@' not in model: # diffusers
                loaded = modelloader.load_reference(model)
                return model if loaded else opts.sd_model_checkpoint
            else: # civitai
                model, url = model.split('@')
                loaded = modelloader.load_civitai(model, url)
                return loaded if loaded is not None else opts.sd_model_checkpoint

        button_set_reference = gr.Button('Change reference', elem_id='change_reference', visible=False)
        button_set_reference.click(
            fn=reference_submit,
            _js="function(v){ return desiredCheckpointName; }",
            inputs=[component_dict['sd_model_checkpoint']],
            outputs=[component_dict['sd_model_checkpoint']],
        )
        component_keys = [k for k in opts.data_labels.keys() if k in component_dict]

        def get_settings_values():
            return [get_value_for_setting(key) for key in component_keys]

        ui_app.load(
            fn=get_settings_values,
            inputs=[],
            outputs=[component_dict[k] for k in component_keys if component_dict[k] is not None],
            queue=False,
        )

    timer.startup.record("ui-defaults")
    loadsave.dump_defaults()
    ui_app.ui_loadsave = loadsave
    return ui_app
