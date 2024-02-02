import os
import time
import gradio as gr
import matplotlib.pyplot as plt
from modules.control import unit
from modules.control import processors # patrickvonplaten controlnet_aux
from modules.control.units import controlnet # lllyasviel ControlNet
from modules.control.units import xs # vislearn ControlNet-XS
from modules.control.units import lite # vislearn ControlNet-XS
from modules.control.units import t2iadapter # TencentARC T2I-Adapter
from modules.control.units import reference # reference pipeline
from modules import errors, shared, progress, sd_samplers, ui_components, ui_symbols, ui_common, ui_sections, generation_parameters_copypaste, call_queue, scripts, masking, ipadapter, images # pylint: disable=ungrouped-imports
from modules import ui_control_helpers as helpers


gr_height = None
max_units = shared.opts.control_max_units
units: list[unit.Unit] = [] # main state variable
debug = shared.log.trace if os.environ.get('SD_CONTROL_DEBUG', None) is not None else lambda *args, **kwargs: None
debug('Trace: CONTROL')


def return_controls(res):
    # return preview, image, video, gallery, text
    debug(f'Control received: type={type(res)} {res}')
    if isinstance(res, str): # error response
        return [None, None, None, None, res]
    elif isinstance(res, tuple): # standard response received as tuple via control_run->yield(output_images, process_image, result_txt)
        preview_image = res[1] # may be None
        output_image = res[0][0] if isinstance(res[0], list) else res[0] # may be image or list of images
        if isinstance(res[0], list):
            output_gallery = res[0] if res[0][0] is not None else []
        else:
            output_gallery = [res[0]] if res[0] is not None else [] # must return list, but can receive single image
        result_txt = res[2] if len(res) > 2 else '' # do we have a message
        output_video = res[3] if len(res) > 3 else None # do we have a video filename
        return [preview_image, output_image, output_video, output_gallery, result_txt]
    else: # unexpected
        return [None, None, None, None, f'Control: Unexpected response: {type(res)}']


def generate_click(job_id: str, active_tab: str, *args):
    while helpers.busy:
        time.sleep(0.01)
    from modules.control.run import control_run
    debug(f'Control: tab="{active_tab}" job={job_id} args={args}')
    shared.state.begin('control')
    progress.add_task_to_queue(job_id)
    with call_queue.queue_lock:
        yield [None, None, None, None, 'Control: starting']
        shared.mem_mon.reset()
        progress.start_task(job_id)
        try:
            for results in control_run(units, helpers.input_source, helpers.input_init, helpers.input_mask, active_tab, True, *args):
                progress.record_results(job_id, results)
                yield return_controls(results)
        except Exception as e:
            shared.log.error(f"Control exception: {e}")
            errors.display(e, 'Control')
            return None, None, None, None, f'Control: Exception: {e}'
        progress.finish_task(job_id)
    shared.state.end()


def create_ui(_blocks: gr.Blocks=None):
    helpers.initialize()

    if shared.backend == shared.Backend.ORIGINAL:
        with gr.Blocks(analytics_enabled = False) as control_ui:
            pass
        return [(control_ui, 'Control', 'control')]

    with gr.Blocks(analytics_enabled = False) as control_ui:
        prompt, styles, negative, btn_generate, btn_paste, btn_extra, prompt_counter, btn_prompt_counter, negative_counter, btn_negative_counter  = ui_sections.create_toprow(is_img2img=False, id_part='control')
        txt_prompt_img = gr.File(label="", elem_id="control_prompt_image", file_count="single", type="binary", visible=False)
        txt_prompt_img.change(fn=images.image_data, inputs=[txt_prompt_img], outputs=[prompt, txt_prompt_img])

        with gr.Group(elem_id="control_interface", equal_height=False):
            with gr.Row(elem_id='control_settings'):

                with gr.Accordion(open=False, label="Input", elem_id="control_input", elem_classes=["small-accordion"]):
                    with gr.Row():
                        show_preview = gr.Checkbox(label="Show preview", value=True, elem_id="control_show_preview")
                    with gr.Row():
                        input_type = gr.Radio(label="Input type", choices=['Control only', 'Init image same as control', 'Separate init image'], value='Control only', type='index', elem_id='control_input_type')
                    with gr.Row():
                        denoising_strength = gr.Slider(minimum=0.01, maximum=1.0, step=0.01, label='Denoising strength', value=0.50, elem_id="control_denoising_strength")

                with gr.Accordion(open=False, label="Size", elem_id="control_size", elem_classes=["small-accordion"]):
                    with gr.Tabs():
                        with gr.Tab('Before'):
                            resize_mode_before, resize_name_before, width_before, height_before, scale_by_before, selected_scale_tab_before = ui_sections.create_resize_inputs('control', [], scale_visible=False, mode='Fixed', accordion=False, latent=True)
                        with gr.Tab('After'):
                            resize_mode_after, resize_name_after, width_after, height_after, scale_by_after, selected_scale_tab_after = ui_sections.create_resize_inputs('control', [], scale_visible=False, mode='Fixed', accordion=False, latent=False)

                with gr.Accordion(open=False, label="Sampler", elem_id="control_sampler", elem_classes=["small-accordion"]):
                    sd_samplers.set_samplers()
                    steps, sampler_index = ui_sections.create_sampler_and_steps_selection(sd_samplers.samplers, "control")

                batch_count, batch_size = ui_sections.create_batch_inputs('control')

                seed, _reuse_seed, subseed, _reuse_subseed, subseed_strength, seed_resize_from_h, seed_resize_from_w = ui_sections.create_seed_inputs('control', reuse_visible=False)

                mask_controls = masking.create_segment_ui()

                cfg_scale, clip_skip, image_cfg_scale, diffusers_guidance_rescale, sag_scale, cfg_end, full_quality, restore_faces, tiling= ui_sections.create_advanced_inputs('control')
                hdr_clamp, hdr_boundary, hdr_threshold, hdr_brightness, hdr_center, hdr_color_correction, hdr_sharpen, hdr_sharpen_ratio, hdr_sharpen_start, hdr_maximize, hdr_max_center, hdr_max_boundry = ui_sections.create_callback_inputs('control')

                with gr.Accordion(open=False, label="Video", elem_id="control_video", elem_classes=["small-accordion"]):
                    with gr.Row():
                        video_skip_frames = gr.Slider(minimum=0, maximum=100, step=1, label='Skip input frames', value=0, elem_id="control_video_skip_frames")
                    with gr.Row():
                        video_type = gr.Dropdown(label='Video file', choices=['None', 'GIF', 'PNG', 'MP4'], value='None')
                        video_duration = gr.Slider(label='Duration', minimum=0.25, maximum=300, step=0.25, value=2, visible=False)
                    with gr.Row():
                        video_loop = gr.Checkbox(label='Loop', value=True, visible=False)
                        video_pad = gr.Slider(label='Pad frames', minimum=0, maximum=24, step=1, value=1, visible=False)
                        video_interpolate = gr.Slider(label='Interpolate frames', minimum=0, maximum=24, step=1, value=0, visible=False)
                    video_type.change(fn=helpers.video_type_change, inputs=[video_type], outputs=[video_duration, video_loop, video_pad, video_interpolate])

                with gr.Accordion(open=False, label="Extensions", elem_id="control_extensions", elem_classes=["small-accordion"]):
                    input_script_args = scripts.scripts_current.setup_ui(parent='control', accordion=False)

            with gr.Row():
                override_settings = ui_common.create_override_inputs('control')

            with gr.Row(variant='compact', elem_id="control_extra_networks", visible=False) as extra_networks_ui:
                from modules import timer, ui_extra_networks
                extra_networks_ui = ui_extra_networks.create_ui(extra_networks_ui, btn_extra, 'control', skip_indexing=shared.opts.extra_network_skip_indexing)
                timer.startup.record('ui-en')

            with gr.Row(elem_id='control_status'):
                result_txt = gr.HTML(elem_classes=['control-result'], elem_id='control-result')

            with gr.Row(elem_id='control-inputs'):
                with gr.Column(scale=9, elem_id='control-input-column', visible=True) as _column_input:
                    gr.HTML('<span id="control-input-button">Control input</p>')
                    with gr.Tabs(elem_classes=['control-tabs'], elem_id='control-tab-input'):
                        with gr.Tab('Image', id='in-image') as tab_image:
                            input_mode = gr.Label(value='select', visible=False)
                            input_image = gr.Image(label="Input", show_label=False, type="pil", source="upload", interactive=True, tool="editor", height=gr_height, visible=True, image_mode='RGB', elem_id='control_input_select', elem_classes=['control-image'])
                            input_resize = gr.Image(label="Input", show_label=False, type="pil", source="upload", interactive=True, tool="select", height=gr_height, visible=False, image_mode='RGB', elem_id='control_input_resize', elem_classes=['control-image'])
                            input_inpaint = gr.Image(label="Input", show_label=False, type="pil", source="upload", interactive=True, tool="sketch", height=gr_height, visible=False, image_mode='RGB', elem_id='control_input_inpaint', brush_radius=32, mask_opacity=0.6, elem_classes=['control-image'])
                            btn_interrogate_clip, btn_interrogate_booru = ui_sections.create_interrogate_buttons('control')
                            with gr.Row():
                                input_buttons = [gr.Button('Select', visible=True, interactive=False), gr.Button('Inpaint', visible=True, interactive=True), gr.Button('Outpaint', visible=True, interactive=True)]
                        with gr.Tab('Video', id='in-video') as tab_video:
                            input_video = gr.Video(label="Input", show_label=False, interactive=True, height=gr_height, elem_classes=['control-image'])
                        with gr.Tab('Batch', id='in-batch') as tab_batch:
                            input_batch = gr.File(label="Input", show_label=False, file_count='multiple', file_types=['image'], type='file', interactive=True, height=gr_height)
                        with gr.Tab('Folder', id='in-folder') as tab_folder:
                            input_folder = gr.File(label="Input", show_label=False, file_count='directory', file_types=['image'], type='file', interactive=True, height=gr_height)
                with gr.Column(scale=9, elem_id='control-init-column', visible=False) as column_init:
                    gr.HTML('<span id="control-init-button">Init input</p>')
                    with gr.Tabs(elem_classes=['control-tabs'], elem_id='control-tab-init'):
                        with gr.Tab('Image', id='init-image') as tab_image_init:
                            init_image = gr.Image(label="Input", show_label=False, type="pil", source="upload", interactive=True, tool="editor", height=gr_height, elem_classes=['control-image'])
                        with gr.Tab('Video', id='init-video') as tab_video_init:
                            init_video = gr.Video(label="Input", show_label=False, interactive=True, height=gr_height, elem_classes=['control-image'])
                        with gr.Tab('Batch', id='init-batch') as tab_batch_init:
                            init_batch = gr.File(label="Input", show_label=False, file_count='multiple', file_types=['image'], type='file', interactive=True, height=gr_height, elem_classes=['control-image'])
                        with gr.Tab('Folder', id='init-folder') as tab_folder_init:
                            init_folder = gr.File(label="Input", show_label=False, file_count='directory', file_types=['image'], type='file', interactive=True, height=gr_height, elem_classes=['control-image'])
                with gr.Column(scale=9, elem_id='control-output-column', visible=True) as _column_output:
                    gr.HTML('<span id="control-output-button">Output</p>')
                    with gr.Tabs(elem_classes=['control-tabs'], elem_id='control-tab-output') as output_tabs:
                        with gr.Tab('Gallery', id='out-gallery'):
                            output_gallery, _output_gen_info, _output_html_info, _output_html_info_formatted, _output_html_log = ui_common.create_output_panel("control", preview=True, prompt=prompt, height=gr_height)
                        with gr.Tab('Image', id='out-image'):
                            output_image = gr.Image(label="Output", show_label=False, type="pil", interactive=False, tool="editor", height=gr_height, elem_id='control_output_image', elem_classes=['control-image'])
                        with gr.Tab('Video', id='out-video'):
                            output_video = gr.Video(label="Output", show_label=False, height=gr_height, elem_id='control_output_video', elem_classes=['control-image'])
                with gr.Column(scale=9, elem_id='control-preview-column', visible=True) as column_preview:
                    gr.HTML('<span id="control-preview-button">Preview</p>')
                    with gr.Tabs(elem_classes=['control-tabs'], elem_id='control-tab-preview'):
                        with gr.Tab('Preview', id='preview-image') as tab_image:
                            preview_process = gr.Image(label="Preview", show_label=False, type="pil", source="upload", interactive=False, height=gr_height, visible=True, elem_id='control_preview', elem_classes=['control-image'])

            with gr.Tabs(elem_id='control-tabs') as _tabs_control_type:

                with gr.Tab('ControlNet') as _tab_controlnet:
                    gr.HTML('<a href="https://github.com/lllyasviel/ControlNet">ControlNet</a>')
                    with gr.Row():
                        extra_controls = [
                            gr.Checkbox(label="Guess mode", value=False, scale=3),
                        ]
                        num_controlnet_units = gr.Slider(label="Units", minimum=1, maximum=max_units, step=1, value=1, scale=1)
                    controlnet_ui_units = [] # list of hidable accordions
                    for i in range(max_units):
                        enabled = True if i==0 else False
                        with gr.Accordion(f'ControlNet unit {i+1}', visible= i < num_controlnet_units.value, elem_classes='control-unit') as unit_ui:
                            with gr.Row():
                                enabled_cb = gr.Checkbox(enabled, container=False, show_label=False)
                                process_id = gr.Dropdown(label="Processor", choices=processors.list_models(), value='None')
                                model_id = gr.Dropdown(label="ControlNet", choices=controlnet.list_models(), value='None')
                                ui_common.create_refresh_button(model_id, controlnet.list_models, lambda: {"choices": controlnet.list_models(refresh=True)}, f'refresh_controlnet_models_{i}')
                                model_strength = gr.Slider(label="Strength", minimum=0.01, maximum=2.0, step=0.01, value=1.0-i/10)
                                control_start = gr.Slider(label="Start", minimum=0.0, maximum=1.0, step=0.05, value=0)
                                control_end = gr.Slider(label="End", minimum=0.0, maximum=1.0, step=0.05, value=1.0)
                                reset_btn = ui_components.ToolButton(value=ui_symbols.reset)
                                image_upload = gr.UploadButton(label=ui_symbols.upload, file_types=['image'], elem_classes=['form', 'gradio-button', 'tool'])
                                process_btn= ui_components.ToolButton(value=ui_symbols.preview)
                                image_preview = gr.Image(label="Input", type="pil", source="upload", height=128, width=128, visible=False, interactive=True, show_label=False, show_download_button=False, container=False)
                        controlnet_ui_units.append(unit_ui)
                        units.append(unit.Unit(
                            unit_type = 'controlnet',
                            enabled = enabled,
                            result_txt = result_txt,
                            enabled_cb = enabled_cb,
                            reset_btn = reset_btn,
                            process_id = process_id,
                            model_id = model_id,
                            model_strength = model_strength,
                            preview_process = preview_process,
                            preview_btn = process_btn,
                            image_upload = image_upload,
                            image_preview = image_preview,
                            control_start = control_start,
                            control_end = control_end,
                            extra_controls = extra_controls,
                            )
                        )
                        if i == 0:
                            units[-1].enabled = True # enable first unit in group
                    num_controlnet_units.change(fn=helpers.display_units, inputs=[num_controlnet_units], outputs=controlnet_ui_units)

                with gr.Tab('IP Adapter') as _tab_ipadapter:
                    with gr.Row():
                        with gr.Column():
                            gr.HTML('<a href="https://github.com/tencent-ailab/IP-Adapter">IP-Adapter</a>')
                            ip_adapter_name = gr.Dropdown(label='Adapter', choices=ipadapter.ADAPTERS, value='None')
                            ip_scale = gr.Slider(label='Scale', minimum=0.0, maximum=1.0, step=0.01, value=0.5)
                        with gr.Column():
                            ip_image = gr.Image(label="Input", show_label=False, type="pil", source="upload", interactive=True, tool="editor", height=256, width=256)

                with gr.Tab('T2I Adapter') as _tab_t2iadapter:
                    gr.HTML('<a href="https://github.com/TencentARC/T2I-Adapter">T2I-Adapter</a>')
                    with gr.Row():
                        extra_controls = [
                            gr.Slider(label="Control factor", minimum=0.0, maximum=1.0, step=0.05, value=1.0, scale=3),
                        ]
                        num_adapter_units = gr.Slider(label="Units", minimum=1, maximum=max_units, step=1, value=1, scale=1)
                    adapter_ui_units = [] # list of hidable accordions
                    for i in range(max_units):
                        enabled = True if i==0 else False
                        with gr.Accordion(f'T2I-Adapter unit {i+1}', visible= i < num_adapter_units.value, elem_classes='control-unit') as unit_ui:
                            with gr.Row():
                                enabled_cb = gr.Checkbox(enabled, container=False, show_label=False)
                                process_id = gr.Dropdown(label="Processor", choices=processors.list_models(), value='None')
                                model_id = gr.Dropdown(label="Adapter", choices=t2iadapter.list_models(), value='None')
                                ui_common.create_refresh_button(model_id, t2iadapter.list_models, lambda: {"choices": t2iadapter.list_models(refresh=True)}, f'refresh_adapter_models_{i}')
                                model_strength = gr.Slider(label="Strength", minimum=0.01, maximum=1.0, step=0.01, value=1.0-i/10)
                                reset_btn = ui_components.ToolButton(value=ui_symbols.reset)
                                image_upload = gr.UploadButton(label=ui_symbols.upload, file_types=['image'], elem_classes=['form', 'gradio-button', 'tool'])
                                process_btn= ui_components.ToolButton(value=ui_symbols.preview)
                                image_preview = gr.Image(label="Input", show_label=False, type="pil", source="upload", interactive=False, height=128, width=128, visible=False)
                        adapter_ui_units.append(unit_ui)
                        units.append(unit.Unit(
                            unit_type = 'adapter',
                            enabled = enabled,
                            result_txt = result_txt,
                            enabled_cb = enabled_cb,
                            reset_btn = reset_btn,
                            process_id = process_id,
                            model_id = model_id,
                            model_strength = model_strength,
                            preview_process = preview_process,
                            preview_btn = process_btn,
                            image_upload = image_upload,
                            image_preview = image_preview,
                            extra_controls = extra_controls,
                            )
                        )
                        if i == 0:
                            units[-1].enabled = True # enable first unit in group
                    num_adapter_units.change(fn=helpers.display_units, inputs=[num_adapter_units], outputs=adapter_ui_units)

                with gr.Tab('XS') as _tab_controlnetxs:
                    gr.HTML('<a href="https://vislearn.github.io/ControlNet-XS/">ControlNet XS</a>')
                    with gr.Row():
                        extra_controls = [
                            gr.Slider(label="Time embedding mix", minimum=0.0, maximum=1.0, step=0.05, value=0.0, scale=3)
                        ]
                        num_controlnet_units = gr.Slider(label="Units", minimum=1, maximum=max_units, step=1, value=1, scale=1)
                    controlnetxs_ui_units = [] # list of hidable accordions
                    for i in range(max_units):
                        enabled = True if i==0 else False
                        with gr.Accordion(f'ControlNet-XS unit {i+1}', visible= i < num_controlnet_units.value, elem_classes='control-unit') as unit_ui:
                            with gr.Row():
                                enabled_cb = gr.Checkbox(enabled, container=False, show_label=False)
                                process_id = gr.Dropdown(label="Processor", choices=processors.list_models(), value='None')
                                model_id = gr.Dropdown(label="ControlNet-XS", choices=xs.list_models(), value='None')
                                ui_common.create_refresh_button(model_id, xs.list_models, lambda: {"choices": xs.list_models(refresh=True)}, f'refresh_xs_models_{i}')
                                model_strength = gr.Slider(label="Strength", minimum=0.01, maximum=1.0, step=0.01, value=1.0-i/10)
                                control_start = gr.Slider(label="Start", minimum=0.0, maximum=1.0, step=0.05, value=0)
                                control_end = gr.Slider(label="End", minimum=0.0, maximum=1.0, step=0.05, value=1.0)
                                reset_btn = ui_components.ToolButton(value=ui_symbols.reset)
                                image_upload = gr.UploadButton(label=ui_symbols.upload, file_types=['image'], elem_classes=['form', 'gradio-button', 'tool'])
                                process_btn= ui_components.ToolButton(value=ui_symbols.preview)
                                image_preview = gr.Image(label="Input", show_label=False, type="pil", source="upload", interactive=False, height=128, width=128, visible=False)
                        controlnetxs_ui_units.append(unit_ui)
                        units.append(unit.Unit(
                            unit_type = 'xs',
                            enabled = enabled,
                            result_txt = result_txt,
                            enabled_cb = enabled_cb,
                            reset_btn = reset_btn,
                            process_id = process_id,
                            model_id = model_id,
                            model_strength = model_strength,
                            preview_process = preview_process,
                            preview_btn = process_btn,
                            image_upload = image_upload,
                            image_preview = image_preview,
                            control_start = control_start,
                            control_end = control_end,
                            extra_controls = extra_controls,
                            )
                        )
                        if i == 0:
                            units[-1].enabled = True # enable first unit in group
                    num_controlnet_units.change(fn=helpers.display_units, inputs=[num_controlnet_units], outputs=controlnetxs_ui_units)

                with gr.Tab('Lite') as _tab_lite:
                    gr.HTML('<a href="https://huggingface.co/kohya-ss/controlnet-lllite">Control LLLite</a>')
                    with gr.Row():
                        extra_controls = [
                        ]
                        num_lite_units = gr.Slider(label="Units", minimum=1, maximum=max_units, step=1, value=1, scale=1)
                    lite_ui_units = [] # list of hidable accordions
                    for i in range(max_units):
                        enabled = True if i==0 else False
                        with gr.Accordion(f'Control-LLLite unit {i+1}', visible= i < num_lite_units.value, elem_classes='control-unit') as unit_ui:
                            with gr.Row():
                                enabled_cb = gr.Checkbox(enabled, container=False, show_label=False)
                                process_id = gr.Dropdown(label="Processor", choices=processors.list_models(), value='None')
                                model_id = gr.Dropdown(label="Model", choices=lite.list_models(), value='None')
                                ui_common.create_refresh_button(model_id, lite.list_models, lambda: {"choices": lite.list_models(refresh=True)}, f'refresh_lite_models_{i}')
                                model_strength = gr.Slider(label="Strength", minimum=0.01, maximum=1.0, step=0.01, value=1.0-i/10)
                                reset_btn = ui_components.ToolButton(value=ui_symbols.reset)
                                image_upload = gr.UploadButton(label=ui_symbols.upload, file_types=['image'], elem_classes=['form', 'gradio-button', 'tool'])
                                image_preview = gr.Image(label="Input", show_label=False, type="pil", source="upload", interactive=False, height=128, width=128, visible=False)
                                process_btn= ui_components.ToolButton(value=ui_symbols.preview)
                        lite_ui_units.append(unit_ui)
                        units.append(unit.Unit(
                            unit_type = 'lite',
                            enabled = enabled,
                            result_txt = result_txt,
                            enabled_cb = enabled_cb,
                            reset_btn = reset_btn,
                            process_id = process_id,
                            model_id = model_id,
                            model_strength = model_strength,
                            preview_process = preview_process,
                            preview_btn = process_btn,
                            image_upload = image_upload,
                            image_preview = image_preview,
                            extra_controls = extra_controls,
                            )
                        )
                        if i == 0:
                            units[-1].enabled = True # enable first unit in group
                    num_lite_units.change(fn=helpers.display_units, inputs=[num_lite_units], outputs=lite_ui_units)

                with gr.Tab('Reference') as _tab_reference:
                    gr.HTML('<a href="https://github.com/Mikubill/sd-webui-controlnet/discussions/1236">ControlNet reference-only control</a>')
                    with gr.Row():
                        extra_controls = [
                            gr.Radio(label="Reference context", choices=['Attention', 'Adain', 'Attention Adain'], value='Attention', interactive=True),
                            gr.Slider(label="Style fidelity", minimum=0.0, maximum=1.0, step=0.05, value=0.5, interactive=True), # prompt vs control importance
                            gr.Slider(label="Reference query weight", minimum=0.0, maximum=1.0, step=0.05, value=1.0, interactive=True),
                            gr.Slider(label="Reference adain weight", minimum=0.0, maximum=2.0, step=0.05, value=1.0, interactive=True),
                        ]
                    for i in range(1): # can only have one reference unit
                        enabled = True if i==0 else False
                        with gr.Accordion(f'Reference unit {i+1}', visible=True, elem_classes='control-unit') as unit_ui:
                            with gr.Row():
                                enabled_cb = gr.Checkbox(enabled, container=False, show_label=False)
                                model_id = gr.Dropdown(label="Reference", choices=reference.list_models(), value='Reference', visible=False)
                                model_strength = gr.Slider(label="Strength", minimum=0.01, maximum=1.0, step=0.01, value=1.0, visible=False)
                                reset_btn = ui_components.ToolButton(value=ui_symbols.reset)
                                image_upload = gr.UploadButton(label=ui_symbols.upload, file_types=['image'], elem_classes=['form', 'gradio-button', 'tool'])
                                image_preview = gr.Image(label="Input", show_label=False, type="pil", source="upload", interactive=False, height=128, width=128, visible=False)
                                process_btn= ui_components.ToolButton(value=ui_symbols.preview)
                        units.append(unit.Unit(
                            unit_type = 'reference',
                            enabled = enabled,
                            result_txt = result_txt,
                            enabled_cb = enabled_cb,
                            reset_btn = reset_btn,
                            process_id = process_id,
                            model_id = model_id,
                            model_strength = model_strength,
                            preview_process = preview_process,
                            preview_btn = process_btn,
                            image_upload = image_upload,
                            image_preview = image_preview,
                            extra_controls = extra_controls,
                            )
                        )
                        if i == 0:
                            units[-1].enabled = True # enable first unit in group

                with gr.Tab('Processor settings') as _tab_settings:
                    with gr.Group(elem_classes=['processor-group']):
                        settings = []
                        with gr.Accordion('HED', open=True, elem_classes=['processor-settings']):
                            settings.append(gr.Checkbox(label="Scribble", value=False))
                        with gr.Accordion('Midas depth', open=True, elem_classes=['processor-settings']):
                            settings.append(gr.Slider(label="Background threshold", minimum=0.0, maximum=1.0, step=0.01, value=0.1))
                            settings.append(gr.Checkbox(label="Depth and normal", value=False))
                        with gr.Accordion('MLSD', open=True, elem_classes=['processor-settings']):
                            settings.append(gr.Slider(label="Score threshold", minimum=0.0, maximum=1.0, step=0.01, value=0.1))
                            settings.append(gr.Slider(label="Distance threshold", minimum=0.0, maximum=1.0, step=0.01, value=0.1))
                        with gr.Accordion('OpenBody', open=True, elem_classes=['processor-settings']):
                            settings.append(gr.Checkbox(label="Body", value=True))
                            settings.append(gr.Checkbox(label="Hands", value=False))
                            settings.append(gr.Checkbox(label="Face", value=False))
                        with gr.Accordion('PidiNet', open=True, elem_classes=['processor-settings']):
                            settings.append(gr.Checkbox(label="Scribble", value=False))
                            settings.append(gr.Checkbox(label="Apply filter", value=False))
                        with gr.Accordion('LineArt', open=True, elem_classes=['processor-settings']):
                            settings.append(gr.Checkbox(label="Coarse", value=False))
                        with gr.Accordion('Leres Depth', open=True, elem_classes=['processor-settings']):
                            settings.append(gr.Checkbox(label="Boost", value=False))
                            settings.append(gr.Slider(label="Near threshold", minimum=0.0, maximum=1.0, step=0.01, value=0.0))
                            settings.append(gr.Slider(label="Background threshold", minimum=0.0, maximum=1.0, step=0.01, value=0.0))
                        with gr.Accordion('MediaPipe Face', open=True, elem_classes=['processor-settings']):
                            settings.append(gr.Slider(label="Max faces", minimum=1, maximum=10, step=1, value=1))
                            settings.append(gr.Slider(label="Min confidence", minimum=0.0, maximum=1.0, step=0.01, value=0.5))
                        with gr.Accordion('Canny', open=True, elem_classes=['processor-settings']):
                            settings.append(gr.Slider(label="Low threshold", minimum=0, maximum=1000, step=1, value=100))
                            settings.append(gr.Slider(label="High threshold", minimum=0, maximum=1000, step=1, value=200))
                        with gr.Accordion('DWPose', open=True, elem_classes=['processor-settings']):
                            settings.append(gr.Radio(label="Model", choices=['Tiny', 'Medium', 'Large'], value='Tiny'))
                            settings.append(gr.Slider(label="Min confidence", minimum=0.0, maximum=1.0, step=0.01, value=0.3))
                        with gr.Accordion('SegmentAnything', open=True, elem_classes=['processor-settings']):
                            settings.append(gr.Radio(label="Model", choices=['Base', 'Large'], value='Base'))
                        with gr.Accordion('Edge', open=True, elem_classes=['processor-settings']):
                            settings.append(gr.Checkbox(label="Parameter free", value=True))
                            settings.append(gr.Radio(label="Mode", choices=['edge', 'gradient'], value='edge'))
                        with gr.Accordion('Zoe Depth', open=True, elem_classes=['processor-settings']):
                            settings.append(gr.Checkbox(label="Gamma corrected", value=False))
                        with gr.Accordion('Marigold Depth', open=True, elem_classes=['processor-settings']):
                            settings.append(gr.Dropdown(label="Color map", choices=['None'] + plt.colormaps(), value='None'))
                            settings.append(gr.Slider(label="Denoising steps", minimum=1, maximum=99, step=1, value=10))
                            settings.append(gr.Slider(label="Ensemble size", minimum=1, maximum=99, step=1, value=10))
                        with gr.Accordion('Depth Anything', open=True, elem_classes=['processor-settings']):
                            settings.append(gr.Dropdown(label="Color map", choices=['none'] + masking.COLORMAP, value='inferno'))
                        for setting in settings:
                            setting.change(fn=processors.update_settings, inputs=settings, outputs=[])

                for btn in input_buttons:
                    btn.click(fn=helpers.copy_input, inputs=[input_mode, btn, input_image, input_resize, input_inpaint], outputs=[input_image, input_resize, input_inpaint], _js='controlInputMode')
                    btn.click(fn=helpers.transfer_input, inputs=[btn], outputs=[input_image, input_resize, input_inpaint] + input_buttons)

                show_preview.change(fn=lambda x: gr.update(visible=x), inputs=[show_preview], outputs=[column_preview])
                input_type.change(fn=lambda x: gr.update(visible=x == 2), inputs=[input_type], outputs=[column_init])
                btn_prompt_counter.click(fn=call_queue.wrap_queued_call(ui_common.update_token_counter), inputs=[prompt, steps], outputs=[prompt_counter])
                btn_negative_counter.click(fn=call_queue.wrap_queued_call(ui_common.update_token_counter), inputs=[negative, steps], outputs=[negative_counter])
                btn_interrogate_clip.click(fn=helpers.interrogate_clip, inputs=[], outputs=[prompt])
                btn_interrogate_booru.click(fn=helpers.interrogate_booru, inputs=[], outputs=[prompt])

                select_fields = [input_mode, input_image, init_image, input_type, input_resize, input_inpaint, input_video, input_batch, input_folder]
                select_output = [output_tabs, result_txt]
                select_dict = dict(
                    fn=helpers.select_input,
                    _js="controlInputMode",
                    inputs=select_fields,
                    outputs=select_output,
                    show_progress=True,
                    queue=False,
                )
                prompt.submit(**select_dict)
                btn_generate.click(**select_dict)
                for ctrl in [input_image, input_resize, input_video, input_batch, input_folder, init_image, init_video, init_batch, init_folder, tab_image, tab_video, tab_batch, tab_folder, tab_image_init, tab_video_init, tab_batch_init, tab_folder_init]:
                    if hasattr(ctrl, 'change'):
                        ctrl.change(**select_dict)
                    if hasattr(ctrl, 'clear'):
                        ctrl.clear(**select_dict)
                for ctrl in [input_inpaint]: # gradio image mode inpaint triggeres endless loop on change event
                    if hasattr(ctrl, 'upload'):
                        ctrl.upload(**select_dict)

                tabs_state = gr.Text(value='none', visible=False)
                input_fields = [
                    input_type,
                    prompt, negative, styles,
                    steps, sampler_index,
                    seed, subseed, subseed_strength, seed_resize_from_h, seed_resize_from_w,
                    cfg_scale, clip_skip, image_cfg_scale, diffusers_guidance_rescale, sag_scale, cfg_end, full_quality, restore_faces, tiling, hdr_clamp, hdr_boundary, hdr_threshold, hdr_brightness, hdr_center, hdr_color_correction, hdr_sharpen, hdr_sharpen_ratio, hdr_sharpen_start, hdr_maximize, hdr_max_center, hdr_max_boundry,
                    resize_mode_before, resize_name_before, width_before, height_before, scale_by_before, selected_scale_tab_before,
                    resize_mode_after, resize_name_after, width_after, height_after, scale_by_after, selected_scale_tab_after,
                    denoising_strength, batch_count, batch_size,
                    video_skip_frames, video_type, video_duration, video_loop, video_pad, video_interpolate,
                    ip_adapter_name, ip_scale, ip_image,
                ]
                output_fields = [
                    preview_process,
                    output_image,
                    output_video,
                    output_gallery,
                    result_txt,
                ]
                control_dict = dict(
                    fn=generate_click,
                    _js="submit_control",
                    inputs=[tabs_state, tabs_state] + input_fields + input_script_args,
                    outputs=output_fields,
                    show_progress=True,
                )
                prompt.submit(**control_dict)
                btn_generate.click(**control_dict)

                paste_fields = [
                    # prompt
                    (prompt, "Prompt"),
                    (negative, "Negative prompt"),
                    # input
                    (denoising_strength, "Denoising strength"),
                    # resize # TODO resize params
                    (width_before, "Size-1"),
                    (height_before, "Size-2"),
                    (resize_mode_before, "Resize mode"),
                    (scale_by_before, "Resize scale"),
                    # sampler
                    (sampler_index, "Sampler"),
                    (steps, "Steps"),
                    # batch
                    (batch_count, "Batch-1"),
                    (batch_size, "Batch-2"),
                    # seed
                    (seed, "Seed"),
                    # mask
                    (mask_controls[1], "Mask only"),
                    (mask_controls[2], "Mask invert"),
                    (mask_controls[3], "Mask blur"),
                    (mask_controls[4], "Mask erode"),
                    (mask_controls[5], "Mask dilate"),
                    (mask_controls[6], "Mask auto"),
                    # advanced
                    (cfg_scale, "CFG scale"),
                    (clip_skip, "Clip skip"),
                    (image_cfg_scale, "Image CFG scale"),
                    (diffusers_guidance_rescale, "CFG rescale"),
                    (full_quality, "Full quality"),
                    (restore_faces, "Face restoration"),
                    (tiling, "Tiling"),
                    # second pass # TODO second pass params
                    # hidden
                    (seed_resize_from_w, "Seed resize from-1"),
                    (seed_resize_from_h, "Seed resize from-2"),
                    *scripts.scripts_control.infotext_fields
                ]
                generation_parameters_copypaste.add_paste_fields("control", input_image, paste_fields, override_settings)
                bindings = generation_parameters_copypaste.ParamBinding(paste_button=btn_paste, tabname="control", source_text_component=prompt, source_image_component=output_gallery)
                generation_parameters_copypaste.register_paste_params_button(bindings)
                masking.bind_controls([input_image, input_inpaint, input_resize], preview_process, output_image)


                if os.environ.get('SD_CONTROL_DEBUG', None) is not None: # debug only
                    from modules.control.test import test_processors, test_controlnets, test_adapters, test_xs, test_lite
                    gr.HTML('<br><h1>Debug</h1><br>')
                    with gr.Row():
                        run_test_processors_btn = gr.Button(value="Test:Processors", variant='primary', elem_classes=['control-button'])
                        run_test_controlnets_btn = gr.Button(value="Test:ControlNets", variant='primary', elem_classes=['control-button'])
                        run_test_xs_btn = gr.Button(value="Test:ControlNets-XS", variant='primary', elem_classes=['control-button'])
                        run_test_adapters_btn = gr.Button(value="Test:Adapters", variant='primary', elem_classes=['control-button'])
                        run_test_lite_btn = gr.Button(value="Test:Control-LLLite", variant='primary', elem_classes=['control-button'])

                        run_test_processors_btn.click(fn=test_processors, inputs=[input_image], outputs=[preview_process, output_image, output_video, output_gallery])
                        run_test_controlnets_btn.click(fn=test_controlnets, inputs=[prompt, negative, input_image], outputs=[preview_process, output_image, output_video, output_gallery])
                        run_test_xs_btn.click(fn=test_xs, inputs=[prompt, negative, input_image], outputs=[preview_process, output_image, output_video, output_gallery])
                        run_test_adapters_btn.click(fn=test_adapters, inputs=[prompt, negative, input_image], outputs=[preview_process, output_image, output_video, output_gallery])
                        run_test_lite_btn.click(fn=test_lite, inputs=[prompt, negative, input_image], outputs=[preview_process, output_image, output_video, output_gallery])

    return [(control_ui, 'Control', 'control')]
