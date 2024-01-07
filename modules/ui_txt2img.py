import gradio as gr
from modules.call_queue import wrap_gradio_gpu_call, wrap_queued_call
from modules import timer, shared, ui_common, ui_symbols, ui_sections, generation_parameters_copypaste
from modules.ui_components import FormRow, FormGroup, ToolButton


def calc_resolution_hires(width, height, hr_scale, hr_resize_x, hr_resize_y, hr_upscaler):
    from modules import processing, devices
    if hr_upscaler == "None":
        return "Hires resize: None"
    p = processing.StableDiffusionProcessingTxt2Img(width=width, height=height, enable_hr=True, hr_scale=hr_scale, hr_resize_x=hr_resize_x, hr_resize_y=hr_resize_y)
    p.init_hr()
    with devices.autocast():
        p.init([""], [0], [0])
    return f"Hires resize: from <span class='resolution'>{p.width}x{p.height}</span> to <span class='resolution'>{p.hr_resize_x or p.hr_upscale_to_x}x{p.hr_resize_y or p.hr_upscale_to_y}</span>"


def create_ui():
    shared.log.debug('UI initialize: txt2img')
    import modules.txt2img # pylint: disable=redefined-outer-name
    modules.scripts.scripts_current = modules.scripts.scripts_txt2img
    modules.scripts.scripts_txt2img.initialize_scripts(is_img2img=False)
    with gr.Blocks(analytics_enabled=False) as _txt2img_interface:
        txt2img_prompt, txt2img_prompt_styles, txt2img_negative_prompt, txt2img_submit, txt2img_paste, txt2img_extra_networks_button, txt2img_token_counter, txt2img_token_button, txt2img_negative_token_counter, txt2img_negative_token_button = ui_sections.create_toprow(is_img2img=False, id_part="txt2img")

        txt_prompt_img = gr.File(label="", elem_id="txt2img_prompt_image", file_count="single", type="binary", visible=False)
        txt_prompt_img.change(fn=modules.images.image_data, inputs=[txt_prompt_img], outputs=[txt2img_prompt, txt_prompt_img])

        with FormRow(variant='compact', elem_id="txt2img_extra_networks", visible=False) as extra_networks_ui:
            from modules import ui_extra_networks
            extra_networks_ui = ui_extra_networks.create_ui(extra_networks_ui, txt2img_extra_networks_button, 'txt2img', skip_indexing=shared.opts.extra_network_skip_indexing)
            timer.startup.record('ui-extra-networks')

        with gr.Row(elem_id="txt2img_interface", equal_height=False):
            with gr.Column(variant='compact', elem_id="txt2img_settings"):

                with FormRow():
                    width = gr.Slider(minimum=64, maximum=4096, step=8, label="Width", value=512, elem_id="txt2img_width")
                    height = gr.Slider(minimum=64, maximum=4096, step=8, label="Height", value=512, elem_id="txt2img_height")
                    res_switch_btn = ToolButton(value=ui_symbols.switch, elem_id="txt2img_res_switch_btn", label="Switch dims")
                    res_switch_btn.click(lambda w, h: (h, w), inputs=[width, height], outputs=[width, height], show_progress=False)

                with FormGroup(elem_classes="settings-accordion"):

                    steps, sampler_index = ui_sections.create_sampler_inputs('txt2img')
                    batch_count, batch_size = ui_sections.create_batch_inputs('txt2img')
                    seed, reuse_seed, subseed, reuse_subseed, subseed_strength, seed_resize_from_h, seed_resize_from_w = ui_sections.create_seed_inputs('txt2img')
                    cfg_scale, clip_skip, image_cfg_scale, diffusers_guidance_rescale, sag_scale, full_quality, restore_faces, tiling, hdr_clamp, hdr_boundary, hdr_threshold, hdr_center, hdr_channel_shift, hdr_full_shift, hdr_maximize, hdr_max_center, hdr_max_boundry = ui_sections.create_advanced_inputs('txt2img')
                    enable_hr, hr_sampler_index, denoising_strength, hr_final_resolution, hr_upscaler, hr_force, hr_second_pass_steps, hr_scale, hr_resize_x, hr_resize_y, refiner_steps, refiner_start, refiner_prompt, refiner_negative = ui_sections.create_hires_inputs('txt2img')
                    override_settings = ui_common.create_override_inputs('txt2img')

                txt2img_script_inputs = modules.scripts.scripts_txt2img.setup_ui(parent='txt2img', accordion=True)

            hr_resolution_preview_inputs = [width, height, hr_scale, hr_resize_x, hr_resize_y, hr_upscaler]
            for preview_input in hr_resolution_preview_inputs:
                preview_input.change(
                    fn=calc_resolution_hires,
                    _js="onCalcResolutionHires",
                    inputs=hr_resolution_preview_inputs,
                    outputs=[hr_final_resolution],
                    show_progress=False,
                )

            txt2img_gallery, txt2img_generation_info, txt2img_html_info, _txt2img_html_info_formatted, txt2img_html_log = ui_common.create_output_panel("txt2img", preview=True, prompt=None)
            ui_common.connect_reuse_seed(seed, reuse_seed, txt2img_generation_info, is_subseed=False)
            ui_common.connect_reuse_seed(subseed, reuse_subseed, txt2img_generation_info, is_subseed=True)

            dummy_component = gr.Textbox(visible=False, value='dummy')
            txt2img_args = [
                dummy_component,
                txt2img_prompt, txt2img_negative_prompt, txt2img_prompt_styles,
                steps, sampler_index, hr_sampler_index,
                full_quality, restore_faces, tiling,
                batch_count, batch_size,
                cfg_scale, image_cfg_scale, diffusers_guidance_rescale, sag_scale,
                clip_skip,
                seed, subseed, subseed_strength, seed_resize_from_h, seed_resize_from_w,
                height, width,
                enable_hr, denoising_strength,
                hr_scale, hr_upscaler, hr_force, hr_second_pass_steps, hr_resize_x, hr_resize_y,
                refiner_steps, refiner_start, refiner_prompt, refiner_negative,
                hdr_clamp, hdr_boundary, hdr_threshold, hdr_center, hdr_channel_shift, hdr_full_shift, hdr_maximize, hdr_max_center, hdr_max_boundry,
                override_settings,
            ]
            txt2img_dict = dict(
                fn=wrap_gradio_gpu_call(modules.txt2img.txt2img, extra_outputs=[None, '', '']),
                _js="submit_txt2img",
                inputs=txt2img_args + txt2img_script_inputs,
                outputs=[
                    txt2img_gallery,
                    txt2img_generation_info,
                    txt2img_html_info,
                    txt2img_html_log,
                ],
                show_progress=False,
            )
            txt2img_prompt.submit(**txt2img_dict)
            txt2img_submit.click(**txt2img_dict)
            txt2img_paste_fields = [
                # prompt
                (txt2img_prompt, "Prompt"),
                (txt2img_negative_prompt, "Negative prompt"),
                # main
                (width, "Size-1"),
                (height, "Size-2"),
                # sampler
                (sampler_index, "Sampler"),
                (steps, "Steps"),
                # batch
                (batch_count, "Batch-1"),
                (batch_size, "Batch-2"),
                # seed
                (seed, "Seed"),
                (subseed, "Variation seed"),
                (subseed_strength, "Variation strength"),
                # advanced
                (cfg_scale, "CFG scale"),
                (clip_skip, "Clip skip"),
                (image_cfg_scale, "Image CFG scale"),
                (diffusers_guidance_rescale, "CFG rescale"),
                (full_quality, "Full quality"),
                (restore_faces, "Face restoration"),
                (tiling, "Tiling"),
                # second pass
                (enable_hr, "Second pass"),
                (hr_sampler_index, "Hires sampler"),
                (denoising_strength, "Denoising strength"),
                (hr_upscaler, "Hires upscaler"),
                (hr_force, "Hires force"),
                (hr_second_pass_steps, "Hires steps"),
                (hr_scale, "Hires upscale"),
                (hr_resize_x, "Hires resize-1"),
                (hr_resize_y, "Hires resize-2"),
                # refiner
                (refiner_start, "Refiner start"),
                (refiner_steps, "Refiner steps"),
                (refiner_prompt, "Prompt2"),
                (refiner_negative, "Negative2"),
                # hidden
                (seed_resize_from_w, "Seed resize from-1"),
                (seed_resize_from_h, "Seed resize from-2"),
                *modules.scripts.scripts_txt2img.infotext_fields
            ]
            generation_parameters_copypaste.add_paste_fields("txt2img", None, txt2img_paste_fields, override_settings)
            txt2img_bindings = generation_parameters_copypaste.ParamBinding(paste_button=txt2img_paste, tabname="txt2img", source_text_component=txt2img_prompt, source_image_component=None)
            generation_parameters_copypaste.register_paste_params_button(txt2img_bindings)

            txt2img_token_button.click(fn=wrap_queued_call(ui_common.update_token_counter), inputs=[txt2img_prompt, steps], outputs=[txt2img_token_counter])
            txt2img_negative_token_button.click(fn=wrap_queued_call(ui_common.update_token_counter), inputs=[txt2img_negative_prompt, steps], outputs=[txt2img_negative_token_counter])

            ui_extra_networks.setup_ui(extra_networks_ui, txt2img_gallery)
