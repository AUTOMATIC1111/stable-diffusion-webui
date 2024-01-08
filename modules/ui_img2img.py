import os
from PIL import Image
import gradio as gr
import numpy as np
from modules.call_queue import wrap_gradio_gpu_call, wrap_queued_call
from modules import shared, ui_common, ui_sections, generation_parameters_copypaste
from modules.ui_components import FormRow, FormGroup


def process_interrogate(interrogation_function, mode, ii_input_files, ii_input_dir, ii_output_dir, *ii_singles):
    mode = int(mode)
    if mode in {0, 1, 3, 4}:
        return [interrogation_function(ii_singles[mode]), None]
    if mode == 2:
        return [interrogation_function(ii_singles[mode]["image"]), None]
    if mode == 5:
        if len(ii_input_files) > 0:
            images = [f.name for f in ii_input_files]
        else:
            if not os.path.isdir(ii_input_dir):
                shared.log.error(f"Interrogate: Input directory not found: {ii_input_dir}")
                return [gr.update(), None]
            images = os.listdir(ii_input_dir)
        if ii_output_dir != "":
            os.makedirs(ii_output_dir, exist_ok=True)
        else:
            ii_output_dir = ii_input_dir
        for image in images:
            img = Image.open(image)
            filename = os.path.basename(image)
            left, _ = os.path.splitext(filename)
            print(interrogation_function(img), file=open(os.path.join(ii_output_dir, f"{left}.txt"), 'a', encoding='utf-8')) # pylint: disable=consider-using-with
    return [gr.update(), None]


def create_ui():
    shared.log.debug('UI initialize: img2img')
    import modules.img2img # pylint: disable=redefined-outer-name
    modules.scripts.scripts_current = modules.scripts.scripts_img2img
    modules.scripts.scripts_img2img.initialize_scripts(is_img2img=True)
    with gr.Blocks(analytics_enabled=False) as _img2img_interface:
        img2img_prompt, img2img_prompt_styles, img2img_negative_prompt, submit, img2img_paste, img2img_extra_networks_button, img2img_token_counter, img2img_token_button, img2img_negative_token_counter, img2img_negative_token_button = ui_sections.create_toprow(is_img2img=True, id_part="img2img")
        img2img_prompt_img = gr.File(label="", elem_id="img2img_prompt_image", file_count="single", type="binary", visible=False)

        with FormRow(variant='compact', elem_id="img2img_extra_networks", visible=False) as extra_networks_ui:
            from modules import ui_extra_networks
            extra_networks_ui_img2img = ui_extra_networks.create_ui(extra_networks_ui, img2img_extra_networks_button, 'img2img', skip_indexing=shared.opts.extra_network_skip_indexing)

        with FormRow(elem_id="img2img_interface", equal_height=False):
            with gr.Column(variant='compact', elem_id="img2img_settings"):
                copy_image_buttons = []
                copy_image_destinations = {}

                def copy_image(img):
                    return img['image'] if isinstance(img, dict) and 'image' in img else img

                def add_copy_image_controls(tab_name, elem):
                    with gr.Row(variant="compact", elem_id=f"img2img_copy_to_{tab_name}"):
                        for title, name in zip(['➠ Image', '➠ Sketch', '➠ Inpaint', '➠ Composite'], ['img2img', 'sketch', 'inpaint', 'inpaint_sketch']):
                            if name == tab_name:
                                gr.Button(title, interactive=False)
                                copy_image_destinations[name] = elem
                                continue
                            button = gr.Button(title)
                            copy_image_buttons.append((button, name, elem))

                with gr.Tabs(elem_id="mode_img2img"):
                    img2img_selected_tab = gr.State(0) # pylint: disable=abstract-class-instantiated
                    with gr.TabItem('Image', id='img2img', elem_id="img2img_img2img_tab") as tab_img2img:
                        init_img = gr.Image(label="Image for img2img", elem_id="img2img_image", show_label=False, source="upload", interactive=True, type="pil", tool="editor", image_mode="RGBA", height=512)
                        interrogate_clip, interrogate_booru = ui_sections.create_interrogate_buttons('img2img')
                        add_copy_image_controls('img2img', init_img)

                    with gr.TabItem('Sketch', id='img2img_sketch', elem_id="img2img_img2img_sketch_tab") as tab_sketch:
                        sketch = gr.Image(label="Image for img2img", elem_id="img2img_sketch", show_label=False, source="upload", interactive=True, type="pil", tool="color-sketch", image_mode="RGBA", height=512)
                        add_copy_image_controls('sketch', sketch)

                    with gr.TabItem('Inpaint', id='inpaint', elem_id="img2img_inpaint_tab") as tab_inpaint:
                        init_img_with_mask = gr.Image(label="Image for inpainting with mask", show_label=False, elem_id="img2maskimg", source="upload", interactive=True, type="pil", tool="sketch", image_mode="RGBA", height=512)
                        add_copy_image_controls('inpaint', init_img_with_mask)

                    with gr.TabItem('Composite', id='inpaint_sketch', elem_id="img2img_inpaint_sketch_tab") as tab_inpaint_color:
                        inpaint_color_sketch = gr.Image(label="Color sketch inpainting", show_label=False, elem_id="inpaint_sketch", source="upload", interactive=True, type="pil", tool="color-sketch", image_mode="RGBA", height=512)
                        inpaint_color_sketch_orig = gr.State(None) # pylint: disable=abstract-class-instantiated
                        add_copy_image_controls('inpaint_sketch', inpaint_color_sketch)

                        def update_orig(image, state):
                            if image is not None:
                                same_size = state is not None and state.size == image.size
                                has_exact_match = np.any(np.all(np.array(image) == np.array(state), axis=-1))
                                edited = same_size and has_exact_match
                                return image if not edited or state is None else state
                            return state

                        inpaint_color_sketch.change(update_orig, [inpaint_color_sketch, inpaint_color_sketch_orig], inpaint_color_sketch_orig)

                    with gr.TabItem('Upload', id='inpaint_upload', elem_id="img2img_inpaint_upload_tab") as tab_inpaint_upload:
                        init_img_inpaint = gr.Image(label="Image for img2img", show_label=False, source="upload", interactive=True, type="pil", elem_id="img_inpaint_base")
                        init_mask_inpaint = gr.Image(label="Mask", source="upload", interactive=True, type="pil", elem_id="img_inpaint_mask")

                    with gr.TabItem('Batch', id='batch', elem_id="img2img_batch_tab") as tab_batch:
                        hidden = '<br>Disabled when launched with --hide-ui-dir-config.' if shared.cmd_opts.hide_ui_dir_config else ''
                        gr.HTML(
                            "<p style='padding-bottom: 1em;' class=\"text-gray-500\">Upload images or process images in a directory" +
                            "<br>Add inpaint batch mask directory to enable inpaint batch processing"
                            f"{hidden}</p>"
                        )
                        img2img_batch_files = gr.Files(label="Batch Process", interactive=True, elem_id="img2img_image_batch")
                        img2img_batch_input_dir = gr.Textbox(label="Inpaint batch input directory", **shared.hide_dirs, elem_id="img2img_batch_input_dir")
                        img2img_batch_output_dir = gr.Textbox(label="Inpaint batch output directory", **shared.hide_dirs, elem_id="img2img_batch_output_dir")
                        img2img_batch_inpaint_mask_dir = gr.Textbox(label="Inpaint batch mask directory", **shared.hide_dirs, elem_id="img2img_batch_inpaint_mask_dir")

                    img2img_tabs = [tab_img2img, tab_sketch, tab_inpaint, tab_inpaint_color, tab_inpaint_upload, tab_batch]
                    for i, tab in enumerate(img2img_tabs):
                        tab.select(fn=lambda tabnum=i: tabnum, inputs=[], outputs=[img2img_selected_tab])

                for button, name, elem in copy_image_buttons:
                    button.click(fn=copy_image, inputs=[elem], outputs=[copy_image_destinations[name]])
                    button.click(fn=lambda: None, _js=f"switch_to_{name.replace(' ', '_')}", inputs=[], outputs=[])

                with FormGroup(elem_classes="settings-accordion"):

                    steps, sampler_index = ui_sections.create_sampler_inputs('img2img')
                    resize_mode, resize_name, width, height, scale_by, selected_scale_tab = ui_sections.create_resize_inputs('img2img', [init_img, sketch])
                    batch_count, batch_size = ui_sections.create_batch_inputs('img2img')
                    seed, reuse_seed, subseed, reuse_subseed, subseed_strength, seed_resize_from_h, seed_resize_from_w = ui_sections.create_seed_inputs('img2img')

                    with gr.Accordion(open=False, label="Denoise", elem_classes=["small-accordion"], elem_id="img2img_denoise_group"):
                        with FormRow():
                            denoising_strength = gr.Slider(minimum=0.0, maximum=0.99, step=0.01, label='Denoising strength', value=0.50, elem_id="img2img_denoising_strength")
                            refiner_start = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, label='Denoise start', value=0.0, elem_id="img2img_refiner_start")

                    cfg_scale, clip_skip, image_cfg_scale, diffusers_guidance_rescale, sag_scale, full_quality, restore_faces, tiling, hdr_clamp, hdr_boundary, hdr_threshold, hdr_center, hdr_channel_shift, hdr_full_shift, hdr_maximize, hdr_max_center, hdr_max_boundry = ui_sections.create_advanced_inputs('img2img')

                    # with FormGroup(elem_id="inpaint_controls", visible=False) as inpaint_controls:
                    with gr.Accordion(open=True, label="Mask", elem_classes=["small-accordion"], elem_id="img2img_mask_group") as inpaint_controls:
                        with FormRow():
                            mask_blur = gr.Slider(label='Mask blur', minimum=0, maximum=64, step=1, value=4, elem_id="img2img_mask_blur")
                            mask_alpha = gr.Slider(label="Mask transparency", visible=False, elem_id="img2img_mask_alpha")
                        with FormRow():
                            with gr.Column():
                                inpainting_mask_invert = gr.Radio(label='Mask mode', choices=['Inpaint masked', 'Inpaint not masked'], value='Inpaint masked', type="index", elem_id="img2img_mask_mode")
                            with gr.Column():
                                inpainting_fill = gr.Radio(label='Masked content', choices=['fill', 'original', 'noise', 'nothing'], value='original', type="index", elem_id="img2img_inpainting_fill")
                        with FormRow():
                            with gr.Column():
                                inpaint_full_res = gr.Radio(label="Inpaint area", choices=["Whole picture", "Only masked"], type="index", value="Whole picture", elem_id="img2img_inpaint_full_res")
                            with gr.Column():
                                inpaint_full_res_padding = gr.Slider(label='Masked padding', minimum=0, maximum=256, step=4, value=32, elem_id="img2img_inpaint_full_res_padding")

                        def select_img2img_tab(tab):
                            return gr.update(visible=tab in [2, 3, 4]), gr.update(visible=tab == 3)

                        for i, elem in enumerate(img2img_tabs):
                            elem.select(fn=lambda tab=i: select_img2img_tab(tab), inputs=[], outputs=[inpaint_controls, mask_alpha]) # pylint: disable=cell-var-from-loop

                override_settings = ui_common.create_override_inputs('img2img')

                with FormGroup(elem_id="img2img_script_container"):
                    img2img_script_inputs = modules.scripts.scripts_img2img.setup_ui(parent='img2img', accordion=True)

            img2img_gallery, img2img_generation_info, img2img_html_info, _img2img_html_info_formatted, img2img_html_log = ui_common.create_output_panel("img2img", prompt=None)

            ui_common.connect_reuse_seed(seed, reuse_seed, img2img_generation_info, is_subseed=False)
            ui_common.connect_reuse_seed(subseed, reuse_subseed, img2img_generation_info, is_subseed=True)

            img2img_prompt_img.change(fn=modules.images.image_data, inputs=[img2img_prompt_img], outputs=[img2img_prompt, img2img_prompt_img])
            dummy_component1 = gr.Textbox(visible=False, value='dummy')
            dummy_component2 = gr.Number(visible=False, value=0)
            img2img_args = [
                dummy_component1, dummy_component2,
                img2img_prompt, img2img_negative_prompt, img2img_prompt_styles,
                init_img,
                sketch,
                init_img_with_mask,
                inpaint_color_sketch,
                inpaint_color_sketch_orig,
                init_img_inpaint,
                init_mask_inpaint,
                steps,
                sampler_index,
                mask_blur, mask_alpha,
                inpainting_fill,
                full_quality, restore_faces, tiling,
                batch_count, batch_size,
                cfg_scale, image_cfg_scale,
                diffusers_guidance_rescale, sag_scale,
                refiner_start,
                clip_skip,
                denoising_strength,
                seed, subseed, subseed_strength, seed_resize_from_h, seed_resize_from_w,
                selected_scale_tab,
                height, width,
                scale_by,
                resize_mode, resize_name,
                inpaint_full_res, inpaint_full_res_padding, inpainting_mask_invert,
                img2img_batch_files, img2img_batch_input_dir, img2img_batch_output_dir, img2img_batch_inpaint_mask_dir,
                hdr_clamp, hdr_boundary, hdr_threshold, hdr_center, hdr_channel_shift, hdr_full_shift, hdr_maximize, hdr_max_center, hdr_max_boundry,
                override_settings,
            ]
            img2img_dict = dict(
                fn=wrap_gradio_gpu_call(modules.img2img.img2img, extra_outputs=[None, '', '']),
                _js="submit_img2img",
                inputs= img2img_args + img2img_script_inputs,
                outputs=[
                    img2img_gallery,
                    img2img_generation_info,
                    img2img_html_info,
                    img2img_html_log,
                ],
                show_progress=False,
            )
            img2img_prompt.submit(**img2img_dict)
            submit.click(**img2img_dict)
            dummy_component = gr.Textbox(visible=False, value='dummy')

            interrogate_args = dict(
                _js="get_img2img_tab_index",
                inputs=[
                    dummy_component,
                    img2img_batch_files,
                    img2img_batch_input_dir,
                    img2img_batch_output_dir,
                    init_img,
                    sketch,
                    init_img_with_mask,
                    inpaint_color_sketch,
                    init_img_inpaint,
                ],
                outputs=[img2img_prompt, dummy_component],
            )
            interrogate_clip.click(fn=lambda *args: process_interrogate(ui_common.interrogate_clip, *args), **interrogate_args)
            interrogate_booru.click(fn=lambda *args: process_interrogate(ui_common.interrogate_booru, *args), **interrogate_args)

            img2img_token_button.click(fn=wrap_queued_call(ui_common.update_token_counter), inputs=[img2img_prompt, steps], outputs=[img2img_token_counter])
            img2img_negative_token_button.click(fn=wrap_queued_call(ui_common.update_token_counter), inputs=[img2img_negative_prompt, steps], outputs=[img2img_negative_token_counter])

            ui_extra_networks.setup_ui(extra_networks_ui_img2img, img2img_gallery)
            img2img_paste_fields = [
                # prompt
                (img2img_prompt, "Prompt"),
                (img2img_negative_prompt, "Negative prompt"),
                # sampler
                (sampler_index, "Sampler"),
                (steps, "Steps"),
                # resize
                (resize_mode, "Resize mode"),
                (width, "Size-1"),
                (height, "Size-2"),
                (scale_by, "Resize scale"),
                # batch
                (batch_count, "Batch-1"),
                (batch_size, "Batch-2"),
                # seed
                (seed, "Seed"),
                (subseed, "Variation seed"),
                (subseed_strength, "Variation strength"),
                # denoise
                (denoising_strength, "Denoising strength"),
                (refiner_start, "Refiner start"),
                # advanced
                (cfg_scale, "CFG scale"),
                (image_cfg_scale, "Image CFG scale"),
                (clip_skip, "Clip skip"),
                (diffusers_guidance_rescale, "CFG rescale"),
                (full_quality, "Full quality"),
                (restore_faces, "Face restoration"),
                (tiling, "Tiling"),
                # inpaint
                (mask_blur, "Mask blur"),
                (mask_alpha, "Mask alpha"),
                (inpainting_mask_invert, "Mask invert"),
                (inpainting_fill, "Masked content"),
                (inpaint_full_res, "Mask area"),
                (inpaint_full_res_padding, "Masked padding"),
                # hidden
                (seed_resize_from_w, "Seed resize from-1"),
                (seed_resize_from_h, "Seed resize from-2"),
                *modules.scripts.scripts_img2img.infotext_fields
            ]
            generation_parameters_copypaste.add_paste_fields("img2img", init_img, img2img_paste_fields, override_settings)
            generation_parameters_copypaste.add_paste_fields("inpaint", init_img_with_mask, img2img_paste_fields, override_settings)
            img2img_bindings = generation_parameters_copypaste.ParamBinding(paste_button=img2img_paste, tabname="img2img", source_text_component=img2img_prompt, source_image_component=None)
            generation_parameters_copypaste.register_paste_params_button(img2img_bindings)
