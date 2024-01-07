import gradio as gr
from modules import shared, modelloader, ui_symbols, ui_common, sd_samplers
from modules.ui_components import FormRow, FormGroup, ToolButton, FormHTML


def create_toprow(is_img2img: bool = False, id_part: str = None):
    def apply_styles(prompt, prompt_neg, styles):
        prompt = shared.prompt_styles.apply_styles_to_prompt(prompt, styles)
        prompt_neg = shared.prompt_styles.apply_negative_styles_to_prompt(prompt_neg, styles)
        return [gr.Textbox.update(value=prompt), gr.Textbox.update(value=prompt_neg), gr.Dropdown.update(value=[])]


    def parse_style(styles):
        return styles.split('|')

    if id_part is None:
        id_part = "img2img" if is_img2img else "txt2img"
    with gr.Row(elem_id=f"{id_part}_toprow", variant="compact"):
        with gr.Column(elem_id=f"{id_part}_prompt_container", scale=6):
            with gr.Row():
                with gr.Column(scale=80):
                    with gr.Row():
                        prompt = gr.Textbox(elem_id=f"{id_part}_prompt", label="Prompt", show_label=False, lines=3, placeholder="Prompt", elem_classes=["prompt"])
            with gr.Row():
                with gr.Column(scale=80):
                    with gr.Row():
                        negative_prompt = gr.Textbox(elem_id=f"{id_part}_neg_prompt", label="Negative prompt", show_label=False, lines=3, placeholder="Negative prompt", elem_classes=["prompt"])
        with gr.Column(scale=1, elem_id=f"{id_part}_actions_column"):
            with gr.Row(elem_id=f"{id_part}_generate_box"):
                submit = gr.Button('Generate', elem_id=f"{id_part}_generate", variant='primary')
            with gr.Row(elem_id=f"{id_part}_generate_line2"):
                interrupt = gr.Button('Stop', elem_id=f"{id_part}_interrupt")
                interrupt.click(fn=lambda: shared.state.interrupt(), _js="requestInterrupt", inputs=[], outputs=[])
                skip = gr.Button('Skip', elem_id=f"{id_part}_skip")
                skip.click(fn=lambda: shared.state.skip(), inputs=[], outputs=[])
                pause = gr.Button('Pause', elem_id=f"{id_part}_pause")
                pause.click(fn=lambda: shared.state.pause(), _js='checkPaused', inputs=[], outputs=[])
            with gr.Row(elem_id=f"{id_part}_tools"):
                button_paste = gr.Button(value='Restore', variant='secondary', elem_id=f"{id_part}_paste") # symbols.paste
                button_clear = gr.Button(value='Clear', variant='secondary', elem_id=f"{id_part}_clear_prompt_btn") # symbols.clear
                button_extra = gr.Button(value='Networks', variant='secondary', elem_id=f"{id_part}_extra_networks_btn") # symbols.networks
                button_clear.click(fn=lambda *x: ['', ''], inputs=[prompt, negative_prompt], outputs=[prompt, negative_prompt], show_progress=False)
            with gr.Row(elem_id=f"{id_part}_counters"):
                token_counter = gr.HTML(value="<span>0/75</span>", elem_id=f"{id_part}_token_counter", elem_classes=["token-counter"])
                token_button = gr.Button(visible=False, elem_id=f"{id_part}_token_button")
                negative_token_counter = gr.HTML(value="<span>0/75</span>", elem_id=f"{id_part}_negative_token_counter", elem_classes=["token-counter"])
                negative_token_button = gr.Button(visible=False, elem_id=f"{id_part}_negative_token_button")
            with gr.Row(elem_id=f"{id_part}_styles_row"):
                styles = gr.Dropdown(label="Styles", elem_id=f"{id_part}_styles", choices=[style.name for style in shared.prompt_styles.styles.values()], value=[], multiselect=True)
                _styles_btn_refresh = ui_common.create_refresh_button(styles, shared.prompt_styles.reload, lambda: {"choices": list(shared.prompt_styles.styles)}, f"{id_part}_styles_refresh")
                # styles_btn_refresh = ToolButton(symbols.refresh, elem_id=f"{id_part}_styles_refresh", visible=True)
                # styles_btn_refresh.click(fn=lambda: gr.update(choices=[style.name for style in shared.prompt_styles.styles.values()]), inputs=[], outputs=[styles])
                styles_btn_select = gr.Button('Select', elem_id=f"{id_part}_styles_select", visible=False)
                styles_btn_select.click(_js="applyStyles", fn=parse_style, inputs=[styles], outputs=[styles])
                styles_btn_apply = ToolButton(ui_symbols.apply, elem_id=f"{id_part}_extra_apply", visible=False)
                styles_btn_apply.click(fn=apply_styles, inputs=[prompt, negative_prompt, styles], outputs=[prompt, negative_prompt, styles])
    return prompt, styles, negative_prompt, submit, button_paste, button_extra, token_counter, token_button, negative_token_counter, negative_token_button


def create_interrogate_buttons(tab):
    button_interrogate = gr.Button(ui_symbols.int_clip, elem_id=f"{tab}_interrogate", elem_classes=['interrogate-clip'])
    button_deepbooru = gr.Button(ui_symbols.int_blip, elem_id=f"{tab}_deepbooru", elem_classes=['interrogate-blip'])
    return button_interrogate, button_deepbooru


def create_sampler_inputs(tab, accordion=True):
    with gr.Accordion(open=False, label="Sampler", elem_id=f"{tab}_sampler", elem_classes=["small-accordion"]) if accordion else gr.Group():
        with FormRow(elem_id=f"{tab}_row_sampler"):
            sd_samplers.set_samplers()
            steps, sampler_index = create_sampler_and_steps_selection(sd_samplers.samplers, tab)
    return steps, sampler_index


def create_batch_inputs(tab):
    with gr.Accordion(open=False, label="Batch", elem_id=f"{tab}_batch", elem_classes=["small-accordion"]):
        with FormRow(elem_id=f"{tab}_row_batch"):
            batch_count = gr.Slider(minimum=1, step=1, label='Batch count', value=1, elem_id=f"{tab}_batch_count")
            batch_size = gr.Slider(minimum=1, maximum=32, step=1, label='Batch size', value=1, elem_id=f"{tab}_batch_size")
            batch_switch_btn = ToolButton(value=ui_symbols.switch, elem_id=f"{tab}_batch_switch_btn", label="Switch dims")
            batch_switch_btn.click(lambda w, h: (h, w), inputs=[batch_count, batch_size], outputs=[batch_count, batch_size], show_progress=False)
    return batch_count, batch_size


def create_seed_inputs(tab, reuse_visible=True):
    with gr.Accordion(open=False, label="Seed", elem_id=f"{tab}_seed_group", elem_classes=["small-accordion"]):
        with FormRow(elem_id=f"{tab}_seed_row", variant="compact"):
            seed = gr.Number(label='Initial seed', value=-1, elem_id=f"{tab}_seed", container=True)
            random_seed = ToolButton(ui_symbols.random, elem_id=f"{tab}_random_seed", label='Random seed')
            reuse_seed = ToolButton(ui_symbols.reuse, elem_id=f"{tab}_reuse_seed", label='Reuse seed', visible=reuse_visible)
        with FormRow(elem_id=f"{tab}_subseed_row", variant="compact", visible=shared.backend==shared.Backend.ORIGINAL):
            subseed = gr.Number(label='Variation', value=-1, elem_id=f"{tab}_subseed", container=True)
            random_subseed = ToolButton(ui_symbols.random, elem_id=f"{tab}_random_subseed")
            reuse_subseed = ToolButton(ui_symbols.reuse, elem_id=f"{tab}_reuse_subseed", visible=reuse_visible)
            subseed_strength = gr.Slider(label='Variation strength', value=0.0, minimum=0, maximum=1, step=0.01, elem_id=f"{tab}_subseed_strength")
        with FormRow(visible=False):
            seed_resize_from_w = gr.Slider(minimum=0, maximum=4096, step=8, label="Resize seed from width", value=0, elem_id=f"{tab}_seed_resize_from_w")
            seed_resize_from_h = gr.Slider(minimum=0, maximum=4096, step=8, label="Resize seed from height", value=0, elem_id=f"{tab}_seed_resize_from_h")
        random_seed.click(fn=lambda: [-1, -1], show_progress=False, inputs=[], outputs=[seed, subseed])
        random_subseed.click(fn=lambda: -1, show_progress=False, inputs=[], outputs=[subseed])
    return seed, reuse_seed, subseed, reuse_subseed, subseed_strength, seed_resize_from_h, seed_resize_from_w


def create_advanced_inputs(tab):
    with gr.Accordion(open=False, label="Advanced", elem_id=f"{tab}_advanced", elem_classes=["small-accordion"]):
        with gr.Group():
            with FormRow():
                cfg_scale = gr.Slider(minimum=0.0, maximum=30.0, step=0.1, label='CFG scale', value=6.0, elem_id=f"{tab}_cfg_scale")
                image_cfg_scale = gr.Slider(minimum=0.0, maximum=30.0, step=0.1, label='Secondary CFG scale', value=6.0, elem_id=f"{tab}_image_cfg_scale")
            with FormRow():
                diffusers_guidance_rescale = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, label='Guidance rescale', value=0.7, elem_id=f"{tab}_image_cfg_rescale", visible=shared.backend == shared.Backend.DIFFUSERS)
                # TODO enable SAG once fixed in diffusers
                # diffusers_sag_scale = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, label='Self-attention guidance', value=0.0, elem_id=f"{tab}_image_sag_scale", visible=shared.backend == shared.Backend.DIFFUSERS)
                diffusers_sag_scale = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, label='Self-attention guidance', value=0.0, elem_id=f"{tab}_image_sag_scale", visible=False)
            with FormRow():
                clip_skip = gr.Slider(label='CLIP skip', value=1, minimum=1, maximum=14, step=1, elem_id=f"{tab}_clip_skip", interactive=True)
        with gr.Group():
            with FormRow():
                full_quality = gr.Checkbox(label='Full quality', value=True, elem_id=f"{tab}_full_quality")
                restore_faces = gr.Checkbox(label='Face restore', value=False, visible=len(shared.face_restorers) > 1, elem_id=f"{tab}_restore_faces")
                tiling = gr.Checkbox(label='Tiling', value=False, elem_id=f"{tab}_tiling", visible=shared.backend == shared.Backend.ORIGINAL)
        with gr.Group(visible=shared.backend == shared.Backend.DIFFUSERS):
            with FormRow():
                hdr_clamp = gr.Checkbox(label='HDR clamp', value=False, elem_id=f"{tab}_hdr_clamp")
                hdr_boundary = gr.Slider(minimum=0.0, maximum=10.0, step=0.1, value=4.0,  label='Range', elem_id=f"{tab}_hdr_boundary")
                hdr_threshold = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.95,  label='Threshold', elem_id=f"{tab}_hdr_threshold")
            with FormRow():
                hdr_center = gr.Checkbox(label='HDR center', value=False, elem_id=f"{tab}_hdr_center")
                hdr_channel_shift = gr.Slider(minimum=0.0, maximum=2.0, step=0.1, value=1.0,  label='Channel shift', elem_id=f"{tab}_hdr_channel_shift")
                hdr_full_shift = gr.Slider(minimum=0.0, maximum=2.0, step=0.1, value=1,  label='Full shift', elem_id=f"{tab}_hdr_full_shift")
            with FormRow():
                hdr_maximize = gr.Checkbox(label='HDR maximize', value=False, elem_id=f"{tab}_hdr_maximize")
                hdr_max_center = gr.Slider(minimum=0.0, maximum=2.0, step=0.1, value=0.6,  label='Center', elem_id=f"{tab}_hdr_max_center")
                hdr_max_boundry = gr.Slider(minimum=0.5, maximum=2.0, step=0.1, value=1.0,  label='Range', elem_id=f"{tab}_hdr_max_boundry")
    return cfg_scale, clip_skip, image_cfg_scale, diffusers_guidance_rescale, diffusers_sag_scale, full_quality, restore_faces, tiling, hdr_clamp, hdr_boundary, hdr_threshold, hdr_center, hdr_channel_shift, hdr_full_shift, hdr_maximize, hdr_max_center, hdr_max_boundry


def create_sampler_and_steps_selection(choices, tabname):
    def set_sampler_original_options(sampler_options, sampler_algo):
        shared.opts.data['schedulers_brownian_noise'] = 'brownian noise' in sampler_options
        shared.opts.data['schedulers_discard_penultimate'] = 'discard penultimate sigma' in sampler_options
        shared.opts.data['schedulers_sigma'] = sampler_algo
        shared.opts.save(shared.config_filename, silent=True)

    def set_sampler_diffuser_options(sampler_options):
        shared.opts.data['schedulers_use_karras'] = 'karras' in sampler_options
        shared.opts.data['schedulers_use_thresholding'] = 'dynamic thresholding' in sampler_options
        shared.opts.data['schedulers_use_loworder'] = 'low order' in sampler_options
        shared.opts.data['schedulers_rescale_betas'] = 'rescale beta' in sampler_options
        shared.opts.save(shared.config_filename, silent=True)

    with FormRow(elem_classes=['flex-break']):
        sampler_index = gr.Dropdown(label='Sampling method', elem_id=f"{tabname}_sampling", choices=[x.name for x in choices], value='Default', type="index")
        steps = gr.Slider(minimum=1, maximum=99, step=1, label="Sampling steps", elem_id=f"{tabname}_steps", value=20)
    if shared.backend == shared.Backend.ORIGINAL:
        with FormRow(elem_classes=['flex-break']):
            choices = ['brownian noise', 'discard penultimate sigma']
            values = []
            values += ['brownian noise'] if shared.opts.data.get('schedulers_brownian_noise', False) else []
            values += ['discard penultimate sigma'] if shared.opts.data.get('schedulers_discard_penultimate', True) else []
            sampler_options = gr.CheckboxGroup(label='Sampler options', choices=choices, value=values, type='value')
        with FormRow(elem_classes=['flex-break']):
            shared.opts.data['schedulers_sigma'] = shared.opts.data.get('schedulers_sigma', 'default')
            sampler_algo = gr.Radio(label='Sigma algorithm', choices=['default', 'karras', 'exponential', 'polyexponential'], value=shared.opts.data['schedulers_sigma'], type='value')
        sampler_options.change(fn=set_sampler_original_options, inputs=[sampler_options, sampler_algo], outputs=[])
        sampler_algo.change(fn=set_sampler_original_options, inputs=[sampler_options, sampler_algo], outputs=[])
    else:
        with FormRow(elem_classes=['flex-break']):
            choices = ['karras', 'dynamic threshold', 'low order', 'rescale beta']
            values = []
            values += ['karras'] if shared.opts.data.get('schedulers_use_karras', True) else []
            values += ['dynamic threshold'] if shared.opts.data.get('schedulers_use_thresholding', False) else []
            values += ['low order'] if shared.opts.data.get('schedulers_use_loworder', True) else []
            values += ['rescale beta'] if shared.opts.data.get('schedulers_rescale_betas', False) else []
            sampler_options = gr.CheckboxGroup(label='Sampler options', choices=choices, value=values, type='value')
        sampler_options.change(fn=set_sampler_diffuser_options, inputs=[sampler_options], outputs=[])
    return steps, sampler_index


def create_hires_inputs(tab):
    with gr.Accordion(open=False, label="Second pass", elem_id=f"{tab}_second_pass", elem_classes=["small-accordion"]):
        with FormGroup():
            with FormRow(elem_id=f"{tab}_hires_row1"):
                enable_hr = gr.Checkbox(label='Enable second pass', value=False, elem_id=f"{tab}_enable_hr")
            with FormRow(elem_id=f"{tab}_hires_row2"):
                hr_sampler_index = gr.Dropdown(label='Secondary sampler', elem_id=f"{tab}_sampling_alt", choices=[x.name for x in sd_samplers.samplers], value='Default', type="index")
                denoising_strength = gr.Slider(minimum=0.0, maximum=0.99, step=0.01, label='Denoising strength', value=0.5, elem_id=f"{tab}_denoising_strength")
            with FormRow(elem_id=f"{tab}_hires_finalres", variant="compact"):
                hr_final_resolution = FormHTML(value="", elem_id=f"{tab}_hr_finalres", label="Upscaled resolution", interactive=False)
            with FormRow(elem_id=f"{tab}_hires_fix_row1", variant="compact"):
                hr_upscaler = gr.Dropdown(label="Upscaler", elem_id=f"{tab}_hr_upscaler", choices=[*shared.latent_upscale_modes, *[x.name for x in shared.sd_upscalers]], value=shared.latent_upscale_default_mode)
                hr_force = gr.Checkbox(label='Force Hires', value=False, elem_id=f"{tab}_hr_force")
            with FormRow(elem_id=f"{tab}_hires_fix_row2", variant="compact"):
                hr_second_pass_steps = gr.Slider(minimum=0, maximum=99, step=1, label='Hires steps', elem_id=f"{tab}_steps_alt", value=20)
                hr_scale = gr.Slider(minimum=1.0, maximum=8.0, step=0.05, label="Upscale by", value=2.0, elem_id=f"{tab}_hr_scale")
            with FormRow(elem_id=f"{tab}_hires_fix_row3", variant="compact"):
                hr_resize_x = gr.Slider(minimum=0, maximum=4096, step=8, label="Resize width to", value=0, elem_id=f"{tab}_hr_resize_x")
                hr_resize_y = gr.Slider(minimum=0, maximum=4096, step=8, label="Resize height to", value=0, elem_id=f"{tab}_hr_resize_y")
        with FormGroup(visible=shared.backend == shared.Backend.DIFFUSERS):
            with FormRow(elem_id=f"{tab}_refiner_row1", variant="compact"):
                refiner_start = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, label='Refiner start', value=0.8, elem_id=f"{tab}_refiner_start")
                refiner_steps = gr.Slider(minimum=0, maximum=99, step=1, label="Refiner steps", elem_id=f"{tab}_refiner_steps", value=5)
            with FormRow(elem_id=f"{tab}_refiner_row3", variant="compact"):
                refiner_prompt = gr.Textbox(value='', label='Secondary prompt', elem_id=f"{tab}_refiner_prompt")
            with FormRow(elem_id="txt2img_refiner_row4", variant="compact"):
                refiner_negative = gr.Textbox(value='', label='Secondary negative prompt', elem_id=f"{tab}_refiner_neg_prompt")
    return enable_hr, hr_sampler_index, denoising_strength, hr_final_resolution, hr_upscaler, hr_force, hr_second_pass_steps, hr_scale, hr_resize_x, hr_resize_y, refiner_steps, refiner_start, refiner_prompt, refiner_negative


def create_resize_inputs(tab, images, scale_visible=True, mode=None, accordion=True, latent=False):
    def resize_from_to_html(width, height, scale_by):
        target_width = int(width * scale_by)
        target_height = int(height * scale_by)
        if not target_width or not target_height:
            return "Hires resize: no image selected"
        return f"Hires resize: from <span class='resolution'>{width}x{height}</span> to <span class='resolution'>{target_width}x{target_height}</span>"

    dummy_component = gr.Number(visible=False, value=0)
    with gr.Accordion(open=False, label="Resize", elem_classes=["small-accordion"], elem_id=f"{tab}_resize_group") if accordion else gr.Group():
        with gr.Row():
            if mode is not None:
                resize_mode = gr.Radio(label="Resize mode", elem_id=f"{tab}_resize_mode", choices=shared.resize_modes, type="index", value=mode, visible=False)
            else:
                resize_mode = gr.Radio(label="Resize mode", elem_id=f"{tab}_resize_mode", choices=shared.resize_modes, type="index", value='None')
        with gr.Row():
            resize_name = gr.Dropdown(label="Resize method", elem_id=f"{tab}_resize_name", choices=([] if not latent else list(shared.latent_upscale_modes)) + [x.name for x in shared.sd_upscalers], value=shared.latent_upscale_default_mode)
            ui_common.create_refresh_button(resize_name, modelloader.load_upscalers, lambda: {"choices": modelloader.load_upscalers()}, 'refresh_upscalers')

        with FormRow(visible=True) as _resize_group:
            with gr.Column(elem_id=f"{tab}_column_size"):
                selected_scale_tab = gr.State(value=0) # pylint: disable=abstract-class-instantiated
                with gr.Tabs():
                    with gr.Tab(label="Resize to") as tab_scale_to:
                        with FormRow():
                            with gr.Column(elem_id=f"{tab}_column_size"):
                                with FormRow():
                                    width = gr.Slider(minimum=64, maximum=8192, step=8, label="Width", value=512, elem_id=f"{tab}_width")
                                    height = gr.Slider(minimum=64, maximum=8192, step=8, label="Height", value=512, elem_id=f"{tab}_height")
                                    res_switch_btn = ToolButton(value=ui_symbols.switch, elem_id=f"{tab}_res_switch_btn")
                                    res_switch_btn.click(lambda w, h: (h, w), inputs=[width, height], outputs=[width, height], show_progress=False)
                                    detect_image_size_btn = ToolButton(value=ui_symbols.detect, elem_id=f"{tab}_detect_image_size_btn")
                                    detect_image_size_btn.click(fn=lambda w, h, _: (w or gr.update(), h or gr.update()), _js="currentImg2imgSourceResolution", inputs=[dummy_component, dummy_component, dummy_component], outputs=[width, height], show_progress=False)

                    with gr.Tab(label="Resize by") as tab_scale_by:
                        scale_by = gr.Slider(minimum=0.05, maximum=8.0, step=0.05, label="Scale", value=1.0, elem_id=f"{tab}_scale")
                        if scale_visible:
                            with FormRow():
                                scale_by_html = FormHTML(resize_from_to_html(0, 0, 0.0), elem_id=f"{tab}_scale_resolution_preview")
                                gr.Slider(label="Unused", elem_id=f"{tab}_unused_scale_by_slider")
                                button_update_resize_to = gr.Button(visible=False, elem_id=f"{tab}_update_resize_to")

                            on_change_args = dict(fn=resize_from_to_html, _js="currentImg2imgSourceResolution", inputs=[dummy_component, dummy_component, scale_by], outputs=scale_by_html, show_progress=False)
                            scale_by.release(**on_change_args)
                            button_update_resize_to.click(**on_change_args)

                    for component in images:
                        component.change(fn=lambda: None, _js="updateImg2imgResizeToTextAfterChangingImage", inputs=[], outputs=[], show_progress=False)

            tab_scale_to.select(fn=lambda: 0, inputs=[], outputs=[selected_scale_tab])
            tab_scale_by.select(fn=lambda: 1, inputs=[], outputs=[selected_scale_tab])
            # resize_mode.change(fn=lambda x: gr.update(visible=x != 0), inputs=[resize_mode], outputs=[_resize_group])
    return resize_mode, resize_name, width, height, scale_by, selected_scale_tab
