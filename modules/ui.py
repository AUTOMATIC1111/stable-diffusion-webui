import json
import mimetypes
import os
import sys
from functools import reduce

import gradio as gr
import gradio.routes
import gradio.utils
import numpy as np
from PIL import Image
from modules.call_queue import wrap_gradio_gpu_call, wrap_queued_call, wrap_gradio_call

from modules import sd_hijack, sd_models, script_callbacks, ui_extensions, deepbooru, sd_vae, extra_networks, ui_common, ui_postprocessing
from modules.ui_components import FormRow, FormColumn, FormGroup, ToolButton, FormHTML # pylint: disable=unused-import
from modules.paths import script_path, data_path
from modules.shared import opts, cmd_opts
import modules.codeformer_model
import modules.generation_parameters_copypaste as parameters_copypaste
import modules.gfpgan_model
import modules.hypernetworks.ui
import modules.scripts
import modules.shared as shared
import modules.errors as errors
import modules.styles
import modules.textual_inversion.ui
from modules import prompt_parser
from modules.sd_hijack import model_hijack
from modules.sd_samplers import samplers, samplers_for_img2img
from modules.textual_inversion import textual_inversion
from modules.generation_parameters_copypaste import image_from_url_text
import modules.extras

errors.install()
mimetypes.init()
mimetypes.add_type('application/javascript', '.js')

if not cmd_opts.share and not cmd_opts.listen:
    # fix gradio phoning home
    gradio.utils.version_check = lambda: None
    gradio.utils.get_local_ip_address = lambda: '127.0.0.1'


def gr_show(visible=True):
    return {"visible": visible, "__type__": "update"}


sample_img2img = "assets/stable-samples/img2img/sketch-mountains-input.jpg"
sample_img2img = sample_img2img if os.path.exists(sample_img2img) else None

# Using constants for these since the variation selector isn't visible.
# Important that they exactly match script.js for tooltip to work.
random_symbol = '\U0001f3b2\ufe0f'  # 🎲️
reuse_symbol = '\u267b\ufe0f'  # ♻️
paste_symbol = '\u2199\ufe0f'  # ↙
refresh_symbol = '\U0001f504'  # 🔄
save_style_symbol = '\U0001f4be'  # 💾
apply_style_symbol = '\U0001f4cb'  # 📋
clear_prompt_symbol = '\U0001f5d1\ufe0f'  # 🗑️
extra_networks_symbol = '\U0001F3B4'  # 🎴
switch_values_symbol = '\U000021C5' # ⇅


def plaintext_to_html(text):
    return ui_common.plaintext_to_html(text)


def send_gradio_gallery_to_image(x):
    if len(x) == 0:
        return None
    return image_from_url_text(x[0])

def visit(x, func, path=""):
    if hasattr(x, 'children'):
        for c in x.children:
            visit(c, func, path)
    elif x.label is not None:
        func(path + "/" + str(x.label), x)


def add_style(name: str, prompt: str, negative_prompt: str):
    if name is None:
        return [gr_show() for x in range(4)]
    style = modules.styles.PromptStyle(name, prompt, negative_prompt)
    shared.prompt_styles.styles[style.name] = style
    shared.prompt_styles.save_styles(shared.opts.styles_dir)
    return [gr.Dropdown.update(visible=True, choices=list(shared.prompt_styles.styles)) for _ in range(2)]


def calc_resolution_hires(enable, width, height, hr_scale, hr_resize_x, hr_resize_y):
    from modules import processing, devices
    if not enable:
        return ""
    p = processing.StableDiffusionProcessingTxt2Img(width=width, height=height, enable_hr=True, hr_scale=hr_scale, hr_resize_x=hr_resize_x, hr_resize_y=hr_resize_y)
    with devices.autocast():
        p.init([""], [0], [0])
    return f"resize: from <span class='resolution'>{p.width}x{p.height}</span> to <span class='resolution'>{p.hr_resize_x or p.hr_upscale_to_x}x{p.hr_resize_y or p.hr_upscale_to_y}</span>"


def apply_styles(prompt, prompt_neg, styles):
    prompt = shared.prompt_styles.apply_styles_to_prompt(prompt, styles)
    prompt_neg = shared.prompt_styles.apply_negative_styles_to_prompt(prompt_neg, styles)
    return [gr.Textbox.update(value=prompt), gr.Textbox.update(value=prompt_neg), gr.Dropdown.update(value=[])]


def process_interrogate(interrogation_function, mode, ii_input_dir, ii_output_dir, *ii_singles):
    if mode in {0, 1, 3, 4}:
        return [interrogation_function(ii_singles[mode]), None]
    elif mode == 2:
        return [interrogation_function(ii_singles[mode]["image"]), None]
    elif mode == 5:
        assert not shared.cmd_opts.hide_ui_dir_config, "Launched with --hide-ui-dir-config, batch img2img disabled"
        images = shared.listfiles(ii_input_dir)
        print(f"Will process {len(images)} images.")
        if ii_output_dir != "":
            os.makedirs(ii_output_dir, exist_ok=True)
        else:
            ii_output_dir = ii_input_dir
        for image in images:
            img = Image.open(image)
            filename = os.path.basename(image)
            left, _ = os.path.splitext(filename)
            print(interrogation_function(img), file=open(os.path.join(ii_output_dir, left + ".txt"), 'a', encoding='utf-8'))

        return [gr.update(), None]


def interrogate(image):
    prompt = shared.interrogator.interrogate(image.convert("RGB"))
    return gr.update() if prompt is None else prompt


def interrogate_deepbooru(image):
    prompt = deepbooru.model.tag(image)
    return gr.update() if prompt is None else prompt


def change_clip_skip(val):
    shared.opts.CLIP_stop_at_last_layers = val


def create_seed_inputs(target_interface):
    with FormRow(elem_id=target_interface + '_seed_row', variant="compact"):
        seed = gr.Number(label='Seed', value=-1, elem_id=target_interface + '_seed')
        seed.style(container=False)
        random_seed = ToolButton(random_symbol, elem_id=target_interface + '_random_seed')
        reuse_seed = ToolButton(reuse_symbol, elem_id=target_interface + '_reuse_seed')
        seed_checkbox = gr.Checkbox(label='Extra', elem_id=target_interface + '_subseed_show', value=False, visible=False)  # Ghost checkbox, so it still gets sent. For compatibility with extensions that call txt2img or img2img manually
    with FormRow(visible=True, elem_id=target_interface + '_subseed_row'):
        subseed = gr.Number(label='Variation seed', value=-1, elem_id=target_interface + '_subseed')
        subseed.style(container=False)
        random_subseed = ToolButton(random_symbol, elem_id=target_interface + '_random_subseed')
        reuse_subseed = ToolButton(reuse_symbol, elem_id=target_interface + '_reuse_subseed')
        subseed_strength = gr.Slider(label='Strength', value=0.0, minimum=0, maximum=1, step=0.01, elem_id=target_interface + '_subseed_strength')
    with FormRow(visible=False):
        seed_resize_from_w = gr.Slider(minimum=0, maximum=2048, step=8, label="Resize seed from width", value=0, elem_id=target_interface + '_seed_resize_from_w')
        seed_resize_from_h = gr.Slider(minimum=0, maximum=2048, step=8, label="Resize seed from height", value=0, elem_id=target_interface + '_seed_resize_from_h')
    random_seed.click(fn=lambda: [-1, -1], show_progress=False, inputs=[], outputs=[seed, subseed])
    random_subseed.click(fn=lambda: -1, show_progress=False, inputs=[], outputs=[subseed])
    return seed, reuse_seed, subseed, reuse_subseed, subseed_strength, seed_resize_from_h, seed_resize_from_w, seed_checkbox


def connect_clear_prompt(button):
    """Given clear button, prompt, and token_counter objects, setup clear prompt button click event"""
    button.click(_js="clear_prompt", fn=None, inputs=[], outputs=[])


def connect_reuse_seed(seed: gr.Number, reuse_seed: gr.Button, generation_info: gr.Textbox, dummy_component, is_subseed):
    """ Connects a 'reuse (sub)seed' button's click event so that it copies last used
        (sub)seed value from generation info the to the seed field. If copying subseed and subseed strength
        was 0, i.e. no variation seed was used, it copies the normal seed value instead."""
    def copy_seed(gen_info_string: str, index):
        res = -1
        try:
            gen_info = json.loads(gen_info_string)
            index -= gen_info.get('index_of_first_image', 0)

            if is_subseed and gen_info.get('subseed_strength', 0) > 0:
                all_subseeds = gen_info.get('all_subseeds', [-1])
                res = all_subseeds[index if 0 <= index < len(all_subseeds) else 0]
            else:
                all_seeds = gen_info.get('all_seeds', [-1])
                res = all_seeds[index if 0 <= index < len(all_seeds) else 0]
        except json.decoder.JSONDecodeError:
            if gen_info_string != '':
                print("Error parsing JSON generation info:", file=sys.stderr)
                print(gen_info_string, file=sys.stderr)
        return [res, gr_show(False)]

    reuse_seed.click(fn=copy_seed, _js="(x, y) => [x, selected_gallery_index()]", show_progress=False, inputs=[generation_info, dummy_component], outputs=[seed, dummy_component])


def update_token_counter(text, steps):
    try:
        text, _ = extra_networks.parse_prompt(text)
        _, prompt_flat_list, _ = prompt_parser.get_multicond_prompt_list([text])
        prompt_schedules = prompt_parser.get_learned_conditioning_prompt_schedules(prompt_flat_list, steps)
    except Exception:
        # a parsing error can happen here during typing, and we don't want to bother the user with
        # messages related to it in console
        prompt_schedules = [[[steps, text]]]
    flat_prompts = reduce(lambda list1, list2: list1+list2, prompt_schedules)
    prompts = [prompt_text for step, prompt_text in flat_prompts]
    token_count, max_length = max([model_hijack.get_prompt_lengths(prompt) for prompt in prompts], key=lambda args: args[0])
    return f"<span class='gr-box gr-text-input'>{token_count}/{max_length}</span>"


def create_toprow(is_img2img):
    id_part = "img2img" if is_img2img else "txt2img"
    with gr.Row(elem_id=f"{id_part}_toprow", variant="compact"):
        with gr.Column(elem_id=f"{id_part}_prompt_container", scale=6):
            with gr.Row():
                with gr.Column(scale=80):
                    with gr.Row():
                        prompt = gr.Textbox(label="Prompt", elem_id=f"{id_part}_prompt", show_label=False, lines=3, placeholder="Prompt (press Ctrl+Enter or Alt+Enter to generate)")
            with gr.Row():
                with gr.Column(scale=80):
                    with gr.Row():
                        negative_prompt = gr.Textbox(label="Negative prompt", elem_id=f"{id_part}_neg_prompt", show_label=False, lines=3, placeholder="Negative prompt (press Ctrl+Enter or Alt+Enter to generate)")
        button_interrogate = None
        button_deepbooru = None
        if is_img2img:
            with gr.Column(scale=1, elem_classes="interrogate-col"):
                button_interrogate = gr.Button('Interrogate\nCLIP', elem_id="interrogate")
                button_deepbooru = gr.Button('Interrogate\nDeepBooru', elem_id="deepbooru")
        with gr.Column(scale=1, elem_id=f"{id_part}_actions_column"):
            with gr.Row(elem_id=f"{id_part}_generate_box", elem_classes="generate-box"):
                interrupt = gr.Button('Stop', elem_id=f"{id_part}_interrupt", elem_classes="generate-box-interrupt")
                skip = gr.Button('Skip', elem_id=f"{id_part}_skip", elem_classes="generate-box-skip")
                submit = gr.Button('Generate', elem_id=f"{id_part}_generate", variant='primary')
                skip.click(fn=lambda: shared.state.skip(), inputs=[], outputs=[])
                interrupt.click(fn=lambda: shared.state.interrupt(), inputs=[], outputs=[])
            with gr.Row(elem_id=f"{id_part}_tools"):
                paste = ToolButton(value=paste_symbol, elem_id="paste")
                clear_prompt_button = ToolButton(value=clear_prompt_symbol, elem_id=f"{id_part}_clear_prompt")
                extra_networks_button = ToolButton(value=extra_networks_symbol, elem_id=f"{id_part}_extra_networks")
                prompt_style_apply = ToolButton(value=apply_style_symbol, elem_id=f"{id_part}_style_apply")
                save_style = ToolButton(value=save_style_symbol, elem_id=f"{id_part}_style_create")
                token_counter = gr.HTML(value="<span>0/75</span>", elem_id=f"{id_part}_token_counter", elem_classes=["token-counter"])
                token_button = gr.Button(visible=False, elem_id=f"{id_part}_token_button")
                negative_token_counter = gr.HTML(value="<span>0/75</span>", elem_id=f"{id_part}_negative_token_counter", elem_classes=["token-counter"])
                negative_token_button = gr.Button(visible=False, elem_id=f"{id_part}_negative_token_button")
                clear_prompt_button.click(fn=lambda *x: x, _js="confirm_clear_prompt", inputs=[prompt, negative_prompt], outputs=[prompt, negative_prompt])
            with gr.Row(elem_id=f"{id_part}_styles_row"):
                prompt_styles = gr.Dropdown(label="Styles", elem_id=f"{id_part}_styles", choices=[k for k, v in shared.prompt_styles.styles.items()], value=[], multiselect=True)
                create_refresh_button(prompt_styles, shared.prompt_styles.reload, lambda: {"choices": [k for k, v in shared.prompt_styles.styles.items()]}, f"refresh_{id_part}_styles")
    return prompt, prompt_styles, negative_prompt, submit, button_interrogate, button_deepbooru, prompt_style_apply, save_style, paste, extra_networks_button, token_counter, token_button, negative_token_counter, negative_token_button


def setup_progressbar(*args, **kwargs): # pylint: disable=unused-argument
    pass


def apply_setting(key, value):
    if value is None:
        return gr.update()
    if shared.cmd_opts.freeze_settings:
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
        return
    valtype = type(opts.data_labels[key].default)
    oldval = opts.data.get(key, None)
    opts.data[key] = valtype(value) if valtype != type(None) else value
    if oldval != value and opts.data_labels[key].onchange is not None:
        opts.data_labels[key].onchange()
    opts.save(shared.config_filename)
    return getattr(opts, key)


def create_refresh_button(refresh_component, refresh_method, refreshed_args, elem_id):
    def refresh():
        refresh_method()
        args = refreshed_args() if callable(refreshed_args) else refreshed_args
        for k, v in args.items():
            setattr(refresh_component, k, v)
        return gr.update(**(args or {}))

    refresh_button = ToolButton(value=refresh_symbol, elem_id=elem_id)
    refresh_button.click(fn=refresh, inputs=[], outputs=[refresh_component])
    return refresh_button


def create_output_panel(tabname, outdir):
    return ui_common.create_output_panel(tabname, outdir)


def create_sampler_and_steps_selection(choices, tabname):
    with FormRow(elem_id=f"sampler_selection_{tabname}"):
        sampler_index = gr.Dropdown(label='Sampling method', elem_id=f"{tabname}_sampling", choices=[x.name for x in choices], value="UniPC" if tabname == 'txt2img' else "Euler a", type="index")
        steps = gr.Slider(minimum=1, maximum=150, step=1, elem_id=f"{tabname}_steps", label="Sampling steps", value=10 if tabname == 'txt2img' else 20)
    return steps, sampler_index


def ordered_ui_categories():
    user_order = {x.strip(): i * 2 + 1 for i, x in enumerate(shared.opts.ui_reorder.split(","))}
    for i, category in sorted(enumerate(shared.ui_reorder_categories), key=lambda x: user_order.get(x[1], x[0] * 2 + 0)):
        yield category


def get_value_for_setting(key):
    value = getattr(opts, key)
    info = opts.data_labels[key]
    args = info.component_args() if callable(info.component_args) else info.component_args or {}
    args = {k: v for k, v in args.items() if k not in {'precision'}}
    return gr.update(value=value, **args)


def create_override_settings_dropdown(tabname, row): # pylint: disable=unused-argument
    dropdown = gr.Dropdown([], label="Override settings", visible=False, elem_id=f"{tabname}_override_settings", multiselect=True)
    dropdown.change(fn=lambda x: gr.Dropdown.update(visible=len(x) > 0), inputs=[dropdown], outputs=[dropdown])
    return dropdown


def create_ui():
    import modules.img2img # pylint: disable=redefined-outer-name
    import modules.txt2img # pylint: disable=redefined-outer-name
    reload_javascript()
    parameters_copypaste.reset()
    modules.scripts.scripts_current = modules.scripts.scripts_txt2img
    modules.scripts.scripts_txt2img.initialize_scripts(is_img2img=False)
    with gr.Blocks(analytics_enabled=False) as txt2img_interface:
        txt2img_prompt, txt2img_prompt_styles, txt2img_negative_prompt, submit, _, _, txt2img_prompt_style_apply, txt2img_save_style, txt2img_paste, extra_networks_button, token_counter, token_button, negative_token_counter, negative_token_button = create_toprow(is_img2img=False)
        dummy_component = gr.Label(visible=False)
        txt_prompt_img = gr.File(label="", elem_id="txt2img_prompt_image", file_count="single", type="binary", visible=False)
        with FormRow(variant='compact', elem_id="txt2img_extra_networks", visible=False) as extra_networks_ui:
            from modules import ui_extra_networks
            extra_networks_ui = ui_extra_networks.create_ui(extra_networks_ui, extra_networks_button, 'txt2img')
        with gr.Row().style(equal_height=False):
            with gr.Column(variant='compact', elem_id="txt2img_settings"):
                for category in ordered_ui_categories():
                    if category == "sampler":
                        steps, sampler_index = create_sampler_and_steps_selection(samplers, "txt2img")
                    elif category == "dimensions":
                        with FormRow():
                            with gr.Column(elem_id="txt2img_column_size", scale=4):
                                with FormRow(elem_id="txt2img_row_dimension"):
                                    width = gr.Slider(minimum=64, maximum=2048, step=8, label="Width", value=512, elem_id="txt2img_width")
                                    height = gr.Slider(minimum=64, maximum=2048, step=8, label="Height", value=512, elem_id="txt2img_height")
                            with gr.Column(elem_id="txt2img_dimensions_row", scale=1, elem_classes="dimensions-tools"):
                                res_switch_btn = ToolButton(value=switch_values_symbol, elem_id="txt2img_res_switch_btn")
                            with gr.Column(elem_id="txt2img_column_batch"):
                                with FormRow(elem_id="txt2img_row_batch"):
                                    batch_count = gr.Slider(minimum=1, step=1, label='Batch count', value=1, elem_id="txt2img_batch_count")
                                    batch_size = gr.Slider(minimum=1, maximum=32, step=1, label='Batch size', value=1, elem_id="txt2img_batch_size")
                    elif category == "cfg":
                        with FormRow():
                            cfg_scale = gr.Slider(minimum=1.0, maximum=30.0, step=0.5, label='CFG Scale', value=6.0, elem_id="txt2img_cfg_scale")
                            clip_skip = gr.Slider(label='CLIP Skip', value=shared.opts.CLIP_stop_at_last_layers, minimum=1, maximum=4, step=1, elem_id='txt2img_clip_skip', interactive=True)
                            clip_skip.change(fn=change_clip_skip, show_progress=False, inputs=clip_skip)
                    elif category == "seed":
                        seed, reuse_seed, subseed, reuse_subseed, subseed_strength, seed_resize_from_h, seed_resize_from_w, seed_checkbox = create_seed_inputs('txt2img')
                    elif category == "checkboxes":
                        with FormRow(elem_classes="checkboxes-row", variant="compact"):
                            restore_faces = gr.Checkbox(label='Restore faces', value=False, visible=len(shared.face_restorers) > 1, elem_id="txt2img_restore_faces")
                            tiling = gr.Checkbox(label='Tiling', value=False, elem_id="txt2img_tiling")
                            enable_hr = gr.Checkbox(label='Hires fix', value=False, elem_id="txt2img_enable_hr")
                            hr_final_resolution = FormHTML(value="", elem_id="txtimg_hr_finalres", label="Upscaled resolution", interactive=False)
                    elif category == "hires_fix":
                        with FormGroup(visible=False, elem_id="txt2img_hires_fix") as hr_options:
                            with FormRow(elem_id="txt2img_hires_fix_row1", variant="compact"):
                                hr_upscaler = gr.Dropdown(label="Upscaler", elem_id="txt2img_hr_upscaler", choices=[*shared.latent_upscale_modes, *[x.name for x in shared.sd_upscalers]], value=shared.latent_upscale_default_mode)
                                hr_second_pass_steps = gr.Slider(minimum=0, maximum=150, step=1, label='Hires steps', value=0, elem_id="txt2img_hires_steps")
                                denoising_strength = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label='Denoising strength', value=0.7, elem_id="txt2img_denoising_strength")
                            with FormRow(elem_id="txt2img_hires_fix_row2", variant="compact"):
                                hr_scale = gr.Slider(minimum=1.0, maximum=4.0, step=0.05, label="Upscale by", value=2.0, elem_id="txt2img_hr_scale")
                                hr_resize_x = gr.Slider(minimum=0, maximum=2048, step=8, label="Resize width to", value=0, elem_id="txt2img_hr_resize_x")
                                hr_resize_y = gr.Slider(minimum=0, maximum=2048, step=8, label="Resize height to", value=0, elem_id="txt2img_hr_resize_y")
                    elif category == "override_settings":
                        with FormRow(elem_id="txt2img_override_settings_row") as row:
                            override_settings = create_override_settings_dropdown('txt2img', row)
                    elif category == "scripts":
                        with FormGroup(elem_id="txt2img_script_container"):
                            custom_inputs = modules.scripts.scripts_txt2img.setup_ui()
            hr_resolution_preview_inputs = [enable_hr, width, height, hr_scale, hr_resize_x, hr_resize_y]
            for preview_input in hr_resolution_preview_inputs:
                preview_input.change(
                    fn=calc_resolution_hires,
                    inputs=hr_resolution_preview_inputs,
                    outputs=[hr_final_resolution],
                    show_progress=False,
                )
                preview_input.change(
                    None,
                    _js="onCalcResolutionHires",
                    inputs=hr_resolution_preview_inputs,
                    outputs=[],
                    show_progress=False,
                )

            txt2img_gallery, generation_info, html_info, html_log = create_output_panel("txt2img", opts.outdir_txt2img_samples)
            connect_reuse_seed(seed, reuse_seed, generation_info, dummy_component, is_subseed=False)
            connect_reuse_seed(subseed, reuse_subseed, generation_info, dummy_component, is_subseed=True)

            txt2img_args = dict(
                fn=wrap_gradio_gpu_call(modules.txt2img.txt2img, extra_outputs=[None, '', '']),
                _js="submit",
                inputs=[
                    dummy_component,
                    txt2img_prompt,
                    txt2img_negative_prompt,
                    txt2img_prompt_styles,
                    steps,
                    sampler_index,
                    restore_faces,
                    tiling,
                    batch_count,
                    batch_size,
                    cfg_scale,
                    seed,
                    subseed, subseed_strength, seed_resize_from_h, seed_resize_from_w,
                    seed_checkbox,  # seed_enable_extras
                    height,
                    width,
                    enable_hr,
                    denoising_strength,
                    hr_scale,
                    hr_upscaler,
                    hr_second_pass_steps,
                    hr_resize_x,
                    hr_resize_y,
                    override_settings,
                ] + custom_inputs,

                outputs=[
                    txt2img_gallery,
                    generation_info,
                    html_info,
                    html_log,
                ],
                show_progress=False,
            )

            txt2img_prompt.submit(**txt2img_args)
            submit.click(**txt2img_args)

            res_switch_btn.click(lambda w, h: (h, w), inputs=[width, height], outputs=[width, height], show_progress=False)

            txt_prompt_img.change(
                fn=modules.images.image_data,
                inputs=[
                    txt_prompt_img
                ],
                outputs=[
                    txt2img_prompt,
                    txt_prompt_img
                ]
            )

            enable_hr.change(
                fn=lambda x: gr_show(x),
                inputs=[enable_hr],
                outputs=[hr_options],
                show_progress = False,
            )

            txt2img_paste_fields = [
                (txt2img_prompt, "Prompt"),
                (txt2img_negative_prompt, "Negative prompt"),
                (steps, "Steps"),
                (sampler_index, "Sampler"),
                (restore_faces, "Face restoration"),
                (cfg_scale, "CFG scale"),
                (seed, "Seed"),
                (width, "Size-1"),
                (height, "Size-2"),
                (batch_size, "Batch size"),
                (subseed, "Variation seed"),
                (subseed_strength, "Variation strength"),
                (seed_resize_from_w, "Seed resize from-1"),
                (seed_resize_from_h, "Seed resize from-2"),
                (denoising_strength, "Denoising strength"),
                (enable_hr, lambda d: "Denoising strength" in d),
                (hr_options, lambda d: gr.Row.update(visible="Denoising strength" in d)),
                (hr_scale, "Hires upscale"),
                (hr_upscaler, "Hires upscaler"),
                (hr_second_pass_steps, "Hires steps"),
                (hr_resize_x, "Hires resize-1"),
                (hr_resize_y, "Hires resize-2"),
                *modules.scripts.scripts_txt2img.infotext_fields
            ]
            parameters_copypaste.add_paste_fields("txt2img", None, txt2img_paste_fields, override_settings)
            parameters_copypaste.register_paste_params_button(parameters_copypaste.ParamBinding(
                paste_button=txt2img_paste, tabname="txt2img", source_text_component=txt2img_prompt, source_image_component=None,
            ))

            txt2img_preview_params = [
                txt2img_prompt,
                txt2img_negative_prompt,
                steps,
                sampler_index,
                cfg_scale,
                seed,
                width,
                height,
            ]

            token_button.click(fn=wrap_queued_call(update_token_counter), inputs=[txt2img_prompt, steps], outputs=[token_counter])
            negative_token_button.click(fn=wrap_queued_call(update_token_counter), inputs=[txt2img_negative_prompt, steps], outputs=[negative_token_counter])

            ui_extra_networks.setup_ui(extra_networks_ui, txt2img_gallery)

    modules.scripts.scripts_current = modules.scripts.scripts_img2img
    modules.scripts.scripts_img2img.initialize_scripts(is_img2img=True)

    with gr.Blocks(analytics_enabled=False) as img2img_interface:
        img2img_prompt, img2img_prompt_styles, img2img_negative_prompt, submit, img2img_interrogate, img2img_deepbooru, img2img_prompt_style_apply, img2img_save_style, img2img_paste, extra_networks_button, token_counter, token_button, negative_token_counter, negative_token_button = create_toprow(is_img2img=True)

        img2img_prompt_img = gr.File(label="", elem_id="img2img_prompt_image", file_count="single", type="binary", visible=False)

        with FormRow(variant='compact', elem_id="img2img_extra_networks", visible=False) as extra_networks_ui:
            from modules import ui_extra_networks
            extra_networks_ui_img2img = ui_extra_networks.create_ui(extra_networks_ui, extra_networks_button, 'img2img')

        with FormRow().style(equal_height=False):
            with gr.Column(variant='compact', elem_id="img2img_settings"):
                copy_image_buttons = []
                copy_image_destinations = {}

                def add_copy_image_controls(tab_name, elem):
                    with gr.Row(variant="compact", elem_id=f"img2img_copy_to_{tab_name}"):
                        gr.HTML("Copy image to: ", elem_id=f"img2img_label_copy_to_{tab_name}")

                        for title, name in zip(['img2img', 'sketch', 'inpaint', 'inpaint sketch'], ['img2img', 'sketch', 'inpaint', 'inpaint_sketch']):
                            if name == tab_name:
                                gr.Button(title, interactive=False)
                                copy_image_destinations[name] = elem
                                continue

                            button = gr.Button(title)
                            copy_image_buttons.append((button, name, elem))

                with gr.Tabs(elem_id="mode_img2img"):
                    with gr.TabItem('img2img', id='img2img', elem_id="img2img_img2img_tab") as tab_img2img:
                        init_img = gr.Image(label="Image for img2img", elem_id="img2img_image", show_label=False, source="upload", interactive=True, type="pil", tool="editor", image_mode="RGBA").style(height=480)
                        add_copy_image_controls('img2img', init_img)

                    with gr.TabItem('Sketch', id='img2img_sketch', elem_id="img2img_img2img_sketch_tab") as tab_sketch:
                        sketch = gr.Image(label="Image for img2img", elem_id="img2img_sketch", show_label=False, source="upload", interactive=True, type="pil", tool="color-sketch", image_mode="RGBA").style(height=480)
                        add_copy_image_controls('sketch', sketch)

                    with gr.TabItem('Inpaint', id='inpaint', elem_id="img2img_inpaint_tab") as tab_inpaint:
                        init_img_with_mask = gr.Image(label="Image for inpainting with mask", show_label=False, elem_id="img2maskimg", source="upload", interactive=True, type="pil", tool="sketch", image_mode="RGBA").style(height=480)
                        add_copy_image_controls('inpaint', init_img_with_mask)

                    with gr.TabItem('Inpaint sketch', id='inpaint_sketch', elem_id="img2img_inpaint_sketch_tab") as tab_inpaint_color:
                        inpaint_color_sketch = gr.Image(label="Color sketch inpainting", show_label=False, elem_id="inpaint_sketch", source="upload", interactive=True, type="pil", tool="color-sketch", image_mode="RGBA").style(height=480)
                        inpaint_color_sketch_orig = gr.State(None) # pylint: disable=abstract-class-instantiated
                        add_copy_image_controls('inpaint_sketch', inpaint_color_sketch)

                        def update_orig(image, state):
                            if image is not None:
                                same_size = state is not None and state.size == image.size
                                has_exact_match = np.any(np.all(np.array(image) == np.array(state), axis=-1))
                                edited = same_size and has_exact_match
                                return image if not edited or state is None else state

                        inpaint_color_sketch.change(update_orig, [inpaint_color_sketch, inpaint_color_sketch_orig], inpaint_color_sketch_orig)

                    with gr.TabItem('Inpaint upload', id='inpaint_upload', elem_id="img2img_inpaint_upload_tab") as tab_inpaint_upload:
                        init_img_inpaint = gr.Image(label="Image for img2img", show_label=False, source="upload", interactive=True, type="pil", elem_id="img_inpaint_base")
                        init_mask_inpaint = gr.Image(label="Mask", source="upload", interactive=True, type="pil", elem_id="img_inpaint_mask")

                    with gr.TabItem('Batch', id='batch', elem_id="img2img_batch_tab") as tab_batch:
                        hidden = '<br>Disabled when launched with --hide-ui-dir-config.' if shared.cmd_opts.hide_ui_dir_config else ''
                        gr.HTML(
                            "<p style='padding-bottom: 1em;' class=\"text-gray-500\">Process images in a directory on the same machine where the server is running." +
                            "<br>Use an empty output directory to save pictures normally instead of writing to the output directory." +
                            "<br>Add inpaint batch mask directory to enable inpaint batch processing."
                            f"{hidden}</p>"
                        )
                        img2img_batch_input_dir = gr.Textbox(label="Input directory", **shared.hide_dirs, elem_id="img2img_batch_input_dir")
                        img2img_batch_output_dir = gr.Textbox(label="Output directory", **shared.hide_dirs, elem_id="img2img_batch_output_dir")
                        img2img_batch_inpaint_mask_dir = gr.Textbox(label="Inpaint batch mask directory (required for inpaint batch processing only)", **shared.hide_dirs, elem_id="img2img_batch_inpaint_mask_dir")

                def copy_image(img):
                    if isinstance(img, dict) and 'image' in img:
                        return img['image']

                    return img

                for button, name, elem in copy_image_buttons:
                    button.click(
                        fn=copy_image,
                        inputs=[elem],
                        outputs=[copy_image_destinations[name]],
                    )
                    button.click(
                        fn=lambda: None,
                        _js="switch_to_"+name.replace(" ", "_"),
                        inputs=[],
                        outputs=[],
                    )

                with FormRow():
                    resize_mode = gr.Radio(label="Resize mode", elem_id="resize_mode", choices=["Just resize", "Crop and resize", "Resize and fill", "Just resize (latent upscale)"], type="index", value="Just resize")

                for category in ordered_ui_categories():
                    if category == "sampler":
                        steps, sampler_index = create_sampler_and_steps_selection(samplers_for_img2img, "img2img")

                    elif category == "dimensions":
                        with FormRow():
                            with FormRow(elem_id="img2img_row_size"):
                                width = gr.Slider(minimum=64, maximum=2048, step=8, label="Width", value=512, elem_id="img2img_width")
                                height = gr.Slider(minimum=64, maximum=2048, step=8, label="Height", value=512, elem_id="img2img_height")

                            with gr.Column(elem_id="img2img_dimensions_row", scale=1, elem_classes="dimensions-tools"):
                                res_switch_btn = ToolButton(value=switch_values_symbol, elem_id="img2img_res_switch_btn")

                            with FormRow(elem_id="img2img_row_batch"):
                                batch_count = gr.Slider(minimum=1, step=1, label='Batch count', value=1, elem_id="img2img_batch_count")
                                batch_size = gr.Slider(minimum=1, maximum=32, step=1, label='Batch size', value=1, elem_id="img2img_batch_size")

                    elif category == "cfg":
                        with FormGroup():
                            with FormRow():
                                cfg_scale = gr.Slider(minimum=1.0, maximum=30.0, step=0.5, label='CFG Scale', value=6.0, elem_id="img2img_cfg_scale")
                                image_cfg_scale = gr.Slider(minimum=0, maximum=3.0, step=0.05, label='Image CFG Scale', value=1.5, elem_id="img2img_image_cfg_scale", visible=shared.sd_model and shared.sd_model.cond_stage_key == "edit")
                                denoising_strength = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label='Denoising strength', value=0.75, elem_id="img2img_denoising_strength")
                                clip_skip = gr.Slider(label='CLIP Skip', value=1, minimum=1, maximum=4, step=1, elem_id='img2img_clip_skip', interactive=True)
                                clip_skip.change(fn=change_clip_skip, show_progress=False, inputs=clip_skip)

                    elif category == "seed":
                        seed, reuse_seed, subseed, reuse_subseed, subseed_strength, seed_resize_from_h, seed_resize_from_w, seed_checkbox = create_seed_inputs('img2img')

                    elif category == "checkboxes":
                        with FormRow(elem_classes="checkboxes-row", variant="compact"):
                            restore_faces = gr.Checkbox(label='Restore faces', value=False, visible=len(shared.face_restorers) > 1, elem_id="img2img_restore_faces")
                            tiling = gr.Checkbox(label='Tiling', value=False, elem_id="img2img_tiling")

                    elif category == "override_settings":
                        with FormRow(elem_id="img2img_override_settings_row") as row:
                            override_settings = create_override_settings_dropdown('img2img', row)

                    elif category == "scripts":
                        with FormGroup(elem_id="img2img_script_container"):
                            custom_inputs = modules.scripts.scripts_img2img.setup_ui()

                    elif category == "inpaint":
                        with FormGroup(elem_id="inpaint_controls", visible=False) as inpaint_controls:
                            with FormRow():
                                mask_blur = gr.Slider(label='Mask blur', minimum=0, maximum=64, step=1, value=4, elem_id="img2img_mask_blur")
                                mask_alpha = gr.Slider(label="Mask transparency", visible=False, elem_id="img2img_mask_alpha")

                            with FormRow():
                                inpainting_mask_invert = gr.Radio(label='Mask mode', choices=['Inpaint masked', 'Inpaint not masked'], value='Inpaint masked', type="index", elem_id="img2img_mask_mode")

                            with FormRow():
                                inpainting_fill = gr.Radio(label='Masked content', choices=['fill', 'original', 'latent noise', 'latent nothing'], value='original', type="index", elem_id="img2img_inpainting_fill")

                            with FormRow():
                                with gr.Column():
                                    inpaint_full_res = gr.Radio(label="Inpaint area", choices=["Whole picture", "Only masked"], type="index", value="Whole picture", elem_id="img2img_inpaint_full_res")

                                with gr.Column(scale=4):
                                    inpaint_full_res_padding = gr.Slider(label='Only masked padding, pixels', minimum=0, maximum=256, step=4, value=32, elem_id="img2img_inpaint_full_res_padding")

                            def select_img2img_tab(tab):
                                return gr.update(visible=tab in [2, 3, 4]), gr.update(visible=tab == 3),

                            for i, elem in enumerate([tab_img2img, tab_sketch, tab_inpaint, tab_inpaint_color, tab_inpaint_upload, tab_batch]):
                                elem.select(
                                    fn=lambda tab=i: select_img2img_tab(tab), # pylint: disable=cell-var-from-loop
                                    inputs=[],
                                    outputs=[inpaint_controls, mask_alpha],
                                )

            img2img_gallery, generation_info, html_info, html_log = create_output_panel("img2img", opts.outdir_img2img_samples)

            connect_reuse_seed(seed, reuse_seed, generation_info, dummy_component, is_subseed=False)
            connect_reuse_seed(subseed, reuse_subseed, generation_info, dummy_component, is_subseed=True)

            img2img_prompt_img.change(
                fn=modules.images.image_data,
                inputs=[
                    img2img_prompt_img
                ],
                outputs=[
                    img2img_prompt,
                    img2img_prompt_img
                ]
            )

            img2img_args = dict(
                fn=wrap_gradio_gpu_call(modules.img2img.img2img, extra_outputs=[None, '', '']),
                _js="submit_img2img",
                inputs=[
                    dummy_component,
                    dummy_component,
                    img2img_prompt,
                    img2img_negative_prompt,
                    img2img_prompt_styles,
                    init_img,
                    sketch,
                    init_img_with_mask,
                    inpaint_color_sketch,
                    inpaint_color_sketch_orig,
                    init_img_inpaint,
                    init_mask_inpaint,
                    steps,
                    sampler_index,
                    mask_blur,
                    mask_alpha,
                    inpainting_fill,
                    restore_faces,
                    tiling,
                    batch_count,
                    batch_size,
                    cfg_scale,
                    image_cfg_scale,
                    denoising_strength,
                    seed,
                    subseed, subseed_strength, seed_resize_from_h, seed_resize_from_w,
                    seed_checkbox,  # seed_enable_extras
                    height,
                    width,
                    resize_mode,
                    inpaint_full_res,
                    inpaint_full_res_padding,
                    inpainting_mask_invert,
                    img2img_batch_input_dir,
                    img2img_batch_output_dir,
                    img2img_batch_inpaint_mask_dir,
                    override_settings,
                ] + custom_inputs,
                outputs=[
                    img2img_gallery,
                    generation_info,
                    html_info,
                    html_log,
                ],
                show_progress=False,
            )

            interrogate_args = dict(
                _js="get_img2img_tab_index",
                inputs=[
                    dummy_component,
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

            img2img_prompt.submit(**img2img_args)
            submit.click(**img2img_args)
            res_switch_btn.click(lambda w, h: (h, w), inputs=[width, height], outputs=[width, height], show_progress=False)

            img2img_interrogate.click(
                fn=lambda *args: process_interrogate(interrogate, *args),
                **interrogate_args,
            )

            img2img_deepbooru.click(
                fn=lambda *args: process_interrogate(interrogate_deepbooru, *args),
                **interrogate_args,
            )

            prompts = [(txt2img_prompt, txt2img_negative_prompt), (img2img_prompt, img2img_negative_prompt)]
            style_dropdowns = [txt2img_prompt_styles, img2img_prompt_styles]
            style_js_funcs = ["update_txt2img_tokens", "update_img2img_tokens"]

            for button, (prompt, negative_prompt) in zip([txt2img_save_style, img2img_save_style], prompts):
                button.click(
                    fn=add_style,
                    _js="ask_for_style_name",
                    # Have to pass empty dummy component here, because the JavaScript and Python function have to accept
                    # the same number of parameters, but we only know the style-name after the JavaScript prompt
                    inputs=[dummy_component, prompt, negative_prompt],
                    outputs=[txt2img_prompt_styles, img2img_prompt_styles],
                )

            for button, (prompt, negative_prompt), styles, js_func in zip([txt2img_prompt_style_apply, img2img_prompt_style_apply], prompts, style_dropdowns, style_js_funcs):
                button.click(
                    fn=apply_styles,
                    _js=js_func,
                    inputs=[prompt, negative_prompt, styles],
                    outputs=[prompt, negative_prompt, styles],
                )

            token_button.click(fn=update_token_counter, inputs=[img2img_prompt, steps], outputs=[token_counter])
            negative_token_button.click(fn=wrap_queued_call(update_token_counter), inputs=[img2img_negative_prompt, steps], outputs=[negative_token_counter])

            ui_extra_networks.setup_ui(extra_networks_ui_img2img, img2img_gallery)

            img2img_paste_fields = [
                (img2img_prompt, "Prompt"),
                (img2img_negative_prompt, "Negative prompt"),
                (steps, "Steps"),
                (sampler_index, "Sampler"),
                (restore_faces, "Face restoration"),
                (cfg_scale, "CFG scale"),
                (image_cfg_scale, "Image CFG scale"),
                (seed, "Seed"),
                (width, "Size-1"),
                (height, "Size-2"),
                (batch_size, "Batch size"),
                (subseed, "Variation seed"),
                (subseed_strength, "Variation strength"),
                (seed_resize_from_w, "Seed resize from-1"),
                (seed_resize_from_h, "Seed resize from-2"),
                (denoising_strength, "Denoising strength"),
                (mask_blur, "Mask blur"),
                *modules.scripts.scripts_img2img.infotext_fields
            ]
            parameters_copypaste.add_paste_fields("img2img", init_img, img2img_paste_fields, override_settings)
            parameters_copypaste.add_paste_fields("inpaint", init_img_with_mask, img2img_paste_fields, override_settings)
            parameters_copypaste.register_paste_params_button(parameters_copypaste.ParamBinding(
                paste_button=img2img_paste, tabname="img2img", source_text_component=img2img_prompt, source_image_component=None,
            ))

    modules.scripts.scripts_current = None

    with gr.Blocks(analytics_enabled=False) as extras_interface:
        ui_postprocessing.create_ui()

    def update_interp_description(value):
        interp_description_css = "<p style='margin-bottom: 2.5em'>{}</p>"
        interp_descriptions = {
            "No interpolation": interp_description_css.format("No interpolation will be used. Requires one model; A. Allows for format conversion and VAE baking."),
            "Weighted sum": interp_description_css.format("A weighted sum will be used for interpolation. Requires two models; A and B. The result is calculated as A * (1 - M) + B * M"),
            "Add difference": interp_description_css.format("The difference between the last two models will be added to the first. Requires three models; A, B and C. The result is calculated as A + (B - C) * M")
        }
        return interp_descriptions[value]

    with gr.Blocks(analytics_enabled=False) as train_interface:
        with gr.Column(elem_id='ti_train_container'):
            with gr.Tabs(elem_id="train_tabs"):

                with gr.Tab(label="Merge models") as modelmerger_interface:
                    with gr.Row().style(equal_height=False):
                        with gr.Column(variant='compact'):
                            interp_description = gr.HTML(value=update_interp_description("Weighted sum"), elem_id="modelmerger_interp_description")

                            with FormRow(elem_id="modelmerger_models"):
                                primary_model_name = gr.Dropdown(modules.sd_models.checkpoint_tiles(), elem_id="modelmerger_primary_model_name", label="Primary model (A)")
                                create_refresh_button(primary_model_name, modules.sd_models.list_models, lambda: {"choices": modules.sd_models.checkpoint_tiles()}, "refresh_checkpoint_A")

                                secondary_model_name = gr.Dropdown(modules.sd_models.checkpoint_tiles(), elem_id="modelmerger_secondary_model_name", label="Secondary model (B)")
                                create_refresh_button(secondary_model_name, modules.sd_models.list_models, lambda: {"choices": modules.sd_models.checkpoint_tiles()}, "refresh_checkpoint_B")

                                tertiary_model_name = gr.Dropdown(modules.sd_models.checkpoint_tiles(), elem_id="modelmerger_tertiary_model_name", label="Tertiary model (C)")
                                create_refresh_button(tertiary_model_name, modules.sd_models.list_models, lambda: {"choices": modules.sd_models.checkpoint_tiles()}, "refresh_checkpoint_C")

                            custom_name = gr.Textbox(label="Custom Name (Optional)", elem_id="modelmerger_custom_name")
                            interp_amount = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, label='Multiplier (M) - set to 0 to get model A', value=0.3, elem_id="modelmerger_interp_amount")
                            interp_method = gr.Radio(choices=["No interpolation", "Weighted sum", "Add difference"], value="Weighted sum", label="Interpolation Method", elem_id="modelmerger_interp_method")
                            interp_method.change(fn=update_interp_description, inputs=[interp_method], outputs=[interp_description])

                            with FormRow():
                                checkpoint_format = gr.Radio(choices=["ckpt", "safetensors"], value="ckpt", label="Checkpoint format", elem_id="modelmerger_checkpoint_format")
                                save_as_half = gr.Checkbox(value=False, label="Save as float16", elem_id="modelmerger_save_as_half")

                            with FormRow():
                                with gr.Column():
                                    config_source = gr.Radio(choices=["A, B or C", "B", "C", "Don't"], value="A, B or C", label="Copy config from", type="index", elem_id="modelmerger_config_method")

                                with gr.Column():
                                    with FormRow():
                                        bake_in_vae = gr.Dropdown(choices=["None"] + list(sd_vae.vae_dict), value="None", label="Bake in VAE", elem_id="modelmerger_bake_in_vae")
                                        create_refresh_button(bake_in_vae, sd_vae.refresh_vae_list, lambda: {"choices": ["None"] + list(sd_vae.vae_dict)}, "modelmerger_refresh_bake_in_vae")

                            with FormRow():
                                discard_weights = gr.Textbox(value="", label="Discard weights with matching name", elem_id="modelmerger_discard_weights")

                            with gr.Row():
                                modelmerger_merge = gr.Button(elem_id="modelmerger_merge", value="Merge", variant='primary')

                        with gr.Column(variant='compact', elem_id="modelmerger_results_container"):
                            with gr.Group(elem_id="modelmerger_results_panel"):
                                modelmerger_result = gr.HTML(elem_id="modelmerger_result", show_label=False)

                with gr.Tab(label="Create embedding"):
                    new_embedding_name = gr.Textbox(label="Name", elem_id="train_new_embedding_name")
                    initialization_text = gr.Textbox(label="Initialization text", value="*", elem_id="train_initialization_text")
                    nvpt = gr.Slider(label="Number of vectors per token", minimum=1, maximum=75, step=1, value=1, elem_id="train_nvpt")
                    overwrite_old_embedding = gr.Checkbox(value=False, label="Overwrite Old Embedding", elem_id="train_overwrite_old_embedding")

                    with gr.Row():
                        with gr.Column(scale=3):
                            gr.HTML(value="")

                        with gr.Column():
                            create_embedding = gr.Button(value="Create embedding", variant='primary', elem_id="train_create_embedding")

                with gr.Tab(label="Create hypernetwork"):
                    new_hypernetwork_name = gr.Textbox(label="Name", elem_id="train_new_hypernetwork_name")
                    new_hypernetwork_sizes = gr.CheckboxGroup(label="Modules", value=["768", "320", "640", "1280"], choices=["768", "1024", "320", "640", "1280"], elem_id="train_new_hypernetwork_sizes")
                    new_hypernetwork_layer_structure = gr.Textbox("1, 2, 1", label="Enter hypernetwork layer structure", placeholder="1st and last digit must be 1. ex:'1, 2, 1'", elem_id="train_new_hypernetwork_layer_structure")
                    new_hypernetwork_activation_func = gr.Dropdown(value="linear", label="Select activation function of hypernetwork. Recommended : Swish / Linear(none)", choices=modules.hypernetworks.ui.keys, elem_id="train_new_hypernetwork_activation_func")
                    new_hypernetwork_initialization_option = gr.Dropdown(value = "Normal", label="Select Layer weights initialization. Recommended: Kaiming for relu-like, Xavier for sigmoid-like, Normal otherwise", choices=["Normal", "KaimingUniform", "KaimingNormal", "XavierUniform", "XavierNormal"], elem_id="train_new_hypernetwork_initialization_option")
                    new_hypernetwork_add_layer_norm = gr.Checkbox(label="Add layer normalization", elem_id="train_new_hypernetwork_add_layer_norm")
                    new_hypernetwork_use_dropout = gr.Checkbox(label="Use dropout", elem_id="train_new_hypernetwork_use_dropout")
                    new_hypernetwork_dropout_structure = gr.Textbox("0, 0, 0", label="Enter hypernetwork Dropout structure (or empty). Recommended : 0~0.35 incrementing sequence: 0, 0.05, 0.15", placeholder="1st and last digit must be 0 and values should be between 0 and 1. ex:'0, 0.01, 0'")
                    overwrite_old_hypernetwork = gr.Checkbox(value=False, label="Overwrite Old Hypernetwork", elem_id="train_overwrite_old_hypernetwork")

                    with gr.Row():
                        with gr.Column(scale=3):
                            gr.HTML(value="")

                        with gr.Column():
                            create_hypernetwork = gr.Button(value="Create hypernetwork", variant='primary', elem_id="train_create_hypernetwork")

                with gr.Tab(label="Preprocess images"):
                    process_src = gr.Textbox(label='Source directory', elem_id="train_process_src")
                    process_dst = gr.Textbox(label='Destination directory', elem_id="train_process_dst")
                    process_width = gr.Slider(minimum=64, maximum=2048, step=8, label="Width", value=512, elem_id="train_process_width")
                    process_height = gr.Slider(minimum=64, maximum=2048, step=8, label="Height", value=512, elem_id="train_process_height")
                    preprocess_txt_action = gr.Dropdown(label='Existing Caption txt Action', value="ignore", choices=["ignore", "copy", "prepend", "append"], elem_id="train_preprocess_txt_action")

                    with gr.Row():
                        process_flip = gr.Checkbox(label='Create flipped copies', elem_id="train_process_flip")
                        process_split = gr.Checkbox(label='Split oversized images', elem_id="train_process_split")
                        process_focal_crop = gr.Checkbox(label='Auto focal point crop', elem_id="train_process_focal_crop")
                        process_multicrop = gr.Checkbox(label='Auto-sized crop', elem_id="train_process_multicrop")
                        process_caption = gr.Checkbox(label='Use BLIP for caption', elem_id="train_process_caption")
                        process_caption_deepbooru = gr.Checkbox(label='Use deepbooru for caption', visible=True, elem_id="train_process_caption_deepbooru")

                    with gr.Row(visible=False) as process_split_extra_row:
                        process_split_threshold = gr.Slider(label='Split image threshold', value=0.5, minimum=0.0, maximum=1.0, step=0.05, elem_id="train_process_split_threshold")
                        process_overlap_ratio = gr.Slider(label='Split image overlap ratio', value=0.2, minimum=0.0, maximum=0.9, step=0.05, elem_id="train_process_overlap_ratio")

                    with gr.Row(visible=False) as process_focal_crop_row:
                        process_focal_crop_face_weight = gr.Slider(label='Focal point face weight', value=0.9, minimum=0.0, maximum=1.0, step=0.05, elem_id="train_process_focal_crop_face_weight")
                        process_focal_crop_entropy_weight = gr.Slider(label='Focal point entropy weight', value=0.15, minimum=0.0, maximum=1.0, step=0.05, elem_id="train_process_focal_crop_entropy_weight")
                        process_focal_crop_edges_weight = gr.Slider(label='Focal point edges weight', value=0.5, minimum=0.0, maximum=1.0, step=0.05, elem_id="train_process_focal_crop_edges_weight")
                        process_focal_crop_debug = gr.Checkbox(label='Create debug image', elem_id="train_process_focal_crop_debug")

                    with gr.Column(visible=False) as process_multicrop_col:
                        gr.Markdown('Each image is center-cropped with an automatically chosen width and height.')
                        with gr.Row():
                            process_multicrop_mindim = gr.Slider(minimum=64, maximum=2048, step=8, label="Dimension lower bound", value=384, elem_id="train_process_multicrop_mindim")
                            process_multicrop_maxdim = gr.Slider(minimum=64, maximum=2048, step=8, label="Dimension upper bound", value=768, elem_id="train_process_multicrop_maxdim")
                        with gr.Row():
                            process_multicrop_minarea = gr.Slider(minimum=64*64, maximum=2048*2048, step=1, label="Area lower bound", value=64*64, elem_id="train_process_multicrop_minarea")
                            process_multicrop_maxarea = gr.Slider(minimum=64*64, maximum=2048*2048, step=1, label="Area upper bound", value=640*640, elem_id="train_process_multicrop_maxarea")
                        with gr.Row():
                            process_multicrop_objective = gr.Radio(["Maximize area", "Minimize error"], value="Maximize area", label="Resizing objective", elem_id="train_process_multicrop_objective")
                            process_multicrop_threshold = gr.Slider(minimum=0, maximum=1, step=0.01, label="Error threshold", value=0.1, elem_id="train_process_multicrop_threshold")

                    with gr.Row():
                        with gr.Column(scale=3):
                            gr.HTML(value="")

                        with gr.Column():
                            with gr.Row():
                                interrupt_preprocessing = gr.Button("Stop", elem_id="train_interrupt_preprocessing")
                            run_preprocess = gr.Button(value="Preprocess", variant='primary', elem_id="train_run_preprocess")

                    process_split.change(
                        fn=lambda show: gr_show(show),
                        inputs=[process_split],
                        outputs=[process_split_extra_row],
                    )

                    process_focal_crop.change(
                        fn=lambda show: gr_show(show),
                        inputs=[process_focal_crop],
                        outputs=[process_focal_crop_row],
                    )

                    process_multicrop.change(
                        fn=lambda show: gr_show(show),
                        inputs=[process_multicrop],
                        outputs=[process_multicrop_col],
                    )

                def get_textual_inversion_template_names():
                    return sorted([x for x in textual_inversion.textual_inversion_templates])

                with gr.Tab(label="Train"):
                    gr.HTML(value="<p style='margin-bottom: 0.7em'>Train an embedding or Hypernetwork; you must specify a directory with a set of 1:1 ratio images</p>")
                    with FormRow():
                        train_embedding_name = gr.Dropdown(label='Embedding', elem_id="train_embedding", choices=sorted(sd_hijack.model_hijack.embedding_db.word_embeddings.keys()))
                        create_refresh_button(train_embedding_name, sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings, lambda: {"choices": sorted(sd_hijack.model_hijack.embedding_db.word_embeddings.keys())}, "refresh_train_embedding_name")

                        train_hypernetwork_name = gr.Dropdown(label='Hypernetwork', elem_id="train_hypernetwork", choices=[x for x in shared.hypernetworks.keys()])
                        create_refresh_button(train_hypernetwork_name, shared.reload_hypernetworks, lambda: {"choices": sorted([x for x in shared.hypernetworks.keys()])}, "refresh_train_hypernetwork_name")

                    with FormRow():
                        embedding_learn_rate = gr.Textbox(label='Embedding Learning rate', placeholder="Embedding Learning rate", value="0.005", elem_id="train_embedding_learn_rate")
                        hypernetwork_learn_rate = gr.Textbox(label='Hypernetwork Learning rate', placeholder="Hypernetwork Learning rate", value="0.00001", elem_id="train_hypernetwork_learn_rate")

                    with FormRow():
                        clip_grad_mode = gr.Dropdown(value="disabled", label="Gradient Clipping", choices=["disabled", "value", "norm"])
                        clip_grad_value = gr.Textbox(placeholder="Gradient clip value", value="0.1", show_label=False)

                    with FormRow():
                        batch_size = gr.Number(label='Batch size', value=1, precision=0, elem_id="train_batch_size")
                        gradient_step = gr.Number(label='Gradient accumulation steps', value=1, precision=0, elem_id="train_gradient_step")

                    dataset_directory = gr.Textbox(label='Dataset directory', placeholder="Path to directory with input images", elem_id="train_dataset_directory")
                    log_directory = gr.Textbox(label='Log directory', placeholder="Path to directory where to write outputs", value="train/log/embeddings", elem_id="train_log_directory")

                    with FormRow():
                        template_file = gr.Dropdown(label='Prompt template', value="style_filewords.txt", elem_id="train_template_file", choices=get_textual_inversion_template_names())
                        create_refresh_button(template_file, textual_inversion.list_textual_inversion_templates, lambda: {"choices": get_textual_inversion_template_names()}, "refrsh_train_template_file")

                    training_width = gr.Slider(minimum=64, maximum=2048, step=8, label="Width", value=512, elem_id="train_training_width")
                    training_height = gr.Slider(minimum=64, maximum=2048, step=8, label="Height", value=512, elem_id="train_training_height")
                    varsize = gr.Checkbox(label="Do not resize images", value=False, elem_id="train_varsize")
                    steps = gr.Number(label='Max steps', value=1000, precision=0, elem_id="train_steps")

                    with FormRow():
                        create_image_every = gr.Number(label='Save an image to log directory every N steps, 0 to disable', value=500, precision=0, elem_id="train_create_image_every")
                        save_embedding_every = gr.Number(label='Save a copy of embedding to log directory every N steps, 0 to disable', value=500, precision=0, elem_id="train_save_embedding_every")

                    use_weight = gr.Checkbox(label="Use PNG alpha channel as loss weight", value=False, elem_id="use_weight")

                    save_image_with_stored_embedding = gr.Checkbox(label='Save images with embedding in PNG chunks', value=True, elem_id="train_save_image_with_stored_embedding")
                    preview_from_txt2img = gr.Checkbox(label='Read parameters (prompt, etc...) from txt2img tab when making previews', value=False, elem_id="train_preview_from_txt2img")

                    shuffle_tags = gr.Checkbox(label="Shuffle tags by ',' when creating prompts.", value=False, elem_id="train_shuffle_tags")
                    tag_drop_out = gr.Slider(minimum=0, maximum=1, step=0.1, label="Drop out tags when creating prompts.", value=0, elem_id="train_tag_drop_out")

                    latent_sampling_method = gr.Radio(label='Choose latent sampling method', value="once", choices=['once', 'deterministic', 'random'], elem_id="train_latent_sampling_method")

                    with gr.Row():
                        train_embedding = gr.Button(value="Train Embedding", variant='primary', elem_id="train_train_embedding")
                        interrupt_training = gr.Button(value="Stop", elem_id="train_interrupt_training")
                        train_hypernetwork = gr.Button(value="Train Hypernetwork", variant='primary', elem_id="train_train_hypernetwork")

                params = script_callbacks.UiTrainTabParams(txt2img_preview_params)

                script_callbacks.ui_train_tabs_callback(params)

            with gr.Column(elem_id='ti_gallery_container'):
                ti_output = gr.Text(elem_id="ti_output", value="", show_label=False)
                _ti_gallery = gr.Gallery(label='Output', show_label=False, elem_id='ti_gallery').style(grid=4)
                _ti_progress = gr.HTML(elem_id="ti_progress", value="")
                ti_outcome = gr.HTML(elem_id="ti_error", value="")

        create_embedding.click(
            fn=modules.textual_inversion.ui.create_embedding,
            inputs=[
                new_embedding_name,
                initialization_text,
                nvpt,
                overwrite_old_embedding,
            ],
            outputs=[
                train_embedding_name,
                ti_output,
                ti_outcome,
            ]
        )

        create_hypernetwork.click(
            fn=modules.hypernetworks.ui.create_hypernetwork,
            inputs=[
                new_hypernetwork_name,
                new_hypernetwork_sizes,
                overwrite_old_hypernetwork,
                new_hypernetwork_layer_structure,
                new_hypernetwork_activation_func,
                new_hypernetwork_initialization_option,
                new_hypernetwork_add_layer_norm,
                new_hypernetwork_use_dropout,
                new_hypernetwork_dropout_structure
            ],
            outputs=[
                train_hypernetwork_name,
                ti_output,
                ti_outcome,
            ]
        )

        run_preprocess.click(
            fn=wrap_gradio_gpu_call(modules.textual_inversion.ui.preprocess, extra_outputs=[gr.update()]),
            _js="start_training_textual_inversion",
            inputs=[
                dummy_component,
                process_src,
                process_dst,
                process_width,
                process_height,
                preprocess_txt_action,
                process_flip,
                process_split,
                process_caption,
                process_caption_deepbooru,
                process_split_threshold,
                process_overlap_ratio,
                process_focal_crop,
                process_focal_crop_face_weight,
                process_focal_crop_entropy_weight,
                process_focal_crop_edges_weight,
                process_focal_crop_debug,
                process_multicrop,
                process_multicrop_mindim,
                process_multicrop_maxdim,
                process_multicrop_minarea,
                process_multicrop_maxarea,
                process_multicrop_objective,
                process_multicrop_threshold,
            ],
            outputs=[
                ti_output,
                ti_outcome,
            ],
        )

        train_embedding.click(
            fn=wrap_gradio_gpu_call(modules.textual_inversion.ui.train_embedding, extra_outputs=[gr.update()]),
            _js="start_training_textual_inversion",
            inputs=[
                dummy_component,
                train_embedding_name,
                embedding_learn_rate,
                batch_size,
                gradient_step,
                dataset_directory,
                log_directory,
                training_width,
                training_height,
                varsize,
                steps,
                clip_grad_mode,
                clip_grad_value,
                shuffle_tags,
                tag_drop_out,
                latent_sampling_method,
                use_weight,
                create_image_every,
                save_embedding_every,
                template_file,
                save_image_with_stored_embedding,
                preview_from_txt2img,
                *txt2img_preview_params,
            ],
            outputs=[
                ti_output,
                ti_outcome,
            ]
        )

        train_hypernetwork.click(
            fn=wrap_gradio_gpu_call(modules.hypernetworks.ui.train_hypernetwork, extra_outputs=[gr.update()]),
            _js="start_training_textual_inversion",
            inputs=[
                dummy_component,
                train_hypernetwork_name,
                hypernetwork_learn_rate,
                batch_size,
                gradient_step,
                dataset_directory,
                log_directory,
                training_width,
                training_height,
                varsize,
                steps,
                clip_grad_mode,
                clip_grad_value,
                shuffle_tags,
                tag_drop_out,
                latent_sampling_method,
                use_weight,
                create_image_every,
                save_embedding_every,
                template_file,
                preview_from_txt2img,
                *txt2img_preview_params,
            ],
            outputs=[
                ti_output,
                ti_outcome,
            ]
        )

        interrupt_training.click(
            fn=lambda: shared.state.interrupt(),
            inputs=[],
            outputs=[],
        )

        interrupt_preprocessing.click(
            fn=lambda: shared.state.interrupt(),
            inputs=[],
            outputs=[],
        )

    def create_setting_component(key, is_quicksettings=False):
        def fun():
            return opts.data[key] if key in opts.data else opts.data_labels[key].default

        info = opts.data_labels[key]
        t = type(info.default)
        args = info.component_args() if callable(info.component_args) else info.component_args
        if info.component is not None:
            comp = info.component
        elif t == str:
            comp = gr.Textbox
        elif t == int:
            comp = gr.Number
        elif t == bool:
            comp = gr.Checkbox
        else:
            raise ValueError(f'bad options item type: {str(t)} for key {key}')
        elem_id = "setting_"+key
        if info.refresh is not None:
            if is_quicksettings:
                res = comp(label=info.label, value=fun(), elem_id=elem_id, **(args or {}))
                create_refresh_button(res, info.refresh, info.component_args, "refresh_" + key)
            else:
                with FormRow():
                    res = comp(label=info.label, value=fun(), elem_id=elem_id, **(args or {}))
                    create_refresh_button(res, info.refresh, info.component_args, "refresh_" + key)
        else:
            res = comp(label=info.label, value=fun(), elem_id=elem_id, **(args or {}))
        return res

    components = []
    component_dict = {}
    shared.settings_components = component_dict

    script_callbacks.ui_settings_callback()
    opts.reorder()

    def run_settings(*args):
        changed = []

        for key, value, comp in zip(opts.data_labels.keys(), args, components):
            assert comp == dummy_component or opts.same_type(value, opts.data_labels[key].default), f"Bad value for setting {key}: {value}; expecting {type(opts.data_labels[key].default).__name__}"

        for key, value, comp in zip(opts.data_labels.keys(), args, components):
            if comp == dummy_component:
                continue

            if opts.set(key, value):
                changed.append(key)

        try:
            opts.save(shared.config_filename)
        except RuntimeError:
            return opts.dumpjson(), f'{len(changed)} settings changed without save: {", ".join(changed)}.'
        return opts.dumpjson(), f'{len(changed)} settings changed{": " if len(changed) > 0 else ""}{", ".join(changed)}.'

    def run_settings_single(value, key):
        if not opts.same_type(value, opts.data_labels[key].default):
            return gr.update(visible=True), opts.dumpjson()

        if not opts.set(key, value):
            return gr.update(value=getattr(opts, key)), opts.dumpjson()

        opts.save(shared.config_filename)

        return get_value_for_setting(key), opts.dumpjson()

    with gr.Blocks(analytics_enabled=False) as settings_interface:
        with gr.Row():
            settings_submit = gr.Button(value="Apply settings", variant='primary', elem_id="settings_submit")
            restart_submit = gr.Button(value="Restart UI", variant='primary', elem_id="restart_submit")
            preview_theme = gr.Button(value="Preview theme", variant='primary', elem_id="settings_preview_theme")
            unload_sd_model = gr.Button(value='Unload checkpoint', variant='primary', elem_id="sett_unload_sd_model")
            reload_sd_model = gr.Button(value='Reload checkpoint', variant='primary', elem_id="sett_reload_sd_model")
            reload_script_bodies = gr.Button(value='Reload scripts', variant='primary', elem_id="settings_reload_script_bodies")

        result = gr.HTML(elem_id="settings_result")

        quicksettings_names = [x.strip() for x in opts.quicksettings.split(",")]
        quicksettings_names = {x: i for i, x in enumerate(quicksettings_names) if x != 'quicksettings'}
        quicksettings_list = []
        previous_section = None
        current_tab = None
        current_row = None
        with gr.Tabs(elem_id="settings"):
            for i, (k, item) in enumerate(opts.data_labels.items()):
                section_must_be_skipped = item.section[0] is None
                if previous_section != item.section and not section_must_be_skipped:
                    elem_id, text = item.section
                    if current_tab is not None:
                        current_row.__exit__()
                        current_tab.__exit__()
                    gr.Group()
                    current_tab = gr.TabItem(elem_id=f"settings_{elem_id}", label=text)
                    current_tab.__enter__()
                    current_row = gr.Column(variant='compact')
                    current_row.__enter__()
                    previous_section = item.section
                if k in quicksettings_names and not shared.cmd_opts.freeze_settings:
                    quicksettings_list.append((i, k, item))
                    components.append(dummy_component)
                elif section_must_be_skipped:
                    components.append(dummy_component)
                else:
                    component = create_setting_component(k)
                    component_dict[k] = component
                    components.append(component)
            if current_tab is not None:
                current_row.__exit__()
                current_tab.__exit__()

            request_notifications = gr.Button(value='Request browser notifications', elem_id="request_notifications", visible=False)
            _show_all_pages = gr.Button(value="Show all pages", variant='primary', elem_id="settings_show_all_pages")
            with gr.TabItem("Licenses"):
                gr.HTML(shared.html("licenses.html"), elem_id="licenses")

        def unload_sd_weights():
            modules.sd_models.unload_model_weights()

        def reload_sd_weights():
            modules.sd_models.reload_model_weights()

        unload_sd_model.click(
            fn=unload_sd_weights,
            inputs=[],
            outputs=[]
        )

        reload_sd_model.click(
            fn=reload_sd_weights,
            inputs=[],
            outputs=[]
        )

        request_notifications.click(
            fn=lambda: None,
            inputs=[],
            outputs=[],
            _js='function(){}'
        )

        def reload_scripts():
            modules.scripts.reload_script_body_only()
            reload_javascript()  # need to refresh the html page

        reload_script_bodies.click(
            fn=reload_scripts,
            inputs=[],
            outputs=[]
        )

        preview_theme.click(
            fn=None,
            _js='preview_theme',
            inputs=[dummy_component],
            outputs=[dummy_component]
        )


    interfaces = [
        (txt2img_interface, "From Text", "txt2img"),
        (img2img_interface, "From Image", "img2img"),
        (extras_interface, "Process Image", "extras"),
        # (pnginfo_interface, "Image Info", "pnginfo"),
        # (modelmerger_interface, "Checkpoint Merger", "modelmerger"),
        (train_interface, "Train", "ti"),
    ]

    interfaces += script_callbacks.ui_tabs_callback()
    interfaces += [(settings_interface, "Settings", "settings")]

    extensions_interface = ui_extensions.create_ui()
    interfaces += [(extensions_interface, "Extensions", "extensions")]

    shared.tab_names = []
    for _interface, label, _ifid in interfaces:
        shared.tab_names.append(label)

    with gr.Blocks(theme=shared.gradio_theme, analytics_enabled=False, title="Stable Diffusion") as demo:
        with gr.Row(elem_id="quicksettings", variant="compact"):
            for i, k, item in sorted(quicksettings_list, key=lambda x: quicksettings_names.get(x[1], x[0])):
                component = create_setting_component(k, is_quicksettings=True)
                component_dict[k] = component

        parameters_copypaste.connect_paste_params_buttons()

        with gr.Tabs(elem_id="tabs") as _tabs:
            for interface, label, ifid in interfaces:
                if label in shared.opts.hidden_tabs:
                    continue
                with gr.TabItem(label, id=ifid, elem_id='tab_' + ifid):
                    interface.render()

        text_settings = gr.Textbox(elem_id="settings_json", value=lambda: opts.dumpjson(), visible=False)
        settings_submit.click(
            fn=wrap_gradio_call(run_settings, extra_outputs=[gr.update()]),
            inputs=components,
            outputs=[text_settings, result],
        )
        restart_submit.click(fn=shared.restart_server, _js="restart_reload")

        for i, k, item in quicksettings_list:
            component = component_dict[k]
            info = opts.data_labels[k]

            component.change(
                fn=lambda value, k=k: run_settings_single(value, key=k),
                inputs=[component],
                outputs=[component, text_settings],
                show_progress=info.refresh is not None,
            )

        text_settings.change(
            fn=lambda: gr.update(visible=shared.sd_model and shared.sd_model.cond_stage_key == "edit"),
            inputs=[],
            outputs=[image_cfg_scale],
        )

        button_set_checkpoint = gr.Button('Change checkpoint', elem_id='change_checkpoint', visible=False)
        button_set_checkpoint.click(
            fn=lambda value, _: run_settings_single(value, key='sd_model_checkpoint'),
            _js="function(v){ var res = desiredCheckpointName; desiredCheckpointName = ''; return [res || v, null]; }",
            inputs=[component_dict['sd_model_checkpoint'], dummy_component],
            outputs=[component_dict['sd_model_checkpoint'], text_settings],
        )

        component_keys = [k for k in opts.data_labels.keys() if k in component_dict]

        def get_settings_values():
            return [get_value_for_setting(key) for key in component_keys]

        demo.load(
            fn=get_settings_values,
            inputs=[],
            outputs=[component_dict[k] for k in component_keys],
            queue=False,
        )

        def modelmerger(*args):
            try:
                results = modules.extras.run_modelmerger(*args)
            except Exception as e:
                errors.display(e, 'model merge')
                modules.sd_models.list_models()  # to remove the potentially missing models from the list
                return [*[gr.Dropdown.update(choices=modules.sd_models.checkpoint_tiles()) for _ in range(4)], f"Error merging checkpoints: {e}"]
            return results

        modelmerger_merge.click(fn=lambda: '', inputs=[], outputs=[modelmerger_result])
        modelmerger_merge.click(
            fn=wrap_gradio_gpu_call(modelmerger, extra_outputs=lambda: [gr.update() for _ in range(4)]),
            _js='modelmerger',
            inputs=[
                dummy_component,
                primary_model_name,
                secondary_model_name,
                tertiary_model_name,
                interp_method,
                interp_amount,
                save_as_half,
                custom_name,
                checkpoint_format,
                config_source,
                bake_in_vae,
                discard_weights,
            ],
            outputs=[
                primary_model_name,
                secondary_model_name,
                tertiary_model_name,
                component_dict['sd_model_checkpoint'],
                modelmerger_result,
            ]
        )

    ui_config_file = cmd_opts.ui_config_file
    ui_settings = {}
    settings_count = len(ui_settings)
    error_loading = False

    try:
        if os.path.exists(ui_config_file):
            with open(ui_config_file, "r", encoding="utf8") as file:
                ui_settings = json.load(file)
    except Exception as e:
        error_loading = True
        errors.display(e, 'loading ui settings')

    def loadsave(path, x):
        def apply_field(obj, field, condition=None, init_field=None):
            key = path + "/" + field

            if getattr(obj, 'custom_script_source', None) is not None:
                key = 'customscript/' + obj.custom_script_source + '/' + key

            if getattr(obj, 'do_not_save_to_config', False):
                return

            saved_value = ui_settings.get(key, None)
            if saved_value is None:
                ui_settings[key] = getattr(obj, field)
            elif condition and not condition(saved_value):
                pass

                # this warning is generally not useful;
                # print(f'Warning: Bad ui setting value: {key}: {saved_value}; Default value "{getattr(obj, field)}" will be used instead.')
            else:
                setattr(obj, field, saved_value)
                if init_field is not None:
                    init_field(saved_value)

        if type(x) in [gr.Slider, gr.Radio, gr.Checkbox, gr.Textbox, gr.Number, gr.Dropdown] and x.visible:
            apply_field(x, 'visible')

        if type(x) == gr.Slider:
            apply_field(x, 'value')
            apply_field(x, 'minimum')
            apply_field(x, 'maximum')
            apply_field(x, 'step')

        if type(x) == gr.Radio:
            apply_field(x, 'value', lambda val: val in x.choices)

        if type(x) == gr.Checkbox:
            apply_field(x, 'value')

        if type(x) == gr.Textbox:
            apply_field(x, 'value')

        if type(x) == gr.Number:
            apply_field(x, 'value')

        if type(x) == gr.Dropdown:
            def check_dropdown(val):
                if getattr(x, 'multiselect', False):
                    return all([value in x.choices for value in val])
                else:
                    return val in x.choices

            apply_field(x, 'value', check_dropdown, getattr(x, 'init_field', None))

    visit(txt2img_interface, loadsave, "txt2img")
    visit(img2img_interface, loadsave, "img2img")
    visit(extras_interface, loadsave, "extras")
    visit(modelmerger_interface, loadsave, "modelmerger")
    visit(train_interface, loadsave, "train")

    if not error_loading and (not os.path.exists(ui_config_file) or settings_count != len(ui_settings)):
        with open(ui_config_file, "w", encoding="utf8") as file:
            json.dump(ui_settings, file, indent=4)

    # Required as a workaround for change() event not triggering when loading values from ui-config.json
    interp_description.value = update_interp_description(interp_method.value)

    return demo


def webpath(fn):
    if fn.startswith(script_path):
        web_path = os.path.relpath(fn, script_path).replace('\\', '/')
    else:
        web_path = os.path.abspath(fn)

    return f'file={web_path}?{os.path.getmtime(fn)}'


def html_head():
    script_js = os.path.join(script_path, "script.js")
    head = f'<script type="text/javascript" src="{webpath(script_js)}"></script>\n'
    for script in modules.scripts.list_scripts("javascript", ".js"):
        head += f'<script type="text/javascript" src="{webpath(script.path)}"></script>\n'
    for script in modules.scripts.list_scripts("javascript", ".mjs"):
        head += f'<script type="module" src="{webpath(script.path)}"></script>\n'
    return head


def html_body():
    body = ''
    # inline = f"{localization.localization_js(shared.opts.localization)};"
    inline = ''
    if cmd_opts.theme is not None:
        inline += f"set_theme('{cmd_opts.theme}');"
    elif opts.gradio_theme == 'black-orange':
        inline += "set_theme('dark');"
    body += f'<script type="text/javascript">{inline}</script>\n'
    return body


def html_css():
    head = ""
    def stylesheet(fn):
        return f'<link rel="stylesheet" property="stylesheet" href="{webpath(fn)}">'
    for cssfile in modules.scripts.list_files_with_name("style.css"):
        if not os.path.isfile(cssfile):
            continue
        head += stylesheet(cssfile)
    if opts.gradio_theme == 'black-orange':
        head += stylesheet(os.path.join(script_path, "javascript", "black-orange.css"))
    if os.path.exists(os.path.join(data_path, "user.css")):
        head += stylesheet(os.path.join(data_path, "user.css"))
    return head


def reload_javascript():
    head = html_head()
    css = html_css()
    body = html_body()

    def template_response(*args, **kwargs):
        res = shared.GradioTemplateResponseOriginal(*args, **kwargs)
        res.body = res.body.replace(b'</head>', f'{head}</head>'.encode("utf8"))
        res.body = res.body.replace(b'</body>', f'{css}{body}</body>'.encode("utf8"))
        res.init_headers()
        return res

    gradio.routes.templates.TemplateResponse = template_response


if not hasattr(shared, 'GradioTemplateResponseOriginal'):
    shared.GradioTemplateResponseOriginal = gradio.routes.templates.TemplateResponse
