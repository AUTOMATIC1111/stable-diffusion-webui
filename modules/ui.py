import json
import mimetypes
import os
from functools import reduce

import gradio as gr
import gradio.routes
import gradio.utils
import numpy as np
from PIL import Image
from modules.call_queue import wrap_gradio_gpu_call, wrap_queued_call, wrap_gradio_call

from modules import sd_hijack, sd_models, script_callbacks, ui_extensions, deepbooru, extra_networks, ui_common, ui_postprocessing, ui_loadsave, ui_train, ui_models
from modules.ui_components import FormRow, FormColumn, FormGroup, ToolButton, FormHTML # pylint: disable=unused-import
from modules.paths import script_path, data_path
from modules.shared import opts, cmd_opts
from modules import prompt_parser
import modules.codeformer_model
import modules.generation_parameters_copypaste as parameters_copypaste
import modules.gfpgan_model
import modules.hypernetworks.ui
import modules.scripts
import modules.shared
import modules.errors
import modules.styles
import modules.extras
import modules.textual_inversion.ui
import modules.sd_samplers


modules.errors.install()
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
paste_symbol = '\U0001F4D8' # '\u2199\ufe0f'  # ↙
refresh_symbol = '\U0001F504' # 🔄
save_style_symbol = '\U0001F6C5' # '\U0001f4be'  # 💾
apply_style_symbol = '\U0001F9F3' # '\U0001f4cb'  # 📋
clear_prompt_symbol = '\U0001F6AE' # '\U0001f5d1\ufe0f'  # 🗑️
extra_networks_symbol = '\U0001F310' # '\U0001F3B4'  # 🎴
switch_values_symbol = '\U000021C5' # ⇅
detect_image_size_symbol = '\U0001F4D0'  # 📐


def create_output_panel(tabname, outdir): # may be referenced by extensions
    a, b, c, _d, e = ui_common.create_output_panel(tabname, outdir)
    return a, b, c, e

def plaintext_to_html(text): # may be referenced by extensions
    return ui_common.plaintext_to_html(text)

def infotext_to_html(text): # may be referenced by extensions
    return ui_common.infotext_to_html(text)


def send_gradio_gallery_to_image(x):
    if len(x) == 0:
        return None
    return parameters_copypaste.image_from_url_text(x[0])


def add_style(name: str, prompt: str, negative_prompt: str):
    if name is None:
        return [gr_show() for x in range(4)]
    style = modules.styles.PromptStyle(name, prompt, negative_prompt)
    modules.shared.prompt_styles.styles[style.name] = style
    modules.shared.prompt_styles.save_styles(modules.shared.opts.styles_dir)
    return [gr.Dropdown.update(visible=True, choices=list(modules.shared.prompt_styles.styles)) for _ in range(2)]


def calc_resolution_hires(enable, width, height, hr_scale, hr_resize_x, hr_resize_y):
    from modules import processing, devices
    if not enable:
        return ""
    if modules.shared.backend == modules.shared.Backend.DIFFUSERS:
        return "Hires resize: disabled"
    p = processing.StableDiffusionProcessingTxt2Img(width=width, height=height, enable_hr=True, hr_scale=hr_scale, hr_resize_x=hr_resize_x, hr_resize_y=hr_resize_y)
    with devices.autocast():
        p.init([""], [0], [0])
    return f"Hires resize: from <span class='resolution'>{p.width}x{p.height}</span> to <span class='resolution'>{p.hr_resize_x or p.hr_upscale_to_x}x{p.hr_resize_y or p.hr_upscale_to_y}</span>"


def resize_from_to_html(width, height, scale_by):
    target_width = int(width * scale_by)
    target_height = int(height * scale_by)
    if not target_width or not target_height:
        return "no image selected"
    if modules.shared.backend == modules.shared.Backend.DIFFUSERS:
        return "Hires resize: disabled"
    return f"Hires resize: from <span class='resolution'>{width}x{height}</span> to <span class='resolution'>{target_width}x{target_height}</span>"


def apply_styles(prompt, prompt_neg, styles):
    prompt = modules.shared.prompt_styles.apply_styles_to_prompt(prompt, styles)
    prompt_neg = modules.shared.prompt_styles.apply_negative_styles_to_prompt(prompt_neg, styles)
    return [gr.Textbox.update(value=prompt), gr.Textbox.update(value=prompt_neg), gr.Dropdown.update(value=[])]


def process_interrogate(interrogation_function, mode, ii_input_files, ii_input_dir, ii_output_dir, *ii_singles):
    if mode in {0, 1, 3, 4}:
        return [interrogation_function(ii_singles[mode]), None]
    if mode == 2:
        return [interrogation_function(ii_singles[mode]["image"]), None]
    if mode == 5:
        if len(ii_input_files) > 0:
            images = [f.name for f in ii_input_files]
        else:
            if not os.path.isdir(ii_input_dir):
                modules.shared.log.error(f"Input directory not found: {ii_input_dir}")
                return
            images = modules.shared.listfiles(ii_input_dir)
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


def interrogate(image):
    prompt = modules.shared.interrogator.interrogate(image.convert("RGB"))
    return gr.update() if prompt is None else prompt


def interrogate_deepbooru(image):
    prompt = deepbooru.model.tag(image)
    return gr.update() if prompt is None else prompt


def create_seed_inputs(target_interface):
    with FormRow(elem_id=f"{target_interface}_seed_row", variant="compact"):
        seed = gr.Number(label='Seed', value=-1, elem_id=f"{target_interface}_seed")
        seed.style(container=False)
        random_seed = ToolButton(random_symbol, elem_id=f"{target_interface}_random_seed", label='Random seed')
        reuse_seed = ToolButton(reuse_symbol, elem_id=f"{target_interface}_reuse_seed", label='Reuse seed')
    with FormRow(visible=True, elem_id=f"{target_interface}_subseed_row"):
        subseed = gr.Number(label='Variation seed', value=-1, elem_id=f"{target_interface}_subseed")
        subseed.style(container=False)
        random_subseed = ToolButton(random_symbol, elem_id=f"{target_interface}_random_subseed")
        reuse_subseed = ToolButton(reuse_symbol, elem_id=f"{target_interface}_reuse_subseed")
        subseed_strength = gr.Slider(label='Variation strength', value=0.0, minimum=0, maximum=1, step=0.01, elem_id=f"{target_interface}_subseed_strength")
    with FormRow(visible=False):
        seed_resize_from_w = gr.Slider(minimum=0, maximum=4096, step=8, label="Resize seed from width", value=0, elem_id=f"{target_interface}_seed_resize_from_w")
        seed_resize_from_h = gr.Slider(minimum=0, maximum=4096, step=8, label="Resize seed from height", value=0, elem_id=f"{target_interface}_seed_resize_from_h")
    random_seed.click(fn=lambda: [-1, -1], show_progress=False, inputs=[], outputs=[seed, subseed])
    random_subseed.click(fn=lambda: -1, show_progress=False, inputs=[], outputs=[subseed])
    return seed, reuse_seed, subseed, reuse_subseed, subseed_strength, seed_resize_from_h, seed_resize_from_w


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
                modules.shared.log.error(f"Error parsing JSON generation info: {gen_info_string}")
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
    if modules.shared.backend == modules.shared.Backend.ORIGINAL:
        token_count, max_length = max([sd_hijack.model_hijack.get_prompt_lengths(prompt) for prompt in prompts], key=lambda args: args[0])
    elif modules.shared.backend == modules.shared.Backend.DIFFUSERS:
        if modules.shared.sd_model is not None and hasattr(modules.shared.sd_model, 'tokenizer'):
            tokenizer = modules.shared.sd_model.tokenizer
            has_bos_token = tokenizer.bos_token_id is not None
            has_eos_token = tokenizer.eos_token_id is not None
            ids = [modules.shared.sd_model.tokenizer(prompt) for prompt in prompts]
            if len(ids) > 0 and hasattr(ids[0], 'input_ids'):
                ids = [x.input_ids for x in ids]
            token_count = max([len(x) for x in ids]) - int(has_bos_token) - int(has_eos_token)
            max_length = tokenizer.model_max_length - int(has_bos_token) - int(has_eos_token)
        else:
            token_count = 0
            max_length = 75
    return f"<span class='gr-box gr-text-input'>{token_count}/{max_length}</span>"


def create_toprow(is_img2img):
    id_part = "img2img" if is_img2img else "txt2img"
    with gr.Row(elem_id=f"{id_part}_toprow", variant="compact"):
        with gr.Column(elem_id=f"{id_part}_prompt_container", scale=6):
            with gr.Row():
                with gr.Column(scale=80):
                    with gr.Row():
                        prompt = gr.Textbox(label="Prompt", elem_id=f"{id_part}_prompt", show_label=False, lines=3, placeholder="Prompt (press Ctrl+Enter or Alt+Enter to generate)", elem_classes=["prompt"])
            with gr.Row():
                with gr.Column(scale=80):
                    with gr.Row():
                        negative_prompt = gr.Textbox(label="Negative prompt", elem_id=f"{id_part}_neg_prompt", show_label=False, lines=3, placeholder="Negative prompt (press Ctrl+Enter or Alt+Enter to generate)", elem_classes=["prompt"])
        button_interrogate = None
        button_deepbooru = None
        if is_img2img:
            with gr.Column(scale=1, elem_classes="interrogate-col"):
                button_interrogate = gr.Button('Interrogate\nCLIP', elem_id="interrogate")
                button_deepbooru = gr.Button('Interrogate\nDeepBooru', elem_id="deepbooru")
        with gr.Column(scale=1, elem_id=f"{id_part}_actions_column"):
            with gr.Row(elem_id=f"{id_part}_generate_box"):
                submit = gr.Button('Generate', elem_id=f"{id_part}_generate", variant='primary')
            with gr.Row(elem_id=f"{id_part}_generate_line2"):
                interrupt = gr.Button('Stop', elem_id=f"{id_part}_interrupt")
                interrupt.click(fn=lambda: modules.shared.state.interrupt(), _js="requestInterrupt", inputs=[], outputs=[])
                skip = gr.Button('Skip', elem_id=f"{id_part}_skip")
                skip.click(fn=lambda: modules.shared.state.skip(), inputs=[], outputs=[])
                pause = gr.Button('Pause', elem_id=f"{id_part}_pause")
                pause.click(fn=lambda: modules.shared.state.pause(), _js='checkPaused', inputs=[], outputs=[])
            with gr.Row(elem_id=f"{id_part}_tools"):
                paste = ToolButton(value=paste_symbol, elem_id="paste")
                clear_prompt_button = ToolButton(value=clear_prompt_symbol, elem_id=f"{id_part}_clear_prompt_btn")
                extra_networks_button = ToolButton(value=extra_networks_symbol, elem_id=f"{id_part}_extra_networks_btn")
                prompt_style_apply = ToolButton(value=apply_style_symbol, elem_id=f"{id_part}_style_apply_btn")
                save_style = ToolButton(value=save_style_symbol, elem_id=f"{id_part}_style_create_btn")
                clear_prompt_button.click(fn=lambda *x: x, _js="confirm_clear_prompt", inputs=[prompt, negative_prompt], outputs=[prompt, negative_prompt])
            with gr.Row(elem_id=f"{id_part}_counters"):
                token_counter = gr.HTML(value="<span>0/75</span>", elem_id=f"{id_part}_token_counter", elem_classes=["token-counter"])
                token_button = gr.Button(visible=False, elem_id=f"{id_part}_token_button")
                negative_token_counter = gr.HTML(value="<span>0/75</span>", elem_id=f"{id_part}_negative_token_counter", elem_classes=["token-counter"])
                negative_token_button = gr.Button(visible=False, elem_id=f"{id_part}_negative_token_button")
            with gr.Row(elem_id=f"{id_part}_styles_row"):
                prompt_styles = gr.Dropdown(label="Styles", elem_id=f"{id_part}_styles", choices=[k for k, v in modules.shared.prompt_styles.styles.items()], value=[], multiselect=True)
                create_refresh_button(prompt_styles, modules.shared.prompt_styles.reload, lambda: {"choices": [k for k, v in modules.shared.prompt_styles.styles.items()]}, f"refresh_{id_part}_styles")
    return prompt, prompt_styles, negative_prompt, submit, button_interrogate, button_deepbooru, prompt_style_apply, save_style, paste, extra_networks_button, token_counter, token_button, negative_token_counter, negative_token_button


def setup_progressbar(*args, **kwargs): # pylint: disable=unused-argument
    pass


def apply_setting(key, value):
    if value is None:
        return gr.update()
    if modules.shared.cmd_opts.freeze:
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
    opts.save(modules.shared.config_filename)
    return getattr(opts, key)


def create_refresh_button(refresh_component, refresh_method, refreshed_args, elem_id):
    return ui_common.create_refresh_button(refresh_component, refresh_method, refreshed_args, elem_id)


def create_sampler_and_steps_selection(choices, tabname, primary: bool = True):
    with FormRow(elem_id=f"sampler_selection_{tabname}{'_alt' if not primary else ''}"):
        sampler_index = gr.Dropdown(label='Sampling method' if primary else 'Secondary sampler', elem_id=f"{tabname}_sampling{'_alt' if not primary else ''}", choices=[x.name for x in choices], value='Default', type="index")
        steps = gr.Slider(minimum=0, maximum=99, step=1, label="Sampling steps" if primary else 'Secondary steps', elem_id=f"{tabname}_steps{'_alt' if not primary else ''}", value=20)
    return steps, sampler_index


def ordered_ui_categories():
    user_order = {x.strip(): i * 2 + 1 for i, x in enumerate(modules.shared.opts.ui_reorder.split(","))}
    for _i, category in sorted(enumerate(modules.shared.ui_reorder_categories), key=lambda x: user_order.get(x[1], x[0] * 2 + 0)):
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


def create_ui(startup_timer = None):
    if startup_timer is None:
        from modules import timer
        startup_timer = timer.Timer()
    import modules.img2img # pylint: disable=redefined-outer-name
    import modules.txt2img # pylint: disable=redefined-outer-name
    reload_javascript()
    parameters_copypaste.reset()
    modules.scripts.scripts_current = modules.scripts.scripts_txt2img
    modules.scripts.scripts_txt2img.initialize_scripts(is_img2img=False)
    with gr.Blocks(analytics_enabled=False) as txt2img_interface:
        txt2img_prompt, txt2img_prompt_styles, txt2img_negative_prompt, submit, _interrogate, _deepbooru, txt2img_prompt_style_apply, txt2img_save_style, txt2img_paste, extra_networks_button, token_counter, token_button, negative_token_counter, negative_token_button = create_toprow(is_img2img=False)
        dummy_component = gr.Label(visible=False)
        txt_prompt_img = gr.File(label="", elem_id="txt2img_prompt_image", file_count="single", type="binary", visible=False)
        with FormRow(variant='compact', elem_id="txt2img_extra_networks", visible=False) as extra_networks_ui:
            from modules import ui_extra_networks
            extra_networks_ui = ui_extra_networks.create_ui(extra_networks_ui, extra_networks_button, 'txt2img', skip_indexing=opts.extra_network_skip_indexing)
        with gr.Row().style(equal_height=False, elem_id="txt2img_interface"):
            with gr.Column(variant='compact', elem_id="txt2img_settings"):
                for category in ordered_ui_categories():
                    if category == "sampler":
                        modules.sd_samplers.set_samplers()
                        steps, sampler_index = create_sampler_and_steps_selection(modules.sd_samplers.samplers, "txt2img", True)
                    elif category == "dimensions":
                        with FormRow():
                            with gr.Column(elem_id="txt2img_column_size", scale=4):
                                with FormRow(elem_id="txt2img_row_dimension"):
                                    width = gr.Slider(minimum=64, maximum=4096, step=8, label="Width", value=512, elem_id="txt2img_width")
                                    height = gr.Slider(minimum=64, maximum=4096, step=8, label="Height", value=512, elem_id="txt2img_height")
                            with gr.Column(elem_id="txt2img_dimensions_row", scale=1, elem_classes="dimensions-tools"):
                                res_switch_btn = ToolButton(value=switch_values_symbol, elem_id="txt2img_res_switch_btn", label="Switch dims")
                            with gr.Column(elem_id="txt2img_column_batch"):
                                with FormRow(elem_id="txt2img_row_batch"):
                                    batch_count = gr.Slider(minimum=1, step=1, label='Batch count', value=1, elem_id="txt2img_batch_count")
                                    batch_size = gr.Slider(minimum=1, maximum=32, step=1, label='Batch size', value=1, elem_id="txt2img_batch_size")
                    elif category == "cfg":
                        with FormRow():
                            cfg_scale = gr.Slider(minimum=1.0, maximum=30.0, step=0.1, label='CFG Scale', value=6.0, elem_id="txt2img_cfg_scale")
                            clip_skip = gr.Slider(label='CLIP skip', value=1, minimum=1, maximum=14, step=1, elem_id='txt2img_clip_skip', interactive=True)
                    elif category == "seed":
                        seed, reuse_seed, subseed, reuse_subseed, subseed_strength, seed_resize_from_h, seed_resize_from_w = create_seed_inputs('txt2img')
                    elif category == "checkboxes":
                        with FormRow(elem_classes="checkboxes-row", variant="compact"):
                            second_pass = gr.Checkbox(label='Second pass', value=False, elem_id="txt2img_enable_hr")
                            restore_faces = gr.Checkbox(label='Restore faces', value=False, visible=len(modules.shared.face_restorers) > 1, elem_id="txt2img_restore_faces")
                            tiling = gr.Checkbox(label='Tiling', value=False, elem_id="txt2img_tiling")
                    elif category == "second_pass":
                        with FormGroup(visible=False, elem_id="txt2img_second_pass") as hr_options:
                            hr_second_pass_steps, latent_index = create_sampler_and_steps_selection(modules.sd_samplers.samplers, "txt2img", False)
                            with FormRow(elem_id="txt2img_hires_fix_row1", variant="compact"):
                                denoising_strength = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label='Denoising strength', value=0.3, elem_id="txt2img_denoising_strength")

                            with FormRow():
                                hr_final_resolution = FormHTML(value="", elem_id="txtimg_hr_finalres", label="Upscaled resolution", interactive=False)
                            with FormRow(elem_id="txt2img_hires_fix_row2", variant="compact"):
                                hr_upscaler = gr.Dropdown(label="Upscaler", elem_id="txt2img_hr_upscaler", choices=[*modules.shared.latent_upscale_modes, *[x.name for x in modules.shared.sd_upscalers]], value=modules.shared.latent_upscale_default_mode)
                                hr_scale = gr.Slider(minimum=1.0, maximum=4.0, step=0.05, label="Upscale by", value=2.0, elem_id="txt2img_hr_scale")
                            with FormRow(elem_id="txt2img_hires_fix_row3", variant="compact"):
                                hr_resize_x = gr.Slider(minimum=0, maximum=4096, step=8, label="Resize width to", value=0, elem_id="txt2img_hr_resize_x")
                                hr_resize_y = gr.Slider(minimum=0, maximum=4096, step=8, label="Resize height to", value=0, elem_id="txt2img_hr_resize_y")

                            with FormRow():
                                hr_refiner = FormHTML(value="Refiner", elem_id="txtimg_hr_finalres", interactive=False)
                            with FormRow(elem_id="txt2img_refiner_row1", variant="compact"):
                                image_cfg_scale = gr.Slider(minimum=1.1, maximum=30.0, step=0.1, label='Secondary CFG Scale', value=6.0, elem_id="txt2img_image_cfg_scale")
                                diffusers_guidance_rescale = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, label='Guidance rescale', value=0.7, elem_id="txt2img_image_cfg_rescale")
                                refiner_denoise_start = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, label='Denoise start', value=0.0, elem_id="txt2img_refiner_denoise_start")
                                refiner_denoise_end = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, label='Denoise end', value=1.0, elem_id="txt2img_refiner_denoise_end")
                            with FormRow(elem_id="txt2img_refiner_row2", variant="compact"):
                                refiner_prompt = gr.Textbox(value='', label='Prompt')
                            with FormRow(elem_id="txt2img_refiner_row3", variant="compact"):
                                refiner_negative = gr.Textbox(value='', label='Negative prompt')

                    elif category == "override_settings":
                        with FormRow(elem_id="txt2img_override_settings_row") as row:
                            override_settings = create_override_settings_dropdown('txt2img', row)
                    elif category == "scripts":
                        with FormGroup(elem_id="txt2img_script_container"):
                            custom_inputs = modules.scripts.scripts_txt2img.setup_ui()
            hr_resolution_preview_inputs = [second_pass, width, height, hr_scale, hr_resize_x, hr_resize_y]
            for preview_input in hr_resolution_preview_inputs:
                preview_input.change(
                    fn=calc_resolution_hires,
                    _js="onCalcResolutionHires",
                    inputs=hr_resolution_preview_inputs,
                    outputs=[hr_final_resolution],
                    show_progress=False,
                )

            txt2img_gallery, generation_info, html_info, _html_info_formatted, html_log = ui_common.create_output_panel("txt2img", opts.outdir_txt2img_samples)
            connect_reuse_seed(seed, reuse_seed, generation_info, dummy_component, is_subseed=False)
            connect_reuse_seed(subseed, reuse_subseed, generation_info, dummy_component, is_subseed=True)

            txt2img_args = dict(
                fn=wrap_gradio_gpu_call(modules.txt2img.txt2img, extra_outputs=[None, '', '']),
                _js="submit",
                inputs=[
                    dummy_component,
                    txt2img_prompt, txt2img_negative_prompt,
                    txt2img_prompt_styles,
                    steps,
                    sampler_index, latent_index,
                    restore_faces, tiling,
                    batch_count, batch_size,
                    cfg_scale, image_cfg_scale,
                    diffusers_guidance_rescale,
                    clip_skip,
                    seed, subseed, subseed_strength, seed_resize_from_h, seed_resize_from_w,
                    height, width,
                    second_pass, denoising_strength,
                    hr_scale, hr_upscaler, hr_second_pass_steps, hr_resize_x, hr_resize_y,
                    refiner_denoise_start, refiner_denoise_end, refiner_prompt, refiner_negative,
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

            txt_prompt_img.change(fn=modules.images.image_data, inputs=[txt_prompt_img], outputs=[txt2img_prompt, txt_prompt_img])

            def enable_hr_change(visible: bool):
                return {"visible": visible, "__type__": "update"}, f'Refiner{": disabled" if modules.shared.sd_refiner is None else ""}'

            second_pass.change(enable_hr_change, inputs=[second_pass], outputs=[hr_options, hr_refiner], show_progress = False)

            txt2img_paste_fields = [
                (txt2img_prompt, "Prompt"),
                (txt2img_negative_prompt, "Negative prompt"),
                (steps, "Steps"),
                (sampler_index, "Sampler"),
                (latent_index, "Latent sampler"),
                (restore_faces, "Face restoration"),
                (cfg_scale, "CFG scale"),
                (clip_skip, "Clip skip"),
                (seed, "Seed"),
                (width, "Size-1"),
                (height, "Size-2"),
                (batch_size, "Batch size"),
                (subseed, "Variation seed"),
                (subseed_strength, "Variation strength"),
                (seed_resize_from_w, "Seed resize from-1"),
                (seed_resize_from_h, "Seed resize from-2"),
                (denoising_strength, "Denoising strength"),
                (second_pass, lambda d: "Denoising strength" in d),
                (hr_options, lambda d: gr.Row.update(visible="Denoising strength" in d)),
                (hr_scale, "Hires upscale"),
                (hr_upscaler, "Hires upscaler"),
                (hr_second_pass_steps, "Hires steps"),
                (hr_resize_x, "Hires resize-1"),
                (hr_resize_y, "Hires resize-2"),
                *modules.scripts.scripts_txt2img.infotext_fields
            ]
            parameters_copypaste.add_paste_fields("txt2img", None, txt2img_paste_fields, override_settings)
            parameters_copypaste.register_paste_params_button(parameters_copypaste.ParamBinding(paste_button=txt2img_paste, tabname="txt2img", source_text_component=txt2img_prompt, source_image_component=None))

            token_button.click(fn=wrap_queued_call(update_token_counter), inputs=[txt2img_prompt, steps], outputs=[token_counter])
            negative_token_button.click(fn=wrap_queued_call(update_token_counter), inputs=[txt2img_negative_prompt, steps], outputs=[negative_token_counter])

            ui_extra_networks.setup_ui(extra_networks_ui, txt2img_gallery)

    startup_timer.record("ui-txt2img")

    modules.scripts.scripts_current = modules.scripts.scripts_img2img
    modules.scripts.scripts_img2img.initialize_scripts(is_img2img=True)
    with gr.Blocks(analytics_enabled=False) as img2img_interface:
        img2img_prompt, img2img_prompt_styles, img2img_negative_prompt, submit, img2img_interrogate, img2img_deepbooru, img2img_prompt_style_apply, img2img_save_style, img2img_paste, extra_networks_button, token_counter, token_button, negative_token_counter, negative_token_button = create_toprow(is_img2img=True)

        img2img_prompt_img = gr.File(label="", elem_id="img2img_prompt_image", file_count="single", type="binary", visible=False)

        with FormRow(variant='compact', elem_id="img2img_extra_networks", visible=False) as extra_networks_ui:
            from modules import ui_extra_networks
            extra_networks_ui_img2img = ui_extra_networks.create_ui(extra_networks_ui, extra_networks_button, 'img2img', skip_indexing=opts.extra_network_skip_indexing)

        with FormRow().style(equal_height=False, elem_id="img2img_interface"):
            with gr.Column(variant='compact', elem_id="img2img_settings"):
                copy_image_buttons = []
                copy_image_destinations = {}

                def add_copy_image_controls(tab_name, elem):
                    with gr.Row(variant="compact", elem_id=f"img2img_copy_to_{tab_name}"):
                        for title, name in zip(['➠ Image', '➠ Sketch', '➠ Inpaint', '➠ Inpaint sketch'], ['img2img', 'sketch', 'inpaint', 'inpaint_sketch']):
                            if name == tab_name:
                                gr.Button(title, interactive=False)
                                copy_image_destinations[name] = elem
                                continue

                            button = gr.Button(title)
                            copy_image_buttons.append((button, name, elem))

                with gr.Tabs(elem_id="mode_img2img"):
                    img2img_selected_tab = gr.State(0) # pylint: disable=abstract-class-instantiated
                    with gr.TabItem('Image', id='img2img', elem_id="img2img_img2img_tab") as tab_img2img:
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
                            return state

                        inpaint_color_sketch.change(update_orig, [inpaint_color_sketch, inpaint_color_sketch_orig], inpaint_color_sketch_orig)

                    with gr.TabItem('Inpaint upload', id='inpaint_upload', elem_id="img2img_inpaint_upload_tab") as tab_inpaint_upload:
                        init_img_inpaint = gr.Image(label="Image for img2img", show_label=False, source="upload", interactive=True, type="pil", elem_id="img_inpaint_base")
                        init_mask_inpaint = gr.Image(label="Mask", source="upload", interactive=True, type="pil", elem_id="img_inpaint_mask")

                    with gr.TabItem('Batch', id='batch', elem_id="img2img_batch_tab") as tab_batch:
                        hidden = '<br>Disabled when launched with --hide-ui-dir-config.' if modules.shared.cmd_opts.hide_ui_dir_config else ''
                        gr.HTML(
                            "<p style='padding-bottom: 1em;' class=\"text-gray-500\">Upload images or process images in a directory" +
                            "<br>Add inpaint batch mask directory to enable inpaint batch processing"
                            f"{hidden}</p>"
                        )
                        img2img_batch_files = gr.Files(label="Batch Process", interactive=True, elem_id="img2img_image_batch")
                        img2img_batch_input_dir = gr.Textbox(label="Inpaint batch input directory", **modules.shared.hide_dirs, elem_id="img2img_batch_input_dir")
                        img2img_batch_output_dir = gr.Textbox(label="Inpaint batch output directory", **modules.shared.hide_dirs, elem_id="img2img_batch_output_dir")
                        img2img_batch_inpaint_mask_dir = gr.Textbox(label="Inpaint batch mask directory", **modules.shared.hide_dirs, elem_id="img2img_batch_inpaint_mask_dir")

                    img2img_tabs = [tab_img2img, tab_sketch, tab_inpaint, tab_inpaint_color, tab_inpaint_upload, tab_batch]

                    for i, tab in enumerate(img2img_tabs):
                        tab.select(fn=lambda tabnum=i: tabnum, inputs=[], outputs=[img2img_selected_tab])

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
                        _js=f"switch_to_{name.replace(' ', '_')}",
                        inputs=[],
                        outputs=[],
                    )

                with FormRow():
                    resize_mode = gr.Radio(label="Resize mode", elem_id="resize_mode", choices=["Resize fixed", "Crop and resize", "Resize and fill", "Resize using Latent upscale"], type="index", value="Resize and fill")

                for category in ordered_ui_categories():
                    if category == "sampler":
                        modules.sd_samplers.set_samplers()
                        steps, sampler_index = create_sampler_and_steps_selection(modules.sd_samplers.samplers_for_img2img, "img2img", True)

                    elif category == "dimensions":
                        with FormRow():
                            with gr.Column(elem_id="img2img_column_size", scale=4):
                                selected_scale_tab = gr.State(value=0) # pylint: disable=abstract-class-instantiated

                                with gr.Tabs():
                                    with gr.Tab(label="Resize to") as tab_scale_to:
                                        with FormRow():
                                            with gr.Column(elem_id="img2img_column_size", scale=4):
                                                with FormRow():
                                                    width = gr.Slider(minimum=64, maximum=4096, step=8, label="Width", value=512, elem_id="img2img_width")
                                                    height = gr.Slider(minimum=64, maximum=4096, step=8, label="Height", value=512, elem_id="img2img_height")
                                            with gr.Column(elem_id="img2img_column_dim", scale=1, elem_classes="dimensions-tools"):
                                                with FormRow():
                                                    res_switch_btn = ToolButton(value=switch_values_symbol, elem_id="img2img_res_switch_btn")
                                                    detect_image_size_btn = ToolButton(value=detect_image_size_symbol, elem_id="img2img_detect_image_size_btn")

                                    with gr.Tab(label="Resize by") as tab_scale_by:
                                        scale_by = gr.Slider(minimum=0.05, maximum=4.0, step=0.05, label="Scale", value=1.0, elem_id="img2img_scale")

                                        with FormRow():
                                            scale_by_html = FormHTML(resize_from_to_html(0, 0, 0.0), elem_id="img2img_scale_resolution_preview")
                                            gr.Slider(label="Unused", elem_id="img2img_unused_scale_by_slider")
                                            button_update_resize_to = gr.Button(visible=False, elem_id="img2img_update_resize_to")

                                    on_change_args = dict(
                                        fn=resize_from_to_html,
                                        _js="currentImg2imgSourceResolution",
                                        inputs=[dummy_component, dummy_component, scale_by],
                                        outputs=scale_by_html,
                                        show_progress=False,
                                    )

                                    scale_by.release(**on_change_args)
                                    button_update_resize_to.click(**on_change_args)

                                    # the code below is meant to update the resolution label after the image in the image selection UI has changed.
                                    # as it is now the event keeps firing continuously for inpaint edits, which ruins the page with constant requests.
                                    # I assume this must be a gradio bug and for now we'll just do it for non-inpaint inputs.
                                    for component in [init_img, sketch]:
                                        component.change(fn=lambda: None, _js="updateImg2imgResizeToTextAfterChangingImage", inputs=[], outputs=[], show_progress=False)

                            tab_scale_to.select(fn=lambda: 0, inputs=[], outputs=[selected_scale_tab])
                            tab_scale_by.select(fn=lambda: 1, inputs=[], outputs=[selected_scale_tab])

                        with FormRow(elem_id="img2img_column_batch"):
                            batch_count = gr.Slider(minimum=1, step=1, label='Batch count', value=1, elem_id="img2img_batch_count")
                            batch_size = gr.Slider(minimum=1, maximum=8, step=1, label='Batch size', value=1, elem_id="img2img_batch_size")
                            clip_skip = gr.Slider(label='CLIP skip', value=1, minimum=1, maximum=4, step=1, elem_id='img2img_clip_skip', interactive=True)

                    elif category == "cfg":
                        with FormGroup():
                            with FormRow():
                                cfg_scale = gr.Slider(minimum=1.0, maximum=30.0, step=0.5, label='CFG Scale', value=6.0, elem_id="img2img_cfg_scale")
                                image_cfg_scale = gr.Slider(minimum=0, maximum=30.0, step=0.05, label='Image CFG Scale', value=1.5, elem_id="img2img_image_cfg_scale")
                                diffusers_guidance_rescale = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, label='Guidance Rescale', value=0.7, elem_id="txt2img_image_cfg_rescale")
                            with FormRow():
                                denoising_strength = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label='Denoising strength', value=0.75, elem_id="img2img_denoising_strength")
                                refiner_denoise_start = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, label='Denoise start', value=0.0, elem_id="txt2img_refiner_denoise_start")
                                refiner_denoise_end = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, label='Denoise end', value=1.0, elem_id="txt2img_refiner_denoise_end")

                    elif category == "seed":
                        seed, reuse_seed, subseed, reuse_subseed, subseed_strength, seed_resize_from_h, seed_resize_from_w = create_seed_inputs('img2img')

                    elif category == "checkboxes":
                        with FormRow(elem_classes="checkboxes-row", variant="compact"):
                            restore_faces = gr.Checkbox(label='Restore faces', value=False, visible=len(modules.shared.face_restorers) > 1, elem_id="img2img_restore_faces")
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
                                return gr.update(visible=tab in [2, 3, 4]), gr.update(visible=tab == 3)

                            for i, elem in enumerate(img2img_tabs):
                                elem.select(
                                    fn=lambda tab=i: select_img2img_tab(tab), # pylint: disable=cell-var-from-loop
                                    inputs=[],
                                    outputs=[inpaint_controls, mask_alpha],
                                )

            img2img_gallery, generation_info, html_info, _html_info_formatted, html_log = ui_common.create_output_panel("img2img", opts.outdir_img2img_samples)

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
                    dummy_component, dummy_component,
                    img2img_prompt, img2img_negative_prompt,
                    img2img_prompt_styles,
                    init_img,
                    sketch,
                    init_img_with_mask,
                    inpaint_color_sketch,
                    inpaint_color_sketch_orig,
                    init_img_inpaint,
                    init_mask_inpaint,
                    steps,
                    sampler_index, latent_index,
                    mask_blur, mask_alpha,
                    inpainting_fill,
                    restore_faces, tiling,
                    batch_count, batch_size,
                    cfg_scale, image_cfg_scale,
                    diffusers_guidance_rescale,
                    refiner_denoise_start, refiner_denoise_end,
                    clip_skip,
                    denoising_strength,
                    seed, subseed, subseed_strength, seed_resize_from_h, seed_resize_from_w,
                    selected_scale_tab,
                    height, width,
                    scale_by,
                    resize_mode,
                    inpaint_full_res, inpaint_full_res_padding, inpainting_mask_invert,
                    img2img_batch_files, img2img_batch_input_dir, img2img_batch_output_dir, img2img_batch_inpaint_mask_dir,
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

            img2img_prompt.submit(**img2img_args)
            submit.click(**img2img_args)
            res_switch_btn.click(lambda w, h: (h, w), inputs=[width, height], outputs=[width, height], show_progress=False)

            detect_image_size_btn.click(
                fn=lambda w, h, _: (w or gr.update(), h or gr.update()),
                _js="currentImg2imgSourceResolution",
                inputs=[dummy_component, dummy_component, dummy_component],
                outputs=[width, height],
                show_progress=False,
            )

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
                (clip_skip, "Clip skip"),
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

    startup_timer.record("ui-img2img")

    modules.scripts.scripts_current = None

    with gr.Blocks(analytics_enabled=False) as extras_interface:
        ui_postprocessing.create_ui()
        startup_timer.record("ui-extras")

    with gr.Blocks(analytics_enabled=False) as train_interface:
        ui_train.create_ui(txt2img_preview_params = [txt2img_prompt, txt2img_negative_prompt, steps, sampler_index, cfg_scale, seed, width, height])
        startup_timer.record("ui-train")

    with gr.Blocks(analytics_enabled=False) as models_interface:
        ui_models.create_ui()
        startup_timer.record("ui-models")

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
            raise ValueError(f'bad options item type: {t} for key {key}')
        elem_id = f"setting_{key}"

        if not is_quicksettings:
            dirtyable_setting = gr.Group(elem_classes="dirtyable", visible=(args or {}).get("visible", True))
            dirtyable_setting.__enter__()
            dirty_indicator = gr.Button(
                "",
                elem_classes="modification-indicator",
                elem_id="modification_indicator_" + key
            )

        if info.refresh is not None:
            if is_quicksettings:
                res = comp(label=info.label, value=fun(), elem_id=elem_id, **(args or {}))
                create_refresh_button(res, info.refresh, info.component_args, f"refresh_{key}")
            else:
                with FormRow():
                    res = comp(label=info.label, value=fun(), elem_id=elem_id, **(args or {}))
                    create_refresh_button(res, info.refresh, info.component_args, f"refresh_{key}")
        else:
            try:
                res = comp(label=info.label, value=fun(), elem_id=elem_id, **(args or {}))
            except Exception as e:
                modules.shared.log.error(f'Error creating setting: {key} {e}')
                res = None

        if res is not None and not is_quicksettings:
            res.change(fn=None, inputs=res, _js=f'(val) => markIfModified("{key}", val)')
            dirty_indicator.click(fn=lambda: getattr(opts, key), outputs=res, show_progress=False)
            dirtyable_setting.__exit__()

        return res

    def create_dirty_indicator(key, keys_to_reset, **kwargs):
        def get_opt_values():
            return [getattr(opts, _key) for _key in keys_to_reset]

        elements_to_reset = [component_dict[_key] for _key in keys_to_reset]
        indicator = gr.Button(
            "",
            elem_classes="modification-indicator",
            elem_id="modification_indicator_" + key,
            **kwargs
        )
        indicator.click(fn=get_opt_values, outputs=elements_to_reset, show_progress=False)
        return indicator

    loadsave = ui_loadsave.UiLoadsave(cmd_opts.ui_config)
    components = []
    component_dict = {}
    modules.shared.settings_components = component_dict

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
            opts.save(modules.shared.config_filename)
            modules.shared.log.info(f'Settings changed: {len(changed)} {changed}')
        except RuntimeError:
            modules.shared.log.error(f'Settings change failed: {len(changed)} {changed}')
            return opts.dumpjson(), f'{len(changed)} Settings changed without save: {", ".join(changed)}'
        return opts.dumpjson(), f'{len(changed)} Settings changed{": " if len(changed) > 0 else ""}{", ".join(changed)}'

    def run_settings_single(value, key):
        if not opts.same_type(value, opts.data_labels[key].default):
            return gr.update(visible=True), opts.dumpjson()
        if not opts.set(key, value):
            return gr.update(value=getattr(opts, key)), opts.dumpjson()
        opts.save(modules.shared.config_filename)
        modules.shared.log.debug(f'Setting changed: key={key}, value={value}')
        return get_value_for_setting(key), opts.dumpjson()

    with gr.Blocks(analytics_enabled=False) as settings_interface:
        with gr.Row():
            settings_submit = gr.Button(value="Apply settings", variant='primary', elem_id="settings_submit")
            restart_submit = gr.Button(value="Restart server", variant='primary', elem_id="restart_submit")
            shutdown_submit = gr.Button(value="Shutdown server", variant='primary', elem_id="shutdown_submit")
            preview_theme = gr.Button(value="Preview theme", variant='primary', elem_id="settings_preview_theme")
            defaults_submit = gr.Button(value="Restore defaults", variant='primary', elem_id="defaults_submit")
            unload_sd_model = gr.Button(value='Unload checkpoint', variant='primary', elem_id="sett_unload_sd_model")
            reload_sd_model = gr.Button(value='Reload checkpoint', variant='primary', elem_id="sett_reload_sd_model")
            # reload_script_bodies = gr.Button(value='Reload scripts', variant='primary', elem_id="settings_reload_script_bodies")
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
                if k in quicksettings_names and not modules.shared.cmd_opts.freeze:
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
            with gr.TabItem("User interface defaults", id="defaults", elem_id="settings_tab_defaults"):
                loadsave.create_ui()
                create_dirty_indicator("tab_defaults", [], interactive=False)
            with gr.TabItem("Licenses", id="licenses", elem_id="settings_tab_licenses"):
                gr.HTML(modules.shared.html("licenses.html"), elem_id="licenses")
                create_dirty_indicator("tab_licenses", [], interactive=False)
            with gr.TabItem("Show all pages", variant='primary', elem_id="settings_show_all_pages"):
                create_dirty_indicator("show_all_pages", [], interactive=False)

        def unload_sd_weights():
            modules.sd_models.unload_model_weights(op='model')
            modules.sd_models.unload_model_weights(op='refiner')

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

        preview_theme.click(
            fn=None,
            _js='preview_theme',
            inputs=[dummy_component],
            outputs=[dummy_component]
        )

    startup_timer.record("ui-settings")

    interfaces = [
        (txt2img_interface, "From Text", "txt2img"),
        (img2img_interface, "From Image", "img2img"),
        (extras_interface, "Process Image", "process"),
        (train_interface, "Train", "train"),
        (models_interface, "Models", "models"),
    ]
    interfaces += script_callbacks.ui_tabs_callback()
    interfaces += [(settings_interface, "Settings", "settings")]
    extensions_interface = ui_extensions.create_ui()
    interfaces += [(extensions_interface, "Extensions", "extensions")]
    startup_timer.record("ui-extensions")

    modules.shared.tab_names = []
    for _interface, label, _ifid in interfaces:
        modules.shared.tab_names.append(label)

    with gr.Blocks(theme=modules.shared.gradio_theme, analytics_enabled=False, title="SD.Next", allowed_paths=[cmd_opts.data_dir]) as demo:
        with gr.Row(elem_id="quicksettings", variant="compact"):
            for _i, k, _item in sorted(quicksettings_list, key=lambda x: quicksettings_names.get(x[1], x[0])):
                component = create_setting_component(k, is_quicksettings=True)
                component_dict[k] = component

        parameters_copypaste.connect_paste_params_buttons()

        with gr.Tabs(elem_id="tabs") as tabs:
            for interface, label, ifid in interfaces:
                if label in modules.shared.opts.hidden_tabs:
                    continue
                with gr.TabItem(label, id=ifid, elem_id=f"tab_{ifid}"):
                    interface.render()
            for interface, _label, ifid in interfaces:
                if ifid in ["extensions", "settings"]:
                    continue
                loadsave.add_block(interface, ifid)
            loadsave.add_component(f"webui/Tabs@{tabs.elem_id}", tabs)
            loadsave.setup_ui()
        if opts.notification_audio_enable and os.path.exists(os.path.join(script_path, opts.notification_audio_path)):
            gr.Audio(interactive=False, value=os.path.join(script_path, opts.notification_audio_path), elem_id="audio_notification", visible=False)

        text_settings = gr.Textbox(elem_id="settings_json", value=lambda: opts.dumpjson(), visible=False)
        settings_submit.click(
            fn=wrap_gradio_call(run_settings, extra_outputs=[gr.update()]),
            inputs=components,
            outputs=[text_settings, result],
        )
        defaults_submit.click(fn=lambda x: modules.shared.restore_defaults(restart=True), _js="restart_reload")
        restart_submit.click(fn=lambda x: modules.shared.restart_server(restart=True), _js="restart_reload")
        shutdown_submit.click(fn=lambda x: modules.shared.restart_server(restart=False), _js="restart_reload")

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

    startup_timer.record("ui-defaults")
    loadsave.dump_defaults()
    demo.ui_loadsave = loadsave
    return demo


def webpath(fn):
    if fn.startswith(script_path):
        web_path = os.path.relpath(fn, script_path).replace('\\', '/')
    else:
        web_path = os.path.abspath(fn)
    return f'file={web_path}?{os.path.getmtime(fn)}'


def html_head():
    script_js = os.path.join(script_path, "javascript", "script.js")
    head = f'<script type="text/javascript" src="{webpath(script_js)}"></script>\n'
    added = []
    for script in modules.scripts.list_scripts("javascript", ".js"):
        if script.path == script_js:
            continue
        head += f'<script type="text/javascript" src="{webpath(script.path)}"></script>\n'
        added.append(script.path)
    for script in modules.scripts.list_scripts("javascript", ".mjs"):
        head += f'<script type="module" src="{webpath(script.path)}"></script>\n'
        added.append(script.path)
    added = [a.replace(script_path, '').replace('\\', '/') for a in added]
    # modules.shared.log.debug(f'Adding JS scripts: {added}')
    return head


def html_body():
    body = ''
    inline = ''
    if opts.theme_style != 'Auto':
        inline += f"set_theme('{opts.theme_style.lower()}');"
    body += f'<script type="text/javascript">{inline}</script>\n'
    return body


def html_css():
    added = []
    def stylesheet(fn):
        added.append(fn)
        return f'<link rel="stylesheet" property="stylesheet" href="{webpath(fn)}">'
    head = stylesheet('javascript/style.css')
    for cssfile in modules.scripts.list_files_with_name("style.css"):
        if not os.path.isfile(cssfile):
            continue
        head += stylesheet(cssfile)
    if opts.gradio_theme == 'black-orange':
        head += stylesheet(os.path.join(script_path, "javascript", "black-orange.css"))
    if os.path.exists(os.path.join(data_path, "user.css")):
        head += stylesheet(os.path.join(data_path, "user.css"))
    added = [a.replace(script_path, '').replace('\\', '/') for a in added]
    # modules.shared.log.debug(f'Adding CSS stylesheets: {added}')
    return head


def reload_javascript():
    head = html_head()
    css = html_css()
    body = html_body()

    def template_response(*args, **kwargs):
        res = modules.shared.GradioTemplateResponseOriginal(*args, **kwargs)
        res.body = res.body.replace(b'</head>', f'{head}</head>'.encode("utf8"))
        res.body = res.body.replace(b'</body>', f'{css}{body}</body>'.encode("utf8"))
        res.init_headers()
        return res

    gradio.routes.templates.TemplateResponse = template_response


def setup_ui_api(app):
    from pydantic import BaseModel, Field # pylint: disable=no-name-in-module
    from typing import List

    class QuicksettingsHint(BaseModel): # pylint: disable=too-few-public-methods
        name: str = Field(title="Name of the quicksettings field")
        label: str = Field(title="Label of the quicksettings field")

    def quicksettings_hint():
        return [QuicksettingsHint(name=k, label=v.label) for k, v in opts.data_labels.items()]

    app.add_api_route("/internal/quicksettings-hint", quicksettings_hint, methods=["GET"], response_model=List[QuicksettingsHint])
    app.add_api_route("/internal/ping", lambda: {}, methods=["GET"])


if not hasattr(modules.shared, 'GradioTemplateResponseOriginal'):
    modules.shared.GradioTemplateResponseOriginal = gradio.routes.templates.TemplateResponse
