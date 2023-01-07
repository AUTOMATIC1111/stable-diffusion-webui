import html
import json
import math
import mimetypes
import os
import platform
import random
import subprocess as sp
import sys
import tempfile
import time
import traceback
from functools import partial, reduce

import gradio as gr
import gradio.routes
import gradio.utils
import numpy as np
from PIL import Image, PngImagePlugin
from modules.call_queue import wrap_gradio_gpu_call, wrap_queued_call, wrap_gradio_call

from modules import sd_hijack, sd_models, localization, script_callbacks, ui_extensions, deepbooru
from modules.ui_components import FormRow, FormGroup, ToolButton, FormHTML
from modules.paths import script_path

from modules.shared import opts, cmd_opts, restricted_opts

import modules.codeformer_model
import modules.generation_parameters_copypaste as parameters_copypaste
import modules.gfpgan_model
import modules.hypernetworks.ui
import modules.scripts
import modules.shared as shared
import modules.styles
import modules.textual_inversion.ui
from modules import prompt_parser
from modules.images import save_image
from modules.sd_hijack import model_hijack
from modules.sd_samplers import samplers, samplers_for_img2img
import modules.textual_inversion.ui
import modules.hypernetworks.ui
from modules.generation_parameters_copypaste import image_from_url_text

# this is a fix for Windows users. Without it, javascript files will be served with text/html content-type and the browser will not show any UI
mimetypes.init()
mimetypes.add_type('application/javascript', '.js')

if not cmd_opts.share and not cmd_opts.listen:
    # fix gradio phoning home
    gradio.utils.version_check = lambda: None
    gradio.utils.get_local_ip_address = lambda: '127.0.0.1'

if cmd_opts.ngrok is not None:
    import modules.ngrok as ngrok
    print('ngrok authtoken detected, trying to connect...')
    ngrok.connect(
        cmd_opts.ngrok,
        cmd_opts.port if cmd_opts.port is not None else 7860,
        cmd_opts.ngrok_region
        )


def gr_show(visible=True):
    return {"visible": visible, "__type__": "update"}


sample_img2img = "assets/stable-samples/img2img/sketch-mountains-input.jpg"
sample_img2img = sample_img2img if os.path.exists(sample_img2img) else None

css_hide_progressbar = """
.wrap .m-12 svg { display:none!important; }
.wrap .m-12::before { content:"Loading..." }
.wrap .z-20 svg { display:none!important; }
.wrap .z-20::before { content:"Loading..." }
.progress-bar { display:none!important; }
.meta-text { display:none!important; }
.meta-text-center { display:none!important; }
"""

# Using constants for these since the variation selector isn't visible.
# Important that they exactly match script.js for tooltip to work.
random_symbol = '\U0001f3b2\ufe0f'  # 🎲️
reuse_symbol = '\u267b\ufe0f'  # ♻️
paste_symbol = '\u2199\ufe0f'  # ↙
folder_symbol = '\U0001f4c2'  # 📂
refresh_symbol = '\U0001f504'  # 🔄
save_style_symbol = '\U0001f4be'  # 💾
apply_style_symbol = '\U0001f4cb'  # 📋
clear_prompt_symbol = '\U0001F5D1'  # 🗑️


def plaintext_to_html(text):
    text = "<p>" + "<br>\n".join([f"{html.escape(x)}" for x in text.split('\n')]) + "</p>"
    return text

def send_gradio_gallery_to_image(x):
    if len(x) == 0:
        return None
    return image_from_url_text(x[0])

def save_files(js_data, images, do_make_zip, index):
    import csv
    filenames = []
    fullfns = []

    #quick dictionary to class object conversion. Its necessary due apply_filename_pattern requiring it
    class MyObject:
        def __init__(self, d=None):
            if d is not None:
                for key, value in d.items():
                    setattr(self, key, value)

    data = json.loads(js_data)

    p = MyObject(data)
    path = opts.outdir_save
    save_to_dirs = opts.use_save_to_dirs_for_ui
    extension: str = opts.samples_format
    start_index = 0

    if index > -1 and opts.save_selected_only and (index >= data["index_of_first_image"]):  # ensures we are looking at a specific non-grid picture, and we have save_selected_only

        images = [images[index]]
        start_index = index

    os.makedirs(opts.outdir_save, exist_ok=True)

    with open(os.path.join(opts.outdir_save, "log.csv"), "a", encoding="utf8", newline='') as file:
        at_start = file.tell() == 0
        writer = csv.writer(file)
        if at_start:
            writer.writerow(["prompt", "seed", "width", "height", "sampler", "cfgs", "steps", "filename", "negative_prompt"])

        for image_index, filedata in enumerate(images, start_index):
            image = image_from_url_text(filedata)

            is_grid = image_index < p.index_of_first_image
            i = 0 if is_grid else (image_index - p.index_of_first_image)

            fullfn, txt_fullfn = save_image(image, path, "", seed=p.all_seeds[i], prompt=p.all_prompts[i], extension=extension, info=p.infotexts[image_index], grid=is_grid, p=p, save_to_dirs=save_to_dirs)

            filename = os.path.relpath(fullfn, path)
            filenames.append(filename)
            fullfns.append(fullfn)
            if txt_fullfn:
                filenames.append(os.path.basename(txt_fullfn))
                fullfns.append(txt_fullfn)

        writer.writerow([data["prompt"], data["seed"], data["width"], data["height"], data["sampler_name"], data["cfg_scale"], data["steps"], filenames[0], data["negative_prompt"]])

    # Make Zip
    if do_make_zip:
        zip_filepath = os.path.join(path, "images.zip")

        from zipfile import ZipFile
        with ZipFile(zip_filepath, "w") as zip_file:
            for i in range(len(fullfns)):
                with open(fullfns[i], mode="rb") as f:
                    zip_file.writestr(filenames[i], f.read())
        fullfns.insert(0, zip_filepath)

    return gr.File.update(value=fullfns, visible=True), plaintext_to_html(f"Saved: {filenames[0]}")


def calc_time_left(progress, threshold, label, force_display, show_eta):
    if progress == 0:
        return ""
    else:
        time_since_start = time.time() - shared.state.time_start
        eta = (time_since_start/progress)
        eta_relative = eta-time_since_start
        if (eta_relative > threshold and show_eta) or force_display:
            if eta_relative > 3600:
                return label + time.strftime('%H:%M:%S', time.gmtime(eta_relative))
            elif eta_relative > 60:
                return label + time.strftime('%M:%S',  time.gmtime(eta_relative))
            else:
                return label + time.strftime('%Ss',  time.gmtime(eta_relative))
        else:
            return ""


def check_progress_call(id_part):
    if shared.state.job_count == 0:
        return "", gr_show(False), gr_show(False), gr_show(False)

    progress = 0

    if shared.state.job_count > 0:
        progress += shared.state.job_no / shared.state.job_count
    if shared.state.sampling_steps > 0:
        progress += 1 / shared.state.job_count * shared.state.sampling_step / shared.state.sampling_steps

    # Show progress percentage and time left at the same moment, and base it also on steps done
    show_eta = progress >= 0.01 or shared.state.sampling_step >= 10

    time_left = calc_time_left(progress, 1, " ETA: ", shared.state.time_left_force_display, show_eta)
    if time_left != "":
        shared.state.time_left_force_display = True

    progress = min(progress, 1)

    progressbar = ""
    if opts.show_progressbar:
        progressbar = f"""<div class='progressDiv'><div class='progress' style="overflow:visible;width:{progress * 100}%;white-space:nowrap;">{"&nbsp;" * 2 + str(int(progress*100))+"%" + time_left if show_eta else ""}</div></div>"""

    image = gr_show(False)
    preview_visibility = gr_show(False)

    if opts.show_progress_every_n_steps != 0:
        shared.state.set_current_image()
        image = shared.state.current_image

        if image is None:
            image = gr.update(value=None)
        else:
            preview_visibility = gr_show(True)

    if shared.state.textinfo is not None:
        textinfo_result = gr.HTML.update(value=shared.state.textinfo, visible=True)
    else:
        textinfo_result = gr_show(False)

    return f"<span id='{id_part}_progress_span' style='display: none'>{time.time()}</span><p>{progressbar}</p>", preview_visibility, image, textinfo_result


def check_progress_call_initial(id_part):
    shared.state.job_count = -1
    shared.state.current_latent = None
    shared.state.current_image = None
    shared.state.textinfo = None
    shared.state.time_start = time.time()
    shared.state.time_left_force_display = False

    return check_progress_call(id_part)


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
    # Save all loaded prompt styles: this allows us to update the storage format in the future more easily, because we
    # reserialize all styles every time we save them
    shared.prompt_styles.save_styles(shared.styles_filename)

    return [gr.Dropdown.update(visible=True, choices=list(shared.prompt_styles.styles)) for _ in range(4)]


def calc_resolution_hires(enable, width, height, hr_scale, hr_resize_x, hr_resize_y):
    from modules import processing, devices

    if not enable:
        return ""

    p = processing.StableDiffusionProcessingTxt2Img(width=width, height=height, enable_hr=True, hr_scale=hr_scale, hr_resize_x=hr_resize_x, hr_resize_y=hr_resize_y)

    with devices.autocast():
        p.init([""], [0], [0])

    return f"resize: from <span class='resolution'>{width}x{height}</span> to <span class='resolution'>{p.hr_upscale_to_x}x{p.hr_upscale_to_y}</span>"


def apply_styles(prompt, prompt_neg, style1_name, style2_name):
    prompt = shared.prompt_styles.apply_styles_to_prompt(prompt, [style1_name, style2_name])
    prompt_neg = shared.prompt_styles.apply_negative_styles_to_prompt(prompt_neg, [style1_name, style2_name])

    return [gr.Textbox.update(value=prompt), gr.Textbox.update(value=prompt_neg), gr.Dropdown.update(value="None"), gr.Dropdown.update(value="None")]


def interrogate(image):
    prompt = shared.interrogator.interrogate(image.convert("RGB"))

    return gr_show(True) if prompt is None else prompt


def interrogate_deepbooru(image):
    prompt = deepbooru.model.tag(image)
    return gr_show(True) if prompt is None else prompt


def create_seed_inputs(target_interface):
    with FormRow(elem_id=target_interface + '_seed_row'):
        seed = (gr.Textbox if cmd_opts.use_textbox_seed else gr.Number)(label='Seed', value=-1, elem_id=target_interface + '_seed')
        seed.style(container=False)
        random_seed = gr.Button(random_symbol, elem_id=target_interface + '_random_seed')
        reuse_seed = gr.Button(reuse_symbol, elem_id=target_interface + '_reuse_seed')

        with gr.Group(elem_id=target_interface + '_subseed_show_box'):
            seed_checkbox = gr.Checkbox(label='Extra', elem_id=target_interface + '_subseed_show', value=False)

    # Components to show/hide based on the 'Extra' checkbox
    seed_extras = []

    with FormRow(visible=False, elem_id=target_interface + '_subseed_row') as seed_extra_row_1:
        seed_extras.append(seed_extra_row_1)
        subseed = gr.Number(label='Variation seed', value=-1, elem_id=target_interface + '_subseed')
        subseed.style(container=False)
        random_subseed = gr.Button(random_symbol, elem_id=target_interface + '_random_subseed')
        reuse_subseed = gr.Button(reuse_symbol, elem_id=target_interface + '_reuse_subseed')
        subseed_strength = gr.Slider(label='Variation strength', value=0.0, minimum=0, maximum=1, step=0.01, elem_id=target_interface + '_subseed_strength')

    with FormRow(visible=False) as seed_extra_row_2:
        seed_extras.append(seed_extra_row_2)
        seed_resize_from_w = gr.Slider(minimum=0, maximum=2048, step=8, label="Resize seed from width", value=0, elem_id=target_interface + '_seed_resize_from_w')
        seed_resize_from_h = gr.Slider(minimum=0, maximum=2048, step=8, label="Resize seed from height", value=0, elem_id=target_interface + '_seed_resize_from_h')

    random_seed.click(fn=lambda: -1, show_progress=False, inputs=[], outputs=[seed])
    random_subseed.click(fn=lambda: -1, show_progress=False, inputs=[], outputs=[subseed])

    def change_visibility(show):
        return {comp: gr_show(show) for comp in seed_extras}

    seed_checkbox.change(change_visibility, show_progress=False, inputs=[seed_checkbox], outputs=seed_extras)

    return seed, reuse_seed, subseed, reuse_subseed, subseed_strength, seed_resize_from_h, seed_resize_from_w, seed_checkbox



def connect_clear_prompt(button):
    """Given clear button, prompt, and token_counter objects, setup clear prompt button click event"""
    button.click(
        _js="clear_prompt",
        fn=None,
        inputs=[],
        outputs=[],
    )


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

        except json.decoder.JSONDecodeError as e:
            if gen_info_string != '':
                print("Error parsing JSON generation info:", file=sys.stderr)
                print(gen_info_string, file=sys.stderr)

        return [res, gr_show(False)]

    reuse_seed.click(
        fn=copy_seed,
        _js="(x, y) => [x, selected_gallery_index()]",
        show_progress=False,
        inputs=[generation_info, dummy_component],
        outputs=[seed, dummy_component]
    )


def update_token_counter(text, steps):
    try:
        _, prompt_flat_list, _ = prompt_parser.get_multicond_prompt_list([text])
        prompt_schedules = prompt_parser.get_learned_conditioning_prompt_schedules(prompt_flat_list, steps)

    except Exception:
        # a parsing error can happen here during typing, and we don't want to bother the user with
        # messages related to it in console
        prompt_schedules = [[[steps, text]]]

    flat_prompts = reduce(lambda list1, list2: list1+list2, prompt_schedules)
    prompts = [prompt_text for step, prompt_text in flat_prompts]
    token_count, max_length = max([model_hijack.get_prompt_lengths(prompt) for prompt in prompts], key=lambda args: args[0])
    style_class = ' class="red"' if (token_count > max_length) else ""
    return f"<span {style_class}>{token_count}/{max_length}</span>"


def create_toprow(is_img2img):
    id_part = "img2img" if is_img2img else "txt2img"

    with gr.Row(elem_id="toprow"):
        with gr.Column(scale=6):
            with gr.Row():
                with gr.Column(scale=80):
                    with gr.Row():
                        prompt = gr.Textbox(label="Prompt", elem_id=f"{id_part}_prompt", show_label=False, lines=2,
                            placeholder="Prompt (press Ctrl+Enter or Alt+Enter to generate)"
                        )

            with gr.Row():
                with gr.Column(scale=80):
                    with gr.Row():
                        negative_prompt = gr.Textbox(label="Negative prompt", elem_id=f"{id_part}_neg_prompt", show_label=False, lines=2,
                            placeholder="Negative prompt (press Ctrl+Enter or Alt+Enter to generate)"
                        )

        with gr.Column(scale=1, elem_id="roll_col"):
            paste = gr.Button(value=paste_symbol, elem_id="paste")
            save_style = gr.Button(value=save_style_symbol, elem_id="style_create")
            prompt_style_apply = gr.Button(value=apply_style_symbol, elem_id="style_apply")
            clear_prompt_button = gr.Button(value=clear_prompt_symbol, elem_id=f"{id_part}_clear_prompt")
            token_counter = gr.HTML(value="<span></span>", elem_id=f"{id_part}_token_counter")
            token_button = gr.Button(visible=False, elem_id=f"{id_part}_token_button")

            clear_prompt_button.click(
                fn=lambda *x: x,
                _js="confirm_clear_prompt",
                inputs=[prompt, negative_prompt],
                outputs=[prompt, negative_prompt],
            )

        button_interrogate = None
        button_deepbooru = None
        if is_img2img:
            with gr.Column(scale=1, elem_id="interrogate_col"):
                button_interrogate = gr.Button('Interrogate\nCLIP', elem_id="interrogate")
                button_deepbooru = gr.Button('Interrogate\nDeepBooru', elem_id="deepbooru")

        with gr.Column(scale=1):
            with gr.Row():
                skip = gr.Button('Skip', elem_id=f"{id_part}_skip")
                interrupt = gr.Button('Interrupt', elem_id=f"{id_part}_interrupt")
                submit = gr.Button('Generate', elem_id=f"{id_part}_generate", variant='primary')

                skip.click(
                    fn=lambda: shared.state.skip(),
                    inputs=[],
                    outputs=[],
                )

                interrupt.click(
                    fn=lambda: shared.state.interrupt(),
                    inputs=[],
                    outputs=[],
                )

            with gr.Row():
                with gr.Column(scale=1, elem_id="style_pos_col"):
                    prompt_style = gr.Dropdown(label="Style 1", elem_id=f"{id_part}_style_index", choices=[k for k, v in shared.prompt_styles.styles.items()], value=next(iter(shared.prompt_styles.styles.keys())))

                with gr.Column(scale=1, elem_id="style_neg_col"):
                    prompt_style2 = gr.Dropdown(label="Style 2", elem_id=f"{id_part}_style2_index", choices=[k for k, v in shared.prompt_styles.styles.items()], value=next(iter(shared.prompt_styles.styles.keys())))

    return prompt, prompt_style, negative_prompt, prompt_style2, submit, button_interrogate, button_deepbooru, prompt_style_apply, save_style, paste, token_counter, token_button


def setup_progressbar(progressbar, preview, id_part, textinfo=None):
    if textinfo is None:
        textinfo = gr.HTML(visible=False)

    check_progress = gr.Button('Check progress', elem_id=f"{id_part}_check_progress", visible=False)
    check_progress.click(
        fn=lambda: check_progress_call(id_part),
        show_progress=False,
        inputs=[],
        outputs=[progressbar, preview, preview, textinfo],
    )

    check_progress_initial = gr.Button('Check progress (first)', elem_id=f"{id_part}_check_progress_initial", visible=False)
    check_progress_initial.click(
        fn=lambda: check_progress_call_initial(id_part),
        show_progress=False,
        inputs=[],
        outputs=[progressbar, preview, preview, textinfo],
    )


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
    return value


def update_generation_info(args):
    generation_info, html_info, img_index = args
    try:
        generation_info = json.loads(generation_info)
        if img_index < 0 or img_index >= len(generation_info["infotexts"]):
            return html_info
        return plaintext_to_html(generation_info["infotexts"][img_index])
    except Exception:
        pass
    # if the json parse or anything else fails, just return the old html_info
    return html_info


def create_refresh_button(refresh_component, refresh_method, refreshed_args, elem_id):
    def refresh():
        refresh_method()
        args = refreshed_args() if callable(refreshed_args) else refreshed_args

        for k, v in args.items():
            setattr(refresh_component, k, v)

        return gr.update(**(args or {}))

    refresh_button = ToolButton(value=refresh_symbol, elem_id=elem_id)
    refresh_button.click(
        fn=refresh,
        inputs=[],
        outputs=[refresh_component]
    )
    return refresh_button


def create_output_panel(tabname, outdir):
    def open_folder(f):
        if not os.path.exists(f):
            print(f'Folder "{f}" does not exist. After you create an image, the folder will be created.')
            return
        elif not os.path.isdir(f):
            print(f"""
WARNING
An open_folder request was made with an argument that is not a folder.
This could be an error or a malicious attempt to run code on your computer.
Requested path was: {f}
""", file=sys.stderr)
            return

        if not shared.cmd_opts.hide_ui_dir_config:
            path = os.path.normpath(f)
            if platform.system() == "Windows":
                os.startfile(path)
            elif platform.system() == "Darwin":
                sp.Popen(["open", path])
            elif "microsoft-standard-WSL2" in platform.uname().release:
                sp.Popen(["wsl-open", path])
            else:
                sp.Popen(["xdg-open", path])

    with gr.Column(variant='panel'):
            with gr.Group():
                result_gallery = gr.Gallery(label='Output', show_label=False, elem_id=f"{tabname}_gallery").style(grid=4)

            generation_info = None
            with gr.Column():
                with gr.Row(elem_id=f"image_buttons_{tabname}"):
                    open_folder_button = gr.Button(folder_symbol, elem_id="hidden_element" if shared.cmd_opts.hide_ui_dir_config else f'open_folder_{tabname}')

                    if tabname != "extras":
                        save = gr.Button('Save', elem_id=f'save_{tabname}')
                        save_zip = gr.Button('Zip', elem_id=f'save_zip_{tabname}')

                    buttons = parameters_copypaste.create_buttons(["img2img", "inpaint", "extras"])

                open_folder_button.click(
                    fn=lambda: open_folder(opts.outdir_samples or outdir),
                    inputs=[],
                    outputs=[],
                )

                if tabname != "extras":
                    with gr.Row():
                        download_files = gr.File(None, file_count="multiple", interactive=False, show_label=False, visible=False, elem_id=f'download_files_{tabname}')

                    with gr.Group():
                        html_info = gr.HTML(elem_id=f'html_info_{tabname}')
                        html_log = gr.HTML(elem_id=f'html_log_{tabname}')

                        generation_info = gr.Textbox(visible=False, elem_id=f'generation_info_{tabname}')
                        if tabname == 'txt2img' or tabname == 'img2img':
                            generation_info_button = gr.Button(visible=False, elem_id=f"{tabname}_generation_info_button")
                            generation_info_button.click(
                                fn=update_generation_info,
                                _js="(x, y) => [x, y, selected_gallery_index()]",
                                inputs=[generation_info, html_info],
                                outputs=[html_info],
                                preprocess=False
                            )

                        save.click(
                            fn=wrap_gradio_call(save_files),
                            _js="(x, y, z, w) => [x, y, false, selected_gallery_index()]",
                            inputs=[
                                generation_info,
                                result_gallery,
                                html_info,
                                html_info,
                            ],
                            outputs=[
                                download_files,
                                html_log,
                            ]
                        )

                        save_zip.click(
                            fn=wrap_gradio_call(save_files),
                            _js="(x, y, z, w) => [x, y, true, selected_gallery_index()]",
                            inputs=[
                                generation_info,
                                result_gallery,
                                html_info,
                                html_info,
                            ],
                            outputs=[
                                download_files,
                                html_log,
                            ]
                        )

                else:
                    html_info_x = gr.HTML(elem_id=f'html_info_x_{tabname}')
                    html_info = gr.HTML(elem_id=f'html_info_{tabname}')
                    html_log = gr.HTML(elem_id=f'html_log_{tabname}')

                parameters_copypaste.bind_buttons(buttons, result_gallery, "txt2img" if tabname == "txt2img" else None)
                return result_gallery, generation_info if tabname != "extras" else html_info_x, html_info, html_log


def create_sampler_and_steps_selection(choices, tabname):
    if opts.samplers_in_dropdown:
        with FormRow(elem_id=f"sampler_selection_{tabname}"):
            sampler_index = gr.Dropdown(label='Sampling method', elem_id=f"{tabname}_sampling", choices=[x.name for x in choices], value=choices[0].name, type="index")
            steps = gr.Slider(minimum=1, maximum=150, step=1, elem_id=f"{tabname}_steps", label="Sampling steps", value=20)
    else:
        with FormGroup(elem_id=f"sampler_selection_{tabname}"):
            steps = gr.Slider(minimum=1, maximum=150, step=1, elem_id=f"{tabname}_steps", label="Sampling steps", value=20)
            sampler_index = gr.Radio(label='Sampling method', elem_id=f"{tabname}_sampling", choices=[x.name for x in choices], value=choices[0].name, type="index")

    return steps, sampler_index


def ordered_ui_categories():
    user_order = {x.strip(): i for i, x in enumerate(shared.opts.ui_reorder.split(","))}

    for i, category in sorted(enumerate(shared.ui_reorder_categories), key=lambda x: user_order.get(x[1], x[0] + 1000)):
        yield category


def create_ui():
    import modules.img2img
    import modules.txt2img

    reload_javascript()

    parameters_copypaste.reset()

    modules.scripts.scripts_current = modules.scripts.scripts_txt2img
    modules.scripts.scripts_txt2img.initialize_scripts(is_img2img=False)

    with gr.Blocks(analytics_enabled=False) as txt2img_interface:
        txt2img_prompt, txt2img_prompt_style, txt2img_negative_prompt, txt2img_prompt_style2, submit, _, _,txt2img_prompt_style_apply, txt2img_save_style, txt2img_paste, token_counter, token_button = create_toprow(is_img2img=False)

        dummy_component = gr.Label(visible=False)
        txt_prompt_img = gr.File(label="", elem_id="txt2img_prompt_image", file_count="single", type="bytes", visible=False)

        with gr.Row(elem_id='txt2img_progress_row'):
            with gr.Column(scale=1):
                pass

            with gr.Column(scale=1):
                progressbar = gr.HTML(elem_id="txt2img_progressbar")
                txt2img_preview = gr.Image(elem_id='txt2img_preview', visible=False)
                setup_progressbar(progressbar, txt2img_preview, 'txt2img')

        with gr.Row().style(equal_height=False):
            with gr.Column(variant='panel', elem_id="txt2img_settings"):
                for category in ordered_ui_categories():
                    if category == "sampler":
                        steps, sampler_index = create_sampler_and_steps_selection(samplers, "txt2img")

                    elif category == "dimensions":
                        with FormRow():
                            with gr.Column(elem_id="txt2img_column_size", scale=4):
                                width = gr.Slider(minimum=64, maximum=2048, step=8, label="Width", value=512, elem_id="txt2img_width")
                                height = gr.Slider(minimum=64, maximum=2048, step=8, label="Height", value=512, elem_id="txt2img_height")

                            if opts.dimensions_and_batch_together:
                                with gr.Column(elem_id="txt2img_column_batch"):
                                    batch_count = gr.Slider(minimum=1, step=1, label='Batch count', value=1, elem_id="txt2img_batch_count")
                                    batch_size = gr.Slider(minimum=1, maximum=8, step=1, label='Batch size', value=1, elem_id="txt2img_batch_size")

                    elif category == "cfg":
                        cfg_scale = gr.Slider(minimum=1.0, maximum=30.0, step=0.5, label='CFG Scale', value=7.0, elem_id="txt2img_cfg_scale")

                    elif category == "seed":
                        seed, reuse_seed, subseed, reuse_subseed, subseed_strength, seed_resize_from_h, seed_resize_from_w, seed_checkbox = create_seed_inputs('txt2img')

                    elif category == "checkboxes":
                        with FormRow(elem_id="txt2img_checkboxes"):
                            restore_faces = gr.Checkbox(label='Restore faces', value=False, visible=len(shared.face_restorers) > 1, elem_id="txt2img_restore_faces")
                            tiling = gr.Checkbox(label='Tiling', value=False, elem_id="txt2img_tiling")
                            enable_hr = gr.Checkbox(label='Hires. fix', value=False, elem_id="txt2img_enable_hr")
                            hr_final_resolution = FormHTML(value="", elem_id="txtimg_hr_finalres", label="Upscaled resolution", interactive=False)

                    elif category == "hires_fix":
                        with FormGroup(visible=False, elem_id="txt2img_hires_fix") as hr_options:
                            with FormRow(elem_id="txt2img_hires_fix_row1"):
                                hr_upscaler = gr.Dropdown(label="Upscaler", elem_id="txt2img_hr_upscaler", choices=[*shared.latent_upscale_modes, *[x.name for x in shared.sd_upscalers]], value=shared.latent_upscale_default_mode)
                                hr_second_pass_steps = gr.Slider(minimum=0, maximum=150, step=1, label='Hires steps', value=0, elem_id="txt2img_hires_steps")
                                denoising_strength = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label='Denoising strength', value=0.7, elem_id="txt2img_denoising_strength")

                            with FormRow(elem_id="txt2img_hires_fix_row2"):
                                hr_scale = gr.Slider(minimum=1.0, maximum=4.0, step=0.05, label="Upscale by", value=2.0, elem_id="txt2img_hr_scale")
                                hr_resize_x = gr.Slider(minimum=0, maximum=2048, step=8, label="Resize width to", value=0, elem_id="txt2img_hr_resize_x")
                                hr_resize_y = gr.Slider(minimum=0, maximum=2048, step=8, label="Resize height to", value=0, elem_id="txt2img_hr_resize_y")

                    elif category == "batch":
                        if not opts.dimensions_and_batch_together:
                            with FormRow(elem_id="txt2img_column_batch"):
                                batch_count = gr.Slider(minimum=1, step=1, label='Batch count', value=1, elem_id="txt2img_batch_count")
                                batch_size = gr.Slider(minimum=1, maximum=8, step=1, label='Batch size', value=1, elem_id="txt2img_batch_size")

                    elif category == "scripts":
                        with FormGroup(elem_id="txt2img_script_container"):
                            custom_inputs = modules.scripts.scripts_txt2img.setup_ui()

            hr_resolution_preview_inputs = [enable_hr, width, height, hr_scale, hr_resize_x, hr_resize_y]
            hr_resolution_preview_args = dict(
                fn=calc_resolution_hires,
                inputs=hr_resolution_preview_inputs,
                outputs=[hr_final_resolution],
                show_progress=False
            )

            for input in hr_resolution_preview_inputs:
                input.change(**hr_resolution_preview_args)

            txt2img_gallery, generation_info, html_info, html_log = create_output_panel("txt2img", opts.outdir_txt2img_samples)
            parameters_copypaste.bind_buttons({"txt2img": txt2img_paste}, None, txt2img_prompt)

            connect_reuse_seed(seed, reuse_seed, generation_info, dummy_component, is_subseed=False)
            connect_reuse_seed(subseed, reuse_subseed, generation_info, dummy_component, is_subseed=True)

            txt2img_args = dict(
                fn=wrap_gradio_gpu_call(modules.txt2img.txt2img, extra_outputs=[None, '', '']),
                _js="submit",
                inputs=[
                    txt2img_prompt,
                    txt2img_negative_prompt,
                    txt2img_prompt_style,
                    txt2img_prompt_style2,
                    steps,
                    sampler_index,
                    restore_faces,
                    tiling,
                    batch_count,
                    batch_size,
                    cfg_scale,
                    seed,
                    subseed, subseed_strength, seed_resize_from_h, seed_resize_from_w, seed_checkbox,
                    height,
                    width,
                    enable_hr,
                    denoising_strength,
                    hr_scale,
                    hr_upscaler,
                    hr_second_pass_steps,
                    hr_resize_x,
                    hr_resize_y,
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
                (subseed_strength, "Variation seed strength"),
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
            parameters_copypaste.add_paste_fields("txt2img", None, txt2img_paste_fields)

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

    modules.scripts.scripts_current = modules.scripts.scripts_img2img
    modules.scripts.scripts_img2img.initialize_scripts(is_img2img=True)

    with gr.Blocks(analytics_enabled=False) as img2img_interface:
        img2img_prompt, img2img_prompt_style, img2img_negative_prompt, img2img_prompt_style2, submit, img2img_interrogate, img2img_deepbooru, img2img_prompt_style_apply, img2img_save_style, img2img_paste,token_counter, token_button = create_toprow(is_img2img=True)

        with gr.Row(elem_id='img2img_progress_row'):
            img2img_prompt_img = gr.File(label="", elem_id="img2img_prompt_image", file_count="single", type="bytes", visible=False)

            with gr.Column(scale=1):
                pass

            with gr.Column(scale=1):
                progressbar = gr.HTML(elem_id="img2img_progressbar")
                img2img_preview = gr.Image(elem_id='img2img_preview', visible=False)
                setup_progressbar(progressbar, img2img_preview, 'img2img')

        with FormRow().style(equal_height=False):
            with gr.Column(variant='panel', elem_id="img2img_settings"):

                with gr.Tabs(elem_id="mode_img2img") as tabs_img2img_mode:
                    with gr.TabItem('img2img', id='img2img', elem_id="img2img_img2img_tab"):
                        init_img = gr.Image(label="Image for img2img", elem_id="img2img_image", show_label=False, source="upload", interactive=True, type="pil", tool=cmd_opts.gradio_img2img_tool, image_mode="RGBA").style(height=480)

                    with gr.TabItem('Inpaint', id='inpaint', elem_id="img2img_inpaint_tab"):
                        init_img_with_mask = gr.Image(label="Image for inpainting with mask", show_label=False, elem_id="img2maskimg", source="upload", interactive=True, type="pil", tool=cmd_opts.gradio_inpaint_tool, image_mode="RGBA").style(height=480)
                        init_img_with_mask_orig = gr.State(None)

                        use_color_sketch = cmd_opts.gradio_inpaint_tool == "color-sketch"
                        if use_color_sketch:
                            def update_orig(image, state):
                                if image is not None:
                                    same_size = state is not None and state.size == image.size
                                    has_exact_match = np.any(np.all(np.array(image) == np.array(state), axis=-1))
                                    edited = same_size and has_exact_match
                                    return image if not edited or state is None else state

                            init_img_with_mask.change(update_orig, [init_img_with_mask, init_img_with_mask_orig], init_img_with_mask_orig)

                        init_img_inpaint = gr.Image(label="Image for img2img", show_label=False, source="upload", interactive=True, type="pil", visible=False, elem_id="img_inpaint_base")
                        init_mask_inpaint = gr.Image(label="Mask", source="upload", interactive=True, type="pil", visible=False, elem_id="img_inpaint_mask")

                        with FormRow():
                            mask_blur = gr.Slider(label='Mask blur', minimum=0, maximum=64, step=1, value=4, elem_id="img2img_mask_blur")
                            mask_alpha = gr.Slider(label="Mask transparency", interactive=use_color_sketch, visible=use_color_sketch, elem_id="img2img_mask_alpha")

                        with FormRow():
                            mask_mode = gr.Radio(label="Mask source", choices=["Draw mask", "Upload mask"], type="index", value="Draw mask", elem_id="mask_mode")
                            inpainting_mask_invert = gr.Radio(label='Mask mode', choices=['Inpaint masked', 'Inpaint not masked'], value='Inpaint masked', type="index", elem_id="img2img_mask_mode")

                        with FormRow():
                            inpainting_fill = gr.Radio(label='Masked content', choices=['fill', 'original', 'latent noise', 'latent nothing'], value='original', type="index", elem_id="img2img_inpainting_fill")

                        with FormRow():
                            with gr.Column():
                                inpaint_full_res = gr.Radio(label="Inpaint area", choices=["Whole picture", "Only masked"], type="index", value="Whole picture", elem_id="img2img_inpaint_full_res")

                            with gr.Column(scale=4):
                                inpaint_full_res_padding = gr.Slider(label='Only masked padding, pixels', minimum=0, maximum=256, step=4, value=32, elem_id="img2img_inpaint_full_res_padding")

                    with gr.TabItem('Batch img2img', id='batch', elem_id="img2img_batch_tab"):
                        hidden = '<br>Disabled when launched with --hide-ui-dir-config.' if shared.cmd_opts.hide_ui_dir_config else ''
                        gr.HTML(f"<p class=\"text-gray-500\">Process images in a directory on the same machine where the server is running.<br>Use an empty output directory to save pictures normally instead of writing to the output directory.{hidden}</p>")
                        img2img_batch_input_dir = gr.Textbox(label="Input directory", **shared.hide_dirs, elem_id="img2img_batch_input_dir")
                        img2img_batch_output_dir = gr.Textbox(label="Output directory", **shared.hide_dirs, elem_id="img2img_batch_output_dir")

                with FormRow():
                    resize_mode = gr.Radio(label="Resize mode", elem_id="resize_mode", choices=["Just resize", "Crop and resize", "Resize and fill", "Just resize (latent upscale)"], type="index", value="Just resize")

                for category in ordered_ui_categories():
                    if category == "sampler":
                        steps, sampler_index = create_sampler_and_steps_selection(samplers_for_img2img, "img2img")

                    elif category == "dimensions":
                        with FormRow():
                            with gr.Column(elem_id="img2img_column_size", scale=4):
                                width = gr.Slider(minimum=64, maximum=2048, step=8, label="Width", value=512, elem_id="img2img_width")
                                height = gr.Slider(minimum=64, maximum=2048, step=8, label="Height", value=512, elem_id="img2img_height")

                            if opts.dimensions_and_batch_together:
                                with gr.Column(elem_id="img2img_column_batch"):
                                    batch_count = gr.Slider(minimum=1, step=1, label='Batch count', value=1, elem_id="img2img_batch_count")
                                    batch_size = gr.Slider(minimum=1, maximum=8, step=1, label='Batch size', value=1, elem_id="img2img_batch_size")

                    elif category == "cfg":
                        with FormGroup():
                            cfg_scale = gr.Slider(minimum=1.0, maximum=30.0, step=0.5, label='CFG Scale', value=7.0, elem_id="img2img_cfg_scale")
                            denoising_strength = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label='Denoising strength', value=0.75, elem_id="img2img_denoising_strength")

                    elif category == "seed":
                        seed, reuse_seed, subseed, reuse_subseed, subseed_strength, seed_resize_from_h, seed_resize_from_w, seed_checkbox = create_seed_inputs('img2img')

                    elif category == "checkboxes":
                        with FormRow(elem_id="img2img_checkboxes"):
                            restore_faces = gr.Checkbox(label='Restore faces', value=False, visible=len(shared.face_restorers) > 1, elem_id="img2img_restore_faces")
                            tiling = gr.Checkbox(label='Tiling', value=False, elem_id="img2img_tiling")

                    elif category == "batch":
                        if not opts.dimensions_and_batch_together:
                            with FormRow(elem_id="img2img_column_batch"):
                                batch_count = gr.Slider(minimum=1, step=1, label='Batch count', value=1, elem_id="img2img_batch_count")
                                batch_size = gr.Slider(minimum=1, maximum=8, step=1, label='Batch size', value=1, elem_id="img2img_batch_size")

                    elif category == "scripts":
                        with FormGroup(elem_id="img2img_script_container"):
                            custom_inputs = modules.scripts.scripts_img2img.setup_ui()

            img2img_gallery, generation_info, html_info, html_log = create_output_panel("img2img", opts.outdir_img2img_samples)
            parameters_copypaste.bind_buttons({"img2img": img2img_paste}, None, img2img_prompt)

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

            mask_mode.change(
                lambda mode, img: {
                    init_img_with_mask: gr_show(mode == 0),
                    init_img_inpaint: gr_show(mode == 1),
                    init_mask_inpaint: gr_show(mode == 1),
                },
                inputs=[mask_mode, init_img_with_mask],
                outputs=[
                    init_img_with_mask,
                    init_img_inpaint,
                    init_mask_inpaint,
                ],
            )

            img2img_args = dict(
                fn=wrap_gradio_gpu_call(modules.img2img.img2img, extra_outputs=[None, '', '']),
                _js="submit_img2img",
                inputs=[
                    dummy_component,
                    img2img_prompt,
                    img2img_negative_prompt,
                    img2img_prompt_style,
                    img2img_prompt_style2,
                    init_img,
                    init_img_with_mask,
                    init_img_with_mask_orig,
                    init_img_inpaint,
                    init_mask_inpaint,
                    mask_mode,
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
                    denoising_strength,
                    seed,
                    subseed, subseed_strength, seed_resize_from_h, seed_resize_from_w, seed_checkbox,
                    height,
                    width,
                    resize_mode,
                    inpaint_full_res,
                    inpaint_full_res_padding,
                    inpainting_mask_invert,
                    img2img_batch_input_dir,
                    img2img_batch_output_dir,
                ] + custom_inputs,
                outputs=[
                    img2img_gallery,
                    generation_info,
                    html_info,
                    html_log,
                ],
                show_progress=False,
            )

            img2img_prompt.submit(**img2img_args)
            submit.click(**img2img_args)

            img2img_interrogate.click(
                fn=interrogate,
                inputs=[init_img],
                outputs=[img2img_prompt],
            )

            img2img_deepbooru.click(
                fn=interrogate_deepbooru,
                inputs=[init_img],
                outputs=[img2img_prompt],
            )

            prompts = [(txt2img_prompt, txt2img_negative_prompt), (img2img_prompt, img2img_negative_prompt)]
            style_dropdowns = [(txt2img_prompt_style, txt2img_prompt_style2), (img2img_prompt_style, img2img_prompt_style2)]
            style_js_funcs = ["update_txt2img_tokens", "update_img2img_tokens"]

            for button, (prompt, negative_prompt) in zip([txt2img_save_style, img2img_save_style], prompts):
                button.click(
                    fn=add_style,
                    _js="ask_for_style_name",
                    # Have to pass empty dummy component here, because the JavaScript and Python function have to accept
                    # the same number of parameters, but we only know the style-name after the JavaScript prompt
                    inputs=[dummy_component, prompt, negative_prompt],
                    outputs=[txt2img_prompt_style, img2img_prompt_style, txt2img_prompt_style2, img2img_prompt_style2],
                )

            for button, (prompt, negative_prompt), (style1, style2), js_func in zip([txt2img_prompt_style_apply, img2img_prompt_style_apply], prompts, style_dropdowns, style_js_funcs):
                button.click(
                    fn=apply_styles,
                    _js=js_func,
                    inputs=[prompt, negative_prompt, style1, style2],
                    outputs=[prompt, negative_prompt, style1, style2],
                )

            token_button.click(fn=update_token_counter, inputs=[img2img_prompt, steps], outputs=[token_counter])

            img2img_paste_fields = [
                (img2img_prompt, "Prompt"),
                (img2img_negative_prompt, "Negative prompt"),
                (steps, "Steps"),
                (sampler_index, "Sampler"),
                (restore_faces, "Face restoration"),
                (cfg_scale, "CFG scale"),
                (seed, "Seed"),
                (width, "Size-1"),
                (height, "Size-2"),
                (batch_size, "Batch size"),
                (subseed, "Variation seed"),
                (subseed_strength, "Variation seed strength"),
                (seed_resize_from_w, "Seed resize from-1"),
                (seed_resize_from_h, "Seed resize from-2"),
                (denoising_strength, "Denoising strength"),
                (mask_blur, "Mask blur"),
                *modules.scripts.scripts_img2img.infotext_fields
            ]
            parameters_copypaste.add_paste_fields("img2img", init_img, img2img_paste_fields)
            parameters_copypaste.add_paste_fields("inpaint", init_img_with_mask, img2img_paste_fields)

    modules.scripts.scripts_current = None

    with gr.Blocks(analytics_enabled=False) as extras_interface:
        with gr.Row().style(equal_height=False):
            with gr.Column(variant='panel'):
                with gr.Tabs(elem_id="mode_extras"):
                    with gr.TabItem('Single Image', elem_id="extras_single_tab"):
                        extras_image = gr.Image(label="Source", source="upload", interactive=True, type="pil", elem_id="extras_image")

                    with gr.TabItem('Batch Process', elem_id="extras_batch_process_tab"):
                        image_batch = gr.File(label="Batch Process", file_count="multiple", interactive=True, type="file", elem_id="extras_image_batch")

                    with gr.TabItem('Batch from Directory', elem_id="extras_batch_directory_tab"):
                        extras_batch_input_dir = gr.Textbox(label="Input directory", **shared.hide_dirs, placeholder="A directory on the same machine where the server is running.", elem_id="extras_batch_input_dir")
                        extras_batch_output_dir = gr.Textbox(label="Output directory", **shared.hide_dirs, placeholder="Leave blank to save images to the default path.", elem_id="extras_batch_output_dir")
                        show_extras_results = gr.Checkbox(label='Show result images', value=True, elem_id="extras_show_extras_results")

                submit = gr.Button('Generate', elem_id="extras_generate", variant='primary')

                with gr.Tabs(elem_id="extras_resize_mode"):
                    with gr.TabItem('Scale by', elem_id="extras_scale_by_tab"):
                        upscaling_resize = gr.Slider(minimum=1.0, maximum=8.0, step=0.05, label="Resize", value=4, elem_id="extras_upscaling_resize")
                    with gr.TabItem('Scale to', elem_id="extras_scale_to_tab"):
                        with gr.Group():
                            with gr.Row():
                                upscaling_resize_w = gr.Number(label="Width", value=512, precision=0, elem_id="extras_upscaling_resize_w")
                                upscaling_resize_h = gr.Number(label="Height", value=512, precision=0, elem_id="extras_upscaling_resize_h")
                            upscaling_crop = gr.Checkbox(label='Crop to fit', value=True, elem_id="extras_upscaling_crop")

                with gr.Group():
                    extras_upscaler_1 = gr.Radio(label='Upscaler 1', elem_id="extras_upscaler_1", choices=[x.name for x in shared.sd_upscalers], value=shared.sd_upscalers[0].name, type="index")

                with gr.Group():
                    extras_upscaler_2 = gr.Radio(label='Upscaler 2', elem_id="extras_upscaler_2", choices=[x.name for x in shared.sd_upscalers], value=shared.sd_upscalers[0].name, type="index")
                    extras_upscaler_2_visibility = gr.Slider(minimum=0.0, maximum=1.0, step=0.001, label="Upscaler 2 visibility", value=1, elem_id="extras_upscaler_2_visibility")

                with gr.Group():
                    gfpgan_visibility = gr.Slider(minimum=0.0, maximum=1.0, step=0.001, label="GFPGAN visibility", value=0, interactive=modules.gfpgan_model.have_gfpgan, elem_id="extras_gfpgan_visibility")

                with gr.Group():
                    codeformer_visibility = gr.Slider(minimum=0.0, maximum=1.0, step=0.001, label="CodeFormer visibility", value=0, interactive=modules.codeformer_model.have_codeformer, elem_id="extras_codeformer_visibility")
                    codeformer_weight = gr.Slider(minimum=0.0, maximum=1.0, step=0.001, label="CodeFormer weight (0 = maximum effect, 1 = minimum effect)", value=0, interactive=modules.codeformer_model.have_codeformer, elem_id="extras_codeformer_weight")

                with gr.Group():
                    upscale_before_face_fix = gr.Checkbox(label='Upscale Before Restoring Faces', value=False, elem_id="extras_upscale_before_face_fix")

            result_images, html_info_x, html_info, html_log = create_output_panel("extras", opts.outdir_extras_samples)

        submit.click(
            fn=wrap_gradio_gpu_call(modules.extras.run_extras, extra_outputs=[None, '']),
            _js="get_extras_tab_index",
            inputs=[
                dummy_component,
                dummy_component,
                extras_image,
                image_batch,
                extras_batch_input_dir,
                extras_batch_output_dir,
                show_extras_results,
                gfpgan_visibility,
                codeformer_visibility,
                codeformer_weight,
                upscaling_resize,
                upscaling_resize_w,
                upscaling_resize_h,
                upscaling_crop,
                extras_upscaler_1,
                extras_upscaler_2,
                extras_upscaler_2_visibility,
                upscale_before_face_fix,
            ],
            outputs=[
                result_images,
                html_info_x,
                html_info,
            ]
        )
        parameters_copypaste.add_paste_fields("extras", extras_image, None)

        extras_image.change(
            fn=modules.extras.clear_cache,
            inputs=[], outputs=[]
        )

    with gr.Blocks(analytics_enabled=False) as pnginfo_interface:
        with gr.Row().style(equal_height=False):
            with gr.Column(variant='panel'):
                image = gr.Image(elem_id="pnginfo_image", label="Source", source="upload", interactive=True, type="pil")

            with gr.Column(variant='panel'):
                html = gr.HTML()
                generation_info = gr.Textbox(visible=False, elem_id="pnginfo_generation_info")
                html2 = gr.HTML()
                with gr.Row():
                    buttons = parameters_copypaste.create_buttons(["txt2img", "img2img", "inpaint", "extras"])
                parameters_copypaste.bind_buttons(buttons, image, generation_info)

        image.change(
            fn=wrap_gradio_call(modules.extras.run_pnginfo),
            inputs=[image],
            outputs=[html, generation_info, html2],
        )

    with gr.Blocks(analytics_enabled=False) as modelmerger_interface:
        with gr.Row().style(equal_height=False):
            with gr.Column(variant='panel'):
                gr.HTML(value="<p>A merger of the two checkpoints will be generated in your <b>checkpoint</b> directory.</p>")

                with gr.Row():
                    primary_model_name = gr.Dropdown(modules.sd_models.checkpoint_tiles(), elem_id="modelmerger_primary_model_name", label="Primary model (A)")
                    create_refresh_button(primary_model_name, modules.sd_models.list_models, lambda: {"choices": modules.sd_models.checkpoint_tiles()}, "refresh_checkpoint_A")

                    secondary_model_name = gr.Dropdown(modules.sd_models.checkpoint_tiles(), elem_id="modelmerger_secondary_model_name", label="Secondary model (B)")
                    create_refresh_button(secondary_model_name, modules.sd_models.list_models, lambda: {"choices": modules.sd_models.checkpoint_tiles()}, "refresh_checkpoint_B")

                    tertiary_model_name = gr.Dropdown(modules.sd_models.checkpoint_tiles(), elem_id="modelmerger_tertiary_model_name", label="Tertiary model (C)")
                    create_refresh_button(tertiary_model_name, modules.sd_models.list_models, lambda: {"choices": modules.sd_models.checkpoint_tiles()}, "refresh_checkpoint_C")

                custom_name = gr.Textbox(label="Custom Name (Optional)", elem_id="modelmerger_custom_name")
                interp_amount = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, label='Multiplier (M) - set to 0 to get model A', value=0.3, elem_id="modelmerger_interp_amount")
                interp_method = gr.Radio(choices=["Weighted sum", "Add difference"], value="Weighted sum", label="Interpolation Method", elem_id="modelmerger_interp_method")

                with gr.Row():
                    checkpoint_format = gr.Radio(choices=["ckpt", "safetensors"], value="ckpt", label="Checkpoint format", elem_id="modelmerger_checkpoint_format")
                    save_as_half = gr.Checkbox(value=False, label="Save as float16", elem_id="modelmerger_save_as_half")

                modelmerger_merge = gr.Button(elem_id="modelmerger_merge", label="Merge", variant='primary')

            with gr.Column(variant='panel'):
                submit_result = gr.Textbox(elem_id="modelmerger_result", show_label=False)

    with gr.Blocks(analytics_enabled=False) as train_interface:
        with gr.Row().style(equal_height=False):
            gr.HTML(value="<p style='margin-bottom: 0.7em'>See <b><a href=\"https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Textual-Inversion\">wiki</a></b> for detailed explanation.</p>")

        with gr.Row().style(equal_height=False):
            with gr.Tabs(elem_id="train_tabs"):

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

                    with gr.Row():
                        with gr.Column(scale=3):
                            gr.HTML(value="")

                        with gr.Column():
                            with gr.Row():
                                interrupt_preprocessing = gr.Button("Interrupt", elem_id="train_interrupt_preprocessing")
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

                with gr.Tab(label="Train"):
                    gr.HTML(value="<p style='margin-bottom: 0.7em'>Train an embedding or Hypernetwork; you must specify a directory with a set of 1:1 ratio images <a href=\"https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Textual-Inversion\" style=\"font-weight:bold;\">[wiki]</a></p>")
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
                    log_directory = gr.Textbox(label='Log directory', placeholder="Path to directory where to write outputs", value="textual_inversion", elem_id="train_log_directory")
                    template_file = gr.Textbox(label='Prompt template file', value=os.path.join(script_path, "textual_inversion_templates", "style_filewords.txt"), elem_id="train_template_file")
                    training_width = gr.Slider(minimum=64, maximum=2048, step=8, label="Width", value=512, elem_id="train_training_width")
                    training_height = gr.Slider(minimum=64, maximum=2048, step=8, label="Height", value=512, elem_id="train_training_height")
                    steps = gr.Number(label='Max steps', value=100000, precision=0, elem_id="train_steps")

                    with FormRow():
                        create_image_every = gr.Number(label='Save an image to log directory every N steps, 0 to disable', value=500, precision=0, elem_id="train_create_image_every")
                        save_embedding_every = gr.Number(label='Save a copy of embedding to log directory every N steps, 0 to disable', value=500, precision=0, elem_id="train_save_embedding_every")

                    save_image_with_stored_embedding = gr.Checkbox(label='Save images with embedding in PNG chunks', value=True, elem_id="train_save_image_with_stored_embedding")
                    preview_from_txt2img = gr.Checkbox(label='Read parameters (prompt, etc...) from txt2img tab when making previews', value=False, elem_id="train_preview_from_txt2img")

                    shuffle_tags = gr.Checkbox(label="Shuffle tags by ',' when creating prompts.", value=False, elem_id="train_shuffle_tags")
                    tag_drop_out = gr.Slider(minimum=0, maximum=1, step=0.1, label="Drop out tags when creating prompts.", value=0, elem_id="train_tag_drop_out")

                    latent_sampling_method = gr.Radio(label='Choose latent sampling method', value="once", choices=['once', 'deterministic', 'random'], elem_id="train_latent_sampling_method")

                    with gr.Row():
                        train_embedding = gr.Button(value="Train Embedding", variant='primary', elem_id="train_train_embedding")
                        interrupt_training = gr.Button(value="Interrupt", elem_id="train_interrupt_training")
                        train_hypernetwork = gr.Button(value="Train Hypernetwork", variant='primary', elem_id="train_train_hypernetwork")

                params = script_callbacks.UiTrainTabParams(txt2img_preview_params)

                script_callbacks.ui_train_tabs_callback(params)

            with gr.Column():
                progressbar = gr.HTML(elem_id="ti_progressbar")
                ti_output = gr.Text(elem_id="ti_output", value="", show_label=False)

                ti_gallery = gr.Gallery(label='Output', show_label=False, elem_id='ti_gallery').style(grid=4)
                ti_preview = gr.Image(elem_id='ti_preview', visible=False)
                ti_progress = gr.HTML(elem_id="ti_progress", value="")
                ti_outcome = gr.HTML(elem_id="ti_error", value="")
                setup_progressbar(progressbar, ti_preview, 'ti', textinfo=ti_progress)

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
                new_hypernetwork_use_dropout
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
                train_embedding_name,
                embedding_learn_rate,
                batch_size,
                gradient_step,
                dataset_directory,
                log_directory,
                training_width,
                training_height,
                steps,
                clip_grad_mode,
                clip_grad_value,
                shuffle_tags,
                tag_drop_out,
                latent_sampling_method,
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
                train_hypernetwork_name,
                hypernetwork_learn_rate,
                batch_size,
                gradient_step,
                dataset_directory,
                log_directory,
                training_width,
                training_height,
                steps,
                clip_grad_mode,
                clip_grad_value,
                shuffle_tags,
                tag_drop_out,
                latent_sampling_method,
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
            raise Exception(f'bad options item type: {str(t)} for key {key}')

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

        return gr.update(value=value), opts.dumpjson()

    with gr.Blocks(analytics_enabled=False) as settings_interface:
        with gr.Row():
            with gr.Column(scale=6):
                settings_submit = gr.Button(value="Apply settings", variant='primary', elem_id="settings_submit")
            with gr.Column():
                restart_gradio = gr.Button(value='Reload UI', variant='primary', elem_id="settings_restart_gradio")

        result = gr.HTML(elem_id="settings_result")

        quicksettings_names = [x.strip() for x in opts.quicksettings.split(",")]
        quicksettings_names = {x: i for i, x in enumerate(quicksettings_names) if x != 'quicksettings'}

        quicksettings_list = []

        previous_section = None
        current_tab = None
        with gr.Tabs(elem_id="settings"):
            for i, (k, item) in enumerate(opts.data_labels.items()):
                section_must_be_skipped = item.section[0] is None

                if previous_section != item.section and not section_must_be_skipped:
                    elem_id, text = item.section

                    if current_tab is not None:
                        current_tab.__exit__()

                    current_tab = gr.TabItem(elem_id="settings_{}".format(elem_id), label=text)
                    current_tab.__enter__()

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
                current_tab.__exit__()

            with gr.TabItem("Actions"):
                request_notifications = gr.Button(value='Request browser notifications', elem_id="request_notifications")
                download_localization = gr.Button(value='Download localization template', elem_id="download_localization")
                reload_script_bodies = gr.Button(value='Reload custom script bodies (No ui updates, No restart)', variant='secondary', elem_id="settings_reload_script_bodies")

            if os.path.exists("html/licenses.html"):
                with open("html/licenses.html", encoding="utf8") as file:
                    with gr.TabItem("Licenses"):
                        gr.HTML(file.read(), elem_id="licenses")

            gr.Button(value="Show all pages", elem_id="settings_show_all_pages")

        request_notifications.click(
            fn=lambda: None,
            inputs=[],
            outputs=[],
            _js='function(){}'
        )

        download_localization.click(
            fn=lambda: None,
            inputs=[],
            outputs=[],
            _js='download_localization'
        )

        def reload_scripts():
            modules.scripts.reload_script_body_only()
            reload_javascript()  # need to refresh the html page

        reload_script_bodies.click(
            fn=reload_scripts,
            inputs=[],
            outputs=[]
        )

        def request_restart():
            shared.state.interrupt()
            shared.state.need_restart = True

        restart_gradio.click(
            fn=request_restart,
            _js='restart_reload',
            inputs=[],
            outputs=[],
        )

    interfaces = [
        (txt2img_interface, "txt2img", "txt2img"),
        (img2img_interface, "img2img", "img2img"),
        (extras_interface, "Extras", "extras"),
        (pnginfo_interface, "PNG Info", "pnginfo"),
        (modelmerger_interface, "Checkpoint Merger", "modelmerger"),
        (train_interface, "Train", "ti"),
    ]

    css = ""

    for cssfile in modules.scripts.list_files_with_name("style.css"):
        if not os.path.isfile(cssfile):
            continue

        with open(cssfile, "r", encoding="utf8") as file:
            css += file.read() + "\n"

    if os.path.exists(os.path.join(script_path, "user.css")):
        with open(os.path.join(script_path, "user.css"), "r", encoding="utf8") as file:
            css += file.read() + "\n"

    if not cmd_opts.no_progressbar_hiding:
        css += css_hide_progressbar

    interfaces += script_callbacks.ui_tabs_callback()
    interfaces += [(settings_interface, "Settings", "settings")]

    extensions_interface = ui_extensions.create_ui()
    interfaces += [(extensions_interface, "Extensions", "extensions")]

    with gr.Blocks(css=css, analytics_enabled=False, title="Stable Diffusion") as demo:
        with gr.Row(elem_id="quicksettings"):
            for i, k, item in sorted(quicksettings_list, key=lambda x: quicksettings_names.get(x[1], x[0])):
                component = create_setting_component(k, is_quicksettings=True)
                component_dict[k] = component

        parameters_copypaste.integrate_settings_paste_fields(component_dict)
        parameters_copypaste.run_bind()

        with gr.Tabs(elem_id="tabs") as tabs:
            for interface, label, ifid in interfaces:
                with gr.TabItem(label, id=ifid, elem_id='tab_' + ifid):
                    interface.render()

        if os.path.exists(os.path.join(script_path, "notification.mp3")):
            audio_notification = gr.Audio(interactive=False, value=os.path.join(script_path, "notification.mp3"), elem_id="audio_notification", visible=False)

        if os.path.exists("html/footer.html"):
            with open("html/footer.html", encoding="utf8") as file:
                footer = file.read()
                footer = footer.format(versions=versions_html())
                gr.HTML(footer, elem_id="footer")

        text_settings = gr.Textbox(elem_id="settings_json", value=lambda: opts.dumpjson(), visible=False)
        settings_submit.click(
            fn=wrap_gradio_call(run_settings, extra_outputs=[gr.update()]),
            inputs=components,
            outputs=[text_settings, result],
        )

        for i, k, item in quicksettings_list:
            component = component_dict[k]

            component.change(
                fn=lambda value, k=k: run_settings_single(value, key=k),
                inputs=[component],
                outputs=[component, text_settings],
            )

        component_keys = [k for k in opts.data_labels.keys() if k in component_dict]

        def get_settings_values():
            return [getattr(opts, key) for key in component_keys]

        demo.load(
            fn=get_settings_values,
            inputs=[],
            outputs=[component_dict[k] for k in component_keys],
        )

        def modelmerger(*args):
            try:
                results = modules.extras.run_modelmerger(*args)
            except Exception as e:
                print("Error loading/saving model file:", file=sys.stderr)
                print(traceback.format_exc(), file=sys.stderr)
                modules.sd_models.list_models()  # to remove the potentially missing models from the list
                return [f"Error merging checkpoints: {e}"] + [gr.Dropdown.update(choices=modules.sd_models.checkpoint_tiles()) for _ in range(4)]
            return results

        modelmerger_merge.click(
            fn=modelmerger,
            inputs=[
                primary_model_name,
                secondary_model_name,
                tertiary_model_name,
                interp_method,
                interp_amount,
                save_as_half,
                custom_name,
                checkpoint_format,
            ],
            outputs=[
                submit_result,
                primary_model_name,
                secondary_model_name,
                tertiary_model_name,
                component_dict['sd_model_checkpoint'],
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
    except Exception:
        error_loading = True
        print("Error loading settings:", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)

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
                print(f'Warning: Bad ui setting value: {key}: {saved_value}; Default value "{getattr(obj, field)}" will be used instead.')
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
            apply_field(x, 'value', lambda val: val in x.choices, getattr(x, 'init_field', None))

    visit(txt2img_interface, loadsave, "txt2img")
    visit(img2img_interface, loadsave, "img2img")
    visit(extras_interface, loadsave, "extras")
    visit(modelmerger_interface, loadsave, "modelmerger")
    visit(train_interface, loadsave, "train")

    if not error_loading and (not os.path.exists(ui_config_file) or settings_count != len(ui_settings)):
        with open(ui_config_file, "w", encoding="utf8") as file:
            json.dump(ui_settings, file, indent=4)

    return demo


def reload_javascript():
    with open(os.path.join(script_path, "script.js"), "r", encoding="utf8") as jsfile:
        javascript = f'<script>{jsfile.read()}</script>'

    scripts_list = modules.scripts.list_scripts("javascript", ".js")

    for basedir, filename, path in scripts_list:
        with open(path, "r", encoding="utf8") as jsfile:
            javascript += f"\n<!-- {filename} --><script>{jsfile.read()}</script>"

    if cmd_opts.theme is not None:
        javascript += f"\n<script>set_theme('{cmd_opts.theme}');</script>\n"

    javascript += f"\n<script>{localization.localization_js(shared.opts.localization)}</script>"

    def template_response(*args, **kwargs):
        res = shared.GradioTemplateResponseOriginal(*args, **kwargs)
        res.body = res.body.replace(
            b'</head>', f'{javascript}</head>'.encode("utf8"))
        res.init_headers()
        return res

    gradio.routes.templates.TemplateResponse = template_response


if not hasattr(shared, 'GradioTemplateResponseOriginal'):
    shared.GradioTemplateResponseOriginal = gradio.routes.templates.TemplateResponse


def versions_html():
    import torch
    import launch

    python_version = ".".join([str(x) for x in sys.version_info[0:3]])
    commit = launch.commit_hash()
    short_commit = commit[0:8]

    if shared.xformers_available:
        import xformers
        xformers_version = xformers.__version__
    else:
        xformers_version = "N/A"

    return f"""
python: <span title="{sys.version}">{python_version}</span>
 • 
torch: {torch.__version__}
 • 
xformers: {xformers_version}
 • 
gradio: {gr.__version__}
 • 
commit: <a href="https://github.com/AUTOMATIC1111/stable-diffusion-webui/commit/{commit}">{short_commit}</a>
"""
