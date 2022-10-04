import base64
import html
import io
import json
import math
import mimetypes
import os
import random
import sys
import time
import traceback
import platform
import subprocess as sp
from functools import reduce

import numpy as np
import torch
from PIL import Image, PngImagePlugin
import piexif

import gradio as gr
import gradio.utils
import gradio.routes

from modules import sd_hijack
from modules.paths import script_path
from modules.shared import opts, cmd_opts
import modules.shared as shared
from modules.sd_samplers import samplers, samplers_for_img2img
from modules.sd_hijack import model_hijack
import modules.ldsr_model
import modules.scripts
import modules.gfpgan_model
import modules.codeformer_model
import modules.styles
import modules.generation_parameters_copypaste
from modules.prompt_parser import get_learned_conditioning_prompt_schedules
from modules.images import apply_filename_pattern, get_next_sequence_number
import modules.textual_inversion.ui

# this is a fix for Windows users. Without it, javascript files will be served with text/html content-type and the bowser will not show any UI
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

css_hide_progressbar = """
.wrap .m-12 svg { display:none!important; }
.wrap .m-12::before { content:"Loading..." }
.progress-bar { display:none!important; }
.meta-text { display:none!important; }
"""

# Using constants for these since the variation selector isn't visible.
# Important that they exactly match script.js for tooltip to work.
random_symbol = '\U0001f3b2\ufe0f'  # 🎲️
reuse_symbol = '\u267b\ufe0f'  # ♻️
art_symbol = '\U0001f3a8'  # 🎨
paste_symbol = '\u2199\ufe0f'  # ↙
folder_symbol = '\U0001f4c2'  # 📂

def plaintext_to_html(text):
    text = "<p>" + "<br>\n".join([f"{html.escape(x)}" for x in text.split('\n')]) + "</p>"
    return text


def image_from_url_text(filedata):
    if type(filedata) == list:
        if len(filedata) == 0:
            return None

        filedata = filedata[0]

    if filedata.startswith("data:image/png;base64,"):
        filedata = filedata[len("data:image/png;base64,"):]

    filedata = base64.decodebytes(filedata.encode('utf-8'))
    image = Image.open(io.BytesIO(filedata))
    return image


def send_gradio_gallery_to_image(x):
    if len(x) == 0:
        return None

    return image_from_url_text(x[0])


def save_files(js_data, images, index):
    import csv    
    filenames = []

    #quick dictionary to class object conversion. Its neccesary due apply_filename_pattern requiring it
    class MyObject:
        def __init__(self, d=None):
            if d is not None:
                for key, value in d.items():
                    setattr(self, key, value)

    data = json.loads(js_data)

    p = MyObject(data)
    path = opts.outdir_save
    save_to_dirs = opts.use_save_to_dirs_for_ui

    if save_to_dirs:
        dirname = apply_filename_pattern(opts.directories_filename_pattern or "[prompt_words]", p, p.seed, p.prompt)
        path = os.path.join(opts.outdir_save, dirname)

    os.makedirs(path, exist_ok=True)
  

    if index > -1 and opts.save_selected_only and (index >= data["index_of_first_image"]):  # ensures we are looking at a specific non-grid picture, and we have save_selected_only

        images = [images[index]]
        infotexts = [data["infotexts"][index]]
    else:
        infotexts = data["infotexts"]

    with open(os.path.join(opts.outdir_save, "log.csv"), "a", encoding="utf8", newline='') as file:
        at_start = file.tell() == 0
        writer = csv.writer(file)
        if at_start:
            writer.writerow(["prompt", "seed", "width", "height", "sampler", "cfgs", "steps", "filename", "negative_prompt"])

        file_decoration = opts.samples_filename_pattern or "[seed]-[prompt_spaces]"
        if file_decoration != "":
            file_decoration = "-" + file_decoration.lower()
        file_decoration = apply_filename_pattern(file_decoration, p, p.seed, p.prompt)
        truncated = (file_decoration[:240] + '..') if len(file_decoration) > 240 else file_decoration
        filename_base = truncated
        extension = opts.samples_format.lower()

        basecount = get_next_sequence_number(path, "")
        for i, filedata in enumerate(images):
            file_number = f"{basecount+i:05}"
            filename = file_number + filename_base + f".{extension}"
            filepath = os.path.join(path, filename)


            if filedata.startswith("data:image/png;base64,"):
                filedata = filedata[len("data:image/png;base64,"):]

            image = Image.open(io.BytesIO(base64.decodebytes(filedata.encode('utf-8'))))
            if opts.enable_pnginfo and extension == 'png':
                pnginfo = PngImagePlugin.PngInfo()
                pnginfo.add_text('parameters', infotexts[i])
                image.save(filepath, pnginfo=pnginfo)
            else:
                image.save(filepath, quality=opts.jpeg_quality)

            if opts.enable_pnginfo and extension in ("jpg", "jpeg", "webp"):
                piexif.insert(piexif.dump({"Exif": {
                    piexif.ExifIFD.UserComment: piexif.helper.UserComment.dump(infotexts[i], encoding="unicode")
                }}), filepath)

            filenames.append(filename)

        writer.writerow([data["prompt"], data["seed"], data["width"], data["height"], data["sampler"], data["cfg_scale"], data["steps"], filenames[0], data["negative_prompt"]])

    return '', '', plaintext_to_html(f"Saved: {filenames[0]}")


def wrap_gradio_call(func, extra_outputs=None):
    def f(*args, extra_outputs_array=extra_outputs, **kwargs):
        run_memmon = opts.memmon_poll_rate > 0 and not shared.mem_mon.disabled
        if run_memmon:
            shared.mem_mon.monitor()
        t = time.perf_counter()

        try:
            res = list(func(*args, **kwargs))
        except Exception as e:
            print("Error completing request", file=sys.stderr)
            print("Arguments:", args, kwargs, file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)

            shared.state.job = ""
            shared.state.job_count = 0

            if extra_outputs_array is None:
                extra_outputs_array = [None, '']

            res = extra_outputs_array + [f"<div class='error'>{plaintext_to_html(type(e).__name__+': '+str(e))}</div>"]

        elapsed = time.perf_counter() - t

        if run_memmon:
            mem_stats = {k: -(v//-(1024*1024)) for k, v in shared.mem_mon.stop().items()}
            active_peak = mem_stats['active_peak']
            reserved_peak = mem_stats['reserved_peak']
            sys_peak = mem_stats['system_peak']
            sys_total = mem_stats['total']
            sys_pct = round(sys_peak/max(sys_total, 1) * 100, 2)

            vram_html = f"<p class='vram'>Torch active/reserved: {active_peak}/{reserved_peak} MiB, <wbr>Sys VRAM: {sys_peak}/{sys_total} MiB ({sys_pct}%)</p>"
        else:
            vram_html = ''

        # last item is always HTML
        res[-1] += f"<div class='performance'><p class='time'>Time taken: <wbr>{elapsed:.2f}s</p>{vram_html}</div>"

        shared.state.interrupted = False
        shared.state.job_count = 0

        return tuple(res)

    return f


def check_progress_call(id_part):
    if shared.state.job_count == 0:
        return "", gr_show(False), gr_show(False), gr_show(False)

    progress = 0

    if shared.state.job_count > 0:
        progress += shared.state.job_no / shared.state.job_count
    if shared.state.sampling_steps > 0:
        progress += 1 / shared.state.job_count * shared.state.sampling_step / shared.state.sampling_steps

    progress = min(progress, 1)

    progressbar = ""
    if opts.show_progressbar:
        progressbar = f"""<div class='progressDiv'><div class='progress' style="width:{progress * 100}%">{str(int(progress*100))+"%" if progress > 0.01 else ""}</div></div>"""

    image = gr_show(False)
    preview_visibility = gr_show(False)

    if opts.show_progress_every_n_steps > 0:
        if shared.parallel_processing_allowed:

            if shared.state.sampling_step - shared.state.current_image_sampling_step >= opts.show_progress_every_n_steps and shared.state.current_latent is not None:
                shared.state.current_image = modules.sd_samplers.sample_to_image(shared.state.current_latent)
                shared.state.current_image_sampling_step = shared.state.sampling_step

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

    return check_progress_call(id_part)


def roll_artist(prompt):
    allowed_cats = set([x for x in shared.artist_db.categories() if len(opts.random_artist_categories)==0 or x in opts.random_artist_categories])
    artist = random.choice([x for x in shared.artist_db.artists if x.category in allowed_cats])

    return prompt + ", " + artist.name if prompt != '' else artist.name


def visit(x, func, path=""):
    if hasattr(x, 'children'):
        for c in x.children:
            visit(c, func, path)
    elif x.label is not None:
        func(path + "/" + str(x.label), x)


def add_style(name: str, prompt: str, negative_prompt: str):
    if name is None:
        return [gr_show(), gr_show()]

    style = modules.styles.PromptStyle(name, prompt, negative_prompt)
    shared.prompt_styles.styles[style.name] = style
    # Save all loaded prompt styles: this allows us to update the storage format in the future more easily, because we
    # reserialize all styles every time we save them
    shared.prompt_styles.save_styles(shared.styles_filename)

    return [gr.Dropdown.update(visible=True, choices=list(shared.prompt_styles.styles)) for _ in range(4)]


def apply_styles(prompt, prompt_neg, style1_name, style2_name):
    prompt = shared.prompt_styles.apply_styles_to_prompt(prompt, [style1_name, style2_name])
    prompt_neg = shared.prompt_styles.apply_negative_styles_to_prompt(prompt_neg, [style1_name, style2_name])

    return [gr.Textbox.update(value=prompt), gr.Textbox.update(value=prompt_neg), gr.Dropdown.update(value="None"), gr.Dropdown.update(value="None")]


def interrogate(image):
    prompt = shared.interrogator.interrogate(image)

    return gr_show(True) if prompt is None else prompt


def create_seed_inputs():
    with gr.Row():
        with gr.Box():
            with gr.Row(elem_id='seed_row'):
                seed = (gr.Textbox if cmd_opts.use_textbox_seed else gr.Number)(label='Seed', value=-1)
                seed.style(container=False)
                random_seed = gr.Button(random_symbol, elem_id='random_seed')
                reuse_seed = gr.Button(reuse_symbol, elem_id='reuse_seed')

        with gr.Box(elem_id='subseed_show_box'):
            seed_checkbox = gr.Checkbox(label='Extra', elem_id='subseed_show', value=False)

    # Components to show/hide based on the 'Extra' checkbox
    seed_extras = []

    with gr.Row(visible=False) as seed_extra_row_1:
        seed_extras.append(seed_extra_row_1)
        with gr.Box():
            with gr.Row(elem_id='subseed_row'):
                subseed = gr.Number(label='Variation seed', value=-1)
                subseed.style(container=False)
                random_subseed = gr.Button(random_symbol, elem_id='random_subseed')
                reuse_subseed = gr.Button(reuse_symbol, elem_id='reuse_subseed')
        subseed_strength = gr.Slider(label='Variation strength', value=0.0, minimum=0, maximum=1, step=0.01)

    with gr.Row(visible=False) as seed_extra_row_2:
        seed_extras.append(seed_extra_row_2)
        seed_resize_from_w = gr.Slider(minimum=0, maximum=2048, step=64, label="Resize seed from width", value=0)
        seed_resize_from_h = gr.Slider(minimum=0, maximum=2048, step=64, label="Resize seed from height", value=0)

    random_seed.click(fn=lambda: -1, show_progress=False, inputs=[], outputs=[seed])
    random_subseed.click(fn=lambda: -1, show_progress=False, inputs=[], outputs=[subseed])

    def change_visibility(show):
        return {comp: gr_show(show) for comp in seed_extras}

    seed_checkbox.change(change_visibility, show_progress=False, inputs=[seed_checkbox], outputs=seed_extras)

    return seed, reuse_seed, subseed, reuse_subseed, subseed_strength, seed_resize_from_h, seed_resize_from_w, seed_checkbox


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
    prompt_schedules = get_learned_conditioning_prompt_schedules([text], steps)
    flat_prompts = reduce(lambda list1, list2: list1+list2, prompt_schedules)
    prompts = [prompt_text for step,prompt_text in flat_prompts]
    tokens, token_count, max_length = max([model_hijack.tokenize(prompt) for prompt in prompts], key=lambda args: args[1])
    style_class = ' class="red"' if (token_count > max_length) else ""
    return f"<span {style_class}>{token_count}/{max_length}</span>"

def create_toprow(is_img2img):
    id_part = "img2img" if is_img2img else "txt2img"

    with gr.Row(elem_id="toprow"):
        with gr.Column(scale=4):
            with gr.Row():
                with gr.Column(scale=80):
                    with gr.Row():
                        prompt = gr.Textbox(label="Prompt", elem_id=f"{id_part}_prompt", show_label=False, placeholder="Prompt", lines=2)

                with gr.Column(scale=1, elem_id="roll_col"):
                    roll = gr.Button(value=art_symbol, elem_id="roll", visible=len(shared.artist_db.artists) > 0)
                    paste = gr.Button(value=paste_symbol, elem_id="paste")
                    token_counter = gr.HTML(value="<span></span>", elem_id=f"{id_part}_token_counter")
                    token_button = gr.Button(visible=False, elem_id=f"{id_part}_token_button")

                with gr.Column(scale=10, elem_id="style_pos_col"):
                    prompt_style = gr.Dropdown(label="Style 1", elem_id=f"{id_part}_style_index", choices=[k for k, v in shared.prompt_styles.styles.items()], value=next(iter(shared.prompt_styles.styles.keys())), visible=len(shared.prompt_styles.styles) > 1)

            with gr.Row():
                with gr.Column(scale=8):
                    negative_prompt = gr.Textbox(label="Negative prompt", elem_id="negative_prompt", show_label=False, placeholder="Negative prompt", lines=2)

                with gr.Column(scale=1, elem_id="style_neg_col"):
                    prompt_style2 = gr.Dropdown(label="Style 2", elem_id=f"{id_part}_style2_index", choices=[k for k, v in shared.prompt_styles.styles.items()], value=next(iter(shared.prompt_styles.styles.keys())), visible=len(shared.prompt_styles.styles) > 1)

        with gr.Column(scale=1):
            with gr.Row():
                interrupt = gr.Button('Interrupt', elem_id=f"{id_part}_interrupt")
                submit = gr.Button('Generate', elem_id=f"{id_part}_generate", variant='primary')

                interrupt.click(
                    fn=lambda: shared.state.interrupt(),
                    inputs=[],
                    outputs=[],
                )

            with gr.Row():
                if is_img2img:
                    interrogate = gr.Button('Interrogate', elem_id="interrogate")
                else:
                    interrogate = None
                prompt_style_apply = gr.Button('Apply style', elem_id="style_apply")
                save_style = gr.Button('Create style', elem_id="style_create")

    return prompt, roll, prompt_style, negative_prompt, prompt_style2, submit, interrogate, prompt_style_apply, save_style, paste, token_counter, token_button


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


def create_ui(wrap_gradio_gpu_call):
    import modules.img2img
    import modules.txt2img

    with gr.Blocks(analytics_enabled=False) as txt2img_interface:
        txt2img_prompt, roll, txt2img_prompt_style, txt2img_negative_prompt, txt2img_prompt_style2, submit, _, txt2img_prompt_style_apply, txt2img_save_style, paste, token_counter, token_button = create_toprow(is_img2img=False)
        dummy_component = gr.Label(visible=False)

        with gr.Row(elem_id='txt2img_progress_row'):
            with gr.Column(scale=1):
                pass

            with gr.Column(scale=1):
                progressbar = gr.HTML(elem_id="txt2img_progressbar")
                txt2img_preview = gr.Image(elem_id='txt2img_preview', visible=False)
                setup_progressbar(progressbar, txt2img_preview, 'txt2img')

        with gr.Row().style(equal_height=False):
            with gr.Column(variant='panel'):
                steps = gr.Slider(minimum=1, maximum=150, step=1, label="Sampling Steps", value=20)
                sampler_index = gr.Radio(label='Sampling method', elem_id="txt2img_sampling", choices=[x.name for x in samplers], value=samplers[0].name, type="index")

                with gr.Group():
                    width = gr.Slider(minimum=64, maximum=2048, step=64, label="Width", value=512)
                    height = gr.Slider(minimum=64, maximum=2048, step=64, label="Height", value=512)

                with gr.Row():
                    restore_faces = gr.Checkbox(label='Restore faces', value=False, visible=len(shared.face_restorers) > 1)
                    tiling = gr.Checkbox(label='Tiling', value=False)
                    enable_hr = gr.Checkbox(label='Highres. fix', value=False)

                with gr.Row(visible=False) as hr_options:
                    scale_latent = gr.Checkbox(label='Scale latent', value=False)
                    denoising_strength = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label='Denoising strength', value=0.7)

                with gr.Row():
                    batch_count = gr.Slider(minimum=1, maximum=cmd_opts.max_batch_count, step=1, label='Batch count', value=1)
                    batch_size = gr.Slider(minimum=1, maximum=8, step=1, label='Batch size', value=1)

                cfg_scale = gr.Slider(minimum=1.0, maximum=30.0, step=0.5, label='CFG Scale', value=7.0)

                seed, reuse_seed, subseed, reuse_subseed, subseed_strength, seed_resize_from_h, seed_resize_from_w, seed_checkbox = create_seed_inputs()

                with gr.Group():
                    custom_inputs = modules.scripts.scripts_txt2img.setup_ui(is_img2img=False)

            with gr.Column(variant='panel'):

                with gr.Group():
                    txt2img_preview = gr.Image(elem_id='txt2img_preview', visible=False)
                    txt2img_gallery = gr.Gallery(label='Output', show_label=False, elem_id='txt2img_gallery').style(grid=4)

                with gr.Group():
                    with gr.Row():
                        save = gr.Button('Save')
                        send_to_img2img = gr.Button('Send to img2img')
                        send_to_inpaint = gr.Button('Send to inpaint')
                        send_to_extras = gr.Button('Send to extras')
                        button_id = "hidden_element" if shared.cmd_opts.hide_ui_dir_config else 'open_folder'
                        open_txt2img_folder = gr.Button(folder_symbol, elem_id=button_id)

                with gr.Group():
                    html_info = gr.HTML()
                    generation_info = gr.Textbox(visible=False)

            connect_reuse_seed(seed, reuse_seed, generation_info, dummy_component, is_subseed=False)
            connect_reuse_seed(subseed, reuse_subseed, generation_info, dummy_component, is_subseed=True)

            txt2img_args = dict(
                fn=wrap_gradio_gpu_call(modules.txt2img.txt2img),
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
                    scale_latent,
                    denoising_strength,
                ] + custom_inputs,
                outputs=[
                    txt2img_gallery,
                    generation_info,
                    html_info
                ],
                show_progress=False,
            )

            txt2img_prompt.submit(**txt2img_args)
            submit.click(**txt2img_args)

            enable_hr.change(
                fn=lambda x: gr_show(x),
                inputs=[enable_hr],
                outputs=[hr_options],
            )

            save.click(
                fn=wrap_gradio_call(save_files),
                _js="(x, y, z) => [x, y, selected_gallery_index()]",
                inputs=[
                    generation_info,
                    txt2img_gallery,
                    html_info,
                ],
                outputs=[
                    html_info,
                    html_info,
                    html_info,
                ]
            )

            roll.click(
                fn=roll_artist,
                _js="update_txt2img_tokens",
                inputs=[
                    txt2img_prompt,
                ],
                outputs=[
                    txt2img_prompt,
                ]
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
            ]
            modules.generation_parameters_copypaste.connect_paste(paste, txt2img_paste_fields, txt2img_prompt)
            token_button.click(fn=update_token_counter, inputs=[txt2img_prompt, steps], outputs=[token_counter])

    with gr.Blocks(analytics_enabled=False) as img2img_interface:
        img2img_prompt, roll, img2img_prompt_style, img2img_negative_prompt, img2img_prompt_style2, submit, img2img_interrogate, img2img_prompt_style_apply, img2img_save_style, paste, token_counter, token_button = create_toprow(is_img2img=True)

        with gr.Row(elem_id='img2img_progress_row'):
            with gr.Column(scale=1):
                pass

            with gr.Column(scale=1):
                progressbar = gr.HTML(elem_id="img2img_progressbar")
                img2img_preview = gr.Image(elem_id='img2img_preview', visible=False)
                setup_progressbar(progressbar, img2img_preview, 'img2img')

        with gr.Row().style(equal_height=False):
            with gr.Column(variant='panel'):

                with gr.Tabs(elem_id="mode_img2img") as tabs_img2img_mode:
                    with gr.TabItem('img2img', id='img2img'):
                        init_img = gr.Image(label="Image for img2img", show_label=False, source="upload", interactive=True, type="pil")

                    with gr.TabItem('Inpaint', id='inpaint'):
                        init_img_with_mask = gr.Image(label="Image for inpainting with mask",  show_label=False, elem_id="img2maskimg", source="upload", interactive=True, type="pil", tool="sketch", image_mode="RGBA")

                        init_img_inpaint = gr.Image(label="Image for img2img", show_label=False, source="upload", interactive=True, type="pil", visible=False, elem_id="img_inpaint_base")
                        init_mask_inpaint = gr.Image(label="Mask", source="upload", interactive=True, type="pil", visible=False, elem_id="img_inpaint_mask")

                        mask_blur = gr.Slider(label='Mask blur', minimum=0, maximum=64, step=1, value=4)

                        with gr.Row():
                            mask_mode = gr.Radio(label="Mask mode", show_label=False, choices=["Draw mask", "Upload mask"], type="index", value="Draw mask", elem_id="mask_mode")
                            inpainting_mask_invert = gr.Radio(label='Masking mode', show_label=False, choices=['Inpaint masked', 'Inpaint not masked'], value='Inpaint masked', type="index")

                        inpainting_fill = gr.Radio(label='Masked content', choices=['fill', 'original', 'latent noise', 'latent nothing'], value='original', type="index")

                        with gr.Row():
                            inpaint_full_res = gr.Checkbox(label='Inpaint at full resolution', value=False)
                            inpaint_full_res_padding = gr.Slider(label='Inpaint at full resolution padding, pixels', minimum=0, maximum=256, step=4, value=32)

                    with gr.TabItem('Batch img2img', id='batch'):
                        hidden = '<br>Disabled when launched with --hide-ui-dir-config.' if shared.cmd_opts.hide_ui_dir_config else ''
                        gr.HTML(f"<p class=\"text-gray-500\">Process images in a directory on the same machine where the server is running.<br>Use an empty output directory to save pictures normally instead of writing to the output directory.{hidden}</p>")
                        img2img_batch_input_dir = gr.Textbox(label="Input directory", **shared.hide_dirs)
                        img2img_batch_output_dir = gr.Textbox(label="Output directory", **shared.hide_dirs)

                with gr.Row():
                    resize_mode = gr.Radio(label="Resize mode", elem_id="resize_mode", show_label=False, choices=["Just resize", "Crop and resize", "Resize and fill"], type="index", value="Just resize")

                steps = gr.Slider(minimum=1, maximum=150, step=1, label="Sampling Steps", value=20)
                sampler_index = gr.Radio(label='Sampling method', choices=[x.name for x in samplers_for_img2img], value=samplers_for_img2img[0].name, type="index")

                with gr.Group():
                    width = gr.Slider(minimum=64, maximum=2048, step=64, label="Width", value=512)
                    height = gr.Slider(minimum=64, maximum=2048, step=64, label="Height", value=512)

                with gr.Row():
                    restore_faces = gr.Checkbox(label='Restore faces', value=False, visible=len(shared.face_restorers) > 1)
                    tiling = gr.Checkbox(label='Tiling', value=False)

                with gr.Row():
                    batch_count = gr.Slider(minimum=1, maximum=cmd_opts.max_batch_count, step=1, label='Batch count', value=1)
                    batch_size = gr.Slider(minimum=1, maximum=8, step=1, label='Batch size', value=1)

                with gr.Group():
                    cfg_scale = gr.Slider(minimum=1.0, maximum=30.0, step=0.5, label='CFG Scale', value=7.0)
                    denoising_strength = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label='Denoising strength', value=0.75)

                seed, reuse_seed, subseed, reuse_subseed, subseed_strength, seed_resize_from_h, seed_resize_from_w, seed_checkbox = create_seed_inputs()

                with gr.Group():
                    custom_inputs = modules.scripts.scripts_img2img.setup_ui(is_img2img=True)

            with gr.Column(variant='panel'):

                with gr.Group():
                    img2img_preview = gr.Image(elem_id='img2img_preview', visible=False)
                    img2img_gallery = gr.Gallery(label='Output', show_label=False, elem_id='img2img_gallery').style(grid=4)

                with gr.Group():
                    with gr.Row():
                        save = gr.Button('Save')
                        img2img_send_to_img2img = gr.Button('Send to img2img')
                        img2img_send_to_inpaint = gr.Button('Send to inpaint')
                        img2img_send_to_extras = gr.Button('Send to extras')
                        button_id = "hidden_element" if shared.cmd_opts.hide_ui_dir_config else 'open_folder'
                        open_img2img_folder = gr.Button(folder_symbol, elem_id=button_id)

                with gr.Group():
                    html_info = gr.HTML()
                    generation_info = gr.Textbox(visible=False)

            connect_reuse_seed(seed, reuse_seed, generation_info, dummy_component, is_subseed=False)
            connect_reuse_seed(subseed, reuse_subseed, generation_info, dummy_component, is_subseed=True)

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
                fn=wrap_gradio_gpu_call(modules.img2img.img2img),
                _js="submit_img2img",
                inputs=[
                    dummy_component,
                    img2img_prompt,
                    img2img_negative_prompt,
                    img2img_prompt_style,
                    img2img_prompt_style2,
                    init_img,
                    init_img_with_mask,
                    init_img_inpaint,
                    init_mask_inpaint,
                    mask_mode,
                    steps,
                    sampler_index,
                    mask_blur,
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
                    html_info
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

            save.click(
                fn=wrap_gradio_call(save_files),
                _js="(x, y, z) => [x, y, selected_gallery_index()]",
                inputs=[
                    generation_info,
                    img2img_gallery,
                    html_info
                ],
                outputs=[
                    html_info,
                    html_info,
                    html_info,
                ]
            )

            roll.click(
                fn=roll_artist,
                _js="update_img2img_tokens",
                inputs=[
                    img2img_prompt,
                ],
                outputs=[
                    img2img_prompt,
                ]
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
            ]
            modules.generation_parameters_copypaste.connect_paste(paste, img2img_paste_fields, img2img_prompt)
            token_button.click(fn=update_token_counter, inputs=[img2img_prompt, steps], outputs=[token_counter])

    with gr.Blocks(analytics_enabled=False) as extras_interface:
        with gr.Row().style(equal_height=False):
            with gr.Column(variant='panel'):
                with gr.Tabs(elem_id="mode_extras"):
                    with gr.TabItem('Single Image'):
                        extras_image = gr.Image(label="Source", source="upload", interactive=True, type="pil")

                    with gr.TabItem('Batch Process'):
                        image_batch = gr.File(label="Batch Process", file_count="multiple", interactive=True, type="file")

                upscaling_resize = gr.Slider(minimum=1.0, maximum=4.0, step=0.05, label="Resize", value=2)

                with gr.Group():
                    extras_upscaler_1 = gr.Radio(label='Upscaler 1', choices=[x.name for x in shared.sd_upscalers], value=shared.sd_upscalers[0].name, type="index")

                with gr.Group():
                    extras_upscaler_2 = gr.Radio(label='Upscaler 2', choices=[x.name for x in shared.sd_upscalers], value=shared.sd_upscalers[0].name, type="index")
                    extras_upscaler_2_visibility = gr.Slider(minimum=0.0, maximum=1.0, step=0.001, label="Upscaler 2 visibility", value=1)

                with gr.Group():
                    gfpgan_visibility = gr.Slider(minimum=0.0, maximum=1.0, step=0.001, label="GFPGAN visibility", value=0, interactive=modules.gfpgan_model.have_gfpgan)

                with gr.Group():
                    codeformer_visibility = gr.Slider(minimum=0.0, maximum=1.0, step=0.001, label="CodeFormer visibility", value=0, interactive=modules.codeformer_model.have_codeformer)
                    codeformer_weight = gr.Slider(minimum=0.0, maximum=1.0, step=0.001, label="CodeFormer weight (0 = maximum effect, 1 = minimum effect)", value=0, interactive=modules.codeformer_model.have_codeformer)

                submit = gr.Button('Generate', elem_id="extras_generate", variant='primary')

            with gr.Column(variant='panel'):
                result_images = gr.Gallery(label="Result", show_label=False)
                html_info_x = gr.HTML()
                html_info = gr.HTML()
                extras_send_to_img2img = gr.Button('Send to img2img')
                extras_send_to_inpaint = gr.Button('Send to inpaint')
                button_id = "hidden_element" if shared.cmd_opts.hide_ui_dir_config else ''
                open_extras_folder = gr.Button('Open output directory', elem_id=button_id)

        submit.click(
            fn=wrap_gradio_gpu_call(modules.extras.run_extras),
            _js="get_extras_tab_index",
            inputs=[
                dummy_component,
                extras_image,
                image_batch,
                gfpgan_visibility,
                codeformer_visibility,
                codeformer_weight,
                upscaling_resize,
                extras_upscaler_1,
                extras_upscaler_2,
                extras_upscaler_2_visibility,
            ],
            outputs=[
                result_images,
                html_info_x,
                html_info,
            ]
        )
     
        extras_send_to_img2img.click(
            fn=lambda x: image_from_url_text(x),
            _js="extract_image_from_gallery_img2img",
            inputs=[result_images],
            outputs=[init_img],
        )
        
        extras_send_to_inpaint.click(
            fn=lambda x: image_from_url_text(x),
            _js="extract_image_from_gallery_img2img",
            inputs=[result_images],
            outputs=[init_img_with_mask],
        )

    with gr.Blocks(analytics_enabled=False) as pnginfo_interface:
        with gr.Row().style(equal_height=False):
            with gr.Column(variant='panel'):
                image = gr.Image(elem_id="pnginfo_image", label="Source", source="upload", interactive=True, type="pil")

            with gr.Column(variant='panel'):
                html = gr.HTML()
                generation_info = gr.Textbox(visible=False)
                html2 = gr.HTML()

                with gr.Row():
                    pnginfo_send_to_txt2img = gr.Button('Send to txt2img')
                    pnginfo_send_to_img2img = gr.Button('Send to img2img')

        image.change(
            fn=wrap_gradio_call(modules.extras.run_pnginfo),
            inputs=[image],
            outputs=[html, generation_info, html2],
        )

    with gr.Blocks() as modelmerger_interface:
        with gr.Row().style(equal_height=False):
            with gr.Column(variant='panel'):
                gr.HTML(value="<p>A merger of the two checkpoints will be generated in your <b>checkpoint</b> directory.</p>")

                with gr.Row():
                    primary_model_name = gr.Dropdown(modules.sd_models.checkpoint_tiles(), elem_id="modelmerger_primary_model_name", label="Primary Model Name")
                    secondary_model_name = gr.Dropdown(modules.sd_models.checkpoint_tiles(), elem_id="modelmerger_secondary_model_name", label="Secondary Model Name")
                custom_name = gr.Textbox(label="Custom Name (Optional)")
                interp_amount = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, label='Interpolation Amount', value=0.3)
                interp_method = gr.Radio(choices=["Weighted Sum", "Sigmoid", "Inverse Sigmoid"], value="Weighted Sum", label="Interpolation Method")
                save_as_half = gr.Checkbox(value=False, label="Safe as float16")
                modelmerger_merge = gr.Button(elem_id="modelmerger_merge", label="Merge", variant='primary')

            with gr.Column(variant='panel'):
                submit_result = gr.Textbox(elem_id="modelmerger_result", show_label=False)

    sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings()

    with gr.Blocks() as textual_inversion_interface:
        with gr.Row().style(equal_height=False):
            with gr.Column():
                with gr.Group():
                    gr.HTML(value="<p style='margin-bottom: 0.7em'>See <b><a href=\"https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Textual-Inversion\">wiki</a></b> for detailed explanation.</p>")

                    gr.HTML(value="<p style='margin-bottom: 0.7em'>Create a new embedding</p>")

                    new_embedding_name = gr.Textbox(label="Name")
                    initialization_text = gr.Textbox(label="Initialization text", value="*")
                    nvpt = gr.Slider(label="Number of vectors per token", minimum=1, maximum=75, step=1, value=1)

                    with gr.Row():
                        with gr.Column(scale=3):
                            gr.HTML(value="")

                        with gr.Column():
                            create_embedding = gr.Button(value="Create", variant='primary')

                with gr.Group():
                    gr.HTML(value="<p style='margin-bottom: 0.7em'>Preprocess images</p>")

                    process_src = gr.Textbox(label='Source directory')
                    process_dst = gr.Textbox(label='Destination directory')

                    with gr.Row():
                        process_flip = gr.Checkbox(label='Flip')
                        process_split = gr.Checkbox(label='Split into two')
                        process_caption = gr.Checkbox(label='Add caption')

                    with gr.Row():
                        with gr.Column(scale=3):
                            gr.HTML(value="")

                        with gr.Column():
                            run_preprocess = gr.Button(value="Preprocess", variant='primary')

                with gr.Group():
                    gr.HTML(value="<p style='margin-bottom: 0.7em'>Train an embedding; must specify a directory with a set of 512x512 images</p>")
                    train_embedding_name = gr.Dropdown(label='Embedding', choices=sorted(sd_hijack.model_hijack.embedding_db.word_embeddings.keys()))
                    learn_rate = gr.Number(label='Learning rate', value=5.0e-03)
                    dataset_directory = gr.Textbox(label='Dataset directory', placeholder="Path to directory with input images")
                    log_directory = gr.Textbox(label='Log directory', placeholder="Path to directory where to write outputs", value="textual_inversion")
                    template_file = gr.Textbox(label='Prompt template file', value=os.path.join(script_path, "textual_inversion_templates", "style_filewords.txt"))
                    steps = gr.Number(label='Max steps', value=100000, precision=0)
                    create_image_every = gr.Number(label='Save an image to log directory every N steps, 0 to disable', value=500, precision=0)
                    save_embedding_every = gr.Number(label='Save a copy of embedding to log directory every N steps, 0 to disable', value=500, precision=0)

                    with gr.Row():
                        with gr.Column(scale=2):
                            gr.HTML(value="")

                        with gr.Column():
                            with gr.Row():
                                interrupt_training = gr.Button(value="Interrupt")
                                train_embedding = gr.Button(value="Train", variant='primary')

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
            ],
            outputs=[
                train_embedding_name,
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
                process_flip,
                process_split,
                process_caption,
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
                learn_rate,
                dataset_directory,
                log_directory,
                steps,
                create_image_every,
                save_embedding_every,
                template_file,
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

    def create_setting_component(key):
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

        return comp(label=info.label, value=fun, **(args or {}))

    components = []
    component_dict = {}

    def open_folder(f):
        if not shared.cmd_opts.hide_ui_dir_config:
            path = os.path.normpath(f)
            if platform.system() == "Windows":
                os.startfile(path)
            elif platform.system() == "Darwin":
                sp.Popen(["open", path])
            else:
                sp.Popen(["xdg-open", path])

    def run_settings(*args):
        changed = 0

        for key, value, comp in zip(opts.data_labels.keys(), args, components):
            if not opts.same_type(value, opts.data_labels[key].default):
                return f"Bad value for setting {key}: {value}; expecting {type(opts.data_labels[key].default).__name__}"

        for key, value, comp in zip(opts.data_labels.keys(), args, components):
            comp_args = opts.data_labels[key].component_args
            if comp_args and isinstance(comp_args, dict) and comp_args.get('visible') is False:
                continue

            oldval = opts.data.get(key, None)
            opts.data[key] = value

            if oldval != value:
                if opts.data_labels[key].onchange is not None:
                    opts.data_labels[key].onchange()

                changed += 1

        opts.save(shared.config_filename)

        return f'{changed} settings changed.', opts.dumpjson()

    with gr.Blocks(analytics_enabled=False) as settings_interface:
        settings_submit = gr.Button(value="Apply settings", variant='primary')
        result = gr.HTML()

        settings_cols = 3
        items_per_col = int(len(opts.data_labels) * 0.9 / settings_cols)

        cols_displayed = 0
        items_displayed = 0
        previous_section = None
        column = None
        with gr.Row(elem_id="settings").style(equal_height=False):
            for i, (k, item) in enumerate(opts.data_labels.items()):

                if previous_section != item.section:
                    if cols_displayed < settings_cols and (items_displayed >= items_per_col or previous_section is None):
                        if column is not None:
                            column.__exit__()

                        column = gr.Column(variant='panel')
                        column.__enter__()

                        items_displayed = 0
                        cols_displayed += 1

                    previous_section = item.section

                    gr.HTML(elem_id="settings_header_text_{}".format(item.section[0]), value='<h1 class="gr-button-lg">{}</h1>'.format(item.section[1]))

                component = create_setting_component(k)
                component_dict[k] = component
                components.append(component)
                items_displayed += 1

        request_notifications = gr.Button(value='Request browser notifications', elem_id="request_notifications")
        request_notifications.click(
            fn=lambda: None,
            inputs=[],
            outputs=[],
            _js='function(){}'
        )

        with gr.Row():
            reload_script_bodies = gr.Button(value='Reload custom script bodies (No ui updates, No restart)', variant='secondary')
            restart_gradio = gr.Button(value='Restart Gradio and Refresh components (Custom Scripts, ui.py, js and css only)', variant='primary')


        def reload_scripts():
            modules.scripts.reload_script_body_only()

        reload_script_bodies.click(
            fn=reload_scripts,
            inputs=[],
            outputs=[],
            _js='function(){}'
        )

        def request_restart():
            settings_interface.gradio_ref.do_restart = True

        restart_gradio.click(
            fn=request_restart,
            inputs=[],
            outputs=[],
            _js='function(){restart_reload()}'
        )
        
        if column is not None:
            column.__exit__()

    interfaces = [
        (txt2img_interface, "txt2img", "txt2img"),
        (img2img_interface, "img2img", "img2img"),
        (extras_interface, "Extras", "extras"),
        (pnginfo_interface, "PNG Info", "pnginfo"),
        (modelmerger_interface, "Checkpoint Merger", "modelmerger"),
        (textual_inversion_interface, "Textual inversion", "ti"),
        (settings_interface, "Settings", "settings"),
    ]

    with open(os.path.join(script_path, "style.css"), "r", encoding="utf8") as file:
        css = file.read()

    if os.path.exists(os.path.join(script_path, "user.css")):
        with open(os.path.join(script_path, "user.css"), "r", encoding="utf8") as file:
            usercss = file.read()
            css += usercss

    if not cmd_opts.no_progressbar_hiding:
        css += css_hide_progressbar

    with gr.Blocks(css=css, analytics_enabled=False, title="Stable Diffusion") as demo:
        
        settings_interface.gradio_ref = demo
        
        with gr.Tabs() as tabs:
            for interface, label, ifid in interfaces:
                with gr.TabItem(label, id=ifid):
                    interface.render()
        
        if os.path.exists(os.path.join(script_path, "notification.mp3")):
            audio_notification = gr.Audio(interactive=False, value=os.path.join(script_path, "notification.mp3"), elem_id="audio_notification", visible=False)

        text_settings = gr.Textbox(elem_id="settings_json", value=lambda: opts.dumpjson(), visible=False)
        settings_submit.click(
            fn=run_settings,
            inputs=components,
            outputs=[result, text_settings],
        )
        
        def modelmerger(*args):
            try:
                results = modules.extras.run_modelmerger(*args)
            except Exception as e:
                print("Error loading/saving model file:", file=sys.stderr)
                print(traceback.format_exc(), file=sys.stderr)
                modules.sd_models.list_models()  # to remove the potentially missing models from the list
                return ["Error loading/saving model file. It doesn't exist or the name contains illegal characters"] + [gr.Dropdown.update(choices=modules.sd_models.checkpoint_tiles()) for _ in range(3)]
            return results

        modelmerger_merge.click(
            fn=modelmerger,
            inputs=[
                primary_model_name,
                secondary_model_name,
                interp_method,
                interp_amount,
                save_as_half,
                custom_name,
            ],
            outputs=[
                submit_result,
                primary_model_name,
                secondary_model_name,
                component_dict['sd_model_checkpoint'],
            ]
        )
        paste_field_names = ['Prompt', 'Negative prompt', 'Steps', 'Face restoration', 'Seed', 'Size-1', 'Size-2']
        txt2img_fields = [field for field,name in txt2img_paste_fields if name in paste_field_names]
        img2img_fields = [field for field,name in img2img_paste_fields if name in paste_field_names]
        send_to_img2img.click(
            fn=lambda img, *args: (image_from_url_text(img),*args),
            _js="(gallery, ...args) => [extract_image_from_gallery_img2img(gallery), ...args]",
            inputs=[txt2img_gallery] + txt2img_fields,
            outputs=[init_img] + img2img_fields,
        )

        send_to_inpaint.click(
            fn=lambda x, *args: (image_from_url_text(x), *args),
            _js="(gallery, ...args) => [extract_image_from_gallery_inpaint(gallery), ...args]",
            inputs=[txt2img_gallery] + txt2img_fields,
            outputs=[init_img_with_mask] + img2img_fields,
        )

        img2img_send_to_img2img.click(
            fn=lambda x: image_from_url_text(x),
            _js="extract_image_from_gallery_img2img",
            inputs=[img2img_gallery],
            outputs=[init_img],
        )

        img2img_send_to_inpaint.click(
            fn=lambda x: image_from_url_text(x),
            _js="extract_image_from_gallery_inpaint",
            inputs=[img2img_gallery],
            outputs=[init_img_with_mask],
        )

        send_to_extras.click(
            fn=lambda x: image_from_url_text(x),
            _js="extract_image_from_gallery_extras",
            inputs=[txt2img_gallery],
            outputs=[extras_image],
        )

        open_txt2img_folder.click(
            fn=lambda: open_folder(opts.outdir_samples or opts.outdir_txt2img_samples),
            inputs=[],
            outputs=[],
        )

        open_img2img_folder.click(
            fn=lambda: open_folder(opts.outdir_samples or opts.outdir_img2img_samples),
            inputs=[],
            outputs=[],
        )

        open_extras_folder.click(
            fn=lambda: open_folder(opts.outdir_samples or opts.outdir_extras_samples),
            inputs=[],
            outputs=[],
        )

        img2img_send_to_extras.click(
            fn=lambda x: image_from_url_text(x),
            _js="extract_image_from_gallery_extras",
            inputs=[img2img_gallery],
            outputs=[extras_image],
        )

        modules.generation_parameters_copypaste.connect_paste(pnginfo_send_to_txt2img, txt2img_paste_fields, generation_info, 'switch_to_txt2img')
        modules.generation_parameters_copypaste.connect_paste(pnginfo_send_to_img2img, img2img_paste_fields, generation_info, 'switch_to_img2img_img2img')

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
        def apply_field(obj, field, condition=None):
            key = path + "/" + field

            if getattr(obj,'custom_script_source',None) is not None:
              key = 'customscript/' + obj.custom_script_source + '/' + key
            
            if getattr(obj, 'do_not_save_to_config', False):
                return
            
            saved_value = ui_settings.get(key, None)
            if saved_value is None:
                ui_settings[key] = getattr(obj, field)
            elif condition is None or condition(saved_value):
                setattr(obj, field, saved_value)

        if type(x) in [gr.Slider, gr.Radio, gr.Checkbox, gr.Textbox, gr.Number] and x.visible:
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
        
    visit(txt2img_interface, loadsave, "txt2img")
    visit(img2img_interface, loadsave, "img2img")
    visit(extras_interface, loadsave, "extras")

    if not error_loading and (not os.path.exists(ui_config_file) or settings_count != len(ui_settings)):
        with open(ui_config_file, "w", encoding="utf8") as file:
            json.dump(ui_settings, file, indent=4)

    return demo


with open(os.path.join(script_path, "script.js"), "r", encoding="utf8") as jsfile:
    javascript = f'<script>{jsfile.read()}</script>'

jsdir = os.path.join(script_path, "javascript")
for filename in sorted(os.listdir(jsdir)):
    with open(os.path.join(jsdir, filename), "r", encoding="utf8") as jsfile:
        javascript += f"\n<script>{jsfile.read()}</script>"


if 'gradio_routes_templates_response' not in globals():
    def template_response(*args, **kwargs):
        res = gradio_routes_templates_response(*args, **kwargs)
        res.body = res.body.replace(b'</head>', f'{javascript}</head>'.encode("utf8"))
        res.init_headers()
        return res

    gradio_routes_templates_response = gradio.routes.templates.TemplateResponse
    gradio.routes.templates.TemplateResponse = template_response
