import base64
import html
import io
import json
import mimetypes
import os
import random
import sys
import time
import traceback

import numpy as np
import torch
from PIL import Image

import gradio as gr
import gradio.utils
import gradio.routes

from modules.paths import script_path
from modules.shared import opts, cmd_opts
import modules.shared as shared
from modules.sd_samplers import samplers, samplers_for_img2img
import modules.realesrgan_model as realesrgan
import modules.scripts
import modules.gfpgan_model
import modules.codeformer_model

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

def plaintext_to_html(text):
    text = "".join([f"<p>{html.escape(x)}</p>\n" for x in text.split('\n')])
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


def save_files(js_data, images):
    import csv

    os.makedirs(opts.outdir_save, exist_ok=True)

    filenames = []

    data = json.loads(js_data)

    with open(os.path.join(opts.outdir_save, "log.csv"), "a", encoding="utf8", newline='') as file:
        at_start = file.tell() == 0
        writer = csv.writer(file)
        if at_start:
            writer.writerow(["prompt", "seed", "width", "height", "sampler", "cfgs", "steps", "filename"])

        filename_base = str(int(time.time() * 1000))
        for i, filedata in enumerate(images):
            filename = filename_base + ("" if len(images) == 1 else "-" + str(i + 1)) + ".png"
            filepath = os.path.join(opts.outdir_save, filename)

            if filedata.startswith("data:image/png;base64,"):
                filedata = filedata[len("data:image/png;base64,"):]

            with open(filepath, "wb") as imgfile:
                imgfile.write(base64.decodebytes(filedata.encode('utf-8')))

            filenames.append(filename)

        writer.writerow([data["prompt"], data["seed"], data["width"], data["height"], data["sampler"], data["cfg_scale"], data["steps"], filenames[0]])

    return '', '', plaintext_to_html(f"Saved: {filenames[0]}")


def wrap_gradio_call(func):
    def f(*args, **kwargs):
        t = time.perf_counter()

        try:
            res = list(func(*args, **kwargs))
        except Exception as e:
            print("Error completing request", file=sys.stderr)
            print("Arguments:", args, kwargs, file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)

            shared.state.job = ""
            shared.state.job_count = 0

            res = [None, '', f"<div class='error'>{plaintext_to_html(type(e).__name__+': '+str(e))}</div>"]

        elapsed = time.perf_counter() - t

        # last item is always HTML
        res[-1] = res[-1] + f"<p class='performance'>Time taken: {elapsed:.2f}s</p>"

        shared.state.interrupted = False

        return tuple(res)

    return f


def check_progress_call():

    if shared.state.job_count == 0:
        return "", gr_show(False), gr_show(False)

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

        if image is None or progress >= 1:
            image = gr.update(value=None)
        else:
            preview_visibility = gr_show(True)

    return f"<span style='display: none'>{time.time()}</span><p>{progressbar}</p>", preview_visibility, image


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


def create_ui(txt2img, img2img, run_extras, run_pnginfo):
    with gr.Blocks(analytics_enabled=False) as txt2img_interface:
        with gr.Row():
            prompt = gr.Textbox(label="Prompt", elem_id="txt2img_prompt", show_label=False, placeholder="Prompt", lines=1)
            negative_prompt = gr.Textbox(label="Negative prompt", elem_id="txt2img_negative_prompt", show_label=False, placeholder="Negative prompt", lines=1, visible=False)
            roll = gr.Button('Roll', elem_id="txt2img_roll", visible=len(shared.artist_db.artists) > 0)
            submit = gr.Button('Generate', elem_id="txt2img_generate", variant='primary')
            check_progress = gr.Button('Check progress', elem_id="check_progress", visible=False)

        with gr.Row().style(equal_height=False):
            with gr.Column(variant='panel'):
                steps = gr.Slider(minimum=1, maximum=150, step=1, label="Sampling Steps", value=20)
                sampler_index = gr.Radio(label='Sampling method', elem_id="txt2img_sampling", choices=[x.name for x in samplers], value=samplers[0].name, type="index")

                with gr.Row():
                    restore_faces = gr.Checkbox(label='Restore faces', value=False, visible=len(shared.face_restorers) > 1)
                    tiling = gr.Checkbox(label='Tiling', value=False)

                with gr.Row():
                    batch_count = gr.Slider(minimum=1, maximum=cmd_opts.max_batch_count, step=1, label='Batch count', value=1)
                    batch_size = gr.Slider(minimum=1, maximum=8, step=1, label='Batch size', value=1)

                cfg_scale = gr.Slider(minimum=1.0, maximum=15.0, step=0.5, label='CFG Scale', value=7.0)

                with gr.Group():
                    height = gr.Slider(minimum=64, maximum=2048, step=64, label="Height", value=512)
                    width = gr.Slider(minimum=64, maximum=2048, step=64, label="Width", value=512)

                seed = gr.Number(label='Seed', value=-1)

                with gr.Group():
                    custom_inputs = modules.scripts.scripts_txt2img.setup_ui(is_img2img=False)

            with gr.Column(variant='panel'):
                with gr.Group():
                    txt2img_preview = gr.Image(elem_id='txt2img_preview', visible=False)
                    txt2img_gallery = gr.Gallery(label='Output', elem_id='txt2img_gallery').style(grid=4)


                with gr.Group():
                    with gr.Row():
                        save = gr.Button('Save')
                        send_to_img2img = gr.Button('Send to img2img')
                        send_to_inpaint = gr.Button('Send to inpaint')
                        send_to_extras = gr.Button('Send to extras')
                        interrupt = gr.Button('Interrupt')

                progressbar = gr.HTML(elem_id="progressbar")

                with gr.Group():
                    html_info = gr.HTML()
                    generation_info = gr.Textbox(visible=False)


            txt2img_args = dict(
                fn=txt2img,
                _js="submit",
                inputs=[
                    prompt,
                    negative_prompt,
                    steps,
                    sampler_index,
                    restore_faces,
                    tiling,
                    batch_count,
                    batch_size,
                    cfg_scale,
                    seed,
                    height,
                    width,
                ] + custom_inputs,
                outputs=[
                    txt2img_gallery,
                    generation_info,
                    html_info
                ]
            )

            prompt.submit(**txt2img_args)
            submit.click(**txt2img_args)

            check_progress.click(
                fn=check_progress_call,
                show_progress=False,
                inputs=[],
                outputs=[progressbar, txt2img_preview, txt2img_preview],
            )


            interrupt.click(
                fn=lambda: shared.state.interrupt(),
                inputs=[],
                outputs=[],
            )

            save.click(
                fn=wrap_gradio_call(save_files),
                inputs=[
                    generation_info,
                    txt2img_gallery,
                ],
                outputs=[
                    html_info,
                    html_info,
                    html_info,
                ]
            )

            roll.click(
                fn=roll_artist,
                inputs=[
                    prompt,
                ],
                outputs=[
                    prompt
                ]
            )


    with gr.Blocks(analytics_enabled=False) as img2img_interface:
        with gr.Row():
            prompt = gr.Textbox(label="Prompt", elem_id="img2img_prompt", show_label=False, placeholder="Prompt", lines=1)
            submit = gr.Button('Generate', elem_id="img2img_generate", variant='primary')
            check_progress = gr.Button('Check progress', elem_id="check_progress", visible=False)

        with gr.Row().style(equal_height=False):

            with gr.Column(variant='panel'):
                with gr.Group():
                    switch_mode = gr.Radio(label='Mode', elem_id="img2img_mode", choices=['Redraw whole image', 'Inpaint a part of image', 'Loopback', 'SD upscale'], value='Redraw whole image', type="index", show_label=False)
                    init_img = gr.Image(label="Image for img2img", source="upload", interactive=True, type="pil")
                    init_img_with_mask = gr.Image(label="Image for inpainting with mask", elem_id="img2maskimg", source="upload", interactive=True, type="pil", tool="sketch", visible=False, image_mode="RGBA")
                    resize_mode = gr.Radio(label="Resize mode", show_label=False, choices=["Just resize", "Crop and resize", "Resize and fill"], type="index", value="Just resize")

                steps = gr.Slider(minimum=1, maximum=150, step=1, label="Sampling Steps", value=20)
                sampler_index = gr.Radio(label='Sampling method', choices=[x.name for x in samplers_for_img2img], value=samplers_for_img2img[0].name, type="index")
                mask_blur = gr.Slider(label='Mask blur', minimum=0, maximum=64, step=1, value=4, visible=False)
                inpainting_fill = gr.Radio(label='Masked content', choices=['fill', 'original', 'latent noise', 'latent nothing'], value='fill', type="index", visible=False)

                with gr.Row():
                    inpaint_full_res = gr.Checkbox(label='Inpaint at full resolution', value=False, visible=False)
                    inpainting_mask_invert = gr.Radio(label='Masking mode', choices=['Inpaint masked', 'Inpaint not masked'], value='Inpaint masked', type="index", visible=False)

                with gr.Row():
                    restore_faces = gr.Checkbox(label='Restore faces', value=False, visible=len(shared.face_restorers) > 1)
                    tiling = gr.Checkbox(label='Tiling', value=False)
                    sd_upscale_overlap = gr.Slider(minimum=0, maximum=256, step=16, label='Tile overlap', value=64, visible=False)

                with gr.Row():
                    sd_upscale_upscaler_name = gr.Radio(label='Upscaler', choices=[x.name for x in shared.sd_upscalers], value=shared.sd_upscalers[0].name, type="index", visible=False)

                with gr.Row():
                    batch_count = gr.Slider(minimum=1, maximum=cmd_opts.max_batch_count, step=1, label='Batch count', value=1)
                    batch_size = gr.Slider(minimum=1, maximum=8, step=1, label='Batch size', value=1)

                with gr.Group():
                    cfg_scale = gr.Slider(minimum=1.0, maximum=15.0, step=0.5, label='CFG Scale', value=7.0)
                    denoising_strength = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label='Denoising strength', value=0.75)
                    denoising_strength_change_factor = gr.Slider(minimum=0.9, maximum=1.1, step=0.01, label='Denoising strength change factor', value=1, visible=False)

                with gr.Group():
                    height = gr.Slider(minimum=64, maximum=2048, step=64, label="Height", value=512)
                    width = gr.Slider(minimum=64, maximum=2048, step=64, label="Width", value=512)

                seed = gr.Number(label='Seed', value=-1)

                with gr.Group():
                    custom_inputs = modules.scripts.scripts_img2img.setup_ui(is_img2img=True)

            with gr.Column(variant='panel'):
                with gr.Group():
                    img2img_preview = gr.Image(elem_id='img2img_preview', visible=False)
                    img2img_gallery = gr.Gallery(label='Output', elem_id='img2img_gallery').style(grid=4)

                with gr.Group():
                    with gr.Row():
                        save = gr.Button('Save')
                        img2img_send_to_img2img = gr.Button('Send to img2img')
                        img2img_send_to_inpaint = gr.Button('Send to inpaint')
                        img2img_send_to_extras = gr.Button('Send to extras')
                        interrupt = gr.Button('Interrupt')

                progressbar = gr.HTML(elem_id="progressbar")

                with gr.Group():
                    html_info = gr.HTML()
                    generation_info = gr.Textbox(visible=False)

            def apply_mode(mode):
                is_classic = mode == 0
                is_inpaint = mode == 1
                is_loopback = mode == 2
                is_upscale = mode == 3

                return {
                    init_img: gr_show(not is_inpaint),
                    init_img_with_mask: gr_show(is_inpaint),
                    mask_blur: gr_show(is_inpaint),
                    inpainting_fill: gr_show(is_inpaint),
                    batch_count: gr_show(not is_upscale),
                    batch_size: gr_show(not is_loopback),
                    sd_upscale_upscaler_name: gr_show(is_upscale),
                    sd_upscale_overlap: gr_show(is_upscale),
                    inpaint_full_res: gr_show(is_inpaint),
                    inpainting_mask_invert: gr_show(is_inpaint),
                    denoising_strength_change_factor: gr_show(is_loopback),
                }

            switch_mode.change(
                apply_mode,
                inputs=[switch_mode],
                outputs=[
                    init_img,
                    init_img_with_mask,
                    mask_blur,
                    inpainting_fill,
                    batch_count,
                    batch_size,
                    sd_upscale_upscaler_name,
                    sd_upscale_overlap,
                    inpaint_full_res,
                    inpainting_mask_invert,
                    denoising_strength_change_factor,
                ]
            )

            img2img_args = dict(
                fn=img2img,
                _js="submit",
                inputs=[
                    prompt,
                    init_img,
                    init_img_with_mask,
                    steps,
                    sampler_index,
                    mask_blur,
                    inpainting_fill,
                    restore_faces,
                    tiling,
                    switch_mode,
                    batch_count,
                    batch_size,
                    cfg_scale,
                    denoising_strength,
                    denoising_strength_change_factor,
                    seed,
                    height,
                    width,
                    resize_mode,
                    sd_upscale_upscaler_name,
                    sd_upscale_overlap,
                    inpaint_full_res,
                    inpainting_mask_invert,
                ] + custom_inputs,
                outputs=[
                    img2img_gallery,
                    generation_info,
                    html_info
                ]
            )

            prompt.submit(**img2img_args)
            submit.click(**img2img_args)

            check_progress.click(
                fn=check_progress_call,
                show_progress=False,
                inputs=[],
                outputs=[progressbar, img2img_preview, img2img_preview],
            )

            interrupt.click(
                fn=lambda: shared.state.interrupt(),
                inputs=[],
                outputs=[],
            )

            save.click(
                fn=wrap_gradio_call(save_files),
                inputs=[
                    generation_info,
                    img2img_gallery,
                ],
                outputs=[
                    html_info,
                    html_info,
                    html_info,
                ]
            )

            send_to_img2img.click(
                fn=lambda x: image_from_url_text(x),
                _js="extract_image_from_gallery",
                inputs=[txt2img_gallery],
                outputs=[init_img],
            )

            send_to_inpaint.click(
                fn=lambda x: image_from_url_text(x),
                _js="extract_image_from_gallery",
                inputs=[txt2img_gallery],
                outputs=[init_img_with_mask],
            )

            img2img_send_to_img2img.click(
                fn=lambda x: image_from_url_text(x),
                _js="extract_image_from_gallery",
                inputs=[img2img_gallery],
                outputs=[init_img],
            )

            img2img_send_to_inpaint.click(
                fn=lambda x: image_from_url_text(x),
                _js="extract_image_from_gallery",
                inputs=[img2img_gallery],
                outputs=[init_img_with_mask],
            )

    with gr.Blocks(analytics_enabled=False) as extras_interface:
        with gr.Row().style(equal_height=False):
            with gr.Column(variant='panel'):
                with gr.Group():
                    image = gr.Image(label="Source", source="upload", interactive=True, type="pil")

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
                result_image = gr.Image(label="Result")
                html_info_x = gr.HTML()
                html_info = gr.HTML()

        extras_args = dict(
            fn=run_extras,
            inputs=[
                image,
                gfpgan_visibility,
                codeformer_visibility,
                codeformer_weight,
                upscaling_resize,
                extras_upscaler_1,
                extras_upscaler_2,
                extras_upscaler_2_visibility,
            ],
            outputs=[
                result_image,
                html_info_x,
                html_info,
            ]
        )

        submit.click(**extras_args)

        send_to_extras.click(
            fn=lambda x: image_from_url_text(x),
            _js="extract_image_from_gallery",
            inputs=[txt2img_gallery],
            outputs=[image],
        )

        img2img_send_to_extras.click(
            fn=lambda x: image_from_url_text(x),
            _js="extract_image_from_gallery",
            inputs=[img2img_gallery],
            outputs=[image],
        )

    pnginfo_interface = gr.Interface(
        wrap_gradio_call(run_pnginfo),
        inputs=[
            gr.Image(label="Source", source="upload", interactive=True, type="pil"),
        ],
        outputs=[
            gr.HTML(),
            gr.HTML(),
            gr.HTML(),
        ],
        allow_flagging="never",
        analytics_enabled=False,
    )

    def create_setting_component(key):
        def fun():
            return opts.data[key] if key in opts.data else opts.data_labels[key].default

        info = opts.data_labels[key]
        t = type(info.default)

        if info.component is not None:
            args = info.component_args() if callable(info.component_args) else info.component_args
            item = info.component(label=info.label, value=fun, **(args or {}))
        elif t == str:
            item = gr.Textbox(label=info.label, value=fun, lines=1)
        elif t == int:
            item = gr.Number(label=info.label, value=fun)
        elif t == bool:
            item = gr.Checkbox(label=info.label, value=fun)
        else:
            raise Exception(f'bad options item type: {str(t)} for key {key}')

        return item

    def run_settings(*args):
        up = []

        for key, value, comp in zip(opts.data_labels.keys(), args, settings_interface.input_components):
            opts.data[key] = value
            up.append(comp.update(value=value))

        opts.save(shared.config_filename)

        return 'Settings saved.', '', ''

    settings_interface = gr.Interface(
        run_settings,
        inputs=[create_setting_component(key) for key in opts.data_labels.keys()],
        outputs=[
            gr.Textbox(label='Result'),
            gr.HTML(),
            gr.HTML(),
        ],
        title=None,
        description=None,
        allow_flagging="never",
        analytics_enabled=False,
    )

    interfaces = [
        (txt2img_interface, "txt2img"),
        (img2img_interface, "img2img"),
        (extras_interface, "Extras"),
        (pnginfo_interface, "PNG Info"),
        (settings_interface, "Settings"),
    ]

    with open(os.path.join(script_path, "style.css"), "r", encoding="utf8") as file:
        css = file.read()

    if not cmd_opts.no_progressbar_hiding:
        css += css_hide_progressbar

    demo = gr.TabbedInterface(
        interface_list=[x[0] for x in interfaces],
        tab_names=[x[1] for x in interfaces],
        analytics_enabled=False,
        css=css,
    )

    ui_config_file = os.path.join(modules.paths.script_path, 'ui-config.json')
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

            saved_value = ui_settings.get(key, None)
            if saved_value is None:
                ui_settings[key] = getattr(obj, field)
            elif condition is None or condition(saved_value):
                setattr(obj, field, saved_value)

        if type(x) == gr.Slider:
            apply_field(x, 'value')
            apply_field(x, 'minimum')
            apply_field(x, 'maximum')
            apply_field(x, 'step')

        if type(x) == gr.Radio:
            apply_field(x, 'value', lambda val: val in x.choices)

    visit(txt2img_interface, loadsave, "txt2img")
    visit(img2img_interface, loadsave, "img2img")

    if not error_loading and (not os.path.exists(ui_config_file) or settings_count != len(ui_settings)):
        with open(ui_config_file, "w", encoding="utf8") as file:
            json.dump(ui_settings, file, indent=4)

    return demo


with open(os.path.join(script_path, "script.js"), "r", encoding="utf8") as jsfile:
    javascript = jsfile.read()


def template_response(*args, **kwargs):
    res = gradio_routes_templates_response(*args, **kwargs)
    res.body = res.body.replace(b'</head>', f'<script>{javascript}</script></head>'.encode("utf8"))
    res.init_headers()
    return res


gradio_routes_templates_response = gradio.routes.templates.TemplateResponse
gradio.routes.templates.TemplateResponse = template_response
