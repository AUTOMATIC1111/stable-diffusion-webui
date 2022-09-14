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
import modules.styles

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


def save_files(js_data, images):
    import csv

    os.makedirs(opts.outdir_save, exist_ok=True)

    filenames = []

    data = json.loads(js_data)

    with open(os.path.join(opts.outdir_save, "log.csv"), "a", encoding="utf8", newline='') as file:
        at_start = file.tell() == 0
        writer = csv.writer(file)
        if at_start:
            writer.writerow(["prompt", "seed", "width", "height", "sampler", "cfgs", "steps", "filename", "negative_prompt"])

        filename_base = str(int(time.time() * 1000))
        for i, filedata in enumerate(images):
            filename = filename_base + ("" if len(images) == 1 else "-" + str(i + 1)) + ".png"
            filepath = os.path.join(opts.outdir_save, filename)

            if filedata.startswith("data:image/png;base64,"):
                filedata = filedata[len("data:image/png;base64,"):]

            with open(filepath, "wb") as imgfile:
                imgfile.write(base64.decodebytes(filedata.encode('utf-8')))

            filenames.append(filename)

        writer.writerow([data["prompt"], data["seed"], data["width"], data["height"], data["sampler"], data["cfg_scale"], data["steps"], filenames[0], data["negative_prompt"]])

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


def create_seed_inputs():
    with gr.Row():
        seed = gr.Number(label='Seed', value=-1)
        subseed = gr.Number(label='Variation seed', value=-1, visible=False)
        seed_checkbox = gr.Checkbox(label="Extra", elem_id="subseed_show", value=False)

    with gr.Row():
        subseed_strength = gr.Slider(label='Variation strength', value=0.0, minimum=0, maximum=1, step=0.01, visible=False)
        seed_resize_from_w = gr.Slider(minimum=0, maximum=2048, step=64, label="Resize seed from width", value=0, visible=False)
        seed_resize_from_h = gr.Slider(minimum=0, maximum=2048, step=64, label="Resize seed from height", value=0, visible=False)

    def change_visiblity(show):

        return {
            subseed: gr_show(show),
            subseed_strength: gr_show(show),
            seed_resize_from_h: gr_show(show),
            seed_resize_from_w: gr_show(show),
        }

    seed_checkbox.change(
        change_visiblity,
        inputs=[seed_checkbox],
        outputs=[
            subseed,
            subseed_strength,
            seed_resize_from_h,
            seed_resize_from_w
        ]
    )

    return seed, subseed, subseed_strength, seed_resize_from_h, seed_resize_from_w


def add_style(name: str, prompt: str, negative_prompt: str):
    if name is None:
        return [gr_show(), gr_show()]

    style = modules.styles.PromptStyle(name, prompt, negative_prompt)
    shared.prompt_styles[style.name] = style
    # Save all loaded prompt styles: this allows us to update the storage format in the future more easily, because we
    # reserialize all styles every time we save them
    modules.styles.save_styles(shared.styles_filename, shared.prompt_styles.values())

    update = {"visible": True, "choices": list(shared.prompt_styles), "__type__": "update"}
    return [update, update]


def interrogate(image):
    prompt = shared.interrogator.interrogate(image)

    return gr_show(True) if prompt is None else prompt

def create_ui(txt2img, img2img, run_extras, run_pnginfo):
    with gr.Blocks(analytics_enabled=False) as txt2img_interface:
        with gr.Row(elem_id="toprow"):
            txt2img_prompt = gr.Textbox(label="Prompt", elem_id="txt2img_prompt", show_label=False, placeholder="Prompt", lines=1)
            txt2img_negative_prompt = gr.Textbox(label="Negative prompt", elem_id="txt2img_negative_prompt", show_label=False, placeholder="Negative prompt", lines=1)
            txt2img_prompt_style = gr.Dropdown(label="Style", show_label=False, elem_id="style_index", choices=[k for k, v in shared.prompt_styles.items()], value=next(iter(shared.prompt_styles.keys())), visible=len(shared.prompt_styles) > 1)
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

                cfg_scale = gr.Slider(minimum=1.0, maximum=30.0, step=0.5, label='CFG Scale', value=7.0)

                with gr.Group():
                    width = gr.Slider(minimum=64, maximum=2048, step=64, label="Width", value=512)
                    height = gr.Slider(minimum=64, maximum=2048, step=64, label="Height", value=512)

                seed, subseed, subseed_strength, seed_resize_from_h, seed_resize_from_w = create_seed_inputs()

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
                        txt2img_save_style = gr.Button('Save prompt as style')

                progressbar = gr.HTML(elem_id="progressbar")

                with gr.Group():
                    html_info = gr.HTML()
                    generation_info = gr.Textbox(visible=False)


            txt2img_args = dict(
                fn=txt2img,
                _js="submit",
                inputs=[
                    txt2img_prompt,
                    txt2img_negative_prompt,
                    txt2img_prompt_style,
                    steps,
                    sampler_index,
                    restore_faces,
                    tiling,
                    batch_count,
                    batch_size,
                    cfg_scale,
                    seed,
                    subseed, subseed_strength, seed_resize_from_h, seed_resize_from_w,
                    height,
                    width,
                ] + custom_inputs,
                outputs=[
                    txt2img_gallery,
                    generation_info,
                    html_info
                ]
            )

            txt2img_prompt.submit(**txt2img_args)
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
                    txt2img_prompt,
                ],
                outputs=[
                    txt2img_prompt,
                ]
            )

    with gr.Blocks(analytics_enabled=False) as img2img_interface:
        with gr.Row(elem_id="toprow"):
            img2img_prompt = gr.Textbox(label="Prompt", elem_id="img2img_prompt", show_label=False, placeholder="Prompt", lines=1)
            img2img_negative_prompt = gr.Textbox(label="Negative prompt", elem_id="img2img_negative_prompt", show_label=False, placeholder="Negative prompt", lines=1)
            img2img_prompt_style = gr.Dropdown(label="Style", show_label=False, elem_id="style_index", choices=[k for k, v in shared.prompt_styles.items()], value=next(iter(shared.prompt_styles.keys())), visible=len(shared.prompt_styles) > 1)
            img2img_interrogate = gr.Button('Interrogate', elem_id="img2img_interrogate", variant='primary')
            submit = gr.Button('Generate', elem_id="img2img_generate", variant='primary')
            check_progress = gr.Button('Check progress', elem_id="check_progress", visible=False)

        with gr.Row().style(equal_height=False):
            with gr.Column(variant='panel'):
                with gr.Group():
                    switch_mode = gr.Radio(label='Mode', elem_id="img2img_mode", choices=['Redraw whole image', 'Inpaint a part of image', 'Loopback', 'SD upscale'], value='Redraw whole image', type="index", show_label=False)
                    init_img = gr.Image(label="Image for img2img", source="upload", interactive=True, type="pil")
                    init_img_with_mask = gr.Image(label="Image for inpainting with mask", elem_id="img2maskimg", source="upload", interactive=True, type="pil", tool="sketch", visible=False, image_mode="RGBA")
                    init_mask = gr.Image(label="Mask", source="upload", interactive=True, type="pil", visible=False)
                    init_img_with_mask_comment = gr.HTML(elem_id="mask_bug_info", value="<small>if the editor shows ERROR, switch to another tab and back, then to another img2img mode above and back</small>", visible=False)

                    with gr.Row():
                        resize_mode = gr.Radio(label="Resize mode", elem_id="resize_mode", show_label=False, choices=["Just resize", "Crop and resize", "Resize and fill"], type="index", value="Just resize")
                        mask_mode = gr.Radio(label="Mask mode", show_label=False, choices=["Draw mask", "Upload mask"], type="index", value="Draw mask")

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
                    cfg_scale = gr.Slider(minimum=1.0, maximum=30.0, step=0.5, label='CFG Scale', value=7.0)
                    denoising_strength = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label='Denoising strength', value=0.75)
                    denoising_strength_change_factor = gr.Slider(minimum=0.9, maximum=1.1, step=0.01, label='Denoising strength change factor', value=1, visible=False)

                with gr.Group():
                    width = gr.Slider(minimum=64, maximum=2048, step=64, label="Width", value=512)
                    height = gr.Slider(minimum=64, maximum=2048, step=64, label="Height", value=512)

                seed, subseed, subseed_strength, seed_resize_from_h, seed_resize_from_w = create_seed_inputs()

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
                        img2img_save_style = gr.Button('Save prompt as style')

                progressbar = gr.HTML(elem_id="progressbar")

                with gr.Group():
                    html_info = gr.HTML()
                    generation_info = gr.Textbox(visible=False)

            def apply_mode(mode, uploadmask):
                is_classic = mode == 0
                is_inpaint = mode == 1
                is_loopback = mode == 2
                is_upscale = mode == 3

                return {
                    init_img: gr_show(not is_inpaint or (is_inpaint and uploadmask == 1)),
                    init_img_with_mask: gr_show(is_inpaint and uploadmask == 0),
                    init_img_with_mask_comment: gr_show(is_inpaint and uploadmask == 0),
                    init_mask: gr_show(is_inpaint and uploadmask == 1),
                    mask_mode: gr_show(is_inpaint),
                    mask_blur: gr_show(is_inpaint),
                    inpainting_fill: gr_show(is_inpaint),
                    batch_size: gr_show(not is_loopback),
                    sd_upscale_upscaler_name: gr_show(is_upscale),
                    sd_upscale_overlap: gr_show(is_upscale),
                    inpaint_full_res: gr_show(is_inpaint),
                    inpainting_mask_invert: gr_show(is_inpaint),
                    denoising_strength_change_factor: gr_show(is_loopback),
                    img2img_interrogate: gr_show(not is_inpaint),
                }

            switch_mode.change(
                apply_mode,
                inputs=[switch_mode, mask_mode],
                outputs=[
                    init_img,
                    init_img_with_mask,
                    init_img_with_mask_comment,
                    init_mask,
                    mask_mode,
                    mask_blur,
                    inpainting_fill,
                    batch_size,
                    sd_upscale_upscaler_name,
                    sd_upscale_overlap,
                    inpaint_full_res,
                    inpainting_mask_invert,
                    denoising_strength_change_factor,
                    img2img_interrogate,
                ]
            )

            mask_mode.change(
                lambda mode: {
                    init_img: gr_show(mode == 1),
                    init_img_with_mask: gr_show(mode == 0),
                    init_mask: gr_show(mode == 1),
                },
                inputs=[mask_mode],
                outputs=[
                    init_img,
                    init_img_with_mask,
                    init_mask,
                ],
            )

            img2img_args = dict(
                fn=img2img,
                _js="submit",
                inputs=[
                    img2img_prompt,
                    img2img_negative_prompt,
                    img2img_prompt_style,
                    init_img,
                    init_img_with_mask,
                    init_mask,
                    mask_mode,
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
                    subseed, subseed_strength, seed_resize_from_h, seed_resize_from_w,
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

            img2img_prompt.submit(**img2img_args)
            submit.click(**img2img_args)

            img2img_interrogate.click(
                fn=interrogate,
                inputs=[init_img],
                outputs=[img2img_prompt],
            )

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

            dummy_component = gr.Label(visible=False)
            for button, (prompt, negative_prompt) in zip([txt2img_save_style, img2img_save_style], [(txt2img_prompt, txt2img_negative_prompt), (img2img_prompt, img2img_negative_prompt)]):
                button.click(
                    fn=add_style,
                    _js="ask_for_style_name",
                    # Have to pass empty dummy component here, because the JavaScript and Python function have to accept
                    # the same number of parameters, but we only know the style-name after the JavaScript prompt
                    inputs=[dummy_component, prompt, negative_prompt],
                    outputs=[txt2img_prompt_style, img2img_prompt_style],
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
    keys = list(opts.data_labels.keys())
    settings_cols = 3
    items_per_col = math.ceil(len(keys) / settings_cols)

    def run_settings(*args):
        up = []

        for key, value, comp in zip(opts.data_labels.keys(), args, components):
            comp_args = opts.data_labels[key].component_args
            if comp_args and isinstance(comp_args, dict) and comp_args.get('visible') is False:
                continue

            opts.data[key] = value
            up.append(comp.update(value=value))

        opts.save(shared.config_filename)

        return 'Settings applied.'

    with gr.Blocks(analytics_enabled=False) as settings_interface:
        submit = gr.Button(value="Apply settings", variant='primary')
        result = gr.HTML()

        with gr.Row(elem_id="settings").style(equal_height=False):
            for colno in range(settings_cols):
                with gr.Column(variant='panel'):
                    for rowno in range(items_per_col):
                        index = rowno + colno * items_per_col

                        if index < len(keys):
                            components.append(create_setting_component(keys[index]))

        submit.click(
            fn=run_settings,
            inputs=components,
            outputs=[result]
        )

    interfaces = [
        (txt2img_interface, "txt2img", "txt2img"),
        (img2img_interface, "img2img", "img2img"),
        (extras_interface, "Extras", "extras"),
        (pnginfo_interface, "PNG Info", "pnginfo"),
        (settings_interface, "Settings", "settings"),
    ]

    with open(os.path.join(script_path, "style.css"), "r", encoding="utf8") as file:
        css = file.read()

    if not cmd_opts.no_progressbar_hiding:
        css += css_hide_progressbar

    with gr.Blocks(css=css, analytics_enabled=False, title="Stable Diffusion") as demo:
        with gr.Tabs() as tabs:
            for interface, label, ifid in interfaces:
                with gr.TabItem(label, id=ifid):
                    interface.render()

        tabs.change(
            fn=lambda x: x,
            inputs=[init_img_with_mask],
            outputs=[init_img_with_mask],
        )

        send_to_img2img.click(
            fn=lambda x: image_from_url_text(x),
            _js="extract_image_from_gallery_img2img",
            inputs=[txt2img_gallery],
            outputs=[init_img],
        )

        send_to_inpaint.click(
            fn=lambda x: image_from_url_text(x),
            _js="extract_image_from_gallery_img2img",
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

        send_to_extras.click(
            fn=lambda x: image_from_url_text(x),
            _js="extract_image_from_gallery_extras",
            inputs=[txt2img_gallery],
            outputs=[image],
        )

        img2img_send_to_extras.click(
            fn=lambda x: image_from_url_text(x),
            _js="extract_image_from_gallery_extras",
            inputs=[img2img_gallery],
            outputs=[image],
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
    visit(extras_interface, loadsave, "extras")

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
