import gradio as gr
from frontend.css_and_js import *
from frontend.css_and_js import css
import frontend.ui_functions as uifn

def draw_gradio_ui(opt, img2img=lambda x: x, txt2img=lambda x: x, txt2img_defaults={}, RealESRGAN=True, GFPGAN=True,
                   txt2img_toggles={}, txt2img_toggle_defaults='k_euler', show_embeddings=False, img2img_defaults={},
                   img2img_toggles={}, img2img_toggle_defaults={}, sample_img2img=None, img2img_mask_modes=None,
                   img2img_resize_modes=None, user_defaults={}, run_GFPGAN=lambda x: x, run_RealESRGAN=lambda x: x):

    with gr.Blocks(css=css(opt), analytics_enabled=False, title="Stable Diffusion WebUI") as demo:
        with gr.Tabs(elem_id='tabss') as tabs:
            with gr.TabItem("Stable Diffusion Text-to-Image Unified", id='txt2img_tab'):
                with gr.Row(elem_id="prompt_row"):
                    txt2img_prompt = gr.Textbox(label="Prompt",
                                                elem_id='prompt_input',
                                                placeholder="A corgi wearing a top hat as an oil painting.",
                                                lines=1,
                                                max_lines=1 if txt2img_defaults['submit_on_enter'] == 'Yes' else 25,
                                                value=txt2img_defaults['prompt'],
                                                show_label=False)
                    txt2img_btn = gr.Button("Generate", elem_id="generate", variant="primary")

                with gr.Row(elem_id='body').style(equal_height=False):
                    with gr.Column():
                        txt2img_height = gr.Slider(minimum=64, maximum=2048, step=64, label="Height",
                                                   value=txt2img_defaults["height"])
                        txt2img_width = gr.Slider(minimum=64, maximum=2048, step=64, label="Width",
                                                  value=txt2img_defaults["width"])
                        txt2img_cfg = gr.Slider(minimum=1.0, maximum=30.0, step=0.5,
                                                label='Classifier Free Guidance Scale (how strongly the image should follow the prompt)',
                                                value=txt2img_defaults['cfg_scale'])
                        txt2img_seed = gr.Textbox(label="Seed (blank to randomize)", lines=1, max_lines=1,
                                                  value=txt2img_defaults["seed"])
                        txt2img_batch_count = gr.Slider(minimum=1, maximum=250, step=1,
                                                        label='Batch count (how many batches of images to generate)',
                                                        value=txt2img_defaults['n_iter'])
                        txt2img_batch_size = gr.Slider(minimum=1, maximum=8, step=1,
                                                       label='Batch size (how many images are in a batch; memory-hungry)',
                                                       value=txt2img_defaults['batch_size'])
                    with gr.Column():
                        output_txt2img_gallery = gr.Gallery(label="Images", elem_id="txt2img_gallery_output").style(grid=[4, 4])

                        with gr.Tabs():
                            with gr.TabItem("Generated image actions", id="text2img_actions_tab"):
                                gr.Markdown(
                                    'Select an image from the gallery, then click one of the buttons below to perform an action.')
                                with gr.Row():
                                    output_txt2img_copy_clipboard = gr.Button("Copy to clipboard").click(fn=None,
                                                                                                         inputs=output_txt2img_gallery,
                                                                                                         outputs=[],
                                                                                                         _js=js_copy_selected_txt2img)
                                    output_txt2img_copy_to_input_btn = gr.Button("Push to img2img")
                                    if RealESRGAN is not None:
                                        output_txt2img_to_upscale_esrgan = gr.Button("Upscale w/ ESRGAN")

                            with gr.TabItem("Output Info", id="text2img_output_info_tab"):
                                output_txt2img_params = gr.Textbox(label="Generation parameters", interactive=False)
                                with gr.Row():
                                    output_txt2img_copy_params = gr.Button("Copy full parameters").click(
                                        inputs=output_txt2img_params, outputs=[],
                                        _js='(x) => navigator.clipboard.writeText(x)', fn=None, show_progress=False)
                                    output_txt2img_seed = gr.Number(label='Seed', interactive=False, visible=False)
                                    output_txt2img_copy_seed = gr.Button("Copy only seed").click(
                                        inputs=output_txt2img_seed, outputs=[],
                                        _js='(x) => navigator.clipboard.writeText(x)', fn=None, show_progress=False)
                                output_txt2img_stats = gr.HTML(label='Stats')
                    with gr.Column():

                        txt2img_steps = gr.Slider(minimum=1, maximum=250, step=1, label="Sampling Steps",
                                                  value=txt2img_defaults['ddim_steps'])
                        txt2img_sampling = gr.Dropdown(label='Sampling method (k_lms is default k-diffusion sampler)',
                                                       choices=["DDIM", "PLMS", 'k_dpm_2_a', 'k_dpm_2', 'k_euler_a',
                                                                'k_euler', 'k_heun', 'k_lms'],
                                                       value=txt2img_defaults['sampler_name'])
                        with gr.Tabs():
                            with gr.TabItem('Simple'):
                                txt2img_submit_on_enter = gr.Radio(['Yes', 'No'],
                                                                   label="Submit on enter? (no means multiline)",
                                                                   value=txt2img_defaults['submit_on_enter'],
                                                                   interactive=True)
                                txt2img_submit_on_enter.change(
                                    lambda x: gr.update(max_lines=1 if x == 'Single' else 25), txt2img_submit_on_enter,
                                    txt2img_prompt)
                            with gr.TabItem('Advanced'):
                                txt2img_toggles = gr.CheckboxGroup(label='', choices=txt2img_toggles,
                                                                   value=txt2img_toggle_defaults, type="index")
                                txt2img_realesrgan_model_name = gr.Dropdown(label='RealESRGAN model',
                                                                            choices=['RealESRGAN_x4plus',
                                                                                     'RealESRGAN_x4plus_anime_6B'],
                                                                            value='RealESRGAN_x4plus',
                                                                            visible=RealESRGAN is not None)  # TODO: Feels like I shouldnt slot it in here.
                                txt2img_ddim_eta = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="DDIM ETA",
                                                             value=txt2img_defaults['ddim_eta'], visible=False)
                        txt2img_embeddings = gr.File(label="Embeddings file for textual inversion",
                                                     visible=show_embeddings)

                txt2img_btn.click(
                    txt2img,
                    [txt2img_prompt, txt2img_steps, txt2img_sampling, txt2img_toggles, txt2img_realesrgan_model_name,
                     txt2img_ddim_eta, txt2img_batch_count, txt2img_batch_size, txt2img_cfg, txt2img_seed,
                     txt2img_height, txt2img_width, txt2img_embeddings],
                    [output_txt2img_gallery, output_txt2img_seed, output_txt2img_params, output_txt2img_stats]
                )
                txt2img_prompt.submit(
                    txt2img,
                    [txt2img_prompt, txt2img_steps, txt2img_sampling, txt2img_toggles, txt2img_realesrgan_model_name,
                     txt2img_ddim_eta, txt2img_batch_count, txt2img_batch_size, txt2img_cfg, txt2img_seed,
                     txt2img_height, txt2img_width, txt2img_embeddings],
                    [output_txt2img_gallery, output_txt2img_seed, output_txt2img_params, output_txt2img_stats]
                )

            with gr.TabItem("Stable Diffusion Image-to-Image Unified", id="img2img_tab"):
                with gr.Row(elem_id="prompt_row"):
                    img2img_prompt = gr.Textbox(label="Prompt",
                                                elem_id='img2img_prompt_input',
                                                placeholder="A fantasy landscape, trending on artstation.",
                                                lines=1,
                                                max_lines=1 if txt2img_defaults['submit_on_enter'] == 'Yes' else 25,
                                                value=img2img_defaults['prompt'],
                                                show_label=False).style()
                    img2img_btn_mask = gr.Button("Generate", variant="primary", visible=False,
                                                 elem_id="img2img_mask_btn")
                    img2img_btn_editor = gr.Button("Generate", variant="primary", elem_id="img2img_edit_btn")
                with gr.Row().style(equal_height=False):
                    with gr.Column():
                        gr.Markdown('#### Img2Img input')
                        img2img_image_editor = gr.Image(value=sample_img2img, source="upload", interactive=True,
                                                        type="pil", tool="select", elem_id="img2img_editor")
                        img2img_image_mask = gr.Image(value=sample_img2img, source="upload", interactive=True,
                                                      type="pil", tool="sketch", visible=False,
                                                      elem_id="img2img_mask")

                        with gr.Row():
                            img2img_image_editor_mode = gr.Radio(choices=["Mask", "Crop"], label="Image Editor Mode",
                                                             value="Crop", elem_id='edit_mode_select')

                            img2img_painterro_btn = gr.Button("Advanced Editor")
                            img2img_copy_from_painterro_btn = gr.Button(value="Get Image from Advanced Editor")
                            img2img_show_help_btn = gr.Button("Show Hints")
                            img2img_hide_help_btn = gr.Button("Hide Hints", visible=False)
                        img2img_help = gr.Markdown(visible=False, value="")



                    with gr.Column():
                        gr.Markdown('#### Img2Img Results')
                        output_img2img_gallery = gr.Gallery(label="Images", elem_id="img2img_gallery_output").style(grid=[4,4,4])
                        with gr.Tabs():
                            with gr.TabItem("Generated image actions", id="img2img_actions_tab"):
                                with gr.Group():
                                    gr.Markdown("Select an image, then press one of the buttons below")
                                    output_img2img_copy_to_clipboard_btn = gr.Button("Copy to clipboard")
                                    output_img2img_copy_to_input_btn = gr.Button("Push to img2img input")
                                    output_img2img_copy_to_mask_btn = gr.Button("Push to img2img input mask")
                                    gr.Markdown("Warning: This will clear your current image and mask settings!")
                            with gr.TabItem("Output info", id="img2img_output_info_tab"):
                                output_img2img_params = gr.Textbox(label="Generation parameters")
                                with gr.Row():
                                    output_img2img_copy_params = gr.Button("Copy full parameters").click(
                                        inputs=output_img2img_params, outputs=[],
                                        _js='(x) => navigator.clipboard.writeText(x)', fn=None, show_progress=False)
                                    output_img2img_seed = gr.Number(label='Seed', interactive=False, visible=False)
                                    output_img2img_copy_seed = gr.Button("Copy only seed").click(
                                        inputs=output_img2img_seed, outputs=[],
                                        _js='(x) => navigator.clipboard.writeText(x)', fn=None, show_progress=False)
                                output_img2img_stats = gr.HTML(label='Stats')
                gr.Markdown('# img2img settings')
                with gr.Row():

                    with gr.Column():
                        img2img_batch_size = gr.Slider(minimum=1, maximum=8, step=1,
                                                       label='Batch size (how many images are in a batch; memory-hungry)',
                                                       value=img2img_defaults['batch_size'])
                        img2img_width = gr.Slider(minimum=64, maximum=2048, step=64, label="Width",
                                                  value=img2img_defaults["width"])
                        img2img_height = gr.Slider(minimum=64, maximum=2048, step=64, label="Height",
                                                   value=img2img_defaults["height"])
                        img2img_seed = gr.Textbox(label="Seed (blank to randomize)", lines=1,
                                                  value=img2img_defaults["seed"])
                        img2img_steps = gr.Slider(minimum=1, maximum=250, step=1, label="Sampling Steps",
                                                  value=img2img_defaults['ddim_steps'])
                        img2img_batch_count = gr.Slider(minimum=1, maximum=250, step=1,
                                                        label='Batch count (how many batches of images to generate)',
                                                        value=img2img_defaults['n_iter'])
                    with gr.Column():
                        img2img_mask = gr.Radio(choices=["Keep masked area", "Regenerate only masked area"],
                                                label="Mask Mode", type="index",
                                                value=img2img_mask_modes[img2img_defaults['mask_mode']], visible=False)
                        img2img_mask_blur_strength = gr.Slider(minimum=1, maximum=10, step=1,
                                                               label="How much blurry should the mask be? (to avoid hard edges)",
                                                               value=3, visible=False)

                        img2img_sampling = gr.Dropdown(label='Sampling method (k_lms is default k-diffusion sampler)',
                                                       choices=["DDIM", 'k_dpm_2_a', 'k_dpm_2', 'k_euler_a', 'k_euler',
                                                                'k_heun', 'k_lms'],
                                                       value=img2img_defaults['sampler_name'])
                        img2img_toggles = gr.CheckboxGroup(label='', choices=img2img_toggles,
                                                           value=img2img_toggle_defaults, type="index")
                        img2img_realesrgan_model_name = gr.Dropdown(label='RealESRGAN model',
                                                                    choices=['RealESRGAN_x4plus',
                                                                             'RealESRGAN_x4plus_anime_6B'],
                                                                    value='RealESRGAN_x4plus',
                                                                    visible=RealESRGAN is not None)  # TODO: Feels like I shouldnt slot it in here.


                        img2img_cfg = gr.Slider(minimum=1.0, maximum=30.0, step=0.5,
                                                label='Classifier Free Guidance Scale (how strongly the image should follow the prompt)',
                                                value=img2img_defaults['cfg_scale'])
                        img2img_denoising = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label='Denoising Strength',
                                                      value=img2img_defaults['denoising_strength'])

                        img2img_resize = gr.Radio(label="Resize mode",
                                                  choices=["Just resize", "Crop and resize", "Resize and fill"],
                                                  type="index",
                                                  value=img2img_resize_modes[img2img_defaults['resize_mode']])
                        img2img_embeddings = gr.File(label="Embeddings file for textual inversion",
                                                     visible=show_embeddings)

                img2img_image_editor_mode.change(
                    uifn.change_image_editor_mode,
                    [img2img_image_editor_mode, img2img_image_editor, img2img_resize, img2img_width, img2img_height],
                    [img2img_image_editor, img2img_image_mask, img2img_btn_editor, img2img_btn_mask,
                     img2img_painterro_btn, img2img_copy_from_painterro_btn, img2img_mask, img2img_mask_blur_strength]
                )

                img2img_image_editor.edit(
                    uifn.update_image_mask,
                    [img2img_image_editor, img2img_resize, img2img_width, img2img_height],
                    img2img_image_mask
                )

                img2img_show_help_btn.click(
                    uifn.show_help,
                    None,
                    [img2img_show_help_btn, img2img_hide_help_btn, img2img_help]
                )

                img2img_hide_help_btn.click(
                    uifn.hide_help,
                    None,
                    [img2img_show_help_btn, img2img_hide_help_btn, img2img_help]
                )

                output_txt2img_copy_to_input_btn.click(
                    uifn.copy_img_to_input,
                    [output_txt2img_gallery],
                    [img2img_image_editor, img2img_image_mask, tabs],
                    _js=js_return_selected_txt2img
                )

                output_img2img_copy_to_input_btn.click(
                    uifn.copy_img_to_edit,
                    [output_img2img_gallery],
                    [img2img_image_editor, tabs, img2img_image_editor_mode],
                    _js=js_return_selected_img2img
                )
                output_img2img_copy_to_mask_btn.click(
                    uifn.copy_img_to_mask,
                    [output_img2img_gallery],
                    [img2img_image_mask, tabs, img2img_image_editor_mode],
                    _js=js_return_selected_img2img
                )

                output_img2img_copy_to_clipboard_btn.click(fn=None, inputs=output_img2img_gallery, outputs=[],
                                                           _js=js_copy_selected_img2img)

                img2img_btn_mask.click(
                    img2img,
                    [img2img_prompt, img2img_image_editor_mode, img2img_image_mask, img2img_mask,
                     img2img_mask_blur_strength, img2img_steps, img2img_sampling, img2img_toggles,
                     img2img_realesrgan_model_name, img2img_batch_count, img2img_batch_size, img2img_cfg,
                     img2img_denoising, img2img_seed, img2img_height, img2img_width, img2img_resize,
                     img2img_embeddings],
                    [output_img2img_gallery, output_img2img_seed, output_img2img_params, output_img2img_stats]
                )
                def img2img_submit_params():
                    return (img2img,
                    [img2img_prompt, img2img_image_editor_mode, img2img_image_editor, img2img_mask,
                     img2img_mask_blur_strength, img2img_steps, img2img_sampling, img2img_toggles,
                     img2img_realesrgan_model_name, img2img_batch_count, img2img_batch_size, img2img_cfg,
                     img2img_denoising, img2img_seed, img2img_height, img2img_width, img2img_resize,
                     img2img_embeddings],
                    [output_img2img_gallery, output_img2img_seed, output_img2img_params, output_img2img_stats])
                img2img_btn_editor.click(*img2img_submit_params())
                img2img_prompt.submit(*img2img_submit_params())

                img2img_painterro_btn.click(None, [img2img_image_editor], None, _js="""(img) => {
                try {
                    Painterro({
                        hiddenTools: ['arrow'],
                        saveHandler: function (image, done) {
                            localStorage.setItem('painterro-image', image.asDataURL());
                            done(true);
                        },
                    }).show(Array.isArray(img) ? img[0] : img);
                } catch(e) {
                    const script = document.createElement('script');
                    script.src = 'https://unpkg.com/painterro@1.2.78/build/painterro.min.js';
                    document.head.appendChild(script);
                    const style = document.createElement('style');
                    style.appendChild(document.createTextNode('.ptro-holder-wrapper { z-index: 9999 !important; }'));
                    document.head.appendChild(style);
                }
                return [];
            }""")

                img2img_copy_from_painterro_btn.click(None, None, [img2img_image_editor, img2img_image_mask], _js="""() => {
                const image = localStorage.getItem('painterro-image')
                return [image, image];
            }""")

            if GFPGAN is not None:
                gfpgan_defaults = {
                    'strength': 100,
                }

                if 'gfpgan' in user_defaults:
                    gfpgan_defaults.update(user_defaults['gfpgan'])

                with gr.TabItem("GFPGAN", id='cfpgan_tab'):
                    gr.Markdown("Fix faces on images")
                    with gr.Row():
                        with gr.Column():
                            gfpgan_source = gr.Image(label="Source", source="upload", interactive=True, type="pil")
                            gfpgan_strength = gr.Slider(minimum=0.0, maximum=1.0, step=0.001, label="Effect strength",
                                                        value=gfpgan_defaults['strength'])
                            gfpgan_btn = gr.Button("Generate", variant="primary")
                        with gr.Column():
                            gfpgan_output = gr.Image(label="Output")
                    gfpgan_btn.click(
                        run_GFPGAN,
                        [gfpgan_source, gfpgan_strength],
                        [gfpgan_output]
                    )
            if RealESRGAN is not None:
                with gr.TabItem("RealESRGAN", id='realesrgan_tab'):
                    gr.Markdown("Upscale images")
                    with gr.Row():
                        with gr.Column():
                            realesrgan_source = gr.Image(label="Source", source="upload", interactive=True, type="pil")
                            realesrgan_model_name = gr.Dropdown(label='RealESRGAN model', choices=['RealESRGAN_x4plus',
                                                                                                   'RealESRGAN_x4plus_anime_6B'],
                                                                value='RealESRGAN_x4plus')
                            realesrgan_btn = gr.Button("Generate")
                        with gr.Column():
                            realesrgan_output = gr.Image(label="Output")
                    realesrgan_btn.click(
                        run_RealESRGAN,
                        [realesrgan_source, realesrgan_model_name],
                        [realesrgan_output]
                    )
                output_txt2img_to_upscale_esrgan.click(
                    uifn.copy_img_to_upscale_esrgan,
                    output_txt2img_gallery,
                    [realesrgan_source, tabs],
                    _js=js_return_selected_txt2img)

        gr.HTML("""
    <div id="90" style="max-width: 100%; font-size: 14px; text-align: center;" class="output-markdown gr-prose border-solid border border-gray-200 rounded gr-panel">
        <p>For help and advanced usage guides, visit the <a href="https://github.com/hlky/stable-diffusion-webui/wiki" target="_blank">Project Wiki</a></p>
        <p>Stable Diffusion WebUI is an open-source project. You can find the latest stable builds on the <a href="https://github.com/hlky/stable-diffusion" target="_blank">main repository</a>.
        If you would like to contribute to development or test bleeding edge builds, you can visit the <a href="https://github.com/hlky/stable-diffusion-webui" target="_blank">developement repository</a>.</p>
    </div>
    """)
    return demo
