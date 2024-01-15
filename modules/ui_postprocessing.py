import json
import gradio as gr
from modules import scripts, shared, ui_common, postprocessing, call_queue
import modules.generation_parameters_copypaste as parameters_copypaste
from modules.call_queue import wrap_gradio_gpu_call, wrap_queued_call, wrap_gradio_call # pylint: disable=unused-import
from modules.extras import run_pnginfo
from modules.ui_common import infotext_to_html


def wrap_pnginfo(image):
    _, geninfo, info = run_pnginfo(image)
    return infotext_to_html(geninfo), info, geninfo


def submit_click(tab_index, extras_image, image_batch, extras_batch_input_dir, extras_batch_output_dir, show_extras_results, save_output, *script_inputs):
    result_images, geninfo, js_info = postprocessing.run_postprocessing(tab_index, extras_image, image_batch, extras_batch_input_dir, extras_batch_output_dir, show_extras_results, *script_inputs, save_output=save_output)
    return result_images, geninfo, json.dumps(js_info), ''


def create_ui():
    tab_index = gr.State(value=0) # pylint: disable=abstract-class-instantiated
    with gr.Row(equal_height=False, variant='compact', elem_classes="extras"):
        with gr.Column(variant='compact'):
            with gr.Tabs(elem_id="mode_extras"):
                with gr.TabItem('Single Image', id="single_image", elem_id="extras_single_tab") as tab_single:
                    extras_image = gr.Image(label="Source", source="upload", interactive=True, type="pil", elem_id="extras_image")
                with gr.TabItem('Process Batch', id="batch_process", elem_id="extras_batch_process_tab") as tab_batch:
                    image_batch = gr.Files(label="Batch process", interactive=True, elem_id="extras_image_batch")
                with gr.TabItem('Process Folder', id="batch_from_directory", elem_id="extras_batch_directory_tab") as tab_batch_dir:
                    extras_batch_input_dir = gr.Textbox(label="Input directory", **shared.hide_dirs, placeholder="A directory on the same machine where the server is running.", elem_id="extras_batch_input_dir")
                    extras_batch_output_dir = gr.Textbox(label="Output directory", **shared.hide_dirs, placeholder="Leave blank to save images to the default path.", elem_id="extras_batch_output_dir")
                    show_extras_results = gr.Checkbox(label='Show result images', value=True, elem_id="extras_show_extras_results")
            with gr.Row():
                buttons = parameters_copypaste.create_buttons(["txt2img", "img2img", "inpaint", "control"])
            with gr.Row():
                save_output = gr.Checkbox(label='Save output', value=True, elem_id="extras_save_output")
            script_inputs = scripts.scripts_postproc.setup_ui()
        with gr.Column():
            id_part = 'extras'
            with gr.Row(elem_id=f"{id_part}_generate_box", elem_classes="generate-box"):
                submit = gr.Button('Generate', elem_id=f"{id_part}_generate", variant='primary')
                interrupt = gr.Button('Stop', elem_id=f"{id_part}_interrupt", variant='secondary')
                interrupt.click(fn=lambda: shared.state.interrupt(), inputs=[], outputs=[])
                skip = gr.Button('Skip', elem_id=f"{id_part}_skip", variant='secondary')
                skip.click(fn=lambda: shared.state.skip(), inputs=[], outputs=[])
            result_images, generation_info, html_info, html_info_formatted, html_log = ui_common.create_output_panel("extras")
            gr.HTML('File metadata')
            exif_info = gr.HTML(elem_id="pnginfo_html_info")
            gen_info = gr.Text(elem_id="pnginfo_gen_info", visible=False)
        for tabname, button in buttons.items():
            parameters_copypaste.register_paste_params_button(parameters_copypaste.ParamBinding(paste_button=button, tabname=tabname, source_text_component=gen_info, source_image_component=extras_image))

    tab_single.select(fn=lambda: 0, inputs=[], outputs=[tab_index])
    tab_batch.select(fn=lambda: 1, inputs=[], outputs=[tab_index])
    tab_batch_dir.select(fn=lambda: 2, inputs=[], outputs=[tab_index])
    extras_image.change(
        fn=wrap_gradio_call(wrap_pnginfo),
        inputs=[extras_image],
        outputs=[html_info_formatted, exif_info, gen_info],
    )
    submit.click(
        _js="submit_postprocessing",
        fn=call_queue.wrap_gradio_gpu_call(submit_click, extra_outputs=[None, '']),
        inputs=[
            tab_index,
            extras_image,
            image_batch,
            extras_batch_input_dir,
            extras_batch_output_dir,
            show_extras_results,
            save_output,
            *script_inputs,
        ],
        outputs=[
            result_images,
            html_info,
            generation_info,
            html_log,
        ]
    )

    parameters_copypaste.add_paste_fields("extras", extras_image, None)

    extras_image.change(
        fn=scripts.scripts_postproc.image_changed,
        inputs=[], outputs=[]
    )
