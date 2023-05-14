import gradio as gr
from modules import scripts_postprocessing, scripts, shared, gfpgan_model, codeformer_model, ui_common, postprocessing, call_queue # pylint: disable=unused-import
import modules.generation_parameters_copypaste as parameters_copypaste
from modules.call_queue import wrap_gradio_gpu_call, wrap_queued_call, wrap_gradio_call # pylint: disable=unused-import
from modules.extras import run_pnginfo


def submit_click(tab_index, extras_image, image_batch, extras_batch_input_dir, extras_batch_output_dir, show_extras_results, *script_inputs):
    result_images, html_info_x, html_info = postprocessing.run_postprocessing(tab_index, extras_image, image_batch, extras_batch_input_dir, extras_batch_output_dir, show_extras_results, *script_inputs)
    if result_images is not None and len(result_images) > 0:
        _html_info, _generation_info, html_info_x = run_pnginfo(result_images[0])
    return result_images, html_info_x, html_info


def create_ui():
    tab_index = gr.State(value=0) # pylint: disable=abstract-class-instantiated

    with gr.Row().style(equal_height=False, variant='compact'):
        with gr.Column(variant='compact'):
            with gr.Tabs(elem_id="mode_extras"):

                with gr.TabItem('Single Image', id="single_image", elem_id="extras_single_tab") as tab_single:
                    extras_image = gr.Image(label="Source", source="upload", interactive=True, type="pil", elem_id="extras_image")

                with gr.TabItem('Process Batch', id="batch_process", elem_id="extras_batch_process_tab") as tab_batch:
                    image_batch = gr.Files(label="Batch Process", interactive=True, elem_id="extras_image_batch")

                with gr.TabItem('Process Folder', id="batch_from_directory", elem_id="extras_batch_directory_tab") as tab_batch_dir:
                    extras_batch_input_dir = gr.Textbox(label="Input directory", **shared.hide_dirs, placeholder="A directory on the same machine where the server is running.", elem_id="extras_batch_input_dir")
                    extras_batch_output_dir = gr.Textbox(label="Output directory", **shared.hide_dirs, placeholder="Leave blank to save images to the default path.", elem_id="extras_batch_output_dir")
                    show_extras_results = gr.Checkbox(label='Show result images', value=True, elem_id="extras_show_extras_results")

            with gr.Row():
                buttons = parameters_copypaste.create_buttons(["txt2img", "img2img", "inpaint"])

            # submit = gr.Button('Generate', elem_id="extras_generate", variant='primary') # TODO: add all

            script_inputs = scripts.scripts_postproc.setup_ui()

        with gr.Column():
            id_part = 'extras'
            with gr.Row(elem_id=f"{id_part}_generate_box", elem_classes="generate-box"):
                submit = gr.Button('Generate', elem_id=f"{id_part}_generate", variant='primary')
                interrupt = gr.Button('Stop', elem_id=f"{id_part}_interrupt", variant='secondary')
                skip = gr.Button('Skip', elem_id=f"{id_part}_skip", variant='secondary')
                skip.click(fn=lambda: shared.state.skip(), inputs=[], outputs=[])
                interrupt.click(fn=lambda: shared.state.interrupt(), inputs=[], outputs=[])

            result_images, html_info_x, html_info, _html_log = ui_common.create_output_panel("extras", shared.opts.outdir_extras_samples)
            html_info = gr.HTML(elem_id="pnginfo_html_info")
            generation_info = gr.Textbox(elem_id="pnginfo_generation_info", label="Parameters", visible=False)
            generation_info_pretty = gr.Textbox(elem_id="pnginfo_generation_info_pretty", label="Parameters")

            gr.HTML('Full metadata')
            html2_info = gr.HTML(elem_id="pnginfo_html2_info")

        for tabname, button in buttons.items():
            parameters_copypaste.register_paste_params_button(parameters_copypaste.ParamBinding(paste_button=button, tabname=tabname, source_text_component=generation_info, source_image_component=extras_image))

    def pretty_geninfo(generation_info: str):
        if generation_info is None:
            return ''
        sections = generation_info.split('Steps:')
        if len(sections) > 1:
            param = sections[0].strip() + '\nSteps:' + sections[1].strip().replace(', ', '\n')
            return param
        return generation_info

    tab_single.select(fn=lambda: 0, inputs=[], outputs=[tab_index])
    tab_batch.select(fn=lambda: 1, inputs=[], outputs=[tab_index])
    tab_batch_dir.select(fn=lambda: 2, inputs=[], outputs=[tab_index])

    generation_info.change(fn=pretty_geninfo, inputs=[generation_info], outputs=[generation_info_pretty])

    extras_image.change(
        fn=wrap_gradio_call(run_pnginfo),
        inputs=[extras_image],
        outputs=[html_info, generation_info, html2_info],
    )

    submit.click(
        fn=call_queue.wrap_gradio_gpu_call(submit_click, extra_outputs=[None, '']),
        inputs=[
            tab_index,
            extras_image,
            image_batch,
            extras_batch_input_dir,
            extras_batch_output_dir,
            show_extras_results,
            *script_inputs
        ],
        outputs=[
            result_images,
            html_info_x,
            html_info,
        ]
    )

    parameters_copypaste.add_paste_fields("extras", extras_image, None)

    extras_image.change(
        fn=scripts.scripts_postproc.image_changed,
        inputs=[], outputs=[]
    )
