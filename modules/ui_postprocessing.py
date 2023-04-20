import gradio as gr
from modules import scripts_postprocessing, scripts, shared, gfpgan_model, codeformer_model, ui_common, postprocessing, call_queue
import modules.generation_parameters_copypaste as parameters_copypaste
from modules.call_queue import wrap_gradio_gpu_call, wrap_queued_call, wrap_gradio_call
from modules.extras import run_pnginfo

def create_ui():
    tab_index = gr.State(value=0)

    with gr.Row().style(equal_height=False, variant='compact'):
        with gr.Column(variant='compact'):
            with gr.Tabs(elem_id="mode_extras"):

                with gr.TabItem('Process Image', elem_id="extras_single_tab") as tab_single:
                    extras_image = gr.Image(label="Source", source="upload", interactive=True, type="pil", elem_id="extras_image")

                with gr.TabItem('Process Batch', elem_id="extras_batch_process_tab") as tab_batch:
                    image_batch = gr.File(label="Batch Process", file_count="multiple", interactive=True, type="file", elem_id="extras_image_batch")

                with gr.TabItem('Process Folder', elem_id="extras_batch_directory_tab") as tab_batch_dir:
                    extras_batch_input_dir = gr.Textbox(label="Input directory", **shared.hide_dirs, placeholder="A directory on the same machine where the server is running.", elem_id="extras_batch_input_dir")
                    extras_batch_output_dir = gr.Textbox(label="Output directory", **shared.hide_dirs, placeholder="Leave blank to save images to the default path.", elem_id="extras_batch_output_dir")
                    show_extras_results = gr.Checkbox(label='Show result images', value=True, elem_id="extras_show_extras_results")

            with gr.Row():
                buttons = parameters_copypaste.create_buttons(["txt2img", "img2img", "inpaint"])

            submit = gr.Button('Generate', elem_id="extras_generate", variant='primary')

            script_inputs = scripts.scripts_postproc.setup_ui()

        with gr.Column():
            result_images, html_info_x, html_info, _html_log = ui_common.create_output_panel("extras", shared.opts.outdir_extras_samples)
            html_info = gr.HTML(elem_id="pnginfo_html_info")
            generation_info = gr.Textbox(elem_id="pnginfo_generation_info", label="Parameters")
            gr.HTML('Full metadata')
            html2_info = gr.HTML(elem_id="pnginfo_html2_info")

        for tabname, button in buttons.items():
            parameters_copypaste.register_paste_params_button(parameters_copypaste.ParamBinding(paste_button=button, tabname=tabname, source_text_component=generation_info, source_image_component=extras_image))

    tab_single.select(fn=lambda: 0, inputs=[], outputs=[tab_index])
    tab_batch.select(fn=lambda: 1, inputs=[], outputs=[tab_index])
    tab_batch_dir.select(fn=lambda: 2, inputs=[], outputs=[tab_index])

    extras_image.change(
        fn=wrap_gradio_call(run_pnginfo),
        inputs=[extras_image],
        outputs=[html_info, generation_info, html2_info],
    )

    submit.click(
        fn=call_queue.wrap_gradio_gpu_call(postprocessing.run_postprocessing, extra_outputs=[None, '']),
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
