import gradio as gr
from modules import scripts_postprocessing, scripts, shared, gfpgan_model, codeformer_model, ui_common, postprocessing, call_queue
import modules.generation_parameters_copypaste as parameters_copypaste

def create_ui():
    tab_index = gr.State(value=0)
    gr.Row(elem_id="extras_2img_prompt_image", visible=False)
    with gr.Row(): 
        with gr.Column(elem_id="extras_2img_results"):
            result_images, html_info_x, html_info, html_log = ui_common.create_output_panel("extras_2img", shared.opts.outdir_extras_samples)        
        gr.Row(elem_id="extras_2img_splitter")           
        with gr.Column(variant='panel', elem_id="extras_2img_settings"):                            
            submit = gr.Button('Upscale', elem_id="extras_generate", variant='primary')        
            with gr.Column(elem_id="extras_2img_settings_scroll"):   
                with gr.Accordion("Image Source", elem_id="extras_accordion", open=True):
                    with gr.Tabs(elem_id="mode_extras"):
                        with gr.TabItem('Single Image', elem_id="extras_single_tab") as tab_single:
                            extras_image = gr.Image(label="Source", source="upload", interactive=True, type="pil", elem_id="extras_image")

                        with gr.TabItem('Batch Process', elem_id="extras_batch_process_tab") as tab_batch:
                            image_batch = gr.File(label="Batch Process", file_count="multiple", interactive=True, type="file", elem_id="extras_image_batch")

                        with gr.TabItem('Batch from Directory', elem_id="extras_batch_directory_tab") as tab_batch_dir:
                            extras_batch_input_dir = gr.Textbox(label="Input directory", **shared.hide_dirs, placeholder="A directory on the same machine where the server is running.", elem_id="extras_batch_input_dir")
                            extras_batch_output_dir = gr.Textbox(label="Output directory", **shared.hide_dirs, placeholder="Leave blank to save images to the default path.", elem_id="extras_batch_output_dir")
                            show_extras_results = gr.Checkbox(label='Show result images', value=True, elem_id="extras_show_extras_results")

                script_inputs = scripts.scripts_postproc.setup_ui()

    tab_single.select(fn=lambda: 0, inputs=[], outputs=[tab_index])
    tab_batch.select(fn=lambda: 1, inputs=[], outputs=[tab_index])
    tab_batch_dir.select(fn=lambda: 2, inputs=[], outputs=[tab_index])

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
