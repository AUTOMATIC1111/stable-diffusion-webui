import os
import gradio as gr
from modules import sd_hijack, script_callbacks, shared
from modules.ui_components import FormRow
from modules.ui_common import create_refresh_button
from modules.ui_sections import create_sampler_inputs
from modules.call_queue import wrap_gradio_gpu_call


def create_ui():
    from modules.textual_inversion import textual_inversion
    import modules.hypernetworks.ui
    dummy_component = gr.Label(visible=False)

    with gr.Row(elem_id="train_tab"):
        with gr.Column(elem_id='train_output_container', scale=1):
            train_output = gr.Text(elem_id="train_output", value="", show_label=False)
            gr.Gallery(label='Output', show_label=False, elem_id='train_gallery', columns=1)
            gr.HTML(elem_id="train_progress", value="")
            train_outcome = gr.HTML(elem_id="train_error", value="")

            with gr.Row(visible=True) as action_pp:
                process_run = gr.Button(value="Preprocess", variant='primary')
                process_stop = gr.Button("Stop")

            with gr.Row(visible=False) as action_ti:
                ti_train = gr.Button(value="Train embedding", variant='primary')
                ti_stop = gr.Button(value="Stop")

            with gr.Row(visible=False) as action_hn:
                hn_train = gr.Button(value="Train hypernetwork", variant='primary')
                hn_stop = gr.Button(value="Stop")

        with gr.Column(elem_id='train_input_container', scale=3):

            with gr.Tabs(elem_id="train_tabs"):
                def gr_show(visible=True):
                    return {"visible": visible, "__type__": "update"}

                def train_tab_change(tab):
                    if tab == 'ti':
                        return gr_show(False), gr_show(True), gr_show(False)
                    elif tab == 'hn':
                        return gr_show(False), gr_show(False), gr_show(True)
                    elif tab == 'pr':
                        return gr_show(True), gr_show(False), gr_show(False)
                    else:
                        return gr_show(False), gr_show(False), gr_show(False)

                ### preview tab

                with gr.Tab(label="Preview settings", id="train_preview_tab") as tab_preview:
                    tab_preview.select(fn=lambda: train_tab_change('pr'), inputs=[], outputs=[action_pp, action_ti, action_hn])
                    prompt = gr.Textbox(label="Prompt", value="", placeholder="Prompt to be used for previews", lines=2)
                    negative = gr.Textbox(label="Negative prompt", value="", placeholder="Negative prompt to be used for previews", lines=2)
                    steps, sampler_index = create_sampler_inputs('train', accordion=False)
                    cfg_scale = gr.Slider(minimum=0.0, maximum=30.0, step=0.1, label='CFG scale', value=6.0)
                    seed = gr.Number(label='Initial seed', value=-1)
                    with gr.Row():
                        width = gr.Slider(minimum=64, maximum=8192, step=8, label="Width", value=512)
                        height = gr.Slider(minimum=64, maximum=8192, step=8, label="Height", value=512)
                    txt2img_preview_params = [prompt, negative, steps, sampler_index, cfg_scale, seed, width, height]

                ### preprocess tab

                with gr.Tab(label="Preprocess images", id="preprocess_images") as tab_preprocess:
                    tab_preprocess.select(fn=lambda: train_tab_change('pp'), inputs=[], outputs=[action_pp, action_ti, action_hn])
                    process_src = gr.Textbox(label='Source directory')
                    process_dst = gr.Textbox(label='Destination directory')
                    with gr.Row():
                        process_width = gr.Slider(minimum=64, maximum=2048, step=8, label="Width", value=512)
                        process_height = gr.Slider(minimum=64, maximum=2048, step=8, label="Height", value=512)
                    preprocess_txt_action = gr.Dropdown(label='Existing caption text action', value="ignore", choices=["ignore", "copy", "prepend", "append"])

                    with gr.Box():
                        gr.HTML('<h2>Preprocessing steps</h2>')
                        process_keep_original_size = gr.Checkbox(label='Keep original size')
                        process_keep_channels = gr.Checkbox(label='Keep original image channels')
                        process_flip = gr.Checkbox(label='Create flipped copies')
                        process_split = gr.Checkbox(label='Split oversized images')
                        process_focal_crop = gr.Checkbox(label='Auto focal point crop')
                        process_multicrop = gr.Checkbox(label='Auto-sized crop')
                        process_caption_only = gr.Checkbox(label='Create captions only')
                        process_caption = gr.Checkbox(label='Create BLIP captions')
                        process_caption_deepbooru = gr.Checkbox(label='Create Deepbooru captions')

                    with gr.Row(visible=False) as process_split_extra_row:
                        process_split_threshold = gr.Slider(label='Split image threshold', value=0.5, minimum=0.0, maximum=1.0, step=0.05)
                        process_overlap_ratio = gr.Slider(label='Split image overlap ratio', value=0.2, minimum=0.0, maximum=0.9, step=0.05)

                    with gr.Row(visible=False) as process_focal_crop_row:
                        process_focal_crop_face_weight = gr.Slider(label='Focal point face weight', value=0.9, minimum=0.0, maximum=1.0, step=0.05)
                        process_focal_crop_entropy_weight = gr.Slider(label='Focal point entropy weight', value=0.15, minimum=0.0, maximum=1.0, step=0.05)
                        process_focal_crop_edges_weight = gr.Slider(label='Focal point edges weight', value=0.5, minimum=0.0, maximum=1.0, step=0.05)
                        process_focal_crop_debug = gr.Checkbox(label='Create debug image')

                    with gr.Column(visible=False) as process_multicrop_col:
                        gr.HTML('<h2>Each image is center-cropped with an automatically chosen width and height</h2>')
                        with gr.Row():
                            process_multicrop_mindim = gr.Slider(minimum=64, maximum=2048, step=8, label="Dimension lower bound", value=384)
                            process_multicrop_maxdim = gr.Slider(minimum=64, maximum=2048, step=8, label="Dimension upper bound", value=768)
                        with gr.Row():
                            process_multicrop_minarea = gr.Slider(minimum=64*64, maximum=2048*2048, step=1, label="Area lower bound", value=64*64)
                            process_multicrop_maxarea = gr.Slider(minimum=64*64, maximum=2048*2048, step=1, label="Area upper bound", value=640*640)
                        with gr.Row():
                            process_multicrop_objective = gr.Radio(["Maximize area", "Minimize error"], value="Maximize area", label="Resizing objective")
                            process_multicrop_threshold = gr.Slider(minimum=0, maximum=1, step=0.01, label="Error threshold", value=0.1)

                    from modules.textual_inversion import ui
                    process_split.change(fn=lambda show: gr_show(show), inputs=[process_split], outputs=[process_split_extra_row])
                    process_focal_crop.change(fn=lambda show: gr_show(show), inputs=[process_focal_crop], outputs=[process_focal_crop_row])
                    process_multicrop.change(fn=lambda show: gr_show(show), inputs=[process_multicrop], outputs=[process_multicrop_col])
                    process_stop.click(fn=lambda: shared.state.interrupt(), inputs=[], outputs=[])
                    process_run.click(
                        fn=wrap_gradio_gpu_call(ui.preprocess, extra_outputs=[gr.update()]),
                        _js="startTrainMonitor",
                        inputs=[
                            dummy_component,
                            process_src,
                            process_dst,
                            process_width,
                            process_height,
                            preprocess_txt_action,
                            process_keep_original_size,
                            process_keep_channels,
                            process_flip,
                            process_split,
                            process_caption_only,
                            process_caption,
                            process_caption_deepbooru,
                            process_split_threshold,
                            process_overlap_ratio,
                            process_focal_crop,
                            process_focal_crop_face_weight,
                            process_focal_crop_entropy_weight,
                            process_focal_crop_edges_weight,
                            process_focal_crop_debug,
                            process_multicrop,
                            process_multicrop_mindim,
                            process_multicrop_maxdim,
                            process_multicrop_minarea,
                            process_multicrop_maxarea,
                            process_multicrop_objective,
                            process_multicrop_threshold,
                        ],
                        outputs=[
                            train_output,
                            train_outcome,
                        ],
                    )

                ### train embedding tab
                with gr.Tab(label="Train embedding", id="train_embedding_tab") as tab_ti:
                    tab_ti.select(fn=lambda: train_tab_change('ti'), inputs=[], outputs=[action_pp, action_ti, action_hn])
                    def get_textual_inversion_template_names():
                        return sorted(textual_inversion.textual_inversion_templates)

                    gr.HTML('<h2>Select existing embedding to continue training or create a new one</h2>')
                    with FormRow():
                        with gr.Column():
                            with gr.Row():
                                ti_name = gr.Dropdown(label='Select embedding', choices=sorted(sd_hijack.model_hijack.embedding_db.word_embeddings.keys()))
                                create_refresh_button(ti_name, sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings, lambda: {"choices": sorted(sd_hijack.model_hijack.embedding_db.word_embeddings.keys())}, "refresh_train_embedding_name")
                        with gr.Column():
                            ti_new_name = gr.Textbox(label="Create emebedding")
                            ti_init_text = gr.Textbox(label="Initialization text", value="*")
                            ti_vectors = gr.Slider(label="Number of vectors per token", minimum=1, maximum=75, step=1, value=1)
                            ti_overwrite = gr.Checkbox(value=False, label="Overwrite Old Embedding")
                            with gr.Row():
                                ti_create = gr.Button(value="Create embedding", variant='secondary')

                    with gr.Box():
                        gr.HTML('<h2>Training parameters</h2>')
                        ti_learn_rate = gr.Textbox(label='Embedding Learning rate', placeholder="Embedding Learning rate", value="0.005")
                        with FormRow():
                            ti_clip_grad_mode = gr.Dropdown(value="disabled", label="Gradient Clipping", choices=["disabled", "value", "norm"])
                            ti_clip_grad_value = gr.Number(label="Gradient clip value", value=0.1)
                        ti_batch_size = gr.Number(label='Batch size', value=1, precision=0)
                        ti_gradient_step = gr.Number(label='Gradient accumulation steps', value=1, precision=0)
                        ti_steps = gr.Number(label='Max steps', value=1000, precision=0)

                    with gr.Box():
                        gr.HTML('<h2>Training images</h2>')
                        ti_dataset_directory = gr.Textbox(label='Dataset directory', placeholder="Path to directory with input images")
                        with FormRow():
                            ti_varsize = gr.Checkbox(label="Do not resize images", value=False)
                            ti_width = gr.Slider(minimum=64, maximum=2048, step=8, label="Width", value=512)
                            ti_height = gr.Slider(minimum=64, maximum=2048, step=8, label="Height", value=512)
                        ti_use_weight = gr.Checkbox(label="Use PNG alpha channel as loss weight", value=False)

                    with gr.Box():
                        gr.HTML('<h2>Dataset processing</h2>')
                        with FormRow():
                            ti_template = gr.Dropdown(label='Prompt template', value="style_filewords.txt", choices=get_textual_inversion_template_names())
                            create_refresh_button(ti_template, textual_inversion.list_textual_inversion_templates, lambda: {"choices": get_textual_inversion_template_names()}, "refrsh_train_template_file")
                        ti_shuffle = gr.Checkbox(label="Shuffle tags", value=False)
                        ti_tag_drop_out = gr.Slider(minimum=0, maximum=1, step=0.1, label="Drop out tags when creating prompts", value=0)
                        ti_latent_sampling_method = gr.Radio(label='Choose latent sampling method', value="once", choices=['once', 'deterministic', 'random'])

                    with gr.Box():
                        gr.HTML('<h2>Training outputs</h2>')
                        with FormRow():
                            ti_create_every = gr.Number(label='Create interim images', value=500, precision=0)
                            ti_save_every = gr.Number(label='Create interim embeddings', value=500, precision=0)
                        ti_save_image_with_stored_embedding = gr.Checkbox(label='Save images with embedding in PNG chunks', value=True)
                        ti_preview_from_txt2img = gr.Checkbox(label='Use current settings for previews', value=False)
                        ti_log_directory = gr.Textbox(label='Log directory', placeholder="Defaults to train/log/embedding", value="")

                    ti_stop.click(fn=lambda: shared.state.interrupt(), inputs=[], outputs=[])

                    ti_create.click(
                        fn=modules.textual_inversion.ui.create_embedding,
                        inputs=[
                            ti_new_name,
                            ti_init_text,
                            ti_vectors,
                            ti_overwrite,
                        ],
                        outputs=[
                            ti_name,
                            train_output,
                            train_outcome,
                        ]
                    )

                    ti_train.click(
                        fn=wrap_gradio_gpu_call(modules.textual_inversion.ui.train_embedding, extra_outputs=[gr.update()]),
                        _js="startTrainMonitor",
                        inputs=[
                            dummy_component,
                            ti_name,
                            ti_learn_rate,
                            ti_batch_size,
                            ti_gradient_step,
                            ti_dataset_directory,
                            ti_log_directory,
                            ti_width,
                            ti_height,
                            ti_varsize,
                            ti_steps,
                            ti_clip_grad_mode,
                            ti_clip_grad_value,
                            ti_shuffle,
                            ti_tag_drop_out,
                            ti_latent_sampling_method,
                            ti_use_weight,
                            ti_create_every,
                            ti_save_every,
                            ti_template,
                            ti_save_image_with_stored_embedding,
                            ti_preview_from_txt2img,
                            *txt2img_preview_params,
                        ],
                        outputs=[
                            train_output,
                            train_outcome,
                        ]
                    )

                ### train hypernetwork tab

                with gr.Tab(label="Train hypernetwork", id="train_hypernetwork_tab") as tab_hn:
                    tab_hn.select(fn=lambda: train_tab_change('hn'), inputs=[], outputs=[action_pp, action_ti, action_hn])
                    gr.HTML('<h2>Select existing hypernetwork to continue training or create a new one</h2>')
                    with FormRow():
                        with gr.Column():
                            with FormRow():
                                hn_name = gr.Dropdown(label='Hypernetwork', choices=sorted(shared.hypernetworks))
                                create_refresh_button(hn_name, shared.reload_hypernetworks, lambda: {"choices": sorted(shared.hypernetworks)}, "refresh_train_hypernetwork_name")
                        with gr.Column():
                            hn_new_name = gr.Textbox(label="Name")
                            hn_new_sizes = gr.CheckboxGroup(label="Modules", value=["768", "320", "640", "1280"], choices=["768", "1024", "320", "640", "1280"])
                            hn_new_layer_structure = gr.Textbox("1, 2, 1", label="Enter hypernetwork layer structure", placeholder="1st and last digit must be 1. ex:'1, 2, 1'")
                            with gr.Row():
                                hn_new_activation_func = gr.Dropdown(value="linear", label="Select activation function of hypernetwork", choices=modules.hypernetworks.ui.keys)
                                hn_new_initialization_option = gr.Dropdown(value = "Normal", label="Select Layer weights initialization", choices=["Normal", "KaimingUniform", "KaimingNormal", "XavierUniform", "XavierNormal"])
                            hn_new_add_layer_norm = gr.Checkbox(label="Add layer normalization")
                            hn_new_use_dropout = gr.Checkbox(label="Use dropout")
                            hn_new_dropout_structure = gr.Textbox("0, 0, 0", label="Enter hypernetwork Dropout structure", placeholder="1st and last digit must be 0 and values should be between 0 and 1. ex:'0, 0.01, 0'")
                            hn_overwrite = gr.Checkbox(value=False, label="Overwrite Old Hypernetwork")
                            with gr.Row():
                                hn_create = gr.Button(value="Create hypernetwork", variant='secondary')

                    with gr.Box():
                        gr.HTML('<h2>Training parameters</h2>')
                        hn_learn_rate = gr.Textbox(label='Hypernetwork Learning rate', placeholder="Hypernetwork Learning rate", value="0.00001")
                        with FormRow():
                            hn_clip_grad_mode = gr.Dropdown(value="disabled", label="Gradient Clipping", choices=["disabled", "value", "norm"])
                            hn_clip_grad_value = gr.Number(label="Gradient clip value", value=0.1)
                        hn_batch_size = gr.Number(label='Batch size', value=1, precision=0)
                        hn_gradient_step = gr.Number(label='Gradient accumulation steps', value=1, precision=0)
                        hn_steps = gr.Number(label='Max steps', value=1000, precision=0)

                    with gr.Box():
                        gr.HTML('<h2>Training images</h2>')
                        hn_dataset_directory = gr.Textbox(label='Dataset directory', placeholder="Path to directory with input images")
                        with FormRow():
                            hn_varsize = gr.Checkbox(label="Do not resize images", value=False)
                            hn_width = gr.Slider(minimum=64, maximum=2048, step=8, label="Width", value=512)
                            hn_height = gr.Slider(minimum=64, maximum=2048, step=8, label="Height", value=512)
                        hn_use_weight = gr.Checkbox(label="Use PNG alpha channel as loss weight", value=False)

                    with gr.Box():
                        gr.HTML('<h2>Dataset processing</h2>')
                        with FormRow():
                            hn_template = gr.Dropdown(label='Prompt template', value="style_filewords.txt", choices=get_textual_inversion_template_names())
                            create_refresh_button(hn_template, textual_inversion.list_textual_inversion_templates, lambda: {"choices": get_textual_inversion_template_names()}, "refrsh_train_template_file")
                        hn_shuffle_tags = gr.Checkbox(label="Shuffle tags by ',' when creating prompts.", value=False)
                        hn_tag_drop_out = gr.Slider(minimum=0, maximum=1, step=0.1, label="Drop out tags when creating prompts", value=0)
                        hn_latent_sampling_method = gr.Radio(label='Choose latent sampling method', value="once", choices=['once', 'deterministic', 'random'])

                    with gr.Box():
                        gr.HTML('<h2>Training outputs</h2>')
                        with FormRow():
                            hn_create_every = gr.Number(label='Create interim images', value=500, precision=0)
                            hn_save_every = gr.Number(label='Create interim hypernetworks', value=500, precision=0)
                        hn_preview_from_txt2img = gr.Checkbox(label='Use current settings for previews', value=False)
                        hn_log_directory = gr.Textbox(label='Log directory', placeholder="Path to directory where to write outputs", value=f"{os.path.join('cmd_opts.data_dir', 'train/log/embeddings')}")

                    hn_stop.click(fn=lambda: shared.state.interrupt(), inputs=[], outputs=[])

                    hn_create.click(
                        fn=modules.hypernetworks.ui.create_hypernetwork,
                        inputs=[
                            hn_new_name,
                            hn_new_sizes,
                            hn_overwrite,
                            hn_new_layer_structure,
                            hn_new_activation_func,
                            hn_new_initialization_option,
                            hn_new_add_layer_norm,
                            hn_new_use_dropout,
                            hn_new_dropout_structure
                        ],
                        outputs=[
                            hn_name,
                            train_output,
                            train_outcome,
                        ]
                    )

                    hn_train.click(
                        fn=wrap_gradio_gpu_call(modules.hypernetworks.ui.train_hypernetwork, extra_outputs=[gr.update()]),
                        _js="startTrainMonitor",
                        inputs=[
                            dummy_component,
                            hn_name,
                            hn_learn_rate,
                            hn_batch_size,
                            hn_gradient_step,
                            hn_dataset_directory,
                            hn_log_directory,
                            hn_width,
                            hn_height,
                            hn_varsize,
                            hn_steps,
                            hn_clip_grad_mode,
                            hn_clip_grad_value,
                            hn_shuffle_tags,
                            hn_tag_drop_out,
                            hn_latent_sampling_method,
                            hn_use_weight,
                            hn_create_every,
                            hn_save_every,
                            hn_template,
                            hn_preview_from_txt2img,
                            *txt2img_preview_params,
                        ],
                        outputs=[
                            train_output,
                            train_outcome,
                        ]
                    )

            params = script_callbacks.UiTrainTabParams(txt2img_preview_params)
            script_callbacks.ui_train_tabs_callback(params)
