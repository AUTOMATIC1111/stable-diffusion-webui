import os
import json
from datetime import datetime
import gradio as gr
from modules import sd_models, sd_vae, extras
from modules.ui_components import FormRow, ToolButton
from modules.ui_common import create_refresh_button
from modules.call_queue import wrap_gradio_gpu_call
from modules.shared import opts, log
import modules.errors


def create_ui():
    dummy_component = gr.Label(visible=False)

    with gr.Row(id="models_tab", elem_id="models_tab"):
        with gr.Column(elem_id='models_output_container', scale=1):
            # models_output = gr.Text(elem_id="models_output", value="", show_label=False)
            gr.HTML(elem_id="models_progress", value="")
            models_outcome = gr.HTML(elem_id="models_error", value="")

        with gr.Column(elem_id='models_input_container', scale=3):

            def gr_show(visible=True):
                return {"visible": visible, "__type__": "update"}

            with gr.Tab(label="Convert"):
                with gr.Row():
                    model_name = gr.Dropdown(sd_models.checkpoint_tiles(), label="Original model")
                    create_refresh_button(model_name, sd_models.list_models, lambda: {"choices": sd_models.checkpoint_tiles()}, "refresh_checkpoint_Z")
                with gr.Row():
                    custom_name = gr.Textbox(label="New model name")
                with gr.Row():
                    precision = gr.Radio(choices=["fp32", "fp16", "bf16"], value="fp16", label="Model precision")
                    m_type = gr.Radio(choices=["disabled", "no-ema", "ema-only"], value="disabled", label="Model pruning methods")
                with gr.Row():
                    checkpoint_formats = gr.CheckboxGroup(choices=["ckpt", "safetensors"], value=["safetensors"], label="Model Format")
                with gr.Row():
                    show_extra_options = gr.Checkbox(label="Show extra options", value=False)
                    fix_clip = gr.Checkbox(label="Fix clip", value=False)
                with gr.Row(visible=False) as extra_options:
                    specific_part_conv = ["copy", "convert", "delete"]
                    unet_conv = gr.Dropdown(specific_part_conv, value="convert", label="unet")
                    text_encoder_conv = gr.Dropdown(specific_part_conv, value="convert", label="text encoder")
                    vae_conv = gr.Dropdown(specific_part_conv, value="convert", label="vae")
                    others_conv = gr.Dropdown(specific_part_conv, value="convert", label="others")

                show_extra_options.change(fn=lambda x: gr_show(x), inputs=[show_extra_options], outputs=[extra_options])

                model_converter_convert = gr.Button(label="Convert", variant='primary')
                model_converter_convert.click(
                    fn=extras.run_modelconvert,
                    inputs=[
                        model_name,
                        checkpoint_formats,
                        precision, m_type, custom_name,
                        unet_conv,
                        text_encoder_conv,
                        vae_conv,
                        others_conv,
                        fix_clip
                    ],
                    outputs=[models_outcome]
                )

            with gr.Tab(label="Merge"):
                with gr.Row().style(equal_height=False):
                    with gr.Column(variant='compact'):
                        with FormRow():
                            custom_name = gr.Textbox(label="New model name")
                        with FormRow():
                            def sd_model_choices():
                                return ['None'] + sd_models.checkpoint_tiles()
                            primary_model_name = gr.Dropdown(sd_model_choices(), label="Primary model", value="None")
                            create_refresh_button(primary_model_name, sd_models.list_models, lambda: {"choices": sd_model_choices()}, "refresh_checkpoint_A")
                            secondary_model_name = gr.Dropdown(sd_model_choices(), label="Secondary model", value="None")
                            create_refresh_button(secondary_model_name, sd_models.list_models, lambda: {"choices": sd_model_choices()}, "refresh_checkpoint_B")
                            tertiary_model_name = gr.Dropdown(sd_model_choices(), label="Tertiary model", value="None")
                            create_refresh_button(tertiary_model_name, sd_models.list_models, lambda: {"choices": sd_model_choices()}, "refresh_checkpoint_C")
                        with FormRow():
                            interp_method = gr.Radio(choices=["No interpolation", "Weighted sum", "Add difference"], value="Weighted sum", label="Interpolation Method")
                            interp_amount = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, label='Interpolation ratio from Primary to Secondary', value=0.5)
                        with FormRow():
                            checkpoint_format = gr.Radio(choices=["ckpt", "safetensors"], value="safetensors", label="Model format")
                        with gr.Box():
                            save_as_half = gr.Radio(choices=["fp16", "fp32"], value="fp16", label="Model precision", type="index")
                        with FormRow():
                            config_source = gr.Radio(choices=["Primary", "Secondary", "Tertiary", "None"], value="Primary", label="Model configuration", type="index")
                        with FormRow():
                            bake_in_vae = gr.Dropdown(choices=["None"] + list(sd_vae.vae_dict), value="None", label="Bake in VAE")
                            create_refresh_button(bake_in_vae, sd_vae.refresh_vae_list, lambda: {"choices": ["None"] + list(sd_vae.vae_dict)}, "modelmerger_refresh_bake_in_vae")
                        with FormRow():
                            discard_weights = gr.Textbox(value="", label="Discard weights with matching name")
                        with FormRow():
                            save_metadata = gr.Checkbox(value=True, label="Save metadata")
                        with gr.Row():
                            modelmerger_merge = gr.Button(value="Merge", variant='primary')

                def modelmerger(*args):
                    try:
                        results = extras.run_modelmerger(*args)
                    except Exception as e:
                        modules.errors.display(e, 'model merge')
                        sd_models.list_models()  # to remove the potentially missing models from the list
                        return [*[gr.Dropdown.update(choices=sd_models.checkpoint_tiles()) for _ in range(4)], f"Error merging checkpoints: {e}"]
                    return results

                modelmerger_merge.click(
                    fn=wrap_gradio_gpu_call(modelmerger, extra_outputs=lambda: [gr.update() for _ in range(4)]),
                    _js='modelmerger',
                    inputs=[
                        dummy_component,
                        primary_model_name,
                        secondary_model_name,
                        tertiary_model_name,
                        interp_method,
                        interp_amount,
                        save_as_half,
                        custom_name,
                        checkpoint_format,
                        config_source,
                        bake_in_vae,
                        discard_weights,
                        save_metadata,
                    ],
                    outputs=[
                        primary_model_name,
                        secondary_model_name,
                        tertiary_model_name,
                        dummy_component,
                        models_outcome,
                    ]
                )

            with gr.Tab(label="Validate"):
                model_headers = ['name', 'type', 'filename', 'hash', 'added', 'size', 'metadata']
                model_data = []

                with gr.Row():
                    model_list_btn = gr.Button(value="List model details", variant='primary')
                    model_checkhash_btn = gr.Button(value="Calculate hash for all models (may take a long time)", variant='primary')
                    model_checkhash_btn.click(fn=sd_models.update_model_hashes, inputs=[], outputs=[models_outcome])
                with gr.Row():
                    model_table = gr.DataFrame(model_data, label = 'Model data', show_label = True, interactive = False, wrap = True, overflow_row_behaviour = 'paginate', max_rows = 10, headers = model_headers)

                def list_models():
                    total_size = 0
                    for m in sd_models.checkpoints_list.values():
                        txt = ''
                        try:
                            stat = os.stat(m.filename)
                            m_name = m.name.replace('.ckpt', '').replace('.safetensors', '')
                            m_type = 'ckpt' if m.name.endswith('.ckpt') else 'safe'
                            m_meta = len(json.dumps(m.metadata)) - 2
                            m_size = round(stat.st_size / 1024 / 1024 / 1024, 3)
                            m_time = datetime.fromtimestamp(stat.st_mtime)
                            model_data.append([m_name, m_type, m.filename, m.hash, m_time, m_size, m_meta])
                            total_size += stat.st_size
                        except Exception as e:
                            txt += f"Error: {m.name} {e}<br>"
                        txt += f"Model list enumerated {len(sd_models.checkpoints_list.keys())} models in {round(total_size / 1024 / 1024 / 1024, 3)} GB<br>"
                    return model_data, txt

                model_list_btn.click(fn=list_models, inputs=[], outputs=[model_table, models_outcome])

            with gr.Tab(label="Huggingface"):
                data = []
                os.environ.setdefault('HF_HUB_DISABLE_EXPERIMENTAL_WARNING', '1')
                os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')
                os.environ.setdefault('HF_HUB_DISABLE_IMPLICIT_TOKEN', '1')
                os.environ.setdefault('HUGGINGFACE_HUB_VERBOSITY', 'warning')

                def hf_search(keyword):
                    import huggingface_hub as hf
                    hf_api = hf.HfApi()
                    model_filter = hf.ModelFilter(
                        model_name=keyword,
                        task='text-to-image',
                        library=['diffusers'],
                    )
                    models = hf_api.list_models(filter=model_filter, full=True, limit=50, sort="downloads", direction=-1)
                    data.clear()
                    for model in models:
                        tags = [t for t in model.tags if not t.startswith('diffusers') and not t.startswith('license') and not t.startswith('arxiv') and len(t) > 2]
                        data.append([model.modelId, model.pipeline_tag, tags, model.downloads, model.lastModified, f'https://huggingface.co/{model.modelId}'])
                    return data

                def hf_select(evt: gr.SelectData, data):
                    return data[evt.index[0]][0]

                def hf_download_model(hub_id: str, token, variant, revision, mirror):
                    from modules.modelloader import download_diffusers_model
                    try:
                        download_diffusers_model(hub_id, cache_dir=opts.diffusers_dir, token=token, variant=variant, revision=revision, mirror=mirror)
                    except Exception as e:
                        log.error(f"Diffuser model downloaded error: model={hub_id} {e}")
                        return f"Diffuser model downloaded error: model={hub_id} {e}"
                    from modules.sd_models import list_models # pylint: disable=W0621
                    list_models()
                    log.info(f"Diffuser model downloaded: model={hub_id}")
                    return f'Diffuser model downloaded: model={hub_id}'

                with gr.Column(scale=6):
                    with gr.Row():
                        hf_search_text = gr.Textbox('', label = 'Seach models', placeholder='search huggingface models')
                        hf_search_btn = ToolButton(value="🔍", label="Search")
                    with gr.Row():
                        with gr.Column(scale=2):
                            with gr.Row():
                                hf_selected = gr.Textbox('', label = 'Select model', placeholder='select model from search results or enter model name manually')
                        with gr.Column(scale=1):
                            with gr.Row():
                                hf_variant = gr.Textbox(opts.cuda_dtype.lower(), label = 'Specify model variant', placeholder='')
                                hf_revision = gr.Textbox('', label = 'Specify model revision', placeholder='')
                    with gr.Row():
                        hf_token = gr.Textbox('', label = 'Huggingface token', placeholder='optional access token for private or gated models')
                        hf_mirror = gr.Textbox('', label = 'Huggingface mirror', placeholder='optional mirror site for downloads')
                with gr.Column(scale=1):
                    hf_download_model_btn = gr.Button(value="Download model", variant='primary')

                with gr.Row():
                    hf_headers = ['Name', 'Pipeline', 'Tags', 'Downloads', 'Updated', 'URL']
                    hf_types = ['str', 'str', 'str', 'number', 'date', 'markdown']
                    hf_results = gr.DataFrame([], label = 'Search results', show_label = True, interactive = False, wrap = True, overflow_row_behaviour = 'paginate', max_rows = 10, headers = hf_headers, datatype = hf_types, type='array')

                hf_search_text.submit(fn=hf_search, inputs=[hf_search_text], outputs=[hf_results])
                hf_search_btn.click(fn=hf_search, inputs=[hf_search_text], outputs=[hf_results])
                hf_results.select(fn=hf_select, inputs=[hf_results], outputs=[hf_selected])
                hf_download_model_btn.click(fn=hf_download_model, inputs=[hf_selected, hf_token, hf_variant, hf_revision, hf_mirror], outputs=[models_outcome])

            # with gr.Tab(label="CivitAI"):
            #    pass
