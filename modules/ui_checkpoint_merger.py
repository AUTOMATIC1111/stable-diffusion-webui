
import gradio as gr

from modules import sd_models, sd_vae, errors, extras, call_queue
from modules.ui_components import FormRow
from modules.ui_common import create_refresh_button


def update_interp_description(value):
    interp_description_css = "<p style='margin-bottom: 2.5em'>{}</p>"
    interp_descriptions = {
        "No interpolation": interp_description_css.format("No interpolation will be used. Requires one model; A. Allows for format conversion and VAE baking."),
        "Weighted sum": interp_description_css.format("A weighted sum will be used for interpolation. Requires two models; A and B. The result is calculated as A * (1 - M) + B * M"),
        "Add difference": interp_description_css.format("The difference between the last two models will be added to the first. Requires three models; A, B and C. The result is calculated as A + (B - C) * M")
    }
    return interp_descriptions[value]


def modelmerger(*args):
    try:
        results = extras.run_modelmerger(*args)
    except Exception as e:
        errors.report("Error loading/saving model file", exc_info=True)
        sd_models.list_models()  # to remove the potentially missing models from the list
        return [*[gr.Dropdown.update(choices=sd_models.checkpoint_tiles()) for _ in range(4)], f"Error merging checkpoints: {e}"]
    return results


class UiCheckpointMerger:
    def __init__(self):
        with gr.Blocks(analytics_enabled=False) as modelmerger_interface:
            with gr.Row(equal_height=False):
                with gr.Column(variant='compact'):
                    self.interp_description = gr.HTML(value=update_interp_description("Weighted sum"), elem_id="modelmerger_interp_description")

                    with FormRow(elem_id="modelmerger_models"):
                        self.primary_model_name = gr.Dropdown(sd_models.checkpoint_tiles(), elem_id="modelmerger_primary_model_name", label="Primary model (A)")
                        create_refresh_button(self.primary_model_name, sd_models.list_models, lambda: {"choices": sd_models.checkpoint_tiles()}, "refresh_checkpoint_A")

                        self.secondary_model_name = gr.Dropdown(sd_models.checkpoint_tiles(), elem_id="modelmerger_secondary_model_name", label="Secondary model (B)")
                        create_refresh_button(self.secondary_model_name, sd_models.list_models, lambda: {"choices": sd_models.checkpoint_tiles()}, "refresh_checkpoint_B")

                        self.tertiary_model_name = gr.Dropdown(sd_models.checkpoint_tiles(), elem_id="modelmerger_tertiary_model_name", label="Tertiary model (C)")
                        create_refresh_button(self.tertiary_model_name, sd_models.list_models, lambda: {"choices": sd_models.checkpoint_tiles()}, "refresh_checkpoint_C")

                    self.custom_name = gr.Textbox(label="Custom Name (Optional)", elem_id="modelmerger_custom_name")
                    self.interp_amount = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, label='Multiplier (M) - set to 0 to get model A', value=0.3, elem_id="modelmerger_interp_amount")
                    self.interp_method = gr.Radio(choices=["No interpolation", "Weighted sum", "Add difference"], value="Weighted sum", label="Interpolation Method", elem_id="modelmerger_interp_method")
                    self.interp_method.change(fn=update_interp_description, inputs=[self.interp_method], outputs=[self.interp_description])

                    with FormRow():
                        self.checkpoint_format = gr.Radio(choices=["ckpt", "safetensors"], value="safetensors", label="Checkpoint format", elem_id="modelmerger_checkpoint_format")
                        self.save_as_half = gr.Checkbox(value=False, label="Save as float16", elem_id="modelmerger_save_as_half")

                    with FormRow():
                        with gr.Column():
                            self.config_source = gr.Radio(choices=["A, B or C", "B", "C", "Don't"], value="A, B or C", label="Copy config from", type="index", elem_id="modelmerger_config_method")

                        with gr.Column():
                            with FormRow():
                                self.bake_in_vae = gr.Dropdown(choices=["None"] + list(sd_vae.vae_dict), value="None", label="Bake in VAE", elem_id="modelmerger_bake_in_vae")
                                create_refresh_button(self.bake_in_vae, sd_vae.refresh_vae_list, lambda: {"choices": ["None"] + list(sd_vae.vae_dict)}, "modelmerger_refresh_bake_in_vae")

                    with FormRow():
                        self.discard_weights = gr.Textbox(value="", label="Discard weights with matching name", elem_id="modelmerger_discard_weights")

                    with gr.Accordion("Metadata", open=False) as metadata_editor:
                        with FormRow():
                            self.save_metadata = gr.Checkbox(value=True, label="Save metadata", elem_id="modelmerger_save_metadata")
                            self.add_merge_recipe = gr.Checkbox(value=True, label="Add merge recipe metadata", elem_id="modelmerger_add_recipe")
                            self.copy_metadata_fields = gr.Checkbox(value=True, label="Copy metadata from merged models", elem_id="modelmerger_copy_metadata")

                        self.metadata_json = gr.TextArea('{}', label="Metadata in JSON format")
                        self.read_metadata = gr.Button("Read metadata from selected checkpoints")

                    with FormRow():
                        self.modelmerger_merge = gr.Button(elem_id="modelmerger_merge", value="Merge", variant='primary')

                with gr.Column(variant='compact', elem_id="modelmerger_results_container"):
                    with gr.Group(elem_id="modelmerger_results_panel"):
                        self.modelmerger_result = gr.HTML(elem_id="modelmerger_result", show_label=False)

        self.metadata_editor = metadata_editor
        self.blocks = modelmerger_interface

    def setup_ui(self, dummy_component, sd_model_checkpoint_component):
        self.checkpoint_format.change(lambda fmt: gr.update(visible=fmt == 'safetensors'), inputs=[self.checkpoint_format], outputs=[self.metadata_editor], show_progress=False)

        self.read_metadata.click(extras.read_metadata, inputs=[self.primary_model_name, self.secondary_model_name, self.tertiary_model_name], outputs=[self.metadata_json])

        self.modelmerger_merge.click(fn=lambda: '', inputs=[], outputs=[self.modelmerger_result])
        self.modelmerger_merge.click(
            fn=call_queue.wrap_gradio_gpu_call(modelmerger, extra_outputs=lambda: [gr.update() for _ in range(4)]),
            _js='modelmerger',
            inputs=[
                dummy_component,
                self.primary_model_name,
                self.secondary_model_name,
                self.tertiary_model_name,
                self.interp_method,
                self.interp_amount,
                self.save_as_half,
                self.custom_name,
                self.checkpoint_format,
                self.config_source,
                self.bake_in_vae,
                self.discard_weights,
                self.save_metadata,
                self.add_merge_recipe,
                self.copy_metadata_fields,
                self.metadata_json,
            ],
            outputs=[
                self.primary_model_name,
                self.secondary_model_name,
                self.tertiary_model_name,
                sd_model_checkpoint_component,
                self.modelmerger_result,
            ]
        )

        # Required as a workaround for change() event not triggering when loading values from ui-config.json
        self.interp_description.value = update_interp_description(self.interp_method.value)

