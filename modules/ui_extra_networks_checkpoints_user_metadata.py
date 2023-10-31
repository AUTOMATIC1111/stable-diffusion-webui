import gradio as gr

from modules import ui_extra_networks_user_metadata, sd_vae, shared
from modules.ui_common import create_refresh_button


class CheckpointUserMetadataEditor(ui_extra_networks_user_metadata.UserMetadataEditor):
    def __init__(self, ui, tabname, page):
        super().__init__(ui, tabname, page)

        self.select_vae = None

    def save_user_metadata(self, name, desc, notes, vae):
        user_metadata = self.get_user_metadata(name)
        user_metadata["description"] = desc
        user_metadata["notes"] = notes
        user_metadata["vae"] = vae

        self.write_user_metadata(name, user_metadata)

    def update_vae(self, name):
        if name == shared.sd_model.sd_checkpoint_info.name_for_extra:
            sd_vae.reload_vae_weights()

    def put_values_into_components(self, name):
        user_metadata = self.get_user_metadata(name)
        values = super().put_values_into_components(name)

        return [
            *values[0:5],
            user_metadata.get('vae', ''),
        ]

    def create_editor(self):
        self.create_default_editor_elems()

        with gr.Row():
            self.select_vae = gr.Dropdown(choices=["Automatic", "None"] + list(sd_vae.vae_dict), value="None", label="Preferred VAE", elem_id="checpoint_edit_user_metadata_preferred_vae")
            create_refresh_button(self.select_vae, sd_vae.refresh_vae_list, lambda: {"choices": ["Automatic", "None"] + list(sd_vae.vae_dict)}, "checpoint_edit_user_metadata_refresh_preferred_vae")

        self.edit_notes = gr.TextArea(label='Notes', lines=4)

        self.create_default_buttons()

        viewed_components = [
            self.edit_name,
            self.edit_description,
            self.html_filedata,
            self.html_preview,
            self.edit_notes,
            self.select_vae,
        ]

        self.button_edit\
            .click(fn=self.put_values_into_components, inputs=[self.edit_name_input], outputs=viewed_components)\
            .then(fn=lambda: gr.update(visible=True), inputs=[], outputs=[self.box])

        edited_components = [
            self.edit_description,
            self.edit_notes,
            self.select_vae,
        ]

        self.setup_save_handler(self.button_save, self.save_user_metadata, edited_components)
        self.button_save.click(fn=self.update_vae, inputs=[self.edit_name_input])

