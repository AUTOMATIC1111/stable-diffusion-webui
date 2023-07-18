import datetime
import html
import json
import os.path

import gradio as gr

from modules import generation_parameters_copypaste, images, sysinfo, errors


class UserMetadataEditor:

    def __init__(self, ui, tabname, page):
        self.ui = ui
        self.tabname = tabname
        self.page = page
        self.id_part = f"{self.tabname}_{self.page.id_page}_edit_user_metadata"

        self.box = None

        self.edit_name_input = None
        self.button_edit = None

        self.edit_name = None
        self.edit_description = None
        self.edit_notes = None
        self.html_filedata = None
        self.html_preview = None
        self.html_status = None

        self.button_cancel = None
        self.button_replace_preview = None
        self.button_save = None

    def get_user_metadata(self, name):
        item = self.page.items.get(name, {})

        user_metadata = item.get('user_metadata', None)
        if user_metadata is None:
            user_metadata = {}
            item['user_metadata'] = user_metadata

        return user_metadata

    def create_extra_default_items_in_left_column(self):
        pass

    def create_default_editor_elems(self):
        with gr.Row():
            with gr.Column(scale=2):
                self.edit_name = gr.HTML(elem_classes="extra-network-name")
                self.edit_description = gr.Textbox(label="Description", lines=4)
                self.html_filedata = gr.HTML()

                self.create_extra_default_items_in_left_column()

            with gr.Column(scale=1, min_width=0):
                self.html_preview = gr.HTML()

    def create_default_buttons(self):

        with gr.Row(elem_classes="edit-user-metadata-buttons"):
            self.button_cancel = gr.Button('Cancel')
            self.button_replace_preview = gr.Button('Replace preview', variant='primary')
            self.button_save = gr.Button('Save', variant='primary')

        self.html_status = gr.HTML(elem_classes="edit-user-metadata-status")

        self.button_cancel.click(fn=None, _js="closePopup")

    def get_card_html(self, name):
        item = self.page.items.get(name, {})

        preview_url = item.get("preview", None)

        if not preview_url:
            filename, _ = os.path.splitext(item["filename"])
            preview_url = self.page.find_preview(filename)
            item["preview"] = preview_url

        if preview_url:
            preview = f'''
            <div class='card standalone-card-preview'>
                <img src="{html.escape(preview_url)}" class="preview">
            </div>
            '''
        else:
            preview = "<div class='card standalone-card-preview'></div>"

        return preview

    def get_metadata_table(self, name):
        item = self.page.items.get(name, {})
        try:
            filename = item["filename"]

            stats = os.stat(filename)
            params = [
                ('File size: ', sysinfo.pretty_bytes(stats.st_size)),
                ('Modified: ', datetime.datetime.fromtimestamp(stats.st_mtime).strftime('%Y-%m-%d %H:%M')),
            ]

            return params
        except Exception as e:
            errors.display(e, f"reading info for {name}")
            return []

    def put_values_into_components(self, name):
        user_metadata = self.get_user_metadata(name)

        try:
            params = self.get_metadata_table(name)
        except Exception as e:
            errors.display(e, f"reading metadata info for {name}")
            params = []

        table = '<table class="file-metadata">' + "".join(f"<tr><th>{name}</th><td>{value}</td></tr>" for name, value in params) + '</table>'

        return html.escape(name), user_metadata.get('description', ''), table, self.get_card_html(name), user_metadata.get('notes', '')

    def write_user_metadata(self, name, metadata):
        item = self.page.items.get(name, {})
        filename = item.get("filename", None)
        basename, ext = os.path.splitext(filename)

        with open(basename + '.json', "w", encoding="utf8") as file:
            json.dump(metadata, file)

    def save_user_metadata(self, name, desc, notes):
        user_metadata = self.get_user_metadata(name)
        user_metadata["description"] = desc
        user_metadata["notes"] = notes

        self.write_user_metadata(name, user_metadata)

    def setup_save_handler(self, button, func, components):
        button\
            .click(fn=func, inputs=[self.edit_name_input, *components], outputs=[])\
            .then(fn=None, _js="function(name){closePopup(); extraNetworksRefreshSingleCard(" + json.dumps(self.page.name) + "," + json.dumps(self.tabname) + ", name);}", inputs=[self.edit_name_input], outputs=[])

    def create_editor(self):
        self.create_default_editor_elems()

        self.edit_notes = gr.TextArea(label='Notes', lines=4)

        self.create_default_buttons()

        self.button_edit\
            .click(fn=self.put_values_into_components, inputs=[self.edit_name_input], outputs=[self.edit_name, self.edit_description, self.html_filedata, self.html_preview, self.edit_notes])\
            .then(fn=lambda: gr.update(visible=True), inputs=[], outputs=[self.box])

        self.setup_save_handler(self.button_save, self.save_user_metadata, [self.edit_description, self.edit_notes])

    def create_ui(self):
        with gr.Box(visible=False, elem_id=self.id_part, elem_classes="edit-user-metadata") as box:
            self.box = box

            self.edit_name_input = gr.Textbox("Edit user metadata card id", visible=False, elem_id=f"{self.id_part}_name")
            self.button_edit = gr.Button("Edit user metadata", visible=False, elem_id=f"{self.id_part}_button")

            self.create_editor()

    def save_preview(self, index, gallery, name):
        if len(gallery) == 0:
            return self.get_card_html(name), "There is no image in gallery to save as a preview."

        item = self.page.items.get(name, {})

        index = int(index)
        index = 0 if index < 0 else index
        index = len(gallery) - 1 if index >= len(gallery) else index

        img_info = gallery[index if index >= 0 else 0]
        image = generation_parameters_copypaste.image_from_url_text(img_info)
        geninfo, items = images.read_info_from_image(image)

        images.save_image_with_geninfo(image, geninfo, item["local_preview"])

        return self.get_card_html(name), ''

    def setup_ui(self, gallery):
        self.button_replace_preview.click(
            fn=self.save_preview,
            _js="function(x, y, z){return [selected_gallery_index(), y, z]}",
            inputs=[self.edit_name_input, gallery, self.edit_name_input],
            outputs=[self.html_preview, self.html_status]
        ).then(
            fn=None,
            _js="function(name){extraNetworksRefreshSingleCard(" + json.dumps(self.page.name) + "," + json.dumps(self.tabname) + ", name);}",
            inputs=[self.edit_name_input],
            outputs=[]
        )



