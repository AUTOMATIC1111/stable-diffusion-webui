import os.path

from modules import shared
import gradio as gr
import json

from modules.generation_parameters_copypaste import image_from_url_text

extra_pages = []


def register_page(page):
    """registers extra networks page for the UI; recommend doing it in on_app_started() callback for extensions"""

    extra_pages.append(page)


class ExtraNetworksPage:
    def __init__(self, title):
        self.title = title
        self.card_page = shared.html("extra-networks-card.html")
        self.allow_negative_prompt = False

    def refresh(self):
        pass

    def create_html(self, tabname):
        items_html = ''

        for item in self.list_items():
            items_html += self.create_html_for_item(item, tabname)

        if items_html == '':
            dirs = "".join([f"<li>{x}</li>" for x in self.allowed_directories_for_previews()])
            items_html = shared.html("extra-networks-no-cards.html").format(dirs=dirs)

        res = "<div class='extra-network-cards'>" + items_html + "</div>"

        return res

    def list_items(self):
        raise NotImplementedError()

    def allowed_directories_for_previews(self):
        return []

    def create_html_for_item(self, item, tabname):
        preview = item.get("preview", None)

        args = {
            "preview_html": "style='background-image: url(" + json.dumps(preview) + ")'" if preview else '',
            "prompt": json.dumps(item["prompt"]),
            "tabname": json.dumps(tabname),
            "local_preview": json.dumps(item["local_preview"]),
            "name": item["name"],
            "allow_negative_prompt": "true" if self.allow_negative_prompt else "false",
        }

        return self.card_page.format(**args)


def intialize():
    extra_pages.clear()


class ExtraNetworksUi:
    def __init__(self):
        self.pages = None
        self.stored_extra_pages = None

        self.button_save_preview = None
        self.preview_target_filename = None

        self.tabname = None


def create_ui(container, button, tabname):
    ui = ExtraNetworksUi()
    ui.pages = []
    ui.stored_extra_pages = extra_pages.copy()
    ui.tabname = tabname

    with gr.Tabs(elem_id=tabname+"_extra_tabs") as tabs:
        button_refresh = gr.Button('Refresh', elem_id=tabname+"_extra_refresh")
        button_close = gr.Button('Close', elem_id=tabname+"_extra_close")

        for page in ui.stored_extra_pages:
            with gr.Tab(page.title):
                page_elem = gr.HTML(page.create_html(ui.tabname))
                ui.pages.append(page_elem)

    ui.button_save_preview = gr.Button('Save preview', elem_id=tabname+"_save_preview", visible=False)
    ui.preview_target_filename = gr.Textbox('Preview save filename', elem_id=tabname+"_preview_filename", visible=False)

    button.click(fn=lambda: gr.update(visible=True), inputs=[], outputs=[container])
    button_close.click(fn=lambda: gr.update(visible=False), inputs=[], outputs=[container])

    def refresh():
        res = []

        for pg in ui.stored_extra_pages:
            pg.refresh()
            res.append(pg.create_html(ui.tabname))

        return res

    button_refresh.click(fn=refresh, inputs=[], outputs=ui.pages)

    return ui


def path_is_parent(parent_path, child_path):
    parent_path = os.path.abspath(parent_path)
    child_path = os.path.abspath(child_path)

    return os.path.commonpath([parent_path]) == os.path.commonpath([parent_path, child_path])


def setup_ui(ui, gallery):
    def save_preview(index, images, filename):
        if len(images) == 0:
            print("There is no image in gallery to save as a preview.")
            return [page.create_html(ui.tabname) for page in ui.stored_extra_pages]

        index = int(index)
        index = 0 if index < 0 else index
        index = len(images) - 1 if index >= len(images) else index

        img_info = images[index if index >= 0 else 0]
        image = image_from_url_text(img_info)

        is_allowed = False
        for extra_page in ui.stored_extra_pages:
            if any([path_is_parent(x, filename) for x in extra_page.allowed_directories_for_previews()]):
                is_allowed = True
                break

        assert is_allowed, f'writing to {filename} is not allowed'

        image.save(filename)

        return [page.create_html(ui.tabname) for page in ui.stored_extra_pages]

    ui.button_save_preview.click(
        fn=save_preview,
        _js="function(x, y, z){console.log(x, y, z); return [selected_gallery_index(), y, z]}",
        inputs=[ui.preview_target_filename, gallery, ui.preview_target_filename],
        outputs=[*ui.pages]
    )
