import os.path

from modules import shared
import gradio as gr
import json
import html

from modules.generation_parameters_copypaste import image_from_url_text

extra_pages = []


def register_page(page):
    """registers extra networks page for the UI; recommend doing it in on_before_ui() callback for extensions"""

    extra_pages.append(page)


class ExtraNetworksPage:
    def __init__(self, title):
        self.title = title
        self.name = title.lower()
        self.card_page = shared.html("extra-networks-card.html")
        self.allow_negative_prompt = False

    def refresh(self):
        pass

    def create_html(self, tabname):
        view = shared.opts.extra_networks_default_view
        items_html = ''

        for item in self.list_items():
            items_html += self.create_html_for_item(item, tabname)

        if items_html == '':
            dirs = "".join([f"<li>{x}</li>" for x in self.allowed_directories_for_previews()])
            items_html = shared.html("extra-networks-no-cards.html").format(dirs=dirs)

        res = f"""
<div id='{tabname}_{self.name}_cards' class='extra-network-{view}'>
{items_html}
</div>
"""

        return res

    def list_items(self):
        raise NotImplementedError()

    def allowed_directories_for_previews(self):
        return []

    def create_html_for_item(self, item, tabname):
        preview = item.get("preview", None)

        args = {
            "preview_html": "style='background-image: url(\"" + html.escape(preview) + "\")'" if preview else '',
            "prompt": item["prompt"],
            "tabname": json.dumps(tabname),
            "local_preview": json.dumps(item["local_preview"]),
            "name": item["name"],
            "card_clicked": '"' + html.escape(f"""return cardClicked({json.dumps(tabname)}, {item["prompt"]}, {"true" if self.allow_negative_prompt else "false"})""") + '"',
            "save_card_preview": '"' + html.escape(f"""return saveCardPreview(event, {json.dumps(tabname)}, {json.dumps(item["local_preview"])})""") + '"',
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


def pages_in_preferred_order(pages):
    tab_order = [x.lower().strip() for x in shared.opts.ui_extra_networks_tab_reorder.split(",")]

    def tab_name_score(name):
        name = name.lower()
        for i, possible_match in enumerate(tab_order):
            if possible_match in name:
                return i

        return len(pages)

    tab_scores = {page.name: (tab_name_score(page.name), original_index) for original_index, page in enumerate(pages)}

    return sorted(pages, key=lambda x: tab_scores[x.name])


def create_ui(container, button, tabname):
    ui = ExtraNetworksUi()
    ui.pages = []
    ui.stored_extra_pages = pages_in_preferred_order(extra_pages.copy())
    ui.tabname = tabname

    with gr.Tabs(elem_id=tabname+"_extra_tabs") as tabs:
        for page in ui.stored_extra_pages:
            with gr.Tab(page.title):
                page_elem = gr.HTML(page.create_html(ui.tabname))
                ui.pages.append(page_elem)

    filter = gr.Textbox('', show_label=False, elem_id=tabname+"_extra_search", placeholder="Search...", visible=False)
    button_refresh = gr.Button('Refresh', elem_id=tabname+"_extra_refresh")
    button_close = gr.Button('Close', elem_id=tabname+"_extra_close")

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
