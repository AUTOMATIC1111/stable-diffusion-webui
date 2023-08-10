import os.path
import urllib.parse
from pathlib import Path

from modules import shared, ui_extra_networks_user_metadata, errors, extra_networks
from modules.images import read_info_from_image, save_image_with_geninfo
import gradio as gr
import json
import html
from fastapi.exceptions import HTTPException

from modules.generation_parameters_copypaste import image_from_url_text
from modules.ui_components import ToolButton

extra_pages = []
allowed_dirs = set()


def register_page(page):
    """registers extra networks page for the UI; recommend doing it in on_before_ui() callback for extensions"""

    extra_pages.append(page)
    allowed_dirs.clear()
    allowed_dirs.update(set(sum([x.allowed_directories_for_previews() for x in extra_pages], [])))


def fetch_file(filename: str = ""):
    from starlette.responses import FileResponse

    if not os.path.isfile(filename):
        raise HTTPException(status_code=404, detail="File not found")

    if not any(Path(x).absolute() in Path(filename).absolute().parents for x in allowed_dirs):
        raise ValueError(f"File cannot be fetched: {filename}. Must be in one of directories registered by extra pages.")

    ext = os.path.splitext(filename)[1].lower()
    if ext not in (".png", ".jpg", ".jpeg", ".webp", ".gif"):
        raise ValueError(f"File cannot be fetched: {filename}. Only png, jpg, webp, and gif.")

    # would profit from returning 304
    return FileResponse(filename, headers={"Accept-Ranges": "bytes"})


def get_metadata(page: str = "", item: str = ""):
    from starlette.responses import JSONResponse

    page = next(iter([x for x in extra_pages if x.name == page]), None)
    if page is None:
        return JSONResponse({})

    metadata = page.metadata.get(item)
    if metadata is None:
        return JSONResponse({})

    return JSONResponse({"metadata": json.dumps(metadata, indent=4, ensure_ascii=False)})


def get_single_card(page: str = "", tabname: str = "", name: str = ""):
    from starlette.responses import JSONResponse

    page = next(iter([x for x in extra_pages if x.name == page]), None)

    try:
        item = page.create_item(name, enable_filter=False)
        page.items[name] = item
    except Exception as e:
        errors.display(e, "creating item for extra network")
        item = page.items.get(name)

    page.read_user_metadata(item)
    item_html = page.create_html_for_item(item, tabname)

    return JSONResponse({"html": item_html})


def add_pages_to_demo(app):
    app.add_api_route("/sd_extra_networks/thumb", fetch_file, methods=["GET"])
    app.add_api_route("/sd_extra_networks/metadata", get_metadata, methods=["GET"])
    app.add_api_route("/sd_extra_networks/get-single-card", get_single_card, methods=["GET"])


def quote_js(s):
    s = s.replace('\\', '\\\\')
    s = s.replace('"', '\\"')
    return f'"{s}"'


class ExtraNetworksPage:
    def __init__(self, title):
        self.title = title
        self.name = title.lower()
        self.id_page = self.name.replace(" ", "_")
        self.card_page = shared.html("extra-networks-card.html")
        self.allow_negative_prompt = False
        self.metadata = {}
        self.items = {}

    def refresh(self):
        pass

    def read_user_metadata(self, item):
        filename = item.get("filename", None)
        metadata = extra_networks.get_user_metadata(filename)

        desc = metadata.get("description", None)
        if desc is not None:
            item["description"] = desc

        item["user_metadata"] = metadata

    def link_preview(self, filename):
        quoted_filename = urllib.parse.quote(filename.replace('\\', '/'))
        mtime = os.path.getmtime(filename)
        return f"./sd_extra_networks/thumb?filename={quoted_filename}&mtime={mtime}"

    def search_terms_from_path(self, filename, possible_directories=None):
        abspath = os.path.abspath(filename)

        for parentdir in (possible_directories if possible_directories is not None else self.allowed_directories_for_previews()):
            parentdir = os.path.abspath(parentdir)
            if abspath.startswith(parentdir):
                return abspath[len(parentdir):].replace('\\', '/')

        return ""

    def create_html(self, tabname):
        items_html = ''

        self.metadata = {}

        subdirs = {}
        for parentdir in [os.path.abspath(x) for x in self.allowed_directories_for_previews()]:
            for root, dirs, _ in sorted(os.walk(parentdir, followlinks=True), key=lambda x: shared.natural_sort_key(x[0])):
                for dirname in sorted(dirs, key=shared.natural_sort_key):
                    x = os.path.join(root, dirname)

                    if not os.path.isdir(x):
                        continue

                    subdir = os.path.abspath(x)[len(parentdir):].replace("\\", "/")
                    while subdir.startswith("/"):
                        subdir = subdir[1:]

                    is_empty = len(os.listdir(x)) == 0
                    if not is_empty and not subdir.endswith("/"):
                        subdir = subdir + "/"

                    if ("/." in subdir or subdir.startswith(".")) and not shared.opts.extra_networks_show_hidden_directories:
                        continue

                    subdirs[subdir] = 1

        if subdirs:
            subdirs = {"": 1, **subdirs}

        subdirs_html = "".join([f"""
<button class='lg secondary gradio-button custom-button{" search-all" if subdir=="" else ""}' onclick='extraNetworksSearchButton("{tabname}_extra_search", event)'>
{html.escape(subdir if subdir!="" else "all")}
</button>
""" for subdir in subdirs])

        self.items = {x["name"]: x for x in self.list_items()}
        for item in self.items.values():
            metadata = item.get("metadata")
            if metadata:
                self.metadata[item["name"]] = metadata

            if "user_metadata" not in item:
                self.read_user_metadata(item)

            items_html += self.create_html_for_item(item, tabname)

        if items_html == '':
            dirs = "".join([f"<li>{x}</li>" for x in self.allowed_directories_for_previews()])
            items_html = shared.html("extra-networks-no-cards.html").format(dirs=dirs)

        self_name_id = self.name.replace(" ", "_")

        res = f"""
<div id='{tabname}_{self_name_id}_subdirs' class='extra-network-subdirs extra-network-subdirs-cards'>
{subdirs_html}
</div>
<div id='{tabname}_{self_name_id}_cards' class='extra-network-cards'>
{items_html}
</div>
"""

        return res

    def create_item(self, name, index=None):
        raise NotImplementedError()

    def list_items(self):
        raise NotImplementedError()

    def allowed_directories_for_previews(self):
        return []

    def create_html_for_item(self, item, tabname):
        """
        Create HTML for card item in tab tabname; can return empty string if the item is not meant to be shown.
        """

        preview = item.get("preview", None)

        onclick = item.get("onclick", None)
        if onclick is None:
            onclick = '"' + html.escape(f"""return cardClicked({quote_js(tabname)}, {item["prompt"]}, {"true" if self.allow_negative_prompt else "false"})""") + '"'

        height = f"height: {shared.opts.extra_networks_card_height}px;" if shared.opts.extra_networks_card_height else ''
        width = f"width: {shared.opts.extra_networks_card_width}px;" if shared.opts.extra_networks_card_width else ''
        background_image = f'<img src="{html.escape(preview)}" class="preview" loading="lazy">' if preview else ''
        metadata_button = ""
        metadata = item.get("metadata")
        if metadata:
            metadata_button = f"<div class='metadata-button card-button' title='Show internal metadata' onclick='extraNetworksRequestMetadata(event, {quote_js(self.name)}, {quote_js(item['name'])})'></div>"

        edit_button = f"<div class='edit-button card-button' title='Edit metadata' onclick='extraNetworksEditUserMetadata(event, {quote_js(tabname)}, {quote_js(self.id_page)}, {quote_js(item['name'])})'></div>"

        local_path = ""
        filename = item.get("filename", "")
        for reldir in self.allowed_directories_for_previews():
            absdir = os.path.abspath(reldir)

            if filename.startswith(absdir):
                local_path = filename[len(absdir):]

        # if this is true, the item must not be shown in the default view, and must instead only be
        # shown when searching for it
        if shared.opts.extra_networks_hidden_models == "Always":
            search_only = False
        else:
            search_only = "/." in local_path or "\\." in local_path

        if search_only and shared.opts.extra_networks_hidden_models == "Never":
            return ""

        sort_keys = " ".join([html.escape(f'data-sort-{k}={v}') for k, v in item.get("sort_keys", {}).items()]).strip()

        args = {
            "background_image": background_image,
            "style": f"'display: none; {height}{width}; font-size: {shared.opts.extra_networks_card_text_scale*100}%'",
            "prompt": item.get("prompt", None),
            "tabname": quote_js(tabname),
            "local_preview": quote_js(item["local_preview"]),
            "name": html.escape(item["name"]),
            "description": (item.get("description") or "" if shared.opts.extra_networks_card_show_desc else ""),
            "card_clicked": onclick,
            "save_card_preview": '"' + html.escape(f"""return saveCardPreview(event, {quote_js(tabname)}, {quote_js(item["local_preview"])})""") + '"',
            "search_term": item.get("search_term", ""),
            "metadata_button": metadata_button,
            "edit_button": edit_button,
            "search_only": " search_only" if search_only else "",
            "sort_keys": sort_keys,
        }

        return self.card_page.format(**args)

    def get_sort_keys(self, path):
        """
        List of default keys used for sorting in the UI.
        """
        pth = Path(path)
        stat = pth.stat()
        return {
            "date_created": int(stat.st_ctime or 0),
            "date_modified": int(stat.st_mtime or 0),
            "name": pth.name.lower(),
        }

    def find_preview(self, path):
        """
        Find a preview PNG for a given path (without extension) and call link_preview on it.
        """

        preview_extensions = ["png", "jpg", "jpeg", "webp"]
        if shared.opts.samples_format not in preview_extensions:
            preview_extensions.append(shared.opts.samples_format)

        potential_files = sum([[path + "." + ext, path + ".preview." + ext] for ext in preview_extensions], [])

        for file in potential_files:
            if os.path.isfile(file):
                return self.link_preview(file)

        return None

    def find_description(self, path):
        """
        Find and read a description file for a given path (without extension).
        """
        for file in [f"{path}.txt", f"{path}.description.txt"]:
            try:
                with open(file, "r", encoding="utf-8", errors="replace") as f:
                    return f.read()
            except OSError:
                pass
        return None

    def create_user_metadata_editor(self, ui, tabname):
        return ui_extra_networks_user_metadata.UserMetadataEditor(ui, tabname, self)


def initialize():
    extra_pages.clear()


def register_default_pages():
    from modules.ui_extra_networks_textual_inversion import ExtraNetworksPageTextualInversion
    from modules.ui_extra_networks_hypernets import ExtraNetworksPageHypernetworks
    from modules.ui_extra_networks_checkpoints import ExtraNetworksPageCheckpoints
    register_page(ExtraNetworksPageTextualInversion())
    register_page(ExtraNetworksPageHypernetworks())
    register_page(ExtraNetworksPageCheckpoints())


class ExtraNetworksUi:
    def __init__(self):
        self.pages = None
        """gradio HTML components related to extra networks' pages"""

        self.page_contents = None
        """HTML content of the above; empty initially, filled when extra pages have to be shown"""

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


def create_ui(interface: gr.Blocks, unrelated_tabs, tabname):
    from modules.ui import switch_values_symbol

    ui = ExtraNetworksUi()
    ui.pages = []
    ui.pages_contents = []
    ui.user_metadata_editors = []
    ui.stored_extra_pages = pages_in_preferred_order(extra_pages.copy())
    ui.tabname = tabname

    related_tabs = []

    for page in ui.stored_extra_pages:
        with gr.Tab(page.title, id=page.id_page) as tab:
            elem_id = f"{tabname}_{page.id_page}_cards_html"
            page_elem = gr.HTML('Loading...', elem_id=elem_id)
            ui.pages.append(page_elem)

            page_elem.change(fn=lambda: None, _js='function(){applyExtraNetworkFilter(' + quote_js(tabname) + '); return []}', inputs=[], outputs=[])

            editor = page.create_user_metadata_editor(ui, tabname)
            editor.create_ui()
            ui.user_metadata_editors.append(editor)

            related_tabs.append(tab)

    edit_search = gr.Textbox('', show_label=False, elem_id=tabname+"_extra_search", elem_classes="search", placeholder="Search...", visible=False, interactive=True)
    dropdown_sort = gr.Dropdown(choices=['Default Sort', 'Date Created', 'Date Modified', 'Name'], value='Default Sort', elem_id=tabname+"_extra_sort", elem_classes="sort", multiselect=False, visible=False, show_label=False, interactive=True, label=tabname+"_extra_sort_order")
    button_sortorder = ToolButton(switch_values_symbol, elem_id=tabname+"_extra_sortorder", elem_classes="sortorder", visible=False)
    button_refresh = gr.Button('Refresh', elem_id=tabname+"_extra_refresh", visible=False)
    checkbox_show_dirs = gr.Checkbox(True, label='Show dirs', elem_id=tabname+"_extra_show_dirs", elem_classes="show-dirs", visible=False)

    ui.button_save_preview = gr.Button('Save preview', elem_id=tabname+"_save_preview", visible=False)
    ui.preview_target_filename = gr.Textbox('Preview save filename', elem_id=tabname+"_preview_filename", visible=False)

    for tab in unrelated_tabs:
        tab.select(fn=lambda: [gr.update(visible=False) for _ in range(5)], inputs=[], outputs=[edit_search, dropdown_sort, button_sortorder, button_refresh, checkbox_show_dirs], show_progress=False)

    for tab in related_tabs:
        tab.select(fn=lambda: [gr.update(visible=True) for _ in range(5)], inputs=[], outputs=[edit_search, dropdown_sort, button_sortorder, button_refresh, checkbox_show_dirs], show_progress=False)

    def pages_html():
        if not ui.pages_contents:
            return refresh()

        return ui.pages_contents

    def refresh():
        for pg in ui.stored_extra_pages:
            pg.refresh()

        ui.pages_contents = [pg.create_html(ui.tabname) for pg in ui.stored_extra_pages]

        return ui.pages_contents

    interface.load(fn=pages_html, inputs=[], outputs=[*ui.pages])
    button_refresh.click(fn=refresh, inputs=[], outputs=ui.pages)

    return ui


def path_is_parent(parent_path, child_path):
    parent_path = os.path.abspath(parent_path)
    child_path = os.path.abspath(child_path)

    return child_path.startswith(parent_path)


def setup_ui(ui, gallery):
    def save_preview(index, images, filename):
        # this function is here for backwards compatibility and likely will be removed soon

        if len(images) == 0:
            print("There is no image in gallery to save as a preview.")
            return [page.create_html(ui.tabname) for page in ui.stored_extra_pages]

        index = int(index)
        index = 0 if index < 0 else index
        index = len(images) - 1 if index >= len(images) else index

        img_info = images[index if index >= 0 else 0]
        image = image_from_url_text(img_info)
        geninfo, items = read_info_from_image(image)

        is_allowed = False
        for extra_page in ui.stored_extra_pages:
            if any(path_is_parent(x, filename) for x in extra_page.allowed_directories_for_previews()):
                is_allowed = True
                break

        assert is_allowed, f'writing to {filename} is not allowed'

        save_image_with_geninfo(image, geninfo, filename)

        return [page.create_html(ui.tabname) for page in ui.stored_extra_pages]

    ui.button_save_preview.click(
        fn=save_preview,
        _js="function(x, y, z){return [selected_gallery_index(), y, z]}",
        inputs=[ui.preview_target_filename, gallery, ui.preview_target_filename],
        outputs=[*ui.pages]
    )

    for editor in ui.user_metadata_editors:
        editor.setup_ui(gallery)


