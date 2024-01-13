import functools
import os.path
import urllib.parse
from pathlib import Path
from typing import Optional, Union
from dataclasses import dataclass

from modules import shared, ui_extra_networks_user_metadata, errors, extra_networks, util
from modules.images import read_info_from_image, save_image_with_geninfo
import gradio as gr
import json
import html
from fastapi.exceptions import HTTPException

from modules.infotext_utils import image_from_url_text
from modules.ui_components import ToolButton

extra_pages = []
allowed_dirs = set()
default_allowed_preview_extensions = ["png", "jpg", "jpeg", "webp", "gif"]

tree_tpl = (
    "<div class='action-list-search'>"
    "<input "
    "id='{tabname}_{tab_id}_extra_search' "
    "class='action-list-search-text' "
    "type='search' "
    "placeholder='Filter files'"
    ">"
    "</div>"
    "<ul class='action-list action-list--tree'>"
    "{content}"
    "</ul>"
)

tree_ul_tpl = (
    "<ul class='action-list action-list--subgroup' data-hidden>"
    "{content}"
    "</ul>"
)

tree_li_dir_tpl = (
    "<li "
    "class='action-list-item action-list-item--has-subitem' "
    "data-path='{path}' "
    "data-tree-entry-type='dir'>"
    "{content}"
    "</li>"
)
tree_li_file_tpl = (
    "<li "
    "id='file-tree-item-{hash}' "
    "class='action-list-item action-list-item--subitem' "
    "data-path='{path}' "
    "data-tree-entry-type='file'>"
    "{content}"
    "</li>"
)

tree_btn_dir_tpl = (
    "<button "
    "class='action-list-content action-list-content-dir' "
    "type='button' "
    "expanded='false' "
    "onclick='extraNetworksTreeOnClick(event, \"{tabname}\", \"{tab_id}\")'>"
    "<span class='action-list-item-action action-list-item-action--leading'>"
    "<i class='action-list-item-action-chevron'></i>"
    "</span>"
    "<span class='action-list-item-visual action-list-item-visual--leading'>🗀</span>"
    "<span class='action-list-item-label action-list-item-label--truncate'>{label}</span>"
    "</button>"
)

tree_btn_file_action_buttons_tpl = (
    "<div "
    "class='copy-path-button card-button' "
    "title='Copy path to clipboard' "
    "onclick='extraNetworksCopyCardPath(event, {path})' "
    "data-clipboard-text={path}>"
    "</div>"
    "<div "
    "class='metadata-button card-button' "
    "title='Show internal metadata' "
    "onclick='extraNetworksRequestMetadata(event, {tab_id}, {filename})'>"
    "</div>"
    "<div "
    "class='edit-button card-button' "
    "title='Edit metadata' "
    "onclick='extraNetworksEditUserMetadata(event, {tabname}, {tab_id}, {filename})'>"
    "</div>"
)

tree_btn_file_tpl = (
    "<span data-filterable-item-text hidden>{filter}</span>"
    "<button "
    "class='action-list-content action-list-content-file' "
    "type='button' "
    "onclick='extraNetworksTreeOnClick(event, \"{tabname}\", \"{tab_id}\")'>"
    "<span class='action-list-item-visual action-list-item-visual--leading'>🗎</span>"
    "<span class='action-list-item-label action-list-item-label--truncate'>{label}</span>"
    "<span class='action-list-item-action action-list-item-action--trailing'>{buttons}</span>"
    "</button>"
)


@functools.cache
def allowed_preview_extensions_with_extra(extra_extensions=None):
    return set(default_allowed_preview_extensions) | set(extra_extensions or [])


def allowed_preview_extensions():
    return allowed_preview_extensions_with_extra((shared.opts.samples_format, ))


@dataclass
class ExtraNetworksItem:
    """Wrapper for dictionaries representing ExtraNetworks items."""
    item: dict


def get_tree(paths: Union[str, list[str]], items: dict[str, ExtraNetworksItem]) -> dict:
    """Recursively builds a directory tree.

    Args:
        paths: Path or list of paths to directories. These paths are treated as roots from which
            the tree will be built.
        items: A dictionary associating filepaths to an ExtraNetworksItem instance.

    Returns:
        The result directory tree.
    """
    if isinstance(paths, (str,)):
        paths = [paths]

    def _get_tree(_paths: list[str], _root: str):
        _res = {}
        for path in _paths:
            relpath = os.path.relpath(path, _root)
            if os.path.isdir(path):
                dir_items = os.listdir(path)
                # Ignore empty directories.
                if not dir_items:
                    continue
                dir_tree = _get_tree([os.path.join(path, x) for x in dir_items], _root)
                # We only want to store non-empty folders in the tree.
                if dir_tree:
                    _res[relpath] = dir_tree
            else:
                if path not in items:
                    continue
                # Add the ExtraNetworksItem to the result.
                _res[relpath] = items[path]
        return _res

    res = {}
    # Handle each root directory separately.
    # Each root WILL have a key/value at the root of the result dict though
    # the value can be an empty dict if the directory is empty. We want these
    # placeholders for empty dirs so we can inform the user later.
    for path in paths:
        root = os.path.dirname(path)
        relpath = os.path.relpath(path, root)
        # Wrap the path in a list since that is what the `_get_tree` expects.
        res[relpath] = _get_tree([path], root)
        if res[relpath]:
            # We need to pull the inner path out one for these root dirs.
            res[relpath] = res[relpath][relpath]

    return res

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

    ext = os.path.splitext(filename)[1].lower()[1:]
    if ext not in allowed_preview_extensions():
        raise ValueError(f"File cannot be fetched: {filename}. Extensions allowed: {allowed_preview_extensions()}.")

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
    item_html = page.create_item_html(tabname, item)

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
        self.extra_networks_pane_template = shared.html("extra-networks-pane.html")
        self.card_page_template = shared.html("extra-networks-card.html")
        self.card_page_minimal_template = shared.html("extra-networks-card-minimal.html")
        self.tree_button_template = shared.html("extra-networks-tree-button.html")
        self.allow_prompt = True
        self.allow_negative_prompt = False
        self.metadata = {}
        self.items = {}
        self.lister = util.MassFileLister()

    def refresh(self):
        pass

    def read_user_metadata(self, item):
        filename = item.get("filename", None)
        metadata = extra_networks.get_user_metadata(filename, lister=self.lister)

        desc = metadata.get("description", None)
        if desc is not None:
            item["description"] = desc

        item["user_metadata"] = metadata

    def link_preview(self, filename):
        quoted_filename = urllib.parse.quote(filename.replace('\\', '/'))
        mtime, _ = self.lister.mctime(filename)
        return f"./sd_extra_networks/thumb?filename={quoted_filename}&mtime={mtime}"

    def search_terms_from_path(self, filename, possible_directories=None):
        abspath = os.path.abspath(filename)
        for parentdir in (possible_directories if possible_directories is not None else self.allowed_directories_for_previews()):
            parentdir = os.path.dirname(os.path.abspath(parentdir))
            if abspath.startswith(parentdir):
                return os.path.relpath(abspath, parentdir)

        return ""

    def create_item_html(self, tabname: str, item: dict, template: Optional[str] = None) -> str:
        """Generates HTML for a single ExtraNetworks Item

        Args:
            tabname: The name of the active tab.
            item: Dictionary containing item information.

        Returns:
            HTML string generated for this item.
            Can be empty if the item is not meant to be shown.
        """
        metadata = item.get("metadata")
        if metadata:
            self.metadata[item["name"]] = metadata

        if "user_metadata" not in item:
            self.read_user_metadata(item)

        preview = item.get("preview", None)
        height = f"height: {shared.opts.extra_networks_card_height}px;" if shared.opts.extra_networks_card_height else ''
        width = f"width: {shared.opts.extra_networks_card_width}px;" if shared.opts.extra_networks_card_width else ''
        background_image = f'<img src="{html.escape(preview)}" class="preview" loading="lazy">' if preview else ''

        onclick = item.get("onclick", None)
        if onclick is None:
            if "negative_prompt" in item:
                onclick = '"' + html.escape(f"""return cardClicked({quote_js(tabname)}, {item["prompt"]}, {item["negative_prompt"]}, {"true" if self.allow_negative_prompt else "false"})""") + '"'
            else:
                onclick = '"' + html.escape(f"""return cardClicked({quote_js(tabname)}, {item["prompt"]}, {'""'}, {"true" if self.allow_negative_prompt else "false"})""") + '"'

        copy_path_button = f"<div class='copy-path-button card-button' title='Copy path to clipboard' onclick='extraNetworksCopyCardPath(event, {quote_js(item['filename'])})' data-clipboard-text='{quote_js(item['filename'])}'></div>"

        metadata_button = ""
        metadata = item.get("metadata")
        if metadata:
            metadata_button = f"<div class='metadata-button card-button' title='Show internal metadata' onclick='extraNetworksRequestMetadata(event, {quote_js(self.name)}, {quote_js(html.escape(item['name']))})'></div>"

        edit_button = f"<div class='edit-button card-button' title='Edit metadata' onclick='extraNetworksEditUserMetadata(event, {quote_js(tabname)}, {quote_js(self.id_page)}, {quote_js(html.escape(item['name']))})'></div>"

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

        sort_keys = " ".join([f'data-sort-{k}="{html.escape(str(v))}"' for k, v in item.get("sort_keys", {}).items()]).strip()

        search_terms_html = ""
        search_term_template = "<span style='{style}' class='{class}'>{search_term}</span>"
        for search_term in item.get("search_terms", []):
            search_terms_html += search_term_template.format(
                **{
                    "style": "display: none;",
                    "class": "search_terms" + (" search_only" if search_only else ""),
                    "search_term": search_term,
                }
            )

        # Some items here might not be used depending on HTML template used.
        args = {
            "background_image": background_image,
            "card_clicked": onclick,
            "copy_path_button": copy_path_button,
            "description": (item.get("description") or "" if shared.opts.extra_networks_card_show_desc else ""),
            "edit_button": edit_button,
            "local_preview": quote_js(item["local_preview"]),
            "metadata_button": metadata_button,
            "name": html.escape(item["name"]),
            "prompt": item.get("prompt", None),
            "save_card_preview": '"' + html.escape(f"""return saveCardPreview(event, {quote_js(tabname)}, {quote_js(item["local_preview"])})""") + '"',
            "search_only": " search_only" if search_only else "",
            "search_terms": search_terms_html,
            "sort_keys": sort_keys,
            "style": f"'display: none; {height}{width}; font-size: {shared.opts.extra_networks_card_text_scale*100}%'",
            "tabname": tabname,
            "tab_id": self.id_page,

        }

        if template:
            return template.format(**args)
        else:
            return self.card_page.format(**args)

    def create_tree_view_html(self, tabname: str) -> str:
        """Generates HTML for displaying folders in a tree view.

        Args:
            tabname: The name of the active tab.

        Returns:
            HTML string generated for this tree view.
        """
        res = ""

        # Generate HTML for the tree.
        roots = self.allowed_directories_for_previews()
        tree_items = {v["filename"]: ExtraNetworksItem(v) for v in self.items.values()}
        tree = get_tree([os.path.abspath(x) for x in roots], items=tree_items)

        if not tree:
            return res

        def _build_tree(data: Optional[dict[str, ExtraNetworksItem]] = None) -> str:
            """Recursively builds HTML for a tree."""
            _res = ""
            if not data:
                return (
                    "<details open disabled class='folder-item-empty'>"
                    "<summary class='folder-item-summary-empty'>Directory is empty</summary>"
                    "</details>"
                )

            for k, v in sorted(data.items(), key=lambda x: shared.natural_sort_key(x[0])):
                if isinstance(v, (ExtraNetworksItem,)):
                    _action_buttons = tree_btn_file_action_buttons_tpl.format(
                        **{
                            "path": quote_js(k),
                            "filename": quote_js(v.item["name"]),
                            "tabname": quote_js(tabname),
                            "tab_id": quote_js(self.id_page),
                        }
                    )
                    _btn = tree_btn_file_tpl.format(
                        **{
                            "label": v.item["name"],
                            "filter": v.item["search_terms"],
                            "tabname": tabname,
                            "tab_id": self.id_page,
                            "buttons": _action_buttons,
                        }
                    )
                    _li = tree_li_file_tpl.format(
                        **{
                            "hash": v.item["shorthash"],
                            "path": k,
                            "type": "file",
                            #"content": _btn,
                            "content": self.create_item_html(tabname, v.item, self.tree_button_template),
                        }
                    )
                    _res += _li
                    #item_html = self.create_item_html(tabname, v.item, self.card_page_minimal_template)
                    #_res += file_template.format(**{"card": item_html})
                else:
                    _btn = tree_btn_dir_tpl.format(
                        **{
                            "label": os.path.basename(k),
                            "tabname": tabname,
                            "tab_id": self.id_page,
                        }
                    )
                    _ul = tree_ul_tpl.format(**{"content": _build_tree(v)})
                    _li = tree_li_dir_tpl.format(**{"content": _btn + _ul, "path": k})
                    _res += _li
            return _res

        # Add each root directory to the tree.
        for k, v in sorted(tree.items(), key=lambda x: shared.natural_sort_key(x[0])):
            # If root is empty, append the "disabled" attribute to the template details tag.
            btn = tree_btn_dir_tpl.format(
                **{
                    "label": os.path.basename(k),
                    "tabname": tabname,
                    "tab_id": self.id_page,
                }
            )
            ul = tree_ul_tpl.format(**{"content": _build_tree(v)})
            li = tree_li_dir_tpl.format(**{"content": btn + ul, "path": k})
            res += li

        return tree_tpl.format(
            **{
                "content": res,
                "tabname": tabname,
                "tab_id": self.id_page,
            }
        )

    def create_card_view_html(self, tabname):
        res = ""
        self.items = {x["name"]: x for x in self.list_items()}
        for item in self.items.values():
            res += self.create_item_html(tabname, item, self.card_page_template)

        if res == "":
            dirs = "".join([f"<li>{x}</li>" for x in self.allowed_directories_for_previews()])
            res = shared.html("extra-networks-no-cards.html").format(dirs=dirs)

        return res

    def create_html(self, tabname):
        self.lister.reset()
        self.metadata = {}
        self.items = {x["name"]: x for x in self.list_items()}

        tree_view_html = self.create_tree_view_html(tabname)
        card_view_html = self.create_card_view_html(tabname)
        network_type_id = self.id_page

        return self.extra_networks_pane_template.format(
            **{
                "tabname": tabname,
                "network_type_id": network_type_id,
                "tree_html": tree_view_html,
                "items_html": card_view_html,
            }
        )

    def create_item(self, name, index=None):
        raise NotImplementedError()

    def list_items(self):
        raise NotImplementedError()

    def allowed_directories_for_previews(self):
        return []

    def get_sort_keys(self, path):
        """
        List of default keys used for sorting in the UI.
        """
        pth = Path(path)
        mtime, ctime = self.lister.mctime(path)
        return {
            "date_created": int(mtime),
            "date_modified": int(ctime),
            "name": pth.name.lower(),
            "path": str(pth.parent).lower(),
        }

    def find_preview(self, path):
        """
        Find a preview PNG for a given path (without extension) and call link_preview on it.
        """

        potential_files = sum([[path + "." + ext, path + ".preview." + ext] for ext in allowed_preview_extensions()], [])

        for file in potential_files:
            if self.lister.exists(file):
                return self.link_preview(file)

        return None

    def find_description(self, path):
        """
        Find and read a description file for a given path (without extension).
        """
        for file in [f"{path}.txt", f"{path}.description.txt"]:
            if not self.lister.exists(file):
                continue

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
        with gr.Tab(page.title, elem_id=f"{tabname}_{page.id_page}", elem_classes=["extra-page"]) as tab:
            with gr.Column(elem_id=f"{tabname}_{page.id_page}_prompts", elem_classes=["extra-page-prompts"]):
                pass

            elem_id = f"{tabname}_{page.id_page}_cards_html"
            page_elem = gr.HTML('Loading...', elem_id=elem_id)
            ui.pages.append(page_elem)
            page_elem.change(
                fn=lambda: None,
                _js=f"function(){{applyExtraNetworkFilter({tabname}_{page.id_page}_extra_search); return []}}",
                inputs=[],
                outputs=[],
            )

            editor = page.create_user_metadata_editor(ui, tabname)
            editor.create_ui()
            ui.user_metadata_editors.append(editor)

            related_tabs.append(tab)

    dropdown_sort = gr.Dropdown(choices=['Path', 'Name', 'Date Created', 'Date Modified', ], value=shared.opts.extra_networks_card_order_field, elem_id=tabname+"_extra_sort", elem_classes="sort", multiselect=False, visible=False, show_label=False, interactive=True, label=tabname+"_extra_sort_order")
    button_sortorder = ToolButton(switch_values_symbol, elem_id=tabname+"_extra_sortorder", elem_classes=["sortorder"] + ([] if shared.opts.extra_networks_card_order == "Ascending" else ["sortReverse"]), visible=False, tooltip="Invert sort order")
    button_refresh = gr.Button('Refresh', elem_id=tabname+"_extra_refresh", visible=False)

    tab_controls = [
        dropdown_sort,
        button_sortorder,
        button_refresh,
    ]

    ui.button_save_preview = gr.Button('Save preview', elem_id=tabname+"_save_preview", visible=False)
    ui.preview_target_filename = gr.Textbox('Preview save filename', elem_id=tabname+"_preview_filename", visible=False)

    for tab in unrelated_tabs:
        tab.select(
            fn=lambda: [gr.update(visible=False) for _ in tab_controls],
            _js=f"function(){{ extraNetworksUnrelatedTabSelected('{tabname}'); }}",
            inputs=[],
            outputs=tab_controls,
            show_progress=False,
        )

    for page, tab in zip(ui.stored_extra_pages, related_tabs):
        allow_prompt = "true" if page.allow_prompt else "false"
        allow_negative_prompt = "true" if page.allow_negative_prompt else "false"

        jscode = (
            "extraNetworksTabSelected("
            f"'{tabname}', "
            f"'{tabname}_{page.id_page}_prompts', "
            f"'{allow_prompt}', "
            f"'{allow_negative_prompt}'"
            ");"
        )

        tab.select(
            fn=lambda: [gr.update(visible=True) for _ in tab_controls],
            _js="function(){ " + jscode + " }",
            inputs=[],
            outputs=tab_controls,
            show_progress=False,
        )

    dropdown_sort.change(fn=lambda: None, _js="function(){ applyExtraNetworkSort('" + tabname + "'); }")

    def create_html():
        ui.pages_contents = [pg.create_html(ui.tabname) for pg in ui.stored_extra_pages]

    def pages_html():
        if not ui.pages_contents:
            create_html()
        return ui.pages_contents

    def refresh():
        for pg in ui.stored_extra_pages:
            pg.refresh()
        create_html()
        return ui.pages_contents

    interface.load(fn=pages_html, inputs=[], outputs=ui.pages)
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
