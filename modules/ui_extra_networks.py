import functools
import html
import json
import os.path
import re
import urllib.parse
from base64 import b64decode
from io import BytesIO
from pathlib import Path
from typing import Callable, Optional

import gradio as gr
from fastapi.exceptions import HTTPException
from PIL import Image
from starlette.responses import FileResponse, JSONResponse, Response

from modules import (errors, extra_networks, shared,
                     ui_extra_networks_user_metadata, util)
from modules.images import read_info_from_image, save_image_with_geninfo
from modules.infotext_utils import image_from_url_text

extra_pages = []
allowed_dirs = set()
default_allowed_preview_extensions = ["png", "jpg", "jpeg", "webp", "gif"]


@functools.cache
def allowed_preview_extensions_with_extra(extra_extensions=None):
    return set(default_allowed_preview_extensions) | set(extra_extensions or [])


def allowed_preview_extensions():
    return allowed_preview_extensions_with_extra((shared.opts.samples_format,))


class ListItem:
    """
    Attributes:
        id [str]: The ID of this list item.
        html [str]: The HTML string for this item.
    """

    def __init__(self, _id: str, _html: str) -> None:
        self.id = _id
        self.html = _html


class CardListItem(ListItem):
    """
    Attributes:
        visible [bool]: Whether the item should be shown in the list.
        sort_keys [dict]: Nested dict where keys are sort modes and values are sort keys.
        search_terms [str]: String containing multiple search terms joined with spaces.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.visible: bool = False
        self.sort_keys = {}
        self.search_terms = ""


class TreeListItem(ListItem):
    """
    Attributes:
        visible [bool]: Whether the item should be shown in the list.
        expanded [bool]: Whether the item children should be shown.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.node: Optional[DirectoryTreeNode] = None
        self.visible: bool = False
        self.expanded: bool = False


class DirectoryTreeNode:
    def __init__(
        self,
        root_dir: str,
        abspath: str,
        parent: Optional["DirectoryTreeNode"] = None,
    ) -> None:
        self.root_dir = root_dir
        self.abspath = abspath
        self.parent = parent

        self.depth = 0
        self.is_dir = False
        self.item = None
        self.relpath = os.path.relpath(self.abspath, self.root_dir)
        self.children: list["DirectoryTreeNode"] = []

        # If a parent is passed, then we add this instance to the parent's children.
        if self.parent is not None:
            self.depth = self.parent.depth + 1
            self.parent.add_child(self)

    def add_child(self, child: "DirectoryTreeNode") -> None:
        self.children.append(child)

    def build(self, items: dict[str, dict]) -> None:
        """Builds a tree of nodes as children of this instance.

        Args:
            items: A dictionary where keys are absolute filepaths for directories/files.
                The values are dictionaries representing extra networks items.
        """
        self.is_dir = os.path.isdir(self.abspath)
        if self.is_dir:
            for x in os.listdir(self.abspath):
                child_path = os.path.join(self.abspath, x)
                # Add all directories but only add files if they are in the items dict.
                if os.path.isdir(child_path) or child_path in items:
                    DirectoryTreeNode(self.root_dir, child_path, self).build(items)
        else:
            self.item = items.get(self.abspath, None)

    def flatten(self, res: dict, dirs_only: bool = False) -> None:
        """Flattens the keys/values of the tree nodes into a dictionary.

        Args:
            res: The dictionary result updated in place. On initial call, should be passed
                as an empty dictionary.
            dirs_only: Whether to only add directories to the result.

        Raises:
            KeyError: If any nodes in the tree have the same ID.
        """
        if self.abspath in res:
            raise KeyError(f"duplicate key: {self.abspath}")

        if not dirs_only or (dirs_only and self.is_dir):
            res[self.abspath] = self

        for child in self.children:
            child.flatten(res, dirs_only)

    def apply(self, fn: Callable) -> None:
        """Recursively calls passed function with instance for entire tree."""
        fn(self)
        for child in self.children:
            child.apply(fn)


def register_page(page):
    """registers extra networks page for the UI

    recommend doing it in on_before_ui() callback for extensions
    """
    extra_pages.append(page)
    allowed_dirs.clear()
    allowed_dirs.update(set(sum([x.allowed_directories_for_previews() for x in extra_pages], [])))


def get_page_by_name(extra_networks_tabname: str = "") -> "ExtraNetworksPage":
    for page in extra_pages:
        if page.extra_networks_tabname == extra_networks_tabname:
            return page

    raise HTTPException(status_code=404, detail=f"Page not found: {extra_networks_tabname}")


def fetch_file(filename: str = ""):
    if not os.path.isfile(filename):
        raise HTTPException(status_code=404, detail="File not found")

    if not any(Path(x).absolute() in Path(filename).absolute().parents for x in allowed_dirs):
        raise ValueError(f"File cannot be fetched: {filename}. Must be in one of directories registered by extra pages.")

    ext = os.path.splitext(filename)[1].lower()[1:]
    if ext not in allowed_preview_extensions():
        raise ValueError(f"File cannot be fetched: {filename}. Extensions allowed: {allowed_preview_extensions()}.")

    # would profit from returning 304
    return FileResponse(filename, headers={"Accept-Ranges": "bytes"})


def fetch_cover_images(extra_networks_tabname: str = "", item: str = "", index: int = 0):
    page = get_page_by_name(extra_networks_tabname)

    metadata = page.metadata.get(item)
    if metadata is None:
        raise HTTPException(status_code=404, detail="File not found")

    cover_images = json.loads(metadata.get("ssmd_cover_images", {}))
    image = cover_images[index] if index < len(cover_images) else None
    if not image:
        raise HTTPException(status_code=404, detail="File not found")

    try:
        image = Image.open(BytesIO(b64decode(image)))
        buffer = BytesIO()
        image.save(buffer, format=image.format)
        return Response(content=buffer.getvalue(), media_type=image.get_format_mimetype())
    except Exception as err:
        raise ValueError(f"File cannot be fetched: {item}. Failed to load cover image.") from err


def init_tree_data(tabname: str = "", extra_networks_tabname: str = "") -> JSONResponse:
    page = get_page_by_name(extra_networks_tabname)

    data = page.generate_tree_view_data(tabname)

    if data is None:
        return JSONResponse({}, status_code=503)

    return JSONResponse(data, status_code=200)


def fetch_tree_data(
    extra_networks_tabname: str = "",
    div_ids: str = "",
) -> JSONResponse:
    page = get_page_by_name(extra_networks_tabname)

    res = {}
    for div_id in div_ids.split(","):
        if div_id in page.tree:
            res[div_id] = page.tree[div_id].html

    return JSONResponse(res)


def fetch_cards_data(
    extra_networks_tabname: str = "",
    div_ids: str = "",
) -> JSONResponse:
    page = get_page_by_name(extra_networks_tabname)

    res = {}
    for div_id in div_ids.split(","):
        if div_id in page.cards:
            res[div_id] = page.cards[div_id].html

    return JSONResponse(res)


def init_cards_data(tabname: str = "", extra_networks_tabname: str = "") -> JSONResponse:
    page = get_page_by_name(extra_networks_tabname)

    data = page.generate_cards_view_data(tabname)

    if data is None:
        return JSONResponse({}, status_code=503)

    return JSONResponse(data, status_code=200)


def page_is_ready(extra_networks_tabname: str = "") -> JSONResponse:
    page = get_page_by_name(extra_networks_tabname)

    try:
        items_list = list(page.list_items())
        if len(page.items) == len(items_list):
            return JSONResponse({}, status_code=200)

        return JSONResponse({"error": "page not ready"}, status_code=503)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


def get_metadata(extra_networks_tabname: str = "", item: str = "") -> JSONResponse:
    try:
        page = get_page_by_name(extra_networks_tabname)
    except HTTPException:
        return JSONResponse({})

    metadata = page.metadata.get(item)
    if metadata is None:
        return JSONResponse({})

    # those are cover images, and they are too big to display in UI as text
    # FIXME: WHY WAS THIS HERE?
    # metadata = {i: metadata[i] for i in metadata if i != 'ssmd_cover_images'}

    return JSONResponse({"metadata": json.dumps(metadata, indent=4, ensure_ascii=False)})


def get_single_card(tabname: str = "", extra_networks_tabname: str = "", name: str = "") -> JSONResponse:
    page = get_page_by_name(extra_networks_tabname)

    try:
        item = page.create_item(name, enable_filter=False)
        page.items[name] = item
    except Exception as exc:
        errors.display(exc, "creating item for extra network")
        item = page.items.get(name, None)

    if item is None:
        return JSONResponse({})

    page.read_user_metadata(item, use_cache=False)
    item_html = page.create_card_html(tabname=tabname, item=item)

    return JSONResponse({"html": item_html})


def add_pages_to_demo(app):
    app.add_api_route("/sd_extra_networks/thumb", fetch_file, methods=["GET"])
    app.add_api_route("/sd_extra_networks/cover-images", fetch_cover_images, methods=["GET"])
    app.add_api_route("/sd_extra_networks/metadata", get_metadata, methods=["GET"])
    app.add_api_route("/sd_extra_networks/get-single-card", get_single_card, methods=["GET"])
    app.add_api_route("/sd_extra_networks/init-tree-data", init_tree_data, methods=["GET"])
    app.add_api_route("/sd_extra_networks/init-cards-data", init_cards_data, methods=["GET"])
    app.add_api_route("/sd_extra_networks/fetch-tree-data", fetch_tree_data, methods=["GET"])
    app.add_api_route("/sd_extra_networks/fetch-cards-data", fetch_cards_data, methods=["GET"])
    app.add_api_route("/sd_extra_networks/page-is-ready", page_is_ready, methods=["GET"])


def quote_js(s):
    s = s.replace("\\", "\\\\")
    s = s.replace('"', '\\"')
    return f'"{s}"'


class ExtraNetworksPage:
    def __init__(self, title):
        self.title = title
        self.name = title.lower()
        # This is the actual name of the extra networks tab (not txt2img/img2img).
        self.extra_networks_tabname = self.name.replace(" ", "_")
        self.allow_prompt = True
        self.allow_negative_prompt = False
        self.metadata = {}
        self.items = {}
        self.cards = {}
        self.tree = {}
        self.tree_roots = {}
        self.lister = util.MassFileLister()
        # HTML Templates
        self.pane_tpl = shared.html("extra-networks-pane.html")
        self.card_tpl = shared.html("extra-networks-card.html")
        self.tree_row_tpl = shared.html("extra-networks-tree-row.html")
        self.btn_copy_path_tpl = shared.html("extra-networks-btn-copy-path.html")
        self.btn_show_metadata_tpl = shared.html("extra-networks-btn-show-metadata.html")
        self.btn_edit_metadata_tpl = shared.html("extra-networks-btn-edit-metadata.html")
        self.btn_dirs_view_item_tpl = shared.html("extra-networks-btn-dirs-view-item.html")
        # Sorted lists
        # These just store ints so it won't use hardly any memory to just sort ahead
        # of time for each sort mode. These are lists of keys for each file.
        self.keys_sorted = {}
        self.keys_by_name = []
        self.keys_by_path = []
        self.keys_by_created = []
        self.keys_by_modified = []

    def refresh(self):
        self.items = {}
        self.cards = {}
        self.tree = {}
        self.tree_roots = {}

    def read_user_metadata(self, item, use_cache=True):
        filename = item.get("filename", None)
        metadata = extra_networks.get_user_metadata(filename, lister=self.lister if use_cache else None)

        desc = metadata.get("description", None)
        if desc is not None:
            item["description"] = desc

        item["user_metadata"] = metadata

    def link_preview(self, filename):
        quoted_filename = urllib.parse.quote(filename.replace("\\", "/"))
        mtime, _ = self.lister.mctime(filename)
        return f"./sd_extra_networks/thumb?filename={quoted_filename}&mtime={mtime}"

    def search_terms_from_path(self, filename, possible_directories=None):
        abspath = os.path.abspath(filename)
        for parentdir in possible_directories if possible_directories is not None else self.allowed_directories_for_previews():
            parentdir = os.path.dirname(os.path.abspath(parentdir))
            if abspath.startswith(parentdir):
                return os.path.relpath(abspath, parentdir)

        return ""

    def build_tree_html_dict_row(
        self,
        tabname: str,
        label: str,
        btn_type: str,
        btn_title: Optional[str] = None,
        data_attributes: Optional[dict] = None,
        dir_is_empty: bool = False,
        item: Optional[dict] = None,
        onclick_extra: Optional[str] = None,
    ) -> str:
        if btn_type not in ["file", "dir"]:
            raise ValueError("Invalid button type:", btn_type)

        if data_attributes is None:
            data_attributes = {}

        label = label.strip()
        # If not specified, title will just reflect the label
        btn_title = btn_title.strip() if btn_title else label

        action_list_item_action_leading = "<i class='tree-list-item-action-chevron'></i>"
        action_list_item_visual_leading = "🗀"
        action_list_item_visual_trailing = ""
        action_list_item_action_trailing = ""

        if dir_is_empty:
            action_list_item_action_leading = "<i class='tree-list-item-action-chevron' style='visibility: hidden'></i>"

        if btn_type == "file":
            action_list_item_visual_leading = "🗎"
            # Action buttons
            if item is not None:
                action_list_item_action_trailing += self.get_button_row(tabname, item)

        data_attributes_str = ""
        for k, v in data_attributes.items():
            if isinstance(v, (bool,)):
                # Boolean data attributes only need a key when true.
                if v:
                    data_attributes_str += f"{k} "
            elif v not in [None, "", "''", '""']:
                data_attributes_str += f"{k}={v} "

        res = self.tree_row_tpl.format(
            **{
                "data_attributes": data_attributes_str,
                "search_terms": "",
                "btn_type": btn_type,
                "btn_title": btn_title,
                "tabname": tabname,
                "onclick_extra": onclick_extra if onclick_extra else "",
                "extra_networks_tabname": self.extra_networks_tabname,
                "action_list_item_action_leading": action_list_item_action_leading,
                "action_list_item_visual_leading": action_list_item_visual_leading,
                "action_list_item_label": label,
                "action_list_item_visual_trailing": action_list_item_visual_trailing,
                "action_list_item_action_trailing": action_list_item_action_trailing,
            }
        )
        res = res.strip()
        res = re.sub(" +", " ", res.replace("\n", ""))
        return res

    def get_button_row(self, tabname: str, item: dict) -> str:
        metadata = item.get("metadata", None)
        name = item.get("name", "")
        filename = item.get("filename", "")

        button_row_tpl = '<div class="button-row">{btn_copy_path}{btn_edit_item}{btn_metadata}</div>'

        btn_copy_path = self.btn_copy_path_tpl.format(filename=filename)
        btn_edit_item = self.btn_edit_metadata_tpl.format(
            tabname=tabname,
            extra_networks_tabname=self.extra_networks_tabname,
            name=name,
        )
        btn_metadata = ""
        if metadata:
            btn_metadata = self.btn_show_metadata_tpl.format(
                extra_networks_tabname=self.extra_networks_tabname,
                name=name,
            )

        return button_row_tpl.format(
            btn_copy_path=btn_copy_path,
            btn_edit_item=btn_edit_item,
            btn_metadata=btn_metadata,
        )

    def create_card_html(
        self,
        tabname: str,
        item: dict,
        div_id: Optional[str] = None,
    ) -> str:
        """Generates HTML for a single ExtraNetworks Item.

        Args:
            tabname: The name of the active tab.
            item: Dictionary containing item information.
            template: Optional template string to use.

        Returns:
            HTML string generated for this item. Can be empty if the item is not meant to be shown.
        """
        style = f"font-size: {shared.opts.extra_networks_card_text_scale*100}%;"
        if shared.opts.extra_networks_card_height:
            style += f"height: {shared.opts.extra_networks_card_height}px;"
        if shared.opts.extra_networks_card_width:
            style += f"width: {shared.opts.extra_networks_card_width}px;"

        background_image = ""
        preview = html.escape(item.get("preview", "") or "")
        if preview:
            background_image = f'<img src="{preview}" class="preview" loading="lazy">'

        onclick = item.get("onclick", None)
        if onclick is None:
            onclick = html.escape(f"extraNetworksCardOnClick(event, '{tabname}_{self.extra_networks_tabname}');")

        button_row = self.get_button_row(tabname, item)

        filename = item.get("filename", "")
        # if this is true, the item must not be shown in the default view,
        # and must instead only be shown when searching for it
        if shared.opts.extra_networks_hidden_models == "Always":
            search_only = False
        else:
            search_only = filename.startswith(".")

        if search_only and shared.opts.extra_networks_hidden_models == "Never":
            return ""

        sort_keys = {}
        for sort_mode, sort_key in item.get("sort_keys", {}).items():
            sort_keys[sort_mode.strip().lower()] = html.escape(str(sort_key))

        search_terms_html = ""
        search_terms_tpl = "<span class='hidden {class}'>{search_term}</span>"
        for search_term in item.get("search_terms", []):
            search_terms_html += search_terms_tpl.format(
                **{
                    "class": f"search_terms{' search_only' if search_only else ''}",
                    "search_term": search_term,
                }
            )

        description = ""
        if shared.opts.extra_networks_card_show_desc:
            description = item.get("description", "") or ""

        if not shared.opts.extra_networks_card_description_is_html:
            description = html.escape(description)

        data_attributes = {
            "data-div-id": div_id if div_id else "",
            "data-name": item.get("name", "").strip(),
            "data-path": item.get("filename", "").strip(),
            "data-hash": item.get("shorthash", None),
            "data-prompt": item.get("prompt", "").strip(),
            "data-neg-prompt": item.get("negative_prompt", "").strip(),
            "data-allow-neg": self.allow_negative_prompt,
            **{f"data-sort-{sort_mode}": sort_key for sort_mode, sort_key in sort_keys.items()},
        }

        data_attributes_str = ""
        for k, v in data_attributes.items():
            if isinstance(v, (bool,)):
                # Boolean data attributes only need a key when true.
                if v:
                    data_attributes_str += f"{k} "
            elif v not in [None, "", "''", '""']:
                data_attributes_str += f"{k}={v} "

        return self.card_tpl.format(
            style=style,
            onclick=onclick,
            data_attributes=data_attributes_str,
            background_image=background_image,
            button_row=button_row,
            search_terms=search_terms_html,
            name=html.escape(item["name"].strip()),
            description=description,
        )

    def generate_cards_view_data(self, tabname: str) -> dict:
        for i, item in enumerate(self.items.values()):
            div_id = str(i)
            card_html = self.create_card_html(tabname=tabname, item=item, div_id=div_id)
            sort_keys = {k.strip().lower().replace(" ", "_"): html.escape(str(v)) for k, v in item.get("sort_keys", {}).items()}
            search_terms = item.get("search_terms", [])
            self.cards[div_id] = CardListItem(div_id, card_html)
            self.cards[div_id].sort_keys = sort_keys
            self.cards[div_id].search_terms = " ".join(search_terms)

        # Sort cards for all sort modes
        sort_modes = ["name", "path", "date_created", "date_modified"]
        for mode in sort_modes:
            self.keys_sorted[mode] = sorted(
                self.cards.keys(),
                key=lambda k: shared.natural_sort_key(self.cards[k].sort_keys[mode]),
            )

        res = {}
        for div_id, card_item in self.cards.items():
            res[div_id] = {
                **{f"sort_{mode}": key for mode, key in card_item.sort_keys.items()},
                "search_terms": card_item.search_terms,
                "visible": True,
            }
        return res

    def generate_tree_view_data(self, tabname: str) -> dict:
        if not self.tree_roots:
            return {}

        # Flatten each root into a single dict
        tree = {}
        for node in self.tree_roots.values():
            subtree = {}
            node.flatten(subtree)
            tree.update(subtree)

        path_to_div_id = {}
        div_id_to_node = {}  # reverse mapping
        # First assign div IDs to each node. Used for parent ID lookup later.
        for i, path in enumerate(sorted(tree.keys(), key=shared.natural_sort_key)):
            div_id = str(i)
            path_to_div_id[path] = div_id
            div_id_to_node[div_id] = tree[path]

        show_files = shared.opts.extra_networks_tree_view_show_files is True
        for div_id, node in div_id_to_node.items():
            tree_item = TreeListItem(div_id, "")
            tree_item.node = node
            parent_id = None
            if node.parent is not None:
                parent_id = path_to_div_id.get(node.parent.abspath, None)

            if node.is_dir:  # directory
                if show_files:
                    dir_is_empty = node.children == []
                else:
                    dir_is_empty = all(not x.is_dir for x in node.children)
                tree_item.html = self.build_tree_html_dict_row(
                    tabname=tabname,
                    label=os.path.basename(node.abspath),
                    btn_type="dir",
                    btn_title=node.abspath,
                    dir_is_empty=dir_is_empty,
                    data_attributes={
                        "data-div-id": div_id,
                        "data-parent-id": parent_id,
                        "data-tree-entry-type": "dir",
                        "data-depth": node.depth,
                        "data-path": node.relpath,
                        "data-expanded": node.parent is None,  # Expand root directories
                    },
                )
                self.tree[div_id] = tree_item
            else:  # file
                if not show_files:
                    # Don't add file if files are disabled in the options.
                    continue

                onclick = node.item.get("onclick", None)
                if onclick is None:
                    onclick = html.escape(f"extraNetworksCardOnClick(event, '{tabname}_{self.extra_networks_tabname}');")

                item_name = node.item.get("name", "").strip()
                tree_item.html = self.build_tree_html_dict_row(
                    tabname=tabname,
                    label=html.escape(item_name),
                    btn_type="file",
                    data_attributes={
                        "data-div-id": div_id,
                        "data-parent-id": parent_id,
                        "data-tree-entry-type": "file",
                        "data-name": item_name,
                        "data-depth": node.depth,
                        "data-path": node.item.get("filename", "").strip(),
                        "data-hash": node.item.get("shorthash", None),
                        "data-prompt": node.item.get("prompt", "").strip(),
                        "data-neg-prompt": node.item.get("negative_prompt", "").strip(),
                        "data-allow-neg": self.allow_negative_prompt,
                    },
                    item=node.item,
                    onclick_extra=onclick,
                )
                self.tree[div_id] = tree_item

        res = {}

        # Expand all root directories and set them to active so they are displayed.
        for path in self.tree_roots.keys():
            div_id = path_to_div_id[path]
            self.tree[div_id].expanded = True
            self.tree[div_id].visible = True
            # Set all direct children to active
            for child_node in self.tree[div_id].node.children:
                self.tree[path_to_div_id[child_node.abspath]].visible = True

        for div_id, tree_item in self.tree.items():
            # Expand root nodes and make them visible.
            expanded = tree_item.node.parent is None
            visible = tree_item.node.parent is None
            parent_id = None
            if tree_item.node.parent is not None:
                parent_id = path_to_div_id[tree_item.node.parent.abspath]
                # Direct children of root nodes should be visible by default.
                if self.tree[parent_id].node.parent is None:
                    visible = True

            res[div_id] = {
                "parent": parent_id,
                "children": [path_to_div_id[child.abspath] for child in tree_item.node.children],
                "visible": visible,
                "expanded": expanded,
            }
        return res

    def create_dirs_view_html(self, tabname: str) -> str:
        """Generates HTML for displaying folders."""
        # Flatten each root into a single dict. Only get the directories for buttons.
        tree = {}
        for node in self.tree_roots.values():
            subtree = {}
            node.flatten(subtree, dirs_only=True)
            tree.update(subtree)

        # Sort the tree nodes by relative paths
        dir_nodes = sorted(
            tree.values(),
            key=lambda x: shared.natural_sort_key(x.relpath),
        )
        dirs_html = "".join(
            [
                self.btn_dirs_view_item_tpl.format(
                    **{
                        "extra_class": "search-all" if node.relpath == "" else "",
                        "tabname_full": f"{tabname}_{self.extra_networks_tabname}",
                        "path": html.escape(node.relpath),
                    }
                )
                for node in dir_nodes
            ]
        )

        return dirs_html

    def create_html(self, tabname, *, empty=False):
        """Generates an HTML string for the current pane.

        The generated HTML uses `extra-networks-pane.html` as a template.

        Args:
            tabname: The name of the active tab.
            empty: create an empty HTML page with no items

        Returns:
            HTML formatted string.
        """
        self.lister.reset()
        self.metadata = {}

        items_list = [] if empty else self.list_items()
        self.items = {x["name"]: x for x in items_list}

        # Populate the instance metadata for each item.
        for item in self.items.values():
            metadata = item.get("metadata")
            if metadata:
                self.metadata[item["name"]] = metadata

            if "user_metadata" not in item:
                self.read_user_metadata(item)

        # Setup the tree dictionary.
        tree_items = {v["filename"]: v for v in self.items.values()}
        # Create a DirectoryTreeNode for each root directory since they might not share
        # a common path.
        for path in self.allowed_directories_for_previews():
            abspath = os.path.abspath(path)
            if not os.path.exists(abspath):
                continue
            self.tree_roots[abspath] = DirectoryTreeNode(os.path.dirname(abspath), abspath, None)
            self.tree_roots[abspath].build(tree_items if shared.opts.extra_networks_tree_view_show_files else {})

        # Generate the html for displaying directory buttons
        dirs_html = self.create_dirs_view_html(tabname)

        sort_mode = shared.opts.extra_networks_card_order_field.lower().strip().replace(" ", "_")
        sort_dir = shared.opts.extra_networks_card_order.lower().strip()
        dirs_view_en = shared.opts.extra_networks_dirs_view_default_enabled
        tree_view_en = shared.opts.extra_networks_tree_view_default_enabled

        return self.pane_tpl.format(
            **{
                "tabname": tabname,
                "extra_networks_tabname": self.extra_networks_tabname,
                "data_sort_dir": sort_dir,
                "btn_sort_mode_path_data_attributes": "data-selected" if sort_mode == "path" else "",
                "btn_sort_mode_name_data_attributes": "data-selected" if sort_mode == "name" else "",
                "btn_sort_mode_date_created_data_attributes": "data-selected" if sort_mode == "date_created" else "",
                "btn_sort_mode_date_modified_data_attributes": "data-selected" if sort_mode == "date_modified" else "",
                "btn_dirs_view_data_attributes": "data-selected" if dirs_view_en else "",
                "btn_tree_view_data_attributes": "data-selected" if tree_view_en else "",
                "dirs_view_hidden_cls": "" if dirs_view_en else "hidden",
                "tree_view_hidden_cls": "" if tree_view_en else "hidden",
                "tree_view_style": f"flex-basis: {shared.opts.extra_networks_tree_view_default_width}px;",
                "cards_view_style": "flex-grow: 1;",
                "dirs_html": dirs_html,
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
            "path": str(pth).lower(),
        }

    def find_preview(self, path):
        """
        Find a preview PNG for a given path (without extension) and call link_preview on it.
        """

        potential_files = sum([[f"{path}.{ext}", f"{path}.preview.{ext}"] for ext in allowed_preview_extensions()], [])

        for file in potential_files:
            if self.lister.exists(file):
                return self.link_preview(file)

        return None

    def find_embedded_preview(self, path, name, metadata):
        """
        Find if embedded preview exists in safetensors metadata and return endpoint for it.
        """

        file = f"{path}.safetensors"
        if (
            self.lister.exists(file)
            and "ssmd_cover_images" in metadata
            and len(list(filter(None, json.loads(metadata["ssmd_cover_images"])))) > 0
        ):
            return f"./sd_extra_networks/cover-images?extra_networks_tabname={self.extra_networks_tabname}&item={name}"

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
    from modules.ui_extra_networks_checkpoints import \
        ExtraNetworksPageCheckpoints
    from modules.ui_extra_networks_hypernets import \
        ExtraNetworksPageHypernetworks
    from modules.ui_extra_networks_textual_inversion import \
        ExtraNetworksPageTextualInversion

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
    ui = ExtraNetworksUi()
    ui.pages = []
    ui.pages_contents = []
    ui.user_metadata_editors = []
    ui.stored_extra_pages = pages_in_preferred_order(extra_pages.copy())
    ui.tabname = tabname

    related_tabs = []

    for page in ui.stored_extra_pages:
        with gr.Tab(page.title, elem_id=f"{tabname}_{page.extra_networks_tabname}", elem_classes=["extra-page"]) as tab:
            with gr.Column(elem_id=f"{tabname}_{page.extra_networks_tabname}_prompts", elem_classes=["extra-page-prompts"]):
                pass

            elem_id = f"{tabname}_{page.extra_networks_tabname}_pane_container"
            page_elem = gr.HTML(page.create_html(tabname, empty=True), elem_id=elem_id)
            ui.pages.append(page_elem)
            editor = page.create_user_metadata_editor(ui, tabname)
            editor.create_ui()
            ui.user_metadata_editors.append(editor)
            related_tabs.append(tab)

    ui.button_save_preview = gr.Button("Save preview", elem_id=f"{tabname}_save_preview", visible=False)
    ui.preview_target_filename = gr.Textbox("Preview save filename", elem_id=f"{tabname}_preview_filename", visible=False)

    for tab in unrelated_tabs:
        tab.select(
            fn=None,
            _js=f"function(){{extraNetworksUnrelatedTabSelected('{tabname}');}}",
            inputs=[],
            outputs=[],
            show_progress=False,
        )

    for page, tab in zip(ui.stored_extra_pages, related_tabs):
        tab.select(
            fn=None,
            _js=(
                "function(){extraNetworksTabSelected("
                f"'{tabname}_{page.extra_networks_tabname}', "
                f"{str(page.allow_prompt).lower()}, "
                f"{str(page.allow_negative_prompt).lower()}"
                ");}"
            ),
            inputs=[],
            outputs=[],
            show_progress=False,
        )

        def refresh():
            for pg in ui.stored_extra_pages:
                pg.refresh()
            create_html()
            return ui.pages_contents

        button_refresh = gr.Button("Refresh", elem_id=f"{tabname}_{page.extra_networks_tabname}_extra_refresh_internal", visible=False)
        button_refresh.click(fn=refresh, inputs=[], outputs=ui.pages,).then(fn=lambda: None, _js="setupAllResizeHandles").then(
            fn=lambda: None,
            _js=f"function(){{extraNetworksRefreshTab('{tabname}_{page.extra_networks_tabname}');}}",
        )

    def create_html():
        ui.pages_contents = [pg.create_html(ui.tabname) for pg in ui.stored_extra_pages]

    def pages_html():
        if not ui.pages_contents:
            create_html()
        return ui.pages_contents

    interface.load(fn=pages_html, inputs=[], outputs=ui.pages,).then(
        fn=lambda: None,
        _js="setupAllResizeHandles",
    )

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

        assert is_allowed, f"writing to {filename} is not allowed"

        save_image_with_geninfo(image, geninfo, filename)

        return [page.create_html(ui.tabname) for page in ui.stored_extra_pages]

    ui.button_save_preview.click(
        fn=save_preview,
        _js="function(x, y, z){return [selected_gallery_index(), y, z]}",
        inputs=[ui.preview_target_filename, gallery, ui.preview_target_filename],
        outputs=[*ui.pages],
    )

    for editor in ui.user_metadata_editors:
        editor.setup_ui(gallery)
