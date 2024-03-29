import functools
import os.path
import urllib.parse
from base64 import b64decode
from io import BytesIO
from pathlib import Path
from typing import Optional, Union
from dataclasses import dataclass
import zlib
import base64
import re
from starlette.responses import JSONResponse, PlainTextResponse

from modules import shared, ui_extra_networks_user_metadata, errors, extra_networks, util
from modules.images import read_info_from_image, save_image_with_geninfo
import gradio as gr
import json
import html
from fastapi.exceptions import HTTPException
from PIL import Image

from modules.infotext_utils import image_from_url_text

extra_pages = []
allowed_dirs = set()
default_allowed_preview_extensions = ["png", "jpg", "jpeg", "webp", "gif"]

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
    """registers extra networks page for the UI

    recommend doing it in on_before_ui() callback for extensions
    """
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


def fetch_cover_images(page: str = "", item: str = "", index: int = 0):
    from starlette.responses import Response

    page = next(iter([x for x in extra_pages if x.name == page]), None)
    if page is None:
        raise HTTPException(status_code=404, detail="File not found")

    metadata = page.metadata.get(item)
    if metadata is None:
        raise HTTPException(status_code=404, detail="File not found")

    cover_images = json.loads(metadata.get('ssmd_cover_images', {}))
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


def get_list_data(
    tabname: str = "",
    extra_networks_tabname: str = "",
    list_type: Optional[str] = None,
) -> PlainTextResponse:
    """Responds to API GET requests on /sd_extra_networks/get-list-data with list data.

    Args:
        tabname:                The primary tab name containing the data.
                                (i.e. txt2img, img2img)
        extra_networks_tabname: The selected extra networks tabname.
                                (i.e. lora, hypernetworks, etc.)
        list_type:              The type of list data to retrieve. This reflects the
                                class name used in `extraNetworksClusterizeList.js`.

    Returns:
        The string data result along with a status code.
        A status_code of 501 is returned on error, 200 on success.
    """
    res = ""
    status_code = 200

    page = next(iter([
        x for x in extra_pages
        if x.extra_networks_tabname == extra_networks_tabname
    ]), None)

    if page is None:
        return PlainTextResponse(res, status_code=501)

    if list_type == "ExtraNetworksClusterizeTreeList":
        res = page.generate_tree_view_data(tabname)
    elif list_type == "ExtraNetworksClusterizeCardsList":
        res = page.generate_cards_view_data(tabname)
    else:
        status_code = 501  # HTTP_501_NOT_IMPLEMENTED

    return PlainTextResponse(res, status_code=status_code)


def get_metadata(page: str = "", item: str = "") -> JSONResponse:
    page = next(iter([x for x in extra_pages if x.name == page]), None)
    if page is None:
        return JSONResponse({})

    metadata = page.metadata.get(item)
    if metadata is None:
        return JSONResponse({})

    metadata = {i:metadata[i] for i in metadata if i != 'ssmd_cover_images'}  # those are cover images, and they are too big to display in UI as text

    return JSONResponse({"metadata": json.dumps(metadata, indent=4, ensure_ascii=False)})


def get_single_card(page: str = "", tabname: str = "", name: str = "") -> JSONResponse:
    page = next(iter([x for x in extra_pages if x.name == page]), None)

    try:
        item = page.create_item(name, enable_filter=False)
        page.items[name] = item
    except Exception as e:
        errors.display(e, "creating item for extra network")
        item = page.items.get(name)

    page.read_user_metadata(item, use_cache=False)
    item_html = page.create_card_html(tabname=tabname, item=item)

    return JSONResponse({"html": item_html})

def add_pages_to_demo(app):
    app.add_api_route("/sd_extra_networks/thumb", fetch_file, methods=["GET"])
    app.add_api_route("/sd_extra_networks/cover-images", fetch_cover_images, methods=["GET"])
    app.add_api_route("/sd_extra_networks/metadata", get_metadata, methods=["GET"])
    app.add_api_route("/sd_extra_networks/get-single-card", get_single_card, methods=["GET"])
    app.add_api_route("/sd_extra_networks/get-list-data", get_list_data, methods=["GET"])


def quote_js(s):
    s = s.replace('\\', '\\\\')
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
        self.tree = {}
        self.lister = util.MassFileLister()
        # HTML Templates
        self.pane_tpl = shared.html("extra-networks-pane.html")
        self.pane_content_tree_tpl = shared.html("extra-networks-pane-tree.html")
        self.pane_content_dirs_tpl = shared.html("extra-networks-pane-dirs.html")
        self.card_tpl = shared.html("extra-networks-card.html")
        self.tree_row_tpl = shared.html("extra-networks-tree-row.html")
        self.btn_copy_path_tpl = shared.html("extra-networks-copy-path-button.html")
        self.btn_metadata_tpl = shared.html("extra-networks-metadata-button.html")
        self.btn_edit_item_tpl = shared.html("extra-networks-edit-item-button.html")
        self.btn_dirs_view_tpl = shared.html("extra-networks-dirs-view-button.html")

    def refresh(self):
        pass

    def read_user_metadata(self, item, use_cache=True):
        filename = item.get("filename", None)
        metadata = extra_networks.get_user_metadata(filename, lister=self.lister if use_cache else None)

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

    def build_tree_html_dict_row(
        self,
        tabname: str,
        label: str,
        btn_type: str,
        data_attributes: Optional[dict] = None,
        btn_title: Optional[str] = None,
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
            elif v not in [None, "", "\'\'", "\"\""]:
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

    def build_tree_html_dict(
        self,
        tree: dict,
        res: dict,
        tabname: str,
        div_id: int,
        depth: int,
        parent_id: Optional[int] = None,
    ) -> int:
        for k, v in sorted(tree.items(), key=lambda x: shared.natural_sort_key(x[0])):
            if not isinstance(v, (ExtraNetworksItem,)):
                # dir
                if div_id in res:
                    raise KeyError("div_id already in res:", div_id)

                dir_is_empty = True
                for _v in v.values():
                    if shared.opts.extra_networks_tree_view_show_files:
                        dir_is_empty = False
                        break
                    elif not isinstance(_v, (ExtraNetworksItem,)):
                        dir_is_empty = False
                        break
                    else:
                        dir_is_empty = True

                res[div_id] = self.build_tree_html_dict_row(
                    tabname=tabname,
                    label=os.path.basename(k),
                    btn_type="dir",
                    btn_title=k,
                    dir_is_empty=dir_is_empty,
                    data_attributes={
                        "data-div-id": div_id,
                        "data-parent-id": parent_id,
                        "data-tree-entry-type": "dir",
                        "data-depth": depth,
                        "data-path": k,
                        "data-expanded": parent_id is None,  # Expand root directories
                    },
                )
                last_div_id = self.build_tree_html_dict(
                    tree=v,
                    res=res,
                    depth=depth + 1,
                    div_id=div_id + 1,
                    parent_id=div_id,
                    tabname=tabname,
                )
                div_id = last_div_id
            else:
                # file
                if not shared.opts.extra_networks_tree_view_show_files:
                    # Don't add file if showing files is disabled in options.
                    continue

                if div_id in res:
                    raise KeyError("div_id already in res:", div_id)

                onclick = v.item.get("onclick", None)
                if onclick is None:
                    # Don't quote prompt/neg_prompt since they are stored as js strings already.
                    onclick = html.escape(f"extraNetworksCardOnClick(event, '{tabname}');")

                res[div_id] = self.build_tree_html_dict_row(
                    tabname=tabname,
                    label=html.escape(v.item.get("name", "").strip()),
                    btn_type="file",
                    data_attributes={
                        "data-div-id": div_id,
                        "data-parent-id": parent_id,
                        "data-tree-entry-type": "file",
                        "data-name": v.item.get("name", "").strip(),
                        "data-depth": depth,
                        "data-path": v.item.get("filename", "").strip(),
                        "data-hash": v.item.get("shorthash", None),
                        "data-prompt": v.item.get("prompt", "").strip(),
                        "data-neg-prompt": v.item.get("negative_prompt", "").strip(),
                        "data-allow-neg": self.allow_negative_prompt,
                    },
                    item=v.item,
                    onclick_extra=onclick,
                )
            div_id += 1
        return div_id

    def get_button_row(self, tabname: str, item: dict) -> str:
        metadata = item.get("metadata", None)
        name = item.get("name", "")
        filename = item.get("filename", "")

        button_row_tpl = '<div class="button-row">{btn_copy_path}{btn_edit_item}{btn_metadata}</div>'

        btn_copy_path = self.btn_copy_path_tpl.format(filename=filename)
        btn_edit_item = self.btn_edit_item_tpl.format(
            tabname=tabname,
            extra_networks_tabname=self.extra_networks_tabname,
            name=name,
        )
        btn_metadata = ""
        if metadata:
            btn_metadata = self.btn_metadata_tpl.format(
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
        div_id: Optional[int] = None,
    ) -> str:
        """Generates HTML for a single ExtraNetworks Item.

        Args:
            tabname: The name of the active tab.
            item: Dictionary containing item information.
            template: Optional template string to use.

        Returns:
            HTML string generated for this item. Can be empty if the item is not meant to be shown.
        """
        preview = item.get("preview", None)
        style_height = f"height: {shared.opts.extra_networks_card_height}px;" if shared.opts.extra_networks_card_height else ''
        style_width = f"width: {shared.opts.extra_networks_card_width}px;" if shared.opts.extra_networks_card_width else ''
        style_font_size = f"font-size: {shared.opts.extra_networks_card_text_scale*100}%;"
        style = style_height + style_width + style_font_size
        background_image = f'<img src="{html.escape(preview)}" class="preview" loading="lazy">' if preview else ''

        onclick = item.get("onclick", None)
        if onclick is None:
            # Don't quote prompt/neg_prompt since they are stored as js strings already.
            onclick = html.escape(f"extraNetworksCardOnClick(event, '{tabname}');")

        button_row = self.get_button_row(tabname, item)

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

        sort_keys = " ".join(
            [
                f'data-sort-{k}="{html.escape(str(v))}"'
                for k, v in item.get("sort_keys", {}).items()
            ]
        ).strip()

        search_terms_html = ""
        search_terms_tpl = "<span class='hidden {class}'>{search_term}</span>"
        for search_term in item.get("search_terms", []):
            search_terms_html += search_terms_tpl.format(
                **{
                    "class": f"search_terms{' search_only' if search_only else ''}",
                    "search_term": search_term,
                }
            )

        description = (item.get("description", "") or "" if shared.opts.extra_networks_card_show_desc else "")
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
        }

        data_attributes_str = ""
        for k, v in data_attributes.items():
            if isinstance(v, (bool,)):
                # Boolean data attributes only need a key when true.
                if v:
                    data_attributes_str += f"{k} "
            elif v not in [None, "", "\'\'", "\"\""]:
                data_attributes_str += f"{k}={v} "

        return self.card_tpl.format(
            style=style,
            onclick=onclick,
            data_attributes=data_attributes_str,
            sort_keys=sort_keys,
            background_image=background_image,
            button_row=button_row,
            search_terms=search_terms_html,
            name=html.escape(item["name"].strip()),
            description=description,
        )

    def generate_tree_view_data(self, tabname: str) -> str:
        """Generates tree view HTML as a base64 encoded zlib compressed json string."""
        res = {}

        if self.tree:
            self.build_tree_html_dict(
                tree=self.tree,
                res=res,
                depth=0,
                div_id=0,
                parent_id=None,
                tabname=tabname,
            )

        return base64.b64encode(
            zlib.compress(
                json.dumps(res, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
            )
        ).decode("utf-8")

    def generate_tree_view_data_div(self, tabname: str) -> str:
        """Generates HTML for displaying folders in a tree view.

        Args:
            tabname: The name of the active tab.

        Returns:
            HTML string generated for this tree view.
        """
        tpl = """<div id="{tabname_full}_tree_list_data"
            class="extra-network-script-data"
            data-tabname-full={tabname_full}
            data-proxy-name=tree_list
            data-json={data}
            hidden></div>"""
        return tpl.format(
            tabname_full=f"{tabname}_{self.extra_networks_tabname}",
            data=self.generate_tree_view_data(tabname),
        )

    def create_dirs_view_html(self, tabname: str) -> str:
        """Generates HTML for displaying folders."""

        def _get_dirs_buttons(tree: dict, res: list) -> None:
            """ Builds a list of directory names from a tree. """
            for k, v in sorted(tree.items(), key=lambda x: shared.natural_sort_key(x[0])):
                if not isinstance(v, (ExtraNetworksItem,)):
                    # dir
                    res.append(k)
                    _get_dirs_buttons(tree=v, res=res)

        dirs = []
        _get_dirs_buttons(tree=self.tree, res=dirs)

        dirs_html = "".join([
            self.btn_dirs_view_tpl.format(**{
                "extra_class": "search-all" if d == "" else "",
                "tabname_full": f"{tabname}_{self.extra_networks_tabname}",
                "path": html.escape(d),
            }) for d in dirs
        ])
        return dirs_html

    def generate_cards_view_data(self, tabname: str) -> str:
        res = {}
        for i, item in enumerate(self.items.values()):
            res[i] = self.create_card_html(tabname=tabname, item=item, div_id=i)

        return base64.b64encode(
            zlib.compress(
                json.dumps(res, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
            )
        ).decode("utf-8")

    def generate_cards_view_data_div(self, tabname: str, none_message: Optional[str]) -> str:
        """Generates HTML for the network Card View section for a tab.

        This HTML goes into the `extra-networks-pane.html` <div> with
        `id='{tabname}_{extra_networks_tabname}_cards`.

        Args:
            tabname: The name of the active tab.
            none_message: HTML text to show when there are no cards.

        Returns:
            HTML formatted string.
        """
        res = self.generate_cards_view_data(tabname)

        return f'<div id="{tabname}_{self.extra_networks_tabname}_cards_list_data" class="extra-network-script-data" data-tabname-full={tabname}_{self.extra_networks_tabname} data-proxy-name=cards_list data-json={res} hidden></div>'

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
        roots = self.allowed_directories_for_previews()
        tree_items = {v["filename"]: ExtraNetworksItem(v) for v in self.items.values()}
        tree = get_tree([os.path.abspath(x) for x in roots], items=tree_items)
        self.tree = tree

        # Generate the html for displaying directory buttons
        dirs_html = self.create_dirs_view_html(tabname)

        sort_mode = shared.opts.extra_networks_card_order_field.lower().strip().replace(" ", "_")
        sort_dir = shared.opts.extra_networks_card_order.lower().strip()
        dirs_view_en = shared.opts.extra_networks_dirs_view_default_enabled
        tree_view_en = shared.opts.extra_networks_tree_view_default_enabled

        return self.pane_tpl.format(**{
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
        })

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
        if self.lister.exists(file) and 'ssmd_cover_images' in metadata and len(list(filter(None, json.loads(metadata['ssmd_cover_images'])))) > 0:
            return f"./sd_extra_networks/cover-images?page={self.extra_networks_tabname}&item={name}"

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

    ui.button_save_preview = gr.Button('Save preview', elem_id=f"{tabname}_save_preview", visible=False)
    ui.preview_target_filename = gr.Textbox('Preview save filename', elem_id=f"{tabname}_preview_filename", visible=False)

    for tab in unrelated_tabs:
        tab.select(fn=None, _js=f"function(){{extraNetworksUnrelatedTabSelected('{tabname}');}}", inputs=[], outputs=[], show_progress=False)

    for page, tab in zip(ui.stored_extra_pages, related_tabs):
        jscode = (
            "function(){"
            f"extraNetworksTabSelected('{tabname}', '{tabname}_{page.extra_networks_tabname}_prompts', {str(page.allow_prompt).lower()}, {str(page.allow_negative_prompt).lower()}, '{tabname}_{page.extra_networks_tabname}');"
            "}"
        )
        tab.select(fn=None, _js=jscode, inputs=[], outputs=[], show_progress=False)

        def refresh():
            for pg in ui.stored_extra_pages:
                pg.refresh()
            create_html()
            return ui.pages_contents

        button_refresh = gr.Button("Refresh", elem_id=f"{tabname}_{page.extra_networks_tabname}_extra_refresh_internal", visible=False)
        button_refresh.click(
            fn=refresh,
            inputs=[],
            outputs=ui.pages,
        ).then(
            fn=lambda: None,
            _js='setupAllResizeHandles'
        ).then(
            fn=lambda: None,
            _js=f"function(){{ extraNetworksRefreshTab('{tabname}_{page.extra_networks_tabname}'); }}",
        )

    def create_html():
        ui.pages_contents = [pg.create_html(ui.tabname) for pg in ui.stored_extra_pages]

    def pages_html():
        if not ui.pages_contents:
            create_html()
        return ui.pages_contents

    interface.load(
        fn=pages_html,
        inputs=[],
        outputs=ui.pages,
    ).then(
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
