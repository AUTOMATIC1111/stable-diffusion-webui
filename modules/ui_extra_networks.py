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

from modules import errors, extra_networks, shared, util
from modules.images import read_info_from_image, save_image_with_geninfo
from modules.infotext_utils import image_from_url_text
from modules.ui_common import OutputPanel
from modules.ui_extra_networks_user_metadata import UserMetadataEditor

extra_pages = []
allowed_dirs = set()
default_allowed_preview_extensions = ["png", "jpg", "jpeg", "webp", "gif"]


class ListItem:
    """
    Attributes:
        id [str]: The ID of this list item.
        html [str]: The HTML string for this item.
    """

    def __init__(self, _id: str, _html: str) -> None:
        self.id = _id
        self.html = _html
        self.node: Optional[DirectoryTreeNode] = None


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
        self.abspath = ""
        self.relpath = ""
        self.rel_parent_dir = ""
        self.sort_keys = {}
        self.search_terms = ""
        self.search_only = False


class TreeListItem(ListItem):
    """
    Attributes:
        visible [bool]: Whether the item should be shown in the list.
        expanded [bool]: Whether the item children should be shown.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.visible: bool = False
        self.expanded: bool = False


class DirectoryTreeNode:
    """
    Attributes:
        root_dir [str]: The root directory used to generate a relative path for this node.
        abspath [str]: The absolute path of this node.
        parent [DirectoryTreeNode]: The parent node of this node.
        depth [int]: The depth of this node in the tree. (folder level)
        is_dir [bool]: Whether this node is a directory or file.
        item [Optional[dict]]: The item data dictionary.
        relpath [str]: Relative path from `root_dir` to this node.
        children [list[DirectoryTreeNode]]: List of direct child nodes of this node.
    """

    def __init__(
        self,
        root_dir: str,
        abspath: str,
        parent: Optional["DirectoryTreeNode"] = None,
    ) -> None:
        self.root_dir = root_dir
        self.abspath = abspath
        self.parent = parent
        self.relpath = os.path.relpath(self.abspath, self.root_dir)
        self.children: list["DirectoryTreeNode"] = []
        self.is_dir = os.path.isdir(self.abspath)
        # If any parent dirs are hidden, this node is also considered hidden.
        self.is_hidden = any(x.startswith(".") for x in self.abspath.split(os.sep))
        self.id = ""
        self.depth = 0
        self.item = None

        self.rel_from_hidden = None
        if self.is_hidden:
            # Get the relative path starting from the first hidden directory.
            parts = self.relpath.split(os.sep)
            idxs = [i for i, x in enumerate(parts) if x.startswith(".")]
            if len(idxs) > 0:
                self.rel_from_hidden = os.path.join(*parts[idxs[0] :])

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
            res:        The dictionary result updated in place. On initial call,
                        should be passed as an empty dictionary.
            dirs_only:  Whether to only add directories to the result.

        Raises:
            KeyError: If any nodes in the tree have the same ID.
        """
        if self.abspath in res:
            raise KeyError(f"duplicate key: {self.abspath}")

        if not dirs_only or (dirs_only and self.is_dir):
            res[self.abspath] = self

        for child in self.children:
            child.flatten(res, dirs_only)

    def to_sorted_list(self, res: list, dirs_first: bool = True) -> None:
        """Sorts the tree by absolute path and groups by directories/files.

        Since we are sorting a directory tree, we always want the directories to come
        before the files. So we have to sort these two lists separately.

        Args:
            res:    The list result updated in place. On initial call, should be passed
                    as an empty list.
        """
        res.append(self)
        files = sorted(
            [x for x in self.children if not x.is_dir],
            key=lambda x: shared.natural_sort_key(os.path.basename(x.abspath)),
        )
        dirs = sorted(
            [x for x in self.children if x.is_dir],
            key=lambda x: shared.natural_sort_key(os.path.basename(x.abspath)),
        )

        children = [*dirs, *files] if dirs_first else [*files, *dirs]
        for child in children:
            child.to_sorted_list(res, dirs_first)

    def apply(self, fn: Callable) -> None:
        """Recursively calls passed function with instance for entire tree."""
        fn(self)
        for child in self.children:
            child.apply(fn)


class ExtraNetworksUi:
    """UI components for Extra Networks

    Attributes:
        button_save_preview:        Gradio button for saving previews.
        pages:                      Gradio HTML elements for an ExtraNetworks page.
        pages_contents:             HTML string content for `pages`.
        preview_target_filename:    Gradio textbox for entering filename.
        related_tabs:               Gradio Tab instances for each ExtraNetworksPage.
        stored_extra_pages:         `ExtraNetworksPage` instance for each page.
        tabname:                    The primary page tab name (i.e. `txt2img`, `img2img`)
        user_metadata_editors:      The metadata editor objects for a page.
    """

    def __init__(self, tabname: str):
        self.tabname = tabname
        # Dict keys are "{tabname}_{page.extra_networks_tabname}"
        self.pages: dict[str, gr.HTML] = {}
        self.pages_contents: dict[str, str] = {}
        self.stored_extra_pages: dict[str, ExtraNetworksPage] = {}
        self.related_tabs: dict[str, gr.Tab] = {}
        self.user_metadata_editors: dict[str, UserMetadataEditor] = {}

        self.unrelated_tabs: list[gr.Tab] = []
        self.button_save_preview: Optional[gr.Button] = None
        self.preview_target_filename: Optional[gr.Textbox] = None

        # Fetch the extra pages and build a map.
        for page in pages_in_preferred_order(extra_pages.copy()):
            self.stored_extra_pages[f"{self.tabname}_{page.extra_networks_tabname}"] = page


class ExtraNetworksPage:
    def __init__(self, title):
        self.title = title
        self.name = title.lower()
        # This is the actual name of the extra networks tab (not txt2img/img2img).
        self.extra_networks_tabname = self.name.replace(" ", "_")
        self.allow_prompt = True
        self.allow_negative_prompt = False
        self.is_ready = False
        self.metadata = {}
        self.items = {}
        self.cards = {}
        self.tree = {}
        self.tree_roots = {}
        self.nodes = {}
        self.lister = util.MassFileLister()
        # HTML Templates
        self.pane_tpl = shared.html("extra-networks-pane.html")
        self.card_tpl = shared.html("extra-networks-card.html")
        self.tree_row_tpl = shared.html("extra-networks-tree-row.html")
        self.btn_copy_path_tpl = shared.html("extra-networks-btn-copy-path.html")
        self.btn_show_metadata_tpl = shared.html("extra-networks-btn-show-metadata.html")
        self.btn_edit_metadata_tpl = shared.html("extra-networks-btn-edit-metadata.html")
        self.btn_dirs_view_item_tpl = shared.html("extra-networks-btn-dirs-view-item.html")
        self.btn_chevron_tpl = shared.html("extra-networks-btn-chevron.html")

    def clear_data(self) -> None:
        self.is_ready = False
        self.metadata = {}
        self.items = {}
        self.cards = {}
        self.tree = {}
        self.tree_roots = {}
        self.nodes = {}

    def refresh(self) -> None:
        # Whenever we refresh, we want to build our datasets from scratch.
        self.clear_data()

    def read_user_metadata(self, item, use_cache=True):
        filename = os.path.normpath(item.get("filename", None))
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

    def build_tree_html_row(
        self,
        tabname: str,
        label: str,
        btn_type: str,
        btn_title: Optional[str] = None,
        indent_html: Optional[str] = None,
        data_attributes: Optional[dict] = None,
        dir_is_empty: bool = False,
        item: Optional[dict] = None,
    ) -> str:
        """Generates HTML for a single row of the Tree View

        Args:
            tabname:
                "txt2img" or "img2img"
            label:
                The text to display for this row.
            btn_type:
                "dir" or "file"
            btn_title:
                Optional hover text for the row. Defaults to `label`.
            data_attributes:
                Dictionary defining data attributes to add to the row's tag.
                Ex: {"one": "1"} would generate <div data-one="1"></div>
            dir_is_empty:
                Whether the directory is empty. Only useful if btn_type=="dir".
            item:
                Dictionary containing item data such as filename, hash, etc.
        """
        if btn_type not in ["file", "dir"]:
            raise ValueError("Invalid button type:", btn_type)

        if data_attributes is None:
            data_attributes = {}

        label = label.strip()
        # If not specified, title will just reflect the label
        btn_title = btn_title.strip() if btn_title else f'"{label}"'

        action_list_item_action_leading = self.btn_chevron_tpl.format(extra_classes="")
        action_list_item_visual_leading = "🗀"
        action_list_item_visual_trailing = ""
        action_list_item_action_trailing = ""

        if dir_is_empty:
            action_list_item_action_leading = self.btn_chevron_tpl.format(extra_classes="invisible")

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
                "indent_spans": indent_html,
                "btn_type": btn_type,
                "btn_title": btn_title,
                "tabname": tabname,
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
        """Generates a row of buttons for use in Tree/Card View items."""
        metadata = item.get("metadata", None)
        name = item.get("name", "")
        filename = os.path.normpath(item.get("filename", ""))

        button_row_tpl = '<div class="button-row">{btn_copy_path}{btn_edit_item}{btn_metadata}</div>'

        btn_copy_path = self.btn_copy_path_tpl.format(clipboard_text=filename)
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
            HTML string generated for this item.
            Can be empty if the item is not meant to be shown.
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

        button_row = self.get_button_row(tabname, item)

        filename = os.path.normpath(item.get("filename", ""))
        # if this is true, the item must not be shown in the default view,
        # and must instead only be shown when searching for it
        show_hidden_models = str(shared.opts.extra_networks_show_hidden_models_cards).strip().lower()
        # If any parent dirs are hidden, the model is also hidden.
        is_hidden = any(x.startswith(".") for x in filename.split(os.sep))
        if show_hidden_models == "never" and is_hidden:
            return ""

        sort_keys = {}
        for sort_mode, sort_key in item.get("sort_keys", {}).items():
            sort_keys[sort_mode.strip().lower()] = html.escape(str(sort_key))

        description = ""
        if shared.opts.extra_networks_card_show_desc:
            description = item.get("description", "") or ""

        if not shared.opts.extra_networks_card_description_is_html:
            description = html.escape(description)

        data_name = item.get("name", "").strip()
        data_path = os.path.normpath(item.get("filename", "").strip())
        data_attributes = {
            "data-div-id": f'"{div_id}"' if div_id else '""',
            "data-name": f'"{data_name}"',
            "data-path": f'"{data_path}"',
            "data-hash": item.get("shorthash", None),
            "data-prompt": item.get("prompt", "").strip(),
            "data-neg-prompt": item.get("negative_prompt", "").strip(),
            "data-allow-neg": self.allow_negative_prompt,
        }

        if self.__class__.__name__ == "ExtraNetworksPageCheckpoints":
            data_attributes["data-is-checkpoint"] = True

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
            data_attributes=data_attributes_str,
            background_image=background_image,
            button_row=button_row,
            name=html.escape(item["name"].strip()),
            description=description,
        )

    def generate_card_view_data(self, tabname: str) -> dict:
        """Generates the datasets and HTML used to display the Card View.

        Returns:
            A dictionary containing necessary info for the client.
            {
                search_keys: array of strings,
                sort_<mode>: string, (for various sort modes),
                visible: True, // all cards are visible by default.
            }
            Return does not contain the HTML since that is fetched by client.
        """
        res = {}

        # Cards require a different sorting method than tree/dirs. We want to present
        # cards where files in a directory are listed before the contents of subdirectories.
        # Thus we need to sort each tree node again and provide the dirs_first=False flag.
        # We then create a mapping between these results and the self.nodes div_ids for sorting.
        sorted_nodes = []
        for node in self.tree_roots.values():
            _sorted_nodes = []
            node.to_sorted_list(_sorted_nodes, dirs_first=False)
            _sorted_nodes = [x for x in _sorted_nodes if not x.is_dir]
            sorted_nodes.extend(_sorted_nodes)

        nodes = {}
        div_id_to_idx = {}
        for i, node in enumerate(sorted_nodes):
            nodes[node.id] = node
            # Mapping from the self.nodes div_ids to the sorted index.
            div_id_to_idx[node.id] = i

        show_hidden_cards = str(shared.opts.extra_networks_show_hidden_models_cards).strip().lower()
        for node in nodes.values():
            search_only = False
            if show_hidden_cards == "always":
                search_only = False
            elif show_hidden_cards == "when searched":
                search_only = node.is_hidden
            elif "never" == show_hidden_cards and node.is_hidden:
                # We never show hidden cards here so don't even add it to the results.
                continue

            card = CardListItem(node.id, "")
            card.node = node
            item = node.item
            card.html = self.create_card_html(tabname=tabname, item=item, div_id=node.id)
            sort_keys = {}
            for k, v in item.get("sort_keys", {}).items():
                sort_keys[k.strip().lower().replace(" ", "_")] = html.escape(str(v))
            # Manual override the "path" sort key using our sorted path indices.
            sort_keys["path"] = div_id_to_idx[node.id]

            search_terms = item.get("search_terms", [])

            card.abspath = os.path.normpath(item.get("filename", ""))
            for path in self.allowed_directories_for_previews():
                parent_dir = os.path.dirname(os.path.abspath(path))
                if card.abspath.startswith(parent_dir):
                    card.relpath = os.path.relpath(card.abspath, parent_dir)
                    break
            card.sort_keys = sort_keys
            card.search_terms = " ".join(search_terms)
            card.search_only = search_only
            card.rel_parent_dir = os.path.dirname(card.relpath)
            if card.node.rel_from_hidden is not None:
                card.rel_parent_dir = os.path.dirname(card.node.rel_from_hidden)

            if card.search_only and card.node.rel_from_hidden is not None:
                # Limit the ways of searching for `search_only` cards so that the user
                # can't search for a parent to a hidden directory to see hidden cards.
                card.search_terms = card.search_terms.replace(node.relpath, card.node.rel_from_hidden)

            self.cards[node.id] = card

        if self.cards is None or not self.cards:
            return {}

        # Sort card div_ids for all sort modes.
        keys_sorted = {}
        sort_modes = self.cards[next(iter(self.cards))].sort_keys.keys()
        for mode in sort_modes:
            keys_sorted[mode] = sorted(
                self.cards.keys(),
                key=lambda k, sm=mode: shared.natural_sort_key(str(self.cards[k].sort_keys[sm])),
            )

        # Now that we have sorted, we can create the cards dataset.
        for div_id, card in self.cards.items():
            res[div_id] = {
                **{f"sort_{mode}": keys_sorted[mode].index(div_id) for mode in card.sort_keys.keys()},
                "rel_parent_dir": card.rel_parent_dir,
                "search_terms": card.search_terms,
                "search_only": card.search_only,
                "visible": not card.search_only,
            }

        return res

    def generate_tree_view_data(self, tabname: str) -> dict:
        """Generates the datasets and HTML used to display the Tree View.

        Returns:
            A dictionary containing necessary info for the client.
            {
                parent: None or div_id,
                children: list of div_id's,
                visible: bool,
                expanded: bool,
            }
            Return does not contain the HTML since that is fetched by client.
        """
        res = {}
        show_files = shared.opts.extra_networks_tree_view_show_files is True

        # Generate indentation for row
        def _gen_indents(node):
            if node.parent is None:
                return []
            _tpl = "<span data-depth='{depth}' data-parent-id='{parent_id}'></span>"
            _res = [_tpl.format(depth=node.depth, parent_id=node.parent.id)]
            _res.extend(_gen_indents(node.parent))
            return _res

        show_hidden_dirs = shared.opts.extra_networks_show_hidden_directories_buttons
        show_hidden_models = shared.opts.extra_networks_show_hidden_models_in_tree_view
        expand_depth = int(shared.opts.extra_networks_tree_view_expand_depth_default)
        for node in self.nodes.values():
            if node.is_hidden and node.is_dir and not show_hidden_dirs:
                continue

            if node.is_hidden and not node.is_dir and show_hidden_dirs and not show_hidden_models:
                continue

            tree_item = TreeListItem(node.id, "")

            if node.depth <= expand_depth:
                tree_item.expanded = True

            if node.depth <= expand_depth + 1:
                tree_item.visible = True

            tree_item.node = node
            parent_id = None
            if node.parent is not None:
                parent_id = node.parent.id

            indent_html = _gen_indents(node)
            indent_html.reverse()
            indent_html = "".join(indent_html)
            indent_html = f"<div class='tree-list-item-indent'>{indent_html}</div>"

            children = []

            if node.is_dir:  # directory
                if show_files:
                    dir_is_empty = node.children == []
                else:
                    dir_is_empty = all(not x.is_dir for x in node.children)

                if node.is_hidden and not show_hidden_models:
                    dir_is_empty = all(not x.is_dir for x in node.children)

                if not dir_is_empty:
                    for child in tree_item.node.children:
                        if child.is_dir:
                            if child.is_hidden:
                                if show_hidden_dirs:
                                    children.append(child.id)
                            else:
                                children.append(child.id)
                        elif show_files:
                            if child.is_hidden:
                                if show_hidden_dirs and show_hidden_models:
                                    children.append(child.id)
                            else:
                                children.append(child.id)

                data_attributes = {
                    "data-div-id": f'"{node.id}"',
                    "data-parent-id": f'"{parent_id}"',
                    "data-tree-entry-type": "dir",
                    "data-depth": node.depth,
                    "data-path": f'"{node.relpath}"',
                    "data-expanded": tree_item.expanded,
                }

                if node.is_hidden:
                    data_attributes["data-directory-filter-override"] = f'"{node.rel_from_hidden}"'

                tree_item.html = self.build_tree_html_row(
                    tabname=tabname,
                    label=os.path.basename(node.abspath),
                    btn_type="dir",
                    btn_title=f'"{node.abspath}"',
                    dir_is_empty=dir_is_empty,
                    indent_html=indent_html,
                    data_attributes=data_attributes,
                )
                self.tree[node.id] = tree_item
            else:  # file
                if not show_files:
                    # Don't add file if files are disabled in the options.
                    continue

                if node.is_hidden and not show_hidden_models:
                    continue

                item_name = node.item.get("name", "").strip()
                data_path = os.path.normpath(node.item.get("filename", "").strip())
                data_attributes = {
                    "data-div-id": f'"{node.id}"',
                    "data-parent-id": f'"{parent_id}"',
                    "data-tree-entry-type": "file",
                    "data-name": f'"{item_name}"',
                    "data-depth": node.depth,
                    "data-path": f'"{data_path}"',
                    "data-hash": node.item.get("shorthash", None),
                    "data-prompt": node.item.get("prompt", "").strip(),
                    "data-neg-prompt": node.item.get("negative_prompt", "").strip(),
                    "data-allow-neg": self.allow_negative_prompt,
                }
                # Special case for checkpoints since they need to switch model on click.
                # The JS code uses this flag to determine if it needs to swith model.
                if self.__class__.__name__ == "ExtraNetworksPageCheckpoints":
                    data_attributes["data-is-checkpoint"] = True

                tree_item.html = self.build_tree_html_row(
                    tabname=tabname,
                    label=html.escape(item_name),
                    btn_type="file",
                    indent_html=indent_html,
                    data_attributes=data_attributes,
                    item=node.item,
                )
                self.tree[node.id] = tree_item

            res[node.id] = {
                "parent": parent_id,
                "children": children,
                "visible": tree_item.visible,
                "expanded": tree_item.expanded,
            }

        return res

    def create_dirs_view_html(self, tabname: str) -> str:
        """Generates HTML for displaying folders."""
        res = []
        div_ids = sorted(self.nodes.keys(), key=shared.natural_sort_key)
        for div_id in div_ids:
            node = self.nodes[div_id]
            # Only process directories. Skip if file.
            if not node.is_dir:
                continue

            if node.is_hidden and not shared.opts.extra_networks_show_hidden_directories_buttons:
                continue

            if node.parent is None:
                label = node.relpath
            else:
                # Strip the root directory from the label to reduce size of buttons.
                parts = [x for x in node.relpath.split(os.sep) if x]
                label = os.path.join(*parts[1:])

            data_attributes = {
                "data-div-id": f'"{node.id}"',
                "data-path": f'"{node.relpath}"',
            }

            if node.is_hidden:
                data_attributes["data-directory-filter-override"] = f'"{node.rel_from_hidden}"'

            data_attributes_str = ""
            for k, v in data_attributes.items():
                if isinstance(v, (bool,)):
                    # Boolean data attributes only need a key when true.
                    if v:
                        data_attributes_str += f"{k} "
                elif v not in [None, "", "''", '""']:
                    data_attributes_str += f"{k}={v} "

            res.append(
                self.btn_dirs_view_item_tpl.format(
                    **{
                        "extra_class": "search-all" if node.relpath == "" else "",
                        "tabname_full": f"{tabname}_{self.extra_networks_tabname}",
                        "title": html.escape(node.abspath),
                        "label": html.escape(label),
                        "data_attributes": data_attributes_str,
                    }
                )
            )

        return "".join(res)

    def create_html(self, tabname: str, *, empty: bool = False) -> str:
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
        tree_items = {os.path.normpath(v["filename"]): v for v in self.items.values()}
        # Create a DirectoryTreeNode for each root directory since they might not share
        # a common path.
        for path in self.allowed_directories_for_previews():
            abspath = os.path.abspath(path)
            if not os.path.exists(abspath):
                continue
            self.tree_roots[abspath] = DirectoryTreeNode(os.path.dirname(abspath), abspath, None)
            self.tree_roots[abspath].build(tree_items)

        card_list_loading_splash_content = "Loading..."
        no_cards_html_dirs = "".join([f"<li>{x}</li>" for x in self.allowed_directories_for_previews()])
        card_list_no_data_splash_content = (
            "<div class='nocards'>"
            "<h1>Nothing here. Add some content to the following directories:</h1>"
            f"<ul>{no_cards_html_dirs}</ul>"
            "</div>"
        )
        tree_list_loading_splash_content = "Loading..."
        tree_list_no_data_splash_content = "No Data"

        # Now use tree roots to generate a mapping of div_ids to nodes.
        # Flatten roots into a single sorted list of nodes.
        # Directories always come before files. After that, natural sort is used.
        sorted_nodes = []
        for node in self.tree_roots.values():
            _sorted_nodes = []
            node.to_sorted_list(_sorted_nodes)
            sorted_nodes.extend(_sorted_nodes)

        for i, node in enumerate(sorted_nodes):
            node.id = str(i)
            self.nodes[node.id] = node

        # Generate the html for displaying directory buttons
        dirs_html = self.create_dirs_view_html(tabname)

        sort_mode = shared.opts.extra_networks_card_order_field.lower().strip().replace(" ", "_")
        sort_dir = shared.opts.extra_networks_card_order.lower().strip()
        dirs_view_en = shared.opts.extra_networks_dirs_view_default_enabled
        tree_view_en = shared.opts.extra_networks_tree_view_default_enabled
        card_view_en = shared.opts.extra_networks_card_view_default_enabled

        return self.pane_tpl.format(
            **{
                "tabname": tabname,
                "extra_networks_tabname": self.extra_networks_tabname,
                "btn_sort_mode_path_data_attributes": "data-selected" if sort_mode == "path" else "",
                "btn_sort_mode_name_data_attributes": "data-selected" if sort_mode == "name" else "",
                "btn_sort_mode_date_created_data_attributes": "data-selected" if sort_mode == "date_created" else "",
                "btn_sort_mode_date_modified_data_attributes": "data-selected" if sort_mode == "date_modified" else "",
                "btn_sort_dir_ascending_data_attributes": "data-selected" if sort_dir == "ascending" else "",
                "btn_sort_dir_descending_data_attributes": "data-selected" if sort_dir == "descending" else "",
                "btn_dirs_view_data_attributes": "data-selected" if dirs_view_en else "",
                "btn_tree_view_data_attributes": "data-selected" if tree_view_en else "",
                "btn_card_view_data_attributes": "data-selected" if card_view_en else "",
                "tree_view_style": f"flex-basis: {shared.opts.extra_networks_tree_view_default_width}px;",
                "card_view_style": "flex-grow: 1;",
                "dirs_html": dirs_html,
                "card_list_loading_splash_content": card_list_loading_splash_content,
                "card_list_no_data_splash_content": card_list_no_data_splash_content,
                "tree_list_loading_splash_content": tree_list_loading_splash_content,
                "tree_list_no_data_splash_content": tree_list_no_data_splash_content,
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

    def create_user_metadata_editor(self, ui, tabname) -> UserMetadataEditor:
        return UserMetadataEditor(ui, tabname, self)


@functools.cache
def allowed_preview_extensions_with_extra(extra_extensions=None):
    return set(default_allowed_preview_extensions) | set(extra_extensions or [])


def allowed_preview_extensions():
    return allowed_preview_extensions_with_extra((shared.opts.samples_format,))


def register_page(page):
    """registers extra networks page for the UI

    recommend doing it in on_before_ui() callback for extensions
    """
    extra_pages.append(page)
    allowed_dirs.clear()
    allowed_dirs.update(set(sum([x.allowed_directories_for_previews() for x in extra_pages], [])))


def get_page_by_name(extra_networks_tabname: str = "") -> "ExtraNetworksPage":
    """Gets a page from extra pages for the specified tabname.

    Raises:
        HTTPException if the tabname is not in the `extra_pages` dict.
    """
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
    """Generates the initial Tree View data and returns a simplified dataset.

    The data returned does not contain any HTML strings.

    Status Codes:
        200 on success
        404 if data isn't ready or tabname doesn't exist.
    """
    page = get_page_by_name(extra_networks_tabname)

    data = page.generate_tree_view_data(tabname)

    return JSONResponse({"data": data, "ready": data is not None})


def init_card_data(tabname: str = "", extra_networks_tabname: str = "") -> JSONResponse:
    """Generates the initial Card View data and returns a simplified dataset.

    The data returned does not contain any HTML strings.

    Status Codes:
        200 on success
        404 if data isn't ready or tabname doesn't exist.
    """
    page = get_page_by_name(extra_networks_tabname)

    data = page.generate_card_view_data(tabname)

    return JSONResponse({"data": data, "ready": data is not None})


def fetch_tree_data(
    extra_networks_tabname: str = "",
    div_ids: str = "",
) -> JSONResponse:
    """Retrieves Tree View HTML strings for the specified `div_ids`.

    Args:
        div_ids: A string with div_ids in CSV format.

    Status Codes:
        200 on success
        404 if tabname doesn't exist
    """
    page = get_page_by_name(extra_networks_tabname)

    res = {}
    missed = []
    for div_id in div_ids.split(","):
        if div_id in page.tree:
            res[div_id] = page.tree[div_id].html
        else:
            missed.append(div_id)

    return JSONResponse({"data": res, "missing_div_ids": missed})


def fetch_card_data(
    extra_networks_tabname: str = "",
    div_ids: str = "",
) -> JSONResponse:
    """Retrieves Card View HTML strings for the specified `div_ids`.

    Args:
        div_ids: A string with div_ids in CSV format.

    Status Codes:
        200 on success
        404 if tabname doesn't exist
    """
    page = get_page_by_name(extra_networks_tabname)

    res = {}
    missed = []
    for div_id in div_ids.split(","):
        if div_id in page.cards:
            res[div_id] = page.cards[div_id].html
        else:
            missed.append(div_id)
    return JSONResponse({"data": res, "missing_div_ids": missed})


def clear_page_data(extra_networks_tabname: str = "") -> JSONResponse:
    """Returns whether the specified page is ready for fetching data.

    Status Codes:
        200 on success
        404 if tabname doesnt exist or other errors.
    """
    page = get_page_by_name(extra_networks_tabname)

    page.clear_data()

    return JSONResponse({})


def page_is_ready(extra_networks_tabname: str = "") -> JSONResponse:
    """Returns whether the specified page is ready for fetching data.

    Status Codes:
        200 on success. response contains ready state.
        404 if tabname doesnt exist.
    """
    page = get_page_by_name(extra_networks_tabname)

    return JSONResponse({"ready": page.is_ready})


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


def get_single_card(
    tabname: str = "",
    extra_networks_tabname: str = "",
    name: str = "",
    div_id: str = "",
) -> JSONResponse:
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
    item_html = page.create_card_html(tabname=tabname, item=item, div_id=div_id)
    # Update the card's HTML in the page's dataset.
    page.cards[div_id].html = item_html

    return JSONResponse({"html": item_html})


def add_pages_to_demo(app):
    app.add_api_route("/sd_extra_networks/thumb", fetch_file, methods=["GET"])
    app.add_api_route("/sd_extra_networks/cover-images", fetch_cover_images, methods=["GET"])
    app.add_api_route("/sd_extra_networks/metadata", get_metadata, methods=["GET"])
    app.add_api_route("/sd_extra_networks/get-single-card", get_single_card, methods=["GET"])
    app.add_api_route("/sd_extra_networks/init-tree-data", init_tree_data, methods=["GET"])
    app.add_api_route("/sd_extra_networks/init-card-data", init_card_data, methods=["GET"])
    app.add_api_route("/sd_extra_networks/fetch-tree-data", fetch_tree_data, methods=["GET"])
    app.add_api_route("/sd_extra_networks/fetch-card-data", fetch_card_data, methods=["GET"])
    app.add_api_route("/sd_extra_networks/page-is-ready", page_is_ready, methods=["GET"])
    app.add_api_route("/sd_extra_networks/clear-page-data", clear_page_data, methods=["GET"])


def quote_js(s):
    s = s.replace("\\", "\\\\")
    s = s.replace('"', '\\"')
    return f'"{s}"'


def initialize():
    extra_pages.clear()


def register_default_pages():
    from modules.ui_extra_networks_checkpoints import ExtraNetworksPageCheckpoints
    from modules.ui_extra_networks_hypernets import ExtraNetworksPageHypernetworks
    from modules.ui_extra_networks_textual_inversion import ExtraNetworksPageTextualInversion

    register_page(ExtraNetworksPageTextualInversion())
    register_page(ExtraNetworksPageHypernetworks())
    register_page(ExtraNetworksPageCheckpoints())


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
    ui = ExtraNetworksUi(tabname)
    ui.unrelated_tabs = unrelated_tabs

    for tabname_full, page in ui.stored_extra_pages.items():
        with gr.Tab(page.title, elem_id=tabname_full, elem_classes=["extra-page"]) as tab:
            with gr.Column(elem_id=f"{tabname_full}_prompts", elem_classes=["extra-page-prompts"]):
                pass

            page_elem = gr.HTML(
                page.create_html(tabname, empty=True),
                elem_id=f"{tabname_full}_pane_container",
            )
            ui.pages[tabname_full] = page_elem
            editor = page.create_user_metadata_editor(ui, tabname)
            editor.create_ui()
            ui.user_metadata_editors[tabname_full] = editor
            ui.related_tabs[tabname_full] = tab

    ui.button_save_preview = gr.Button(
        "Save preview",
        elem_id=f"{tabname}_save_preview",
        visible=False,
    )
    ui.preview_target_filename = gr.Textbox(
        "Preview save filename",
        elem_id=f"{tabname}_preview_filename",
        visible=False,
    )

    for tab in ui.unrelated_tabs:
        tab.select(
            fn=None,
            _js=f"function(){{extraNetworksUnrelatedTabSelected('{ui.tabname}');}}",
            inputs=[],
            outputs=[],
            show_progress=False,
        )

    for tabname_full, page in ui.stored_extra_pages.items():
        tab = ui.related_tabs[tabname_full]
        tab.select(
            fn=None,
            _js=(
                "function(){extraNetworksTabSelected("
                f"'{tabname_full}', "
                f"{str(page.allow_prompt).lower()}, "
                f"{str(page.allow_negative_prompt).lower()}"
                ");}"
            ),
            inputs=[],
            outputs=[],
            show_progress=False,
        )

        def refresh(tabname_full):
            page = ui.stored_extra_pages[tabname_full]
            page.is_ready = False
            page.refresh()
            ui.pages_contents[tabname_full] = page.create_html(ui.tabname)
            page.is_ready = True
            return list(ui.pages_contents.values())

        button_refresh = gr.Button(
            "Refresh",
            elem_id=f"{tabname_full}_extra_refresh_internal",
            visible=False,
        )
        button_refresh.click(fn=functools.partial(refresh, tabname_full), inputs=[], outputs=list(ui.pages.values()),).then(
            fn=lambda: None,
            _js="setupAllResizeHandles",
        ).then(
            fn=lambda: None,
            _js=f"function(){{extraNetworksRefreshTab('{tabname_full}');}}",
        )

    def create_html():
        for tabname_full, page in ui.stored_extra_pages.items():
            page.is_ready = False
            ui.pages_contents[tabname_full] = page.create_html(ui.tabname)
            page.is_ready = True

    def pages_html():
        if not ui.pages_contents:
            create_html()
        return list(ui.pages_contents.values())

    interface.load(fn=pages_html, inputs=[], outputs=list(ui.pages.values()),).then(
        fn=lambda: None,
        _js="setupAllResizeHandles",
    )

    return ui


def path_is_parent(parent_path, child_path):
    parent_path = os.path.abspath(parent_path)
    child_path = os.path.abspath(child_path)

    return child_path.startswith(parent_path)


def setup_ui(ui: ExtraNetworksUi, gallery: OutputPanel):
    def save_preview(index, images, filename):
        # this function is here for backwards compatibility and likely will be removed soon

        if len(images) == 0:
            print("There is no image in gallery to save as a preview.")
            return [page.create_html(ui.tabname) for page in ui.stored_extra_pages.values()]

        index = int(index)
        index = 0 if index < 0 else index
        index = len(images) - 1 if index >= len(images) else index

        img_info = images[index if index >= 0 else 0]
        image = image_from_url_text(img_info)
        geninfo, items = read_info_from_image(image)

        is_allowed = False
        for page in ui.stored_extra_pages.values():
            if any(path_is_parent(x, filename) for x in page.allowed_directories_for_previews()):
                is_allowed = True
                break

        assert is_allowed, f"writing to {filename} is not allowed"

        save_image_with_geninfo(image, geninfo, filename)

        return [page.create_html(ui.tabname) for page in ui.stored_extra_pages.values()]

    ui.button_save_preview.click(
        fn=save_preview,
        _js="function(x, y, z){return [selected_gallery_index(), y, z]}",
        inputs=[ui.preview_target_filename, gallery, ui.preview_target_filename],
        outputs=[*list(ui.pages.values())],
    )

    for editor in ui.user_metadata_editors.values():
        editor.setup_ui(gallery)
