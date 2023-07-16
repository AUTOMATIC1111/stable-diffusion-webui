import os
import networks

from modules import shared, ui_extra_networks
from modules.ui_extra_networks import quote_js
from ui_edit_user_metadata import LoraUserMetadataEditor


class ExtraNetworksPageLora(ui_extra_networks.ExtraNetworksPage):
    def __init__(self):
        super().__init__('Lora')

    def refresh(self):
        networks.list_available_networks()

    def create_item(self, name, index=None):
        lora_on_disk = networks.available_networks.get(name)

        path, ext = os.path.splitext(lora_on_disk.filename)

        alias = lora_on_disk.get_alias()

        # in 1.5 filename changes to be full filename instead of path without extension, and metadata is dict instead of json string
        item = {
            "name": name,
            "filename": lora_on_disk.filename,
            "preview": self.find_preview(path),
            "description": self.find_description(path),
            "search_term": self.search_terms_from_path(lora_on_disk.filename),
            "local_preview": f"{path}.{shared.opts.samples_format}",
            "metadata": lora_on_disk.metadata,
            "sort_keys": {'default': index, **self.get_sort_keys(lora_on_disk.filename)},
        }

        self.read_user_metadata(item)
        activation_text = item["user_metadata"].get("activation text")
        preferred_weight = item["user_metadata"].get("preferred weight", 0.0)
        item["prompt"] = quote_js(f"<lora:{alias}:") + " + " + (str(preferred_weight) if preferred_weight else "opts.extra_networks_default_multiplier") + " + " + quote_js(">")

        if activation_text:
            item["prompt"] += " + " + quote_js(" " + activation_text)

        return item

    def list_items(self):
        for index, name in enumerate(networks.available_networks):
            item = self.create_item(name, index)
            yield item

    def allowed_directories_for_previews(self):
        return [shared.cmd_opts.lora_dir]

    def create_user_metadata_editor(self, ui, tabname):
        return LoraUserMetadataEditor(ui, tabname, self)
