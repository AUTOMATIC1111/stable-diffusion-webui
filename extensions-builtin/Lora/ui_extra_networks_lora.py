import json
import os
import lora

from modules import shared, ui_extra_networks
from ui_edit_user_metadata import LoraUserMetadataEditor


class ExtraNetworksPageLora(ui_extra_networks.ExtraNetworksPage):
    def __init__(self):
        super().__init__('Lora')

    def refresh(self):
        lora.list_available_loras()

    def create_item(self, name, index=None):
        lora_on_disk = lora.available_loras.get(name)

        path, ext = os.path.splitext(lora_on_disk.filename)

        alias = lora_on_disk.get_alias()

        item = {
            "name": name,
            "filename": lora_on_disk.filename,
            "preview": self.find_preview(path),
            "description": self.find_description(path),
            "search_term": self.search_terms_from_path(lora_on_disk.filename),
            "local_preview": f"{path}.{shared.opts.samples_format}",
            "metadata": json.dumps(lora_on_disk.metadata, indent=4) if lora_on_disk.metadata else None,
            "sort_keys": {'default': index, **self.get_sort_keys(lora_on_disk.filename)},
        }

        self.read_user_metadata(item)
        activation_text = item["user_metadata"].get("activation text")
        preferred_weight = item["user_metadata"].get("preferred weight", 0.0)
        item["prompt"] = json.dumps(f"<lora:{alias}:") + " + " + (str(preferred_weight) if preferred_weight else "opts.extra_networks_default_multiplier") + " + " + json.dumps(">")

        if activation_text:
            item["prompt"] += " + " + json.dumps(" " + activation_text)

        return item

    def list_items(self):
        for index, name in enumerate(lora.available_loras):
            item = self.create_item(name, index)
            yield item

    def allowed_directories_for_previews(self):
        return [shared.cmd_opts.lora_dir]

    def create_user_metadata_editor(self, ui, tabname):
        return LoraUserMetadataEditor(ui, tabname, self)
