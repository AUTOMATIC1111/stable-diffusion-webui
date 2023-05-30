import json
import os
import lora

from modules import shared, ui_extra_networks


user_lora_dir = os.path.join(os.path.dirname(os.path.dirname(shared.cmd_opts.lora_dir)), "user-models", 'Lora')
if not os.path.isdir(user_lora_dir):
    user_lora_dir = None

class ExtraNetworksPageLora(ui_extra_networks.ExtraNetworksPage):
    def __init__(self):
        super().__init__('Lora')

    def refresh(self):
        lora.list_available_loras()

    def list_items(self):
        for name, lora_on_disk in lora.available_loras.items():
            path, ext = os.path.splitext(lora_on_disk.filename)

            alias = lora_on_disk.get_alias()

            yield {
                "name": name,
                "filename": path,
                "preview": self.find_preview(path),
                "description": self.find_description(path),
                "search_term": self.search_terms_from_path(lora_on_disk.filename),
                "prompt": json.dumps(f"<lora:{alias}:") + " + opts.extra_networks_default_multiplier + " + json.dumps(">"),
                "local_preview": f"{path}.{shared.opts.samples_format}",
                "metadata": json.dumps(lora_on_disk.metadata, indent=4) if lora_on_disk.metadata else None,
            }

    def allowed_directories_for_previews(self):
        previews = [shared.cmd_opts.lora_dir]
        if user_lora_dir:
            previews.append(user_lora_dir)
        return previews
