import json
import os
import lora

from modules import shared, ui_extra_networks


class ExtraNetworksPageLora(ui_extra_networks.ExtraNetworksPage):
    def __init__(self):
        super().__init__('Lora')

    def refresh(self):
        lora.list_available_loras()

    def list_items(self):
        for name, lora_on_disk in lora.available_loras.items():
            path, ext = os.path.splitext(lora_on_disk.filename)
            previews = [path + ".png", path + ".preview.png", path + ".jpg", path + ".preview.jpg", path + ".jpeg", path + ".preview.jpeg", path + ".webp", path + ".preview.webp"]

            preview = None
            for file in previews:
                if os.path.isfile(file):
                    preview = self.link_preview(file)
                    break

            yield {
                "name": name,
                "filename": path,
                "preview": preview,
                "search_term": self.search_terms_from_path(lora_on_disk.filename),
                "prompt": json.dumps(f"<lora:{name}:{str(shared.opts.extra_networks_default_multiplier)}>"),
                "local_preview": f"{path}.{shared.opts.samples_format}",
            }

    def allowed_directories_for_previews(self):
        return [shared.cmd_opts.lora_dir]
