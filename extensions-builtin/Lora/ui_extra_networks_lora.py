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
            previews = [path + ".png", path + ".preview.png"]

            preview = None
            for file in previews:
                if os.path.isfile(file):
                    preview = "./file=" + file.replace('\\', '/') + "?mtime=" + str(os.path.getmtime(file))
                    break

            yield {
                "name": name,
                "filename": path,
                "preview": preview,
                "prompt": json.dumps(f"<lora:{name}:") + " + opts.extra_networks_default_multiplier + " + json.dumps(">"),
                "local_preview": path + ".png",
            }

    def allowed_directories_for_previews(self):
        return [shared.cmd_opts.lora_dir]

