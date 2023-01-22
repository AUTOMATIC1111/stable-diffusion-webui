import json
import os

from modules import shared, ui_extra_networks


class ExtraNetworksPageHypernetworks(ui_extra_networks.ExtraNetworksPage):
    def __init__(self):
        super().__init__('Hypernetworks')

    def refresh(self):
        shared.reload_hypernetworks()

    def list_items(self):
        for name, path in shared.hypernetworks.items():
            path, ext = os.path.splitext(path)
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
                "prompt": json.dumps(f"<hypernet:{name}:") + " + opts.extra_networks_default_multiplier + " + json.dumps(">"),
                "local_preview": path + ".png",
            }

    def allowed_directories_for_previews(self):
        return [shared.cmd_opts.hypernetwork_dir]

