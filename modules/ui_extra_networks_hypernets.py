import json
import os
from modules import shared, ui_extra_networks


class ExtraNetworksPageHypernetworks(ui_extra_networks.ExtraNetworksPage):
    def __init__(self):
        super().__init__('Hypernetwork')

    def refresh(self):
        shared.reload_hypernetworks()

    def list_items(self):
        for name, path in shared.hypernetworks.items():
            try:
                name = os.path.relpath(os.path.splitext(path)[0], shared.opts.hypernetwork_dir)
                yield {
                    "type": 'Hypernetwork',
                    "name": name,
                    "filename": path,
                    "preview": self.find_preview(path),
                    "description": self.find_description(path),
                    "info": self.find_info(path),
                    "prompt": json.dumps(f"<hypernet:{os.path.basename(name)}:{shared.opts.extra_networks_default_multiplier}>"),
                    "local_preview": f"{os.path.splitext(path)[0]}.{shared.opts.samples_format}",
                    "mtime": os.path.getmtime(path),
                    "size": os.path.getsize(path),
                }
            except Exception as e:
                shared.log.debug(f"Extra networks error: type=hypernetwork file={path} {e}")

    def allowed_directories_for_previews(self):
        return [shared.opts.hypernetwork_dir]
