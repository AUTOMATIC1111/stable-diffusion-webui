import os

from modules import shared, ui_extra_networks
from modules.ui_extra_networks import quote_js


class ExtraNetworksPageHypernetworks(ui_extra_networks.ExtraNetworksPage):
    def __init__(self):
        super().__init__('Hypernetworks')

    def refresh(self):
        shared.reload_hypernetworks()

    def create_item(self, name, index=None):
        full_path = shared.hypernetworks[name]
        path, ext = os.path.splitext(full_path)

        return {
            "name": name,
            "filename": full_path,
            "preview": self.find_preview(path),
            "description": self.find_description(path),
            "search_term": self.search_terms_from_path(path),
            "prompt": quote_js(f"<hypernet:{name}:") + " + opts.extra_networks_default_multiplier + " + quote_js(">"),
            "local_preview": f"{path}.preview.{shared.opts.samples_format}",
            "sort_keys": {'default': index, **self.get_sort_keys(path + ext)},
        }

    def list_items(self):
        for index, name in enumerate(shared.hypernetworks):
            yield self.create_item(name, index)

    def allowed_directories_for_previews(self):
        return [shared.cmd_opts.hypernetwork_dir]

