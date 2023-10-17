import html
import json
import os
from modules import shared, ui_extra_networks, sd_vae, hashes


class ExtraNetworksPageVAEs(ui_extra_networks.ExtraNetworksPage):
    def __init__(self):
        super().__init__('VAE')

    def refresh(self):
        shared.refresh_vaes()

    def list_items(self):
        for name, filename in sd_vae.vae_dict.items():
            try:
                fn = os.path.splitext(filename)[0]
                record = {
                    "type": 'VAE',
                    "name": name,
                    "title": name,
                    "filename": fn,
                    "hash": hashes.sha256_from_cache(filename, f"vae/{fn}"),
                    "search_term": self.search_terms_from_path(fn),
                    "preview": self.find_preview(fn),
                    "local_preview": f"{fn}.{shared.opts.samples_format}",
                    "description": self.find_description(fn),
                    "info": self.find_info(fn),
                    "metadata": {},
                    "onclick": '"' + html.escape(f"""return selectVAE({json.dumps(name)})""") + '"',
                }
                yield record
            except Exception as e:
                shared.log.debug(f"Extra networks error: type=vae file={filename} {e}")

    def allowed_directories_for_previews(self):
        return [v for v in [shared.opts.vae_dir] if v is not None]
