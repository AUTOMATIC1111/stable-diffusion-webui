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
                record = {
                    "type": 'VAE',
                    "name": name,
                    "title": name,
                    "filename": filename,
                    "hash": hashes.sha256_from_cache(filename, f"vae/{filename}"),
                    "preview": self.find_preview(filename),
                    "local_preview": f"{os.path.splitext(filename)[0]}.{shared.opts.samples_format}",
                    "metadata": {},
                    "onclick": '"' + html.escape(f"""return selectVAE({json.dumps(name)})""") + '"',
                    "mtime": os.path.getmtime(filename),
                    "size": os.path.getsize(filename),
                }
                record["info"] = self.find_info(filename)
                record["description"] = self.find_description(filename, record["info"])
                yield record
            except Exception as e:
                shared.log.debug(f"Extra networks error: type=vae file={filename} {e}")

    def allowed_directories_for_previews(self):
        return [v for v in [shared.opts.vae_dir] if v is not None]
