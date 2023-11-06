import os
import json
import network
import networks
from modules import shared, ui_extra_networks


class ExtraNetworksPageLora(ui_extra_networks.ExtraNetworksPage):
    def __init__(self):
        super().__init__('Lora')

    def refresh(self):
        networks.list_available_networks()

    def create_item(self, name):
        l = networks.available_networks.get(name)
        try:
            path, _ext = os.path.splitext(l.filename)
            name = os.path.splitext(os.path.relpath(l.filename, shared.cmd_opts.lora_dir))[0]

            if shared.backend == shared.Backend.ORIGINAL:
                if l.sd_version == network.SdVersion.SDXL:
                    return None
            elif shared.backend == shared.Backend.DIFFUSERS:
                if shared.sd_model_type == 'none': # return all when model is not loaded
                    pass
                elif shared.sd_model_type == 'sdxl':
                    if l.sd_version == network.SdVersion.SD1 or l.sd_version == network.SdVersion.SD2:
                        return None
                elif shared.sd_model_type == 'sd':
                    if l.sd_version == network.SdVersion.SDXL:
                        return None

            # tags from model metedata
            possible_tags = l.metadata.get('ss_tag_frequency', {}) if l.metadata is not None else {}
            if isinstance(possible_tags, str):
                possible_tags = {}
            tags = {}
            for k, v in possible_tags.items():
                words = k.split('_', 1) if '_' in k else [v, k]
                words = [str(w).replace('.json', '') for w in words]
                if words[0] == '{}':
                    words[0] = 0
                tags[' '.join(words[1:])] = words[0]

            item = {
                "type": 'Lora',
                "name": name,
                "filename": l.filename,
                "hash": l.shorthash,
                "search_term": self.search_terms_from_path(l.filename) + ' '.join(tags.keys()),
                "preview": self.find_preview(l.filename),
                "prompt": json.dumps(f" <lora:{l.get_alias()}:{shared.opts.extra_networks_default_multiplier}>"),
                "local_preview": f"{path}.{shared.opts.samples_format}",
                "metadata": json.dumps(l.metadata, indent=4) if l.metadata else None,
                "mtime": os.path.getmtime(l.filename),
                "size": os.path.getsize(l.filename),
            }
            info = self.find_info(l.filename)
            item["info"] = info
            item["description"] = self.find_description(l.filename, info) # use existing info instead of double-read

            # tags from user metadata
            possible_tags = info.get('tags', [])
            if not isinstance(possible_tags, list):
                possible_tags = [v for v in possible_tags.values()]
            for v in possible_tags:
                tags[v] = 0
            item["tags"] = tags

            return item
        except Exception as e:
            shared.log.debug(f"Extra networks error: type=lora file={name} {e}")
            return None

    def list_items(self):
        for _index, name in enumerate(networks.available_networks):
            item = self.create_item(name)
            if item is not None:
                yield item

    def allowed_directories_for_previews(self):
        return [shared.cmd_opts.lora_dir, shared.cmd_opts.lyco_dir]
