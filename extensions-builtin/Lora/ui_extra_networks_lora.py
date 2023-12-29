import os
import json
import concurrent
import network
import networks
from modules import shared, ui_extra_networks


class ExtraNetworksPageLora(ui_extra_networks.ExtraNetworksPage):
    def __init__(self):
        super().__init__('Lora')
        self.list_time = 0

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

            item = {
                "type": 'Lora',
                "name": name,
                "filename": l.filename,
                "hash": l.shorthash,
                "preview": self.find_preview(l.filename),
                "prompt": json.dumps(f" <lora:{l.get_alias()}:{shared.opts.extra_networks_default_multiplier}>"),
                "local_preview": f"{path}.{shared.opts.samples_format}",
                "metadata": json.dumps(l.metadata, indent=4) if l.metadata else None,
                "mtime": os.path.getmtime(l.filename),
                "size": os.path.getsize(l.filename),
            }
            info = self.find_info(l.filename)

            tags = {}
            possible_tags = l.metadata.get('ss_tag_frequency', {}) if l.metadata is not None else {} # tags from model metedata
            if isinstance(possible_tags, str):
                possible_tags = {}
            for k, v in possible_tags.items():
                words = k.split('_', 1) if '_' in k else [v, k]
                words = [str(w).replace('.json', '') for w in words]
                if words[0] == '{}':
                    words[0] = 0
                tag = ' '.join(words[1:])
                tags[tag] = words[0]
            versions = info.get('modelVersions', []) # trigger words from info json
            for v in versions:
                possible_tags = v.get('trainedWords', [])
                if isinstance(possible_tags, list):
                    for tag in possible_tags:
                        if tag not in tags:
                            tags[tag] = 0
            search = {}
            possible_tags = info.get('tags', []) # tags from info json
            if not isinstance(possible_tags, list):
                possible_tags = [v for v in possible_tags.values()]
            for v in possible_tags:
                search[v] = 0
            if len(list(tags)) == 0:
                tags = search

            bad_chars = [';', ':', '<', ">", "*", '?', '\'', '\"']
            clean_tags = {}
            for k, v in tags.items():
                tag = ''.join(i for i in k if not i in bad_chars)
                clean_tags[tag] = v

            item["info"] = info
            item["description"] = self.find_description(l.filename, info) # use existing info instead of double-read
            item["tags"] = clean_tags
            item["search_term"] = f'{self.search_terms_from_path(l.filename)} {" ".join(tags.keys())} {" ".join(search.keys())}'

            return item
        except Exception as e:
            shared.log.debug(f"Extra networks error: type=lora file={name} {e}")
            return None

    def list_items(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=shared.max_workers) as executor:
            future_items = {executor.submit(self.create_item, net): net for net in networks.available_networks}
            for future in concurrent.futures.as_completed(future_items):
                item = future.result()
                if item is not None:
                    yield item

    def allowed_directories_for_previews(self):
        return [shared.cmd_opts.lora_dir, shared.cmd_opts.lyco_dir]
