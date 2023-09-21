import os
import json
import lora

from modules import shared, ui_extra_networks


class ExtraNetworksPageLora(ui_extra_networks.ExtraNetworksPage):
    def __init__(self):
        super().__init__('Lora')

    def refresh(self):
        lora.list_available_loras()

    def list_items(self):
        for name, l in lora.available_loras.items():
            path, _ext = os.path.splitext(l.filename)
            possible_tags = l.metadata.get('ss_tag_frequency', {}) if l.metadata is not None else {}
            if isinstance(possible_tags, str):
                possible_tags = {}
            tags = {}
            for k, v in possible_tags.items():
                words = k.split('_', 1) if '_' in k else [v, k]
                tags[' '.join(words[1:])] = words[0]
            name = os.path.splitext(os.path.relpath(l.filename, shared.cmd_opts.lora_dir))[0]
            yield {
                "name": name,
                "filename": l.filename,
                "hash": l.shorthash,
                "search_term": self.search_terms_from_path(l.filename) + ' '.join(tags.keys()),
                "preview": self.find_preview(path),
                "description": self.find_description(path),
                "info": self.find_info(path),
                "prompt": json.dumps(f" <lora:{l.get_alias()}:{shared.opts.extra_networks_default_multiplier}>"),
                "local_preview": f"{path}.{shared.opts.samples_format}",
                "metadata": json.dumps(l.metadata, indent=4) if l.metadata else None,
                "tags": tags,
            }

    def allowed_directories_for_previews(self):
        return [shared.cmd_opts.lora_dir]
