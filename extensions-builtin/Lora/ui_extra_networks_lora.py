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
        for name, l in lora.available_loras.items():
            path, _ext = os.path.splitext(l.filename)
            alias = l.get_alias()
            prompt = f" <lora:{alias}:{shared.opts.extra_networks_default_multiplier}>"
            prompt = json.dumps(prompt)
            metadata =  json.dumps(l.metadata, indent=4) if l.metadata else None
            possible_tags = l.metadata.get('ss_tag_frequency', {}) if l.metadata is not None else {}
            if isinstance(possible_tags, str):
                possible_tags = {}
                shared.log.debug(f'Lora has invalid metadata: {path}')
            tags = {}
            for tag in possible_tags.keys():
                if '_' not in tag:
                    tag = f'0_{tag}'
                words = tag.split('_', 1)
                tags[' '.join(words[1:])] = words[0]
            # shared.log.debug(f'Lora: {path}: name={name} alias={alias} tags={tags}')
            yield {
                "name": name,
                "filename": path,
                "fullname": l.filename,
                "hash": l.shorthash,
                "preview": self.find_preview(path),
                "description": self.find_description(path),
                "info": self.find_info(path),
                "search_term": self.search_terms_from_path(l.filename) + ' '.join(tags.keys()),
                "prompt": prompt,
                "local_preview": f"{path}.{shared.opts.samples_format}",
                "metadata": metadata,
                "tags": tags,
            }

    def allowed_directories_for_previews(self):
        return [shared.cmd_opts.lora_dir]
