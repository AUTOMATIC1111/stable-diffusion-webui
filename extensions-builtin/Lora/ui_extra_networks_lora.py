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
            path, _ext = os.path.splitext(lora_on_disk.filename)
            alias = lora_on_disk.get_alias()
            prompt = (json.dumps(f"<lora:{alias}") + " + " + json.dumps(f':{shared.opts.extra_networks_default_multiplier}') + " + " + json.dumps(">"))
            metadata =  json.dumps(lora_on_disk.metadata, indent=4) if lora_on_disk.metadata else None
            possible_tags = lora_on_disk.metadata.get('ss_tag_frequency', {}) if lora_on_disk.metadata is not None else {}
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
                "preview": self.find_preview(path),
                "description": self.find_description(path),
                "search_term": self.search_terms_from_path(lora_on_disk.filename),
                "prompt": prompt,
                "local_preview": f"{path}.{shared.opts.samples_format}",
                "metadata": metadata,
                "tags": tags,
            }

    def allowed_directories_for_previews(self):
        return [shared.cmd_opts.lora_dir]
