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
        # alias = lora_on_disk.get_alias()
        try:
            path, _ext = os.path.splitext(l.filename)
            possible_tags = l.metadata.get('ss_tag_frequency', {}) if l.metadata is not None else {}
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
            if isinstance(possible_tags, str):
                possible_tags = {}
            tags = {}
            for k, v in possible_tags.items():
                words = k.split('_', 1) if '_' in k else [v, k]
                words = [str(w).replace('.json', '') for w in words]
                if words[0] == '{}':
                    words[0] = 0
                tags[' '.join(words[1:])] = words[0]
            name = os.path.splitext(os.path.relpath(l.filename, shared.cmd_opts.lora_dir))[0]
            item = {
                "type": 'Lora',
                "name": name,
                "filename": l.filename,
                "hash": l.shorthash,
                "search_term": self.search_terms_from_path(l.filename) + ' '.join(tags.keys()),
                "preview": self.find_preview(l.filename),
                "description": self.find_description(l.filename),
                "info": self.find_info(l.filename),
                "prompt": json.dumps(f" <lora:{l.get_alias()}:{shared.opts.extra_networks_default_multiplier}>"),
                "local_preview": f"{path}.{shared.opts.samples_format}",
                "metadata": json.dumps(l.metadata, indent=4) if l.metadata else None,
                "tags": tags,
            }
            return item
        except Exception as e:
            shared.log.debug(f"Extra networks error: type=lora file={name} {e}")
            return None

        """
        item = {
            "name": name,
            "filename": lora_on_disk.filename,
            "shorthash": lora_on_disk.shorthash,
            "preview": self.find_preview(path),
            "description": self.find_description(path),
            "search_term": self.search_terms_from_path(lora_on_disk.filename) + " " + (lora_on_disk.hash or ""),
            "local_preview": f"{path}.{shared.opts.samples_format}",
            "metadata": lora_on_disk.metadata,
            "sort_keys": {'default': index, **self.get_sort_keys(lora_on_disk.filename)},
            "sd_version": lora_on_disk.sd_version.name,
        }
        self.read_user_metadata(item)
        activation_text = item["user_metadata"].get("activation text")
        preferred_weight = item["user_metadata"].get("preferred weight", 0.0)
        item["prompt"] = quote_js(f"<lora:{alias}:") + " + " + (str(preferred_weight) if preferred_weight else "opts.extra_networks_default_multiplier") + " + " + quote_js(">")
        if activation_text:
            item["prompt"] += " + " + quote_js(" " + activation_text)
        sd_version = item["user_metadata"].get("sd version")
        if sd_version in network.SdVersion.__members__:
            item["sd_version"] = sd_version
            sd_version = network.SdVersion[sd_version]
        else:
            sd_version = lora_on_disk.sd_version
        if shared.opts.lora_show_all or not enable_filter:
            pass
        elif sd_version == network.SdVersion.Unknown:
            model_version = network.SdVersion.SDXL if shared.sd_model.is_sdxl else network.SdVersion.SD2 if shared.sd_model.is_sd2 else network.SdVersion.SD1
            if model_version.name in shared.opts.lora_hide_unknown_for_versions:
                return None
        elif shared.sd_model.is_sdxl and sd_version != network.SdVersion.SDXL:
            return None
        elif shared.sd_model.is_sd2 and sd_version != network.SdVersion.SD2:
            return None
        elif shared.sd_model.is_sd1 and sd_version != network.SdVersion.SD1:
            return None
        return item
        """

    def list_items(self):
        for _index, name in enumerate(networks.available_networks):
            item = self.create_item(name)
            if item is not None:
                yield item

    def allowed_directories_for_previews(self):
        return [shared.cmd_opts.lora_dir, shared.cmd_opts.lyco_dir]
