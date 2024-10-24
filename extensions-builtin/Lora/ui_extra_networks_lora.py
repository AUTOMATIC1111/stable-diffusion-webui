import os
import html
import datetime
import math
import matplotlib as mpl
import colorsys

import network
import networks

from modules import shared, ui_extra_networks
from modules.ui_extra_networks import quote_js
from ui_edit_user_metadata import LoraUserMetadataEditor, build_tags


class ExtraNetworksPageLora(ui_extra_networks.ExtraNetworksPage):
    def __init__(self):
        super().__init__('Lora')

    def refresh(self):
        networks.list_available_networks()
        super().refresh()

    def create_item(self, name, index=None, enable_filter=True):
        lora_on_disk = networks.available_networks.get(name)
        if lora_on_disk is None:
            return

        path, ext = os.path.splitext(lora_on_disk.filename)

        alias = lora_on_disk.get_alias()

        search_terms = [self.search_terms_from_path(lora_on_disk.filename)]
        if lora_on_disk.hash:
            search_terms.append(lora_on_disk.hash)
        item = {
            "name": name,
            "filename": lora_on_disk.filename,
            "shorthash": lora_on_disk.shorthash,
            "preview": self.find_preview(path) or self.find_embedded_preview(path, name, lora_on_disk.metadata),
            "description": self.find_description(path),
            "search_terms": search_terms,
            "local_preview": f"{path}.{shared.opts.samples_format}",
            "metadata": lora_on_disk.metadata,
            "sort_keys": {'default': index, **self.get_sort_keys(lora_on_disk.filename)},
            "sd_version": lora_on_disk.sd_version.name,
        }

        self.read_user_metadata(item)
        activation_text = item["user_metadata"].get("activation text")
        preferred_weight = item["user_metadata"].get("preferred weight", 0.0)
        prompt = f"<lora:{alias}:{str(preferred_weight) if preferred_weight else shared.opts.extra_networks_default_multiplier}>"
        if activation_text:
            prompt += f" {activation_text}"
        item["prompt"] = quote_js(prompt)

        negative_prompt = item["user_metadata"].get("negative text")
        item["negative_prompt"] = quote_js("")
        if negative_prompt:
            item["negative_prompt"] = quote_js('(' + negative_prompt + ':1)')

        sd_version = item["user_metadata"].get("sd version")
        if sd_version in network.SdVersion.__members__:
            item["sd_version"] = sd_version
            sd_version = network.SdVersion[sd_version]
        else:
            sd_version = lora_on_disk.sd_version

        if shared.opts.lora_show_all or not enable_filter or not shared.sd_model:
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

    def list_items(self):
        # instantiate a list to protect against concurrent modification
        names = list(networks.available_networks)
        for index, name in enumerate(names):
            item = self.create_item(name, index)
            if item is not None:
                yield item

    def allowed_directories_for_previews(self):
        return [shared.cmd_opts.lora_dir, shared.cmd_opts.lyco_dir_backcompat]

    def create_user_metadata_editor(self, ui, tabname):
        return LoraUserMetadataEditor(ui, tabname, self)

    def get_model_detail_metadata_table(self, model_name: str) -> str:
        res = super().get_model_detail_metadata_table(model_name)

        metadata = self.metadata.get(model_name)
        if metadata is None:
            metadata = {}

        keys = {
            'ss_output_name': "Output name:",
            'ss_sd_model_name': "Model:",
            'ss_clip_skip': "Clip skip:",
            'ss_network_module': "Kohya module:",
        }

        params = []

        for k, lbl in keys.items():
            v = metadata.get(k, None)
            if v is not None and str(v) != "None":
                params.append((lbl, html.escape(v)))

        ss_training_started_at = metadata.get('ss_training_started_at')
        if ss_training_started_at:
            date_trained = datetime.datetime.utcfromtimestamp(
                float(ss_training_started_at)
            ).strftime('%Y-%m-%d %H:%M')
            params.append(("Date trained:", date_trained))

        ss_bucket_info = metadata.get("ss_bucket_info")
        if ss_bucket_info and "buckets" in ss_bucket_info:
            resolutions = {}
            for _, bucket in ss_bucket_info["buckets"].items():
                resolution = bucket["resolution"]
                resolution = f'{resolution[1]}x{resolution[0]}'
                resolutions[resolution] = resolutions.get(resolution, 0) + int(bucket["count"])

            resolutions_list = sorted(resolutions.keys(), key=resolutions.get, reverse=True)
            resolutions_text = html.escape(", ".join(resolutions_list))
            resolutions_text = (
                "<div class='styled-scrollbar' style='overflow-x: auto'>"
                f"{resolutions_text}"
                "</div>"
            )
            params.append(("Resolutions:", resolutions_text))

        image_count = 0
        for v in metadata.get("ss_dataset_dirs", {}).values():
            image_count += int(v.get("img_count", 0))

        if image_count:
            params.append(("Dataset size:", image_count))

        tbl_metadata = "".join([f"<tr><th>{tr[0]}</th><td>{tr[1]}</td>" for tr in params])

        return res + tbl_metadata

    def get_model_detail_extra_html(self, model_name: str) -> str:
        """Generates HTML to show in the details view."""
        res = ""

        item = self.items.get(model_name, {})
        metadata = item.get("metadata", {}) or {}
        user_metadata = item.get("user_metadata", {}) or {}

        sd_version = item.get("sd_version", None)
        preferred_weight = user_metadata.get("preferred weight", None)
        activation_text = user_metadata.get("activation text", None)
        negative_text = user_metadata.get("negative text", None)

        rows = []

        if sd_version is not None:
            rows.append(("SD Version:", sd_version))

        if preferred_weight is not None:
            rows.append(("Preferred weight:", preferred_weight))

        if activation_text is not None:
            rows.append(("Activation text:", activation_text))

        if negative_text is not None:
            rows.append(("Negative propmt:", negative_text))

        rows_html = "".join([f"<tr><th>{tr[0]}</th><td>{tr[1]}</td>" for tr in rows])

        if rows_html:
            res += "<h3>User Metadata</h3>"
            res += f"<table><tbody>{rows_html}</tbody></table>"

        tags = build_tags(metadata)
        if tags is None or len(tags) == 0:
            res += "<h3>Model Tags</h3>"
            res += "<div class='model-info--tags'>Metadata contains no tags</div>"
            return res

        min_tag = min(int(x[1]) for x in tags)
        max_tag = max(int(x[1]) for x in tags)

        cmap = mpl.colormaps["coolwarm"]

        def _clamp(x: float, min_val: float, max_val: float) -> float:
            return max(min_val, min(x, max_val))

        def _get_fg_color(r, g, b) -> str:
            return "#000000" if (r * 0.299 + g * 0.587 + b * 0.114) > 0.5 else "#FFFFFF"

        tag_elems = []
        for (tag_name, tag_count) in tags:
            # Normalize tag count
            tag_count = int(tag_count)
            if min_tag == max_tag:  # Prevent DivideByZero error.
                cmap_idx = cmap.N // 2
            else:
                cmap_idx = math.floor(
                    (tag_count - min_tag) / (max_tag - min_tag) * (cmap.N - 1)
                )

            # Get the bg color based on tag count and a contrasting fg color.
            base_color = cmap(cmap_idx)
            base_color = [_clamp(x, 0, 1) for x in base_color]
            base_fg_color = _get_fg_color(*base_color[:3])
            # Now get a slightly darker background for the tag count bg color.
            h, lum, s = colorsys.rgb_to_hls(*base_color[:3])
            lum = max(min(lum * 0.7, 1.0), 0.0)
            dark_color = colorsys.hls_to_rgb(h, lum, s)
            dark_color = [_clamp(x, 0, 1) for x in dark_color]
            dark_fg_color = _get_fg_color(*dark_color[:3])
            # Convert the colors to a hex string.
            base_color = mpl.colors.rgb2hex(base_color)
            dark_color = mpl.colors.rgb2hex(dark_color)
            # Finally, generate the HTML for this tag.
            tag_style = f"background: {base_color};"
            name_style = f"color: {base_fg_color};"
            count_style = f"background: {dark_color}; color: {dark_fg_color};"

            tag_elems.append((
                f"<span class='model-info--tag' style='{tag_style}'>"
                f"<span class='model-info--tag-name' style='{name_style}'>{tag_name}</span>"
                f"<span class='model-info--tag-count' style='{count_style}'>{tag_count}</span>"
                "</span>"
            ))
        res += "<h3>Model Tags</h3>"
        res += f"<div class='model-info--tags'>{''.join(tag_elems)}</div>"
        return res
