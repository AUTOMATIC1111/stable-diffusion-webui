import os
import html
import json

from modules import shared, ui_extra_networks


class ExtraNetworksPageStyles(ui_extra_networks.ExtraNetworksPage):
    def __init__(self):
        super().__init__('Styles')

    def refresh(self):
        shared.prompt_styles.reload()

    def list_items(self):
        styles = list(shared.prompt_styles.styles)
        for style in styles:
            path = os.path.join(shared.opts.styles_dir, style)
            txt = f'Prompt: {shared.prompt_styles.styles[style].prompt}'
            negative = shared.prompt_styles.styles[style].negative_prompt
            if negative is not None and len(negative) > 0:
                txt += f'\nNegative: {negative}'
            yield {
                "name": style,
                "search_term": path,
                "filename": path,
                "preview": self.find_preview(path),
                "description": txt,
                "onclick": '"' + html.escape(f"""return selectStyle({json.dumps(style)})""") + '"',
                "local_preview": f"{path}.{shared.opts.samples_format}",
            }

    def allowed_directories_for_previews(self):
        return [v for v in [shared.opts.styles_dir] if v is not None]
