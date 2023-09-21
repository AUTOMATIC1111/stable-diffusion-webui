import os
import html
import json
from modules import shared, ui_extra_networks


class ExtraNetworksPageStyles(ui_extra_networks.ExtraNetworksPage):
    def __init__(self):
        super().__init__('Style')

    def refresh(self):
        shared.prompt_styles.reload()

    """
    import io
    import base64
    from PIL import Image

    def image2str(image):
        buff = io.BytesIO()
        image.save(buff, format="JPEG", quality=80)
        encoded = base64.b64encode(buff.getvalue())
        return encoded

    def str2image(data):
        buff = io.BytesIO(base64.b64decode(data))
        return Image.open(buff)

    def save_preview(self, index, images, filename):
        from modules.generation_parameters_copypaste import image_from_url_text
        try:
            image = image_from_url_text(images[int(index)])
        except Exception:
            shared.log.error(f'Extra network save preview: {filename} no image')
            return
        if image.width > 512 or image.height > 512:
            image = image.convert('RGB').thumbnail((512, 512), Image.HAMMING)
        for k in shared.prompt_styles.styles.keys():
            if k == filename:
                shared.prompt_styles.styles[k].preview = image2str(image)
                break

    def save_description(self, filename, desc):
        pass
    """

    def list_items(self):
        for k, style in shared.prompt_styles.styles.items():
            fn = os.path.splitext(style.filename)[0]
            txt = f'Prompt: {style.prompt}'
            if len(style.negative_prompt) > 0:
                txt += f'\nNegative: {style.negative_prompt}'
            yield {
                "name": style.name,
                "title": k,
                "filename": style.filename,
                "search_term": f'{txt} {self.search_terms_from_path(style.name)}',
                "preview": self.find_preview(fn),
                "local_preview": f"{fn}.{shared.opts.samples_format}",
                "description": txt,
                "onclick": '"' + html.escape(f"""return selectStyle({json.dumps(style.name)})""") + '"',
            }

    def allowed_directories_for_previews(self):
        return [v for v in [shared.opts.styles_dir] if v is not None]
