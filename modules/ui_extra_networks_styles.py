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

    def parse_desc(self, desc):
        lines = desc.strip().split("\n")
        params = { 'name': '', 'description': '', 'prompt': '', 'negative': '', 'extra': ''}
        found = ''
        for line in lines:
            line = line.strip()
            if line.lower().startswith('name:'):
                found = 'name'
                params['name'] = line[5:].strip()
            elif line.lower().startswith('description:'):
                found = 'description'
                params['description'] = line[12:].strip()
            elif line.lower().startswith('prompt:'):
                found = 'prompt'
                params['prompt'] = line[7:].strip()
            elif line.lower().startswith('negative:'):
                found = 'negative'
                params['negative'] = line[9:].strip()
            elif line.lower().startswith('extra:'):
                found = 'extra'
                params['extra'] = line[6:].strip()
            elif found != '':
                params[found] += '\n' + line
        if params['name'] == '':
            return None
        if params['description'] == '':
            params['description'] = params['name']
        return params

    def create_style(self, params):
        from modules.images import FilenameGenerator
        from hashlib import sha256
        namegen = FilenameGenerator(p=None, seed=None, prompt=params.get('Prompt', ''), image=None, grid=False)
        name = namegen.prompt_words()
        sha = sha256(json.dumps(name).encode()).hexdigest()[0:8]
        fn = os.path.join(shared.opts.styles_dir, sha + '.json')
        item = {
            "type": 'Style',
            "name": name,
            "title": name,
            "filename": fn,
            "search_term": f'{self.search_terms_from_path(name)}',
            "preview": self.find_preview(name),
            "description": '',
            "prompt": params.get('Prompt', ''),
            "negative": params.get('Negative prompt', ''),
            "extra": '', # TODO add extras to styles
            "local_preview": f"{name}.{shared.opts.samples_format}",
        }
        return item

    def list_items(self):
        for k, style in shared.prompt_styles.styles.items():
            try:
                fn = os.path.splitext(getattr(style, 'filename', ''))[0]
                name = getattr(style, 'name', '')
                if name == '':
                    continue
                txt = f'Prompt: {getattr(style, "prompt", "")}'
                if len(getattr(style, 'negative_prompt', '')) > 0:
                    txt += f'\nNegative: {style.negative_prompt}'
                yield {
                    "type": 'Style',
                    "name": name,
                    "title": k,
                    "filename": style.filename,
                    "search_term": f'{txt} {self.search_terms_from_path(name)}',
                    "preview": style.preview if getattr(style, 'preview', None) is not None and style.preview.startswith('data:') else self.find_preview(fn),
                    "description": style.description if getattr(style, 'description', None) is not None and len(style.description) > 0 else txt,
                    "prompt": getattr(style, 'prompt', ''),
                    "negative": getattr(style, 'negative_prompt', ''),
                    "extra": getattr(style, 'extra', ''),
                    "local_preview": f"{fn}.{shared.opts.samples_format}",
                    "onclick": '"' + html.escape(f"""return selectStyle({json.dumps(name)})""") + '"',
                }
            except Exception as e:
                shared.log.debug(f"Extra networks error: type=style file={k} {e}")


    def allowed_directories_for_previews(self):
        return [v for v in [shared.opts.styles_dir] if v is not None]
