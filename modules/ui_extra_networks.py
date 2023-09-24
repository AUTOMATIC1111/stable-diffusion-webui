import re
import time
import json
import html
import os.path
import urllib.parse
import threading
from datetime import datetime
from types import SimpleNamespace
from pathlib import Path
from html.parser import HTMLParser
from collections import OrderedDict
import gradio as gr
from PIL import Image
from starlette.responses import FileResponse, JSONResponse
from modules import shared, scripts, modelloader
from modules.ui_components import ToolButton
import modules.ui_symbols as symbols


allowed_dirs = []
dir_cache = {} # key=path, value=(mtime, listdir(path))
refresh_time = None
extra_pages = shared.extra_networks


def listdir(path):
    if not os.path.exists(path):
        return []
    if path in dir_cache and os.path.getmtime(path) == dir_cache[path][0]:
        return dir_cache[path][1]
    else:
        dir_cache[path] = (os.path.getmtime(path), [os.path.join(path, f) for f in os.listdir(path)])
        return dir_cache[path][1]


def register_page(page):
    # registers extra networks page for the UI; recommend doing it in on_before_ui() callback for extensions
    shared.extra_networks.append(page)
    allowed_dirs.clear()
    for page in shared.extra_networks:
        for folder in page.allowed_directories_for_previews():
            if folder not in allowed_dirs:
                allowed_dirs.append(os.path.abspath(folder))


def init_api(app):

    def fetch_file(filename: str = ""):
        if not os.path.exists(filename):
            return JSONResponse({ "error": f"file {filename}: not found" }, status_code=404)
        if filename.startswith('html/') or filename.startswith('models/'):
            return FileResponse(filename, headers={"Accept-Ranges": "bytes"})
        if not any(Path(folder).absolute() in Path(filename).absolute().parents for folder in allowed_dirs):
            return JSONResponse({ "error": f"file {filename}: must be in one of allowed directories" }, status_code=403)
        if os.path.splitext(filename)[1].lower() not in (".png", ".jpg", ".jpeg", ".webp"):
            return JSONResponse({"error": f"file {filename}: not an image file"}, status_code=403)
        return FileResponse(filename, headers={"Accept-Ranges": "bytes"})

    def get_metadata(page: str = "", item: str = ""):
        page = next(iter([x for x in shared.extra_networks if x.name == page]), None)
        if page is None:
            return JSONResponse({ 'metadata': 'none' })
        metadata = page.metadata.get(item, 'none')
        if metadata is None:
            metadata = ''
        # shared.log.debug(f"Extra networks metadata: page='{page}' item={item} len={len(metadata)}")
        return JSONResponse({"metadata": metadata})

    def get_info(page: str = "", item: str = ""):
        page = next(iter([x for x in get_pages() if x.name == page]), None)
        if page is None:
            return JSONResponse({ 'info': 'none' })
        item = next(iter([x for x in page.items if x['name'] == item]), None)
        if item is None:
            return JSONResponse({ 'info': 'none' })
        info = page.find_info(item['filename'])
        if info is None:
            info = {}
        # shared.log.debug(f"Extra networks info: page='{page.name}' item={item['name']} len={len(info)}")
        return JSONResponse({"info": info})

    def get_desc(page: str = "", item: str = ""):
        page = next(iter([x for x in get_pages() if x.name == page]), None)
        if page is None:
            return JSONResponse({ 'description': 'none' })
        item = next(iter([x for x in page.items if x['name'] == item]), None)
        if item is None:
            return JSONResponse({ 'description': 'none' })
        desc = page.find_description(item['filename'])
        if desc is None:
            desc = ''
        # shared.log.debug(f"Extra networks desc: page='{page.name}' item={item['name']} len={len(desc)}")
        return JSONResponse({"description": desc})

    app.add_api_route("/sd_extra_networks/thumb", fetch_file, methods=["GET"])
    app.add_api_route("/sd_extra_networks/metadata", get_metadata, methods=["GET"])
    app.add_api_route("/sd_extra_networks/info", get_info, methods=["GET"])
    app.add_api_route("/sd_extra_networks/description", get_desc, methods=["GET"])


class ExtraNetworksPage:
    def __init__(self, title):
        self.title = title
        self.name = title.lower()
        self.allow_negative_prompt = False
        self.metadata = {}
        self.info = {}
        self.html = ''
        self.items = []
        self.missing_thumbs = []
        self.refresh_time = None
        # class additional is to keep old extensions happy
        self.card = '''
            <div class='card' onclick={card_click} title='{name}' data-tab='{tabname}' data-page='{page}' data-name='{name}' data-filename='{filename}' data-tags='{tags}'>
                <div class='overlay'>
                    <span style="display:none" class='search_term'>{search_term}</span>
                    <div class='tags'></div>
                    <div class='name'>{title}</div>
                </div>
                <div class='actions'>
                    <span title="Get details" onclick="showCardDetails(event)">&#x1f6c8;</span>
                    <div class='additional'><ul></ul></div>
                </div>
                <img class='preview' src='{preview}' style='width: {width}px; height: {height}px; object-fit: {fit}' loading='lazy'></img>
            </div>
        '''

    def refresh(self):
        pass

    def create_xyz_grid(self):
        xyz_grid = [x for x in scripts.scripts_data if x.script_class.__module__ == "xyz_grid.py"][0].module

        def add_prompt(p, opt, x):
            for item in [x for x in self.items if x["name"] == opt]:
                try:
                    p.prompt = f'{p.prompt} {eval(item["prompt"])}' # pylint: disable=eval-used
                except Exception as e:
                    shared.log.error(f'Cannot evaluate extra network prompt: {item["prompt"]} {e}')

        if not any(self.title in x.label for x in xyz_grid.axis_options):
            if self.title == 'Model':
                return
            opt = xyz_grid.AxisOption(f"[Network] {self.title}", str, add_prompt, choices=lambda: [x["name"] for x in self.items])
            xyz_grid.axis_options.append(opt)

    def link_preview(self, filename):
        quoted_filename = urllib.parse.quote(filename.replace('\\', '/'))
        mtime = os.path.getmtime(filename)
        preview = f"./sd_extra_networks/thumb?filename={quoted_filename}&mtime={mtime}"
        return preview

    def search_terms_from_path(self, filename):
        return filename.replace('\\', '/')

    def is_empty(self, folder):
        for f in listdir(folder):
            _fn, ext = os.path.splitext(f)
            if ext.lower() in ['.ckpt', '.safetensors', '.pt', '.json'] or os.path.isdir(os.path.join(folder, f)):
                return False
        return True

    def create_thumb(self):
        created = 0
        for f in self.missing_thumbs:
            fn, _ext = os.path.splitext(f)
            fn = fn.replace('.preview', '')
            fn = f'{fn}.thumb.jpg'
            if os.path.exists(fn):
                continue
            try:
                img = Image.open(f)
                if img.width > 1024 or img.height > 1024 or os.path.getsize(f) > 65536:
                    img = img.convert('RGB')
                    img.thumbnail((512, 512), Image.HAMMING)
                    img.save(fn, quality=50)
                    img.close()
                    created += 1
            except Exception as e:
                shared.log.error(f'Extra network error creating thumbnail: {f} {e}')
        if created > 0:
            shared.log.info(f"Extra network thumbnails: {self.name} created={created}")
            self.missing_thumbs.clear()

    def create_page(self, tabname, skip = False):
        if self.refresh_time is not None and self.refresh_time > refresh_time: # cached page
            return self.html
        t0 = time.time()
        self_name_id = self.name.replace(" ", "_")
        if skip:
            return f"<div id='{tabname}_{self_name_id}_subdirs' class='extra-network-subdirs'></div><div id='{tabname}_{self_name_id}_cards' class='extra-network-cards'>Extra network page not ready<br>Click refresh to try again</div>"
        subdirs = {}
        allowed_folders = [os.path.abspath(x) for x in self.allowed_directories_for_previews()]
        for parentdir, dirs in {d: modelloader.directory_directories(d) for d in allowed_folders}.items():
            for tgt in dirs.keys():
                if shared.opts.diffusers_dir in tgt:
                    subdirs[os.path.basename(shared.opts.diffusers_dir)] = 1
                if 'models--' in tgt:
                    continue
                subdir = tgt[len(parentdir):].replace("\\", "/")
                while subdir.startswith("/"):
                    subdir = subdir[1:]
                if not self.is_empty(tgt):
                    subdirs[subdir] = 1
        subdirs = OrderedDict(sorted(subdirs.items()))
        subdirs_html = "<button class='lg secondary gradio-button custom-button search-all' onclick='extraNetworksSearchButton(event)'>all</button><br>"
        subdirs_html += "".join([f"<button class='lg secondary gradio-button custom-button' onclick='extraNetworksSearchButton(event)'>{html.escape(subdir)}</button><br>" for subdir in subdirs if subdir != ''])
        self.html = ''
        try:
            self.items = list(self.list_items())
            self.refresh_time = time.time()
        except Exception as e:
            self.items = []
            shared.log.error(f'Extra networks error listing items: class={self.__class__} tab={tabname} {e}')
        self.create_xyz_grid()
        htmls = []
        for item in self.items:
            self.metadata[item["name"]] = item.get("metadata", {})
            htmls.append(self.create_html(item, tabname))
        self.html += ''.join(htmls)
        if len(subdirs_html) > 0 or len(self.html) > 0:
            self.html = f"<div id='{tabname}_{self_name_id}_subdirs' class='extra-network-subdirs'>{subdirs_html}</div><div id='{tabname}_{self_name_id}_cards' class='extra-network-cards'>{self.html}</div>"
        else:
            return ''
        t1 = time.time()
        shared.log.debug(f"Extra networks: page='{self.name}' items={len(self.items)} subdirs={len(subdirs)} tab={tabname} dirs={self.allowed_directories_for_previews()} time={round(t1-t0, 2)}")
        threading.Thread(target=self.create_thumb).start()

    def list_items(self):
        raise NotImplementedError

    def allowed_directories_for_previews(self):
        return []

    def create_html(self, item, tabname):
        try:
            args = {
                "tabname": tabname,
                "page": self.name,
                "name": item["name"],
                "title": item["name"].replace('_', ' '),
                "filename": item["filename"],
                "tags": '|'.join([item.get("tags")] if isinstance(item.get("tags", {}), str) else list(item.get("tags", {}).keys())),
                "preview": html.escape(item.get("preview", self.link_preview('html/card-no-preview.png'))),
                "width": shared.opts.extra_networks_card_size,
                "height": shared.opts.extra_networks_card_size if shared.opts.extra_networks_card_square else 'auto',
                "fit": shared.opts.extra_networks_card_fit,
                "prompt": item.get("prompt", None),
                "search_term": item.get("search_term", ""),
                "description": item.get("description") or "",
                "card_click": item.get("onclick", '"' + html.escape(f'return cardClicked({item.get("prompt", None)}, {"true" if self.allow_negative_prompt else "false"})') + '"'),
            }
            alias = item.get("alias", None)
            if alias is not None:
                args['title'] += f'\nAlias: {alias}'
            return self.card.format(**args)
        except Exception as e:
            shared.log.error(f'Extra networks item error: page={tabname} item={item["name"]} {e}')
            return ""

    def find_preview_file(self, path):
        fn = os.path.splitext(path)[0]
        preview_extensions = ["jpg", "jpeg", "png", "webp", "tiff", "jp2"]
        for file in [f'{fn}{mid}{ext}' for ext in preview_extensions for mid in ['.thumb.', '.preview.', '.']]:
            if os.path.exists(file):
                return file
        return 'html/card-no-preview.png'

    def find_preview(self, path):
        fn = os.path.splitext(path)[0]
        preview_extensions = ["jpg", "jpeg", "png", "webp", "tiff", "jp2"]
        for file in [f'{fn}{mid}{ext}' for ext in preview_extensions for mid in ['.thumb.', '.preview.', '.']]:
            if os.path.exists(file):
                if '.thumb.' not in file:
                    self.missing_thumbs.append(file)
                return self.link_preview(file)
        return self.link_preview('html/card-no-preview.png')

    def find_description(self, path):
        class HTMLFilter(HTMLParser):
            text = ""
            def handle_data(self, data):
                self.text += data
            def handle_endtag(self, tag):
                if tag == 'p':
                    self.text += '\n'

        fn = os.path.splitext(path)[0]
        for file in [f"{fn}.txt", f"{fn}.description.txt"]:
            if os.path.exists(file):
                try:
                    with open(file, "r", encoding="utf-8", errors="replace") as f:
                        txt = f.read()
                        txt = re.sub('[<>]', '', txt)
                        return txt
                except OSError:
                    pass
        info = self.find_info(path)
        desc = info.get('description', '') or ''
        f = HTMLFilter()
        f.feed(desc)
        return f.text

    def find_info(self, path):
        fn = os.path.splitext(path)[0] + '.json'
        if os.path.exists(fn):
            return shared.readfile(fn, silent=True)
        return {}


def initialize():
    shared.extra_networks.clear()


def register_pages():
    from modules.ui_extra_networks_textual_inversion import ExtraNetworksPageTextualInversion
    from modules.ui_extra_networks_hypernets import ExtraNetworksPageHypernetworks
    from modules.ui_extra_networks_checkpoints import ExtraNetworksPageCheckpoints
    from modules.ui_extra_networks_styles import ExtraNetworksPageStyles
    register_page(ExtraNetworksPageCheckpoints())
    register_page(ExtraNetworksPageStyles())
    register_page(ExtraNetworksPageTextualInversion())
    register_page(ExtraNetworksPageHypernetworks())


def get_pages():
    pages = []
    if 'All' in shared.opts.extra_networks:
        pages = shared.extra_networks
    else:
        titles = [page.title for page in shared.extra_networks]
        for page in shared.opts.extra_networks:
            try:
                idx = titles.index(page)
            except ValueError:
                continue
            pages.append(shared.extra_networks[idx])
    return pages


class ExtraNetworksUi:
    def __init__(self):
        self.tabname: str = None
        self.pages: list(str) = None
        self.visible: gr.State = None
        self.state: gr.Textbox = None
        self.details: gr.Group = None
        self.tabs: gr.Tabs = None
        self.gallery: gr.Gallery = None
        self.description: gr.Textbox = None
        self.search: gr.Textbox = None
        self.button_details: gr.Button = None
        self.details_components: list = []
        self.last_item: dict = None


def create_ui(container, button_parent, tabname, skip_indexing = False):
    ui = ExtraNetworksUi()
    ui.tabname = tabname
    ui.pages = []
    ui.state = gr.Textbox('{}', elem_id=tabname+"_extra_state", visible=False)
    ui.visible = gr.State(value=False) # pylint: disable=abstract-class-instantiated
    ui.details = gr.Group(elem_id=tabname+"_extra_details", visible=False)
    ui.tabs = gr.Tabs(elem_id=tabname+"_extra_tabs")
    ui.button_details = gr.Button('Details', elem_id=tabname+"_extra_details_btn", visible=False)
    state = {}

    def get_item(state):
        if state is None or not hasattr(state, 'page') or not hasattr(state, 'item'):
            return None, None
        page = next(iter([x for x in get_pages() if x.title == state.page]), None)
        if page is None:
            return None, None
        item = next(iter([x for x in page.items if x["name"] == state.item]), None)
        if item is None:
            return page, None
        item = SimpleNamespace(**item)
        return page, item

    # main event that is triggered when js updates state text field with json values, used to communicate js -> python
    def state_change(state_text):
        try:
            nonlocal state
            state = SimpleNamespace(**json.loads(state_text))
        except Exception as e:
            shared.log.error(f'Extra networks state error: {e}')
            return
        _page, item = get_item(state)
        # shared.log.debug(f'Extra network: op={state.op} page={page.title if page is not None else None} item={item.filename if item is not None else None}')
        ui.last_item = item

    def toggle_visibility(is_visible):
        is_visible = not is_visible
        return is_visible, gr.update(visible=is_visible), gr.update(variant=("secondary-down" if is_visible else "secondary"))

    with ui.details:
        details_close = ToolButton(symbols.close, elem_id=tabname+"_extra_details_close")
        details_close.click(fn=lambda: gr.update(visible=False), inputs=[], outputs=[ui.details])
        with gr.Row():
            with gr.Column(scale=1):
                text = gr.HTML('<div>title</div>')
                ui.details_components.append(text)
            with gr.Column(scale=1):
                img = gr.Image(value=None, show_label=False, interactive=False, container=True)
                ui.details_components.append(img)
                with gr.Row():
                    btn_save_img = gr.Button('Replace', elem_classes=['small-button'])
                    btn_delete_img = gr.Button('Delete', elem_classes=['small-button'])
        with gr.Tabs():
            with gr.Tab('Description'):
                desc = gr.Textbox('', show_label=False, lines=8, placeholder="Extra network description...")
                ui.details_components.append(desc)
                with gr.Row():
                    btn_save_desc = gr.Button('Save', elem_classes=['small-button'])
                    btn_delete_desc = gr.Button('Delete', elem_classes=['small-button'])
            with gr.Tab('Model metadata'):
                info = gr.JSON({}, show_label=False, lines=8)
                ui.details_components.append(info)
                with gr.Row():
                    btn_save_info = gr.Button('Save', elem_classes=['small-button'])
                    btn_delete_info = gr.Button('Delete', elem_classes=['small-button'])
            with gr.Tab('Embedded metadata'):
                meta = gr.JSON({}, show_label=False, lines=8)
                ui.details_components.append(meta)

    with ui.tabs:
        button_refresh = ToolButton(symbols.refresh, elem_id=tabname+"_extra_refresh")
        button_close = ToolButton(symbols.close, elem_id=tabname+"_extra_close")
        ui.search = gr.Textbox('', show_label=False, elem_id=tabname+"_extra_search", placeholder="Search...", elem_classes="textbox", lines=2)
        ui.description = gr.Textbox('', show_label=False, elem_id=tabname+"_description", elem_classes="textbox", lines=2, interactive=False)

        if ui.tabname == 'txt2img': # refresh only once
            global refresh_time # pylint: disable=global-statement
            refresh_time = time.time()
        for page in get_pages():
            page.create_page(ui.tabname, skip_indexing)
            with gr.Tab(page.title, id=page.title.lower().replace(" ", "_"), elem_classes="extra-networks-tab"):
                hmtl = gr.HTML(page.html, elem_id=f'{tabname}{page.name}_extra_page', elem_classes="extra-networks-page")
                ui.pages.append(hmtl)

    def fn_save_img():
        if ui.last_item is None or ui.last_item.local_preview is None:
            return 'html/card-no-preview.png'
        images = list(ui.gallery.temp_files) # gallery cannot be used as input component so looking at most recently registered temp files
        if len(images) < 1:
            shared.log.warning(f'Extra network no image: item={ui.last_item.name}')
            return 'html/card-no-preview.png'
        try:
            images.sort(key=lambda f: os.path.getmtime(f), reverse=True)
            image = Image.open(images[0])
        except Exception as e:
            shared.log.error(f'Extra network error opening image: item={ui.last_item.name} {e}')
            return 'html/card-no-preview.png'
        fn_delete_img()
        if image.width > 512 or image.height > 512:
            image = image.convert('RGB')
            image.thumbnail((512, 512), Image.HAMMING)
        image.save(ui.last_item.local_preview, quality=50)
        shared.log.debug(f'Extra network save image: item={ui.last_item.name} filename={ui.last_item.local_preview}')
        return image

    def fn_delete_img():
        preview_extensions = ["jpg", "jpeg", "png", "webp", "tiff", "jp2"]
        fn = os.path.splitext(ui.last_item.filename)[0]
        for file in [f'{fn}{mid}{ext}' for ext in preview_extensions for mid in ['.thumb.', '.preview.', '.']]:
            if os.path.exists(file):
                os.remove(file)
                shared.log.debug(f'Extra network delete image: item={ui.last_item.name} filename={file}')
        return 'html/card-no-preview.png'

    def fn_save_desc(desc):
        fn = os.path.splitext(ui.last_item.filename)[0] + '.txt'
        with open(fn, 'w', encoding='utf-8') as f:
            f.write(desc)
        shared.log.debug(f'Extra network save desc: item={ui.last_item.name} filename={fn}')
        return desc

    def fn_delete_desc(desc):
        if ui.last_item is None:
            return desc
        fn = os.path.splitext(ui.last_item.filename)[0] + '.txt'
        if os.path.exists(fn):
            shared.log.debug(f'Extra network delete desc: item={ui.last_item.name} filename={fn}')
            os.remove(fn)
            return ''
        return desc

    def fn_save_info(info):
        fn = os.path.splitext(ui.last_item.filename)[0] + '.json'
        shared.writefile(info, fn, silent=True)
        shared.log.debug(f'Extra network save desc: item={ui.last_item.name} filename={fn}')
        return info

    def fn_delete_info(info):
        if ui.last_item is None:
            return info
        fn = os.path.splitext(ui.last_item.filename)[0] + '.json'
        if os.path.exists(fn):
            shared.log.debug(f'Extra network delete info: item={ui.last_item.name} filename={fn}')
            os.remove(fn)
            return ''
        return info

    btn_save_img.click(fn=fn_save_img, inputs=[], outputs=[img])
    btn_delete_img.click(fn=fn_delete_img, inputs=[], outputs=[img])
    btn_save_desc.click(fn=fn_save_desc, inputs=[desc], outputs=[desc])
    btn_delete_desc.click(fn=fn_delete_desc, inputs=[desc], outputs=[desc])
    btn_save_info.click(fn=fn_save_info, inputs=[info], outputs=[info])
    btn_delete_info.click(fn=fn_delete_info, inputs=[info], outputs=[info])

    def show_details(text, img, desc, info, meta):
        page, item = get_item(state)
        if item is not None and os.path.exists(item.filename):
            stat = os.stat(item.filename)
            desc = item.description
            fullinfo = shared.readfile(os.path.splitext(item.filename)[0] + '.json', silent=True)
            if 'modelVersions' in fullinfo: # sanitize massive objects
                fullinfo['modelVersions'] = []
            info = fullinfo
            meta = page.metadata.get(item.name, {}) or {}
            if type(meta) is str:
                try:
                    meta = json.loads(meta)
                except Exception:
                    meta = {}
            img = page.find_preview_file(item.filename)
            lora = ''
            model = ''
            if page.title == 'Model':
                merge = len(list(meta.get('sd_merge_models', {})))
                if merge > 0:
                    model += f'<tr><td>Merge models</td><td>{merge} recipes</td></tr>'
                if meta.get('modelspec.architecture', None) is not None:
                    model += f'''
                        <tr><td>Architecture</td><td>{meta.get('modelspec.architecture', 'N/A')}</td></tr>
                        <tr><td>Title</td><td>{meta.get('modelspec.title', 'N/A')}</td></tr>
                        <tr><td>Resolution</td><td>{meta.get('modelspec.resolution', 'N/A')}</td></tr>
                    '''
            if page.title == 'Lora':
                tags = getattr(item, 'tags', {})
                tags = [f'{name}:{tags[name]}' for i, name in enumerate(tags)]
                tags = ' '.join(tags)
                lora = f'''
                    <tr><td>Tags</td><td>{tags}</td></tr>
                    <tr><td>Base model</td><td>{meta.get('ss_sd_model_name', 'N/A')}</td></tr>
                    <tr><td>Resolution</td><td>{meta.get('ss_resolution', 'N/A')}</td></tr>
                    <tr><td>Training images</td><td>{meta.get('ss_num_train_images', 'N/A')}</td></tr>
                    <tr><td>Comment</td><td>{meta.get('ss_training_comment', 'N/A')}</td></tr>
                '''
            text = f'''
                <h2 style="border-bottom: 1px solid var(--button-primary-border-color); margin-bottom: 1em; margin-top: -1.3em !important;">{item.name}</h2>
                <table style="width: 100%; line-height: 1.3em;"><tbody>
                    <tr><td>Type</td><td>{page.title}</td></tr>
                    <tr><td>Alias</td><td>{getattr(item, 'alias', 'N/A')}</td></tr>
                    <tr><td>Filename</td><td>{item.filename}</td></tr>
                    <tr><td>Hash</td><td>{getattr(item, 'hash', 'N/A')}</td></tr>
                    <tr><td>Size</td><td>{round(stat.st_size/1024/1024, 2)} MB</td></tr>
                    <tr><td>Last modified</td><td>{datetime.fromtimestamp(stat.st_mtime)}</td></tr>
                    {lora}
                    {model}
                </tbody></table>
            '''
        return [text, img, desc, info, meta, gr.update(visible=item is not None)]

    def en_refresh(title):
        pages = []
        for page in get_pages():
            if title is None or title == '' or title == page.title or len(page.html) == 0:
                page.refresh()
                page.refresh_time = None
                page.create_page(ui.tabname)
                shared.log.debug(f"Refreshing Extra networks: page='{page.title}' items={len(page.items)} tab={ui.tabname}")
            pages.append(page.html)
        ui.search.update(value = ui.search.value)
        return pages

    button_parent.click(fn=toggle_visibility, inputs=[ui.visible], outputs=[ui.visible, container, button_parent])
    button_close.click(fn=toggle_visibility, inputs=[ui.visible], outputs=[ui.visible, container])
    button_refresh.click(_js='getENActivePage', fn=en_refresh, inputs=[ui.search], outputs=ui.pages)
    ui.state.change(state_change, inputs=[ui.state], outputs=[])
    ui.button_details.click(show_details, _js="getCardDetails", inputs=ui.details_components, outputs=ui.details_components + [ui.details])
    return ui


def setup_ui(ui, gallery):
    ui.gallery = gallery
