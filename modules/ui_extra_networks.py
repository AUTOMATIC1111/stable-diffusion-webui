import re
import time
import json
import html
import os.path
import urllib.parse
import threading
from pathlib import Path
from collections import OrderedDict
import gradio as gr
from PIL import Image
from starlette.responses import FileResponse, JSONResponse
from modules import shared, scripts, modelloader
from modules.generation_parameters_copypaste import image_from_url_text
from modules.ui_components import ToolButton
import modules.ui_symbols as symbols


extra_pages = []
allowed_dirs = []
dir_cache = {} # key=path, value=(mtime, listdir(path))
refresh_time = None


def listdir(path):
    if not os.path.exists(path):
        return []
    if path in dir_cache and os.path.getmtime(path) == dir_cache[path][0]:
        return dir_cache[path][1]
    else:
        dir_cache[path] = (
            os.path.getmtime(path),
            [os.path.join(path, f) for f in os.listdir(path)]
        )
        return dir_cache[path][1]


def register_page(page):
    # registers extra networks page for the UI; recommend doing it in on_before_ui() callback for extensions
    extra_pages.append(page)
    allowed_dirs.clear()
    for page in extra_pages:
        for folder in page.allowed_directories_for_previews():
            if folder not in allowed_dirs:
                allowed_dirs.append(os.path.abspath(folder))


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
    page = next(iter([x for x in extra_pages if x.name == page]), None)
    if page is None:
        return JSONResponse({ 'metadata': 'none' })
    metadata = page.metadata.get(item, 'none')
    if metadata is None:
        metadata = ''
    shared.log.debug(f"Extra networks metadata: page='{page}' item={item} len={len(metadata)}")
    return JSONResponse({"metadata": metadata})


def get_info(page: str = "", item: str = ""):
    page = next(iter([x for x in extra_pages if x.name == page]), None)
    if page is None:
        return JSONResponse({ 'info': 'none' })
    info = page.info.get(item, 'none')
    if info is None:
        info = ''
    shared.log.debug(f"Extra networks info: page='{page}' item={item} len={len(info)}")
    return JSONResponse({"info": info})


def add_pages_to_demo(app):
    app.add_api_route("/sd_extra_networks/thumb", fetch_file, methods=["GET"])
    app.add_api_route("/sd_extra_networks/metadata", get_metadata, methods=["GET"])
    app.add_api_route("/sd_extra_networks/info", get_info, methods=["GET"])


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
            <div class='card' onclick={card_click} title='{title}' data-filename='{local_preview}' data-description='{description}' data-tags='{tags}'>
                <div class='overlay'>
                    <span style="display:none" class='search_term'>{search_term}</span>
                    <div class='name'>{name}</div>
                    <div class='tags'></div>
                    <div class='actions'>
                        <div class='additional'><ul></ul></div>
                        <span title="Save current image as preview image" onclick={card_save_preview}>⏺️</span>
                        <span title="Save current description" onclick={card_save_desc}>🛅</span>
                        {card_extra}
                    </div>
                </div>
                <img class='preview' src='{preview}' style='width: {width}px; height: {height}px; object-fit: {fit}' loading='{loading}'></img>
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
            if self.title == 'Checkpoints':
                return
            opt = xyz_grid.AxisOption(f"[Network] {self.title}", str, add_prompt, choices=lambda: [x["name"] for x in self.items])
            xyz_grid.axis_options.append(opt)

    def link_preview(self, filename):
        quoted_filename = urllib.parse.quote(filename.replace('\\', '/'))
        mtime = os.path.getmtime(filename)
        preview = f"./sd_extra_networks/thumb?filename={quoted_filename}&mtime={mtime}"
        return preview

    def search_terms_from_path(self, filename, possible_directories=None):
        abspath = os.path.abspath(filename)
        for parentdir in (possible_directories if possible_directories is not None else self.allowed_directories_for_previews()):
            parentdir = os.path.abspath(parentdir)
            if abspath.startswith(parentdir):
                return abspath[len(parentdir):].replace('\\', '/')
        return ""

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
            self.info[item["name"]] = item.get('info', None) or self.find_info(item['filename'])
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
                "tabname": json.dumps(tabname),
                "name": item["name"].replace('_', ' '),
                "title": item["name"],
                "tags": '|'.join([item.get("tags")] if isinstance(item.get("tags", {}), str) else list(item.get("tags", {}).keys())),
                "preview": html.escape(item.get("preview", None)),
                "width": shared.opts.extra_networks_card_size,
                "height": shared.opts.extra_networks_card_size if shared.opts.extra_networks_card_square else 'auto',
                "fit": shared.opts.extra_networks_card_fit,
                "loading": "lazy" if shared.opts.extra_networks_card_lazy else "eager",
                "prompt": item.get("prompt", None),
                "search_term": item.get("search_term", ""),
                "description": item.get("description") or "",
                "local_preview": item["local_preview"],
                "card_click": item.get("onclick", '"' + html.escape(f'return cardClicked({item.get("prompt", None)}, {"true" if self.allow_negative_prompt else "false"})') + '"'),
                "card_save_preview": '"' + html.escape('return saveCardPreview(event)') + '"',
                "card_save_desc": '"' + html.escape('return saveCardDescription(event)') + '"',
                "card_extra": "",
            }
            metadata = item.get("metadata", None)
            if metadata is not None and len(metadata) > 0:
                card_read_meta = '"' + html.escape(f'return readCardMetadata(event, {json.dumps(self.name)}, {json.dumps(item["name"])})') + '"'
                args['card_extra'] += f'<span title="Read metadata" onclick={card_read_meta}>📘</span>'
            info = item.get("info", None)
            if info is not None and len(info) > 0:
                card_read_info = '"' + html.escape(f'return readCardInformation(event, {json.dumps(self.name)}, {json.dumps(item["name"])})') + '"'
                args['card_extra'] += f'<span title="Read info" onclick={card_read_info}>ℹ️</span>' # noqa
            alias = item.get("alias", None)
            if alias is not None:
                args['title'] += f'\nAlias: {alias}'
            return self.card.format(**args)
        except Exception as e:
            shared.log.error(f'Extra networks item error: page={tabname} item={item["name"]} {e}')
            return ""

    def find_preview(self, path):
        preview_extensions = ["jpg", "jpeg", "png", "webp", "tiff", "jp2"]
        files = listdir(os.path.dirname(path))
        for file in [f'{path}{mid}{ext}' for ext in preview_extensions for mid in ['.thumb.', '.preview.', '.']]:
            if file in files:
                if '.thumb.' not in file:
                    self.missing_thumbs.append(file)
                return self.link_preview(file)
        return self.link_preview('html/card-no-preview.png')

    def find_description(self, path):
        files = listdir(os.path.dirname(path))
        for file in [f"{path}.txt", f"{path}.description.txt"]:
            if file in files:
                try:
                    with open(file, "r", encoding="utf-8", errors="replace") as f:
                        txt = f.read()
                        txt = re.sub('[<>]', '', txt)
                        return txt
                except OSError:
                    pass
        return ''

    def find_info(self, path):
        basename, _ext = os.path.splitext(path)
        files = listdir(os.path.dirname(path))
        for file in [f"{path}.info", f"{path}.civitai.info", f"{basename}.info", f"{basename}.civitai.info"]:
            if file in files:
                try:
                    with open(file, "r", encoding="utf-8", errors="replace") as f:
                        txt = f.read()
                        txt = re.sub('[<>]', '', txt)
                        return txt
                except OSError:
                    pass
        return ''

    def save_preview(self, index, images, filename):
        try:
            image = image_from_url_text(images[int(index)])
        except Exception as e:
            shared.log.error(f'Extra network save preview: {filename} {e}')
            return
        is_allowed = False
        for page in extra_pages:
            if any(path_is_parent(x, filename) for x in page.allowed_directories_for_previews()):
                is_allowed = True
                break
        if not is_allowed:
            shared.log.error(f'Extra network save preview: {filename} not allowed')
            return
        if image.width > 512 or image.height > 512:
            image = image.convert('RGB')
            image.thumbnail((512, 512), Image.HAMMING)
        image.save(filename, quality=50)
        fn, _ext = os.path.splitext(filename)
        thumb = fn + '.thumb.jpg'
        if os.path.exists(thumb):
            shared.log.debug(f'Extra network delete thumbnail: {thumb}')
            os.remove(thumb)
        shared.log.info(f'Extra network save preview: {filename}')

    def save_description(self, filename, desc):
        lastDotIndex = filename.rindex('.')
        filename = filename[0:lastDotIndex]+".txt"
        if desc != "":
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(desc)
                shared.log.info(f'Extra network save description: {filename} {desc}')
            except Exception as e:
                shared.log.error(f'Extra network save description: {filename} {e}')


def initialize():
    extra_pages.clear()


def register_pages():
    from modules.ui_extra_networks_textual_inversion import ExtraNetworksPageTextualInversion
    from modules.ui_extra_networks_hypernets import ExtraNetworksPageHypernetworks
    from modules.ui_extra_networks_checkpoints import ExtraNetworksPageCheckpoints
    from modules.ui_extra_networks_styles import ExtraNetworksPageStyles
    register_page(ExtraNetworksPageCheckpoints())
    register_page(ExtraNetworksPageStyles())
    register_page(ExtraNetworksPageTextualInversion())
    register_page(ExtraNetworksPageHypernetworks())


class ExtraNetworksUi:
    def __init__(self):
        self.pages = None
        self.button_save_preview = None
        self.preview_target_filename = None
        self.button_save_description = None
        self.button_read_description = None
        self.description_target_filename = None
        self.description = None
        self.tags = None
        self.tabname = None
        self.search = None


def create_ui(container, button, tabname, skip_indexing = False):
    ui = ExtraNetworksUi()
    ui.pages = []
    ui.tabname = tabname
    with gr.Tabs(elem_id=tabname+"_extra_tabs"):
        button_refresh = ToolButton(symbols.refresh, elem_id=tabname+"_extra_refresh")
        button_close = ToolButton(symbols.close, elem_id=tabname+"_extra_close")
        ui.search = gr.Textbox('', show_label=False, elem_id=tabname+"_extra_search", placeholder="Search...", elem_classes="textbox", lines=2)
        ui.description = gr.Textbox('', show_label=False, elem_id=tabname+"_description", placeholder="Save/Replace Extra Network Description...", elem_classes="textbox", lines=2)
        ui.preview_target_filename = gr.Textbox('Preview save filename', elem_id=tabname+"_preview_filename", visible=False)
        ui.button_save_preview = gr.Button('Save preview', elem_id=tabname+"_save_preview", visible=False)
        ui.description_target_filename = gr.Textbox('Description save filename', elem_id=tabname+"_description_filename", visible=False)
        ui.button_save_description = gr.Button('Save description', elem_id=tabname+"_save_description", visible=False)

        if ui.tabname == 'txt2img': # refresh only once
            global refresh_time # pylint: disable=global-statement
            refresh_time = time.time()
        for page in extra_pages:
            page.create_page(ui.tabname, skip_indexing)
            with gr.Tab(page.title, id=page.title.lower().replace(" ", "_"), elem_classes="extra-networks-tab"):
                page_elem = gr.HTML(page.html, elem_id=f'{tabname}{page.name}_extra_page', elem_classes="extra-networks-page")
                page_elem.change(fn=lambda: None, _js=f'() => refreshExtraNetworks("{tabname}")', inputs=[], outputs=[])
                ui.pages.append(page_elem)

    def toggle_visibility(is_visible):
        is_visible = not is_visible
        return is_visible, gr.update(visible=is_visible), gr.update(variant=("secondary-down" if is_visible else "secondary"))

    def en_refresh(title):
        res = []
        for page in extra_pages:
            if title is None or title == '' or title == page.title or len(page.html) == 0:
                page.refresh()
                page.refresh_time = None
                page.create_page(ui.tabname)
                shared.log.debug(f"Refreshing Extra networks: page='{page.title}' items={len(page.items)} tab={ui.tabname}")
            res.append(page.html)
        ui.search.update(value = ui.search.value)
        return res

    state_visible = gr.State(value=False) # pylint: disable=abstract-class-instantiated
    button.click(fn=toggle_visibility, inputs=[state_visible], outputs=[state_visible, container, button])
    button_close.click(fn=toggle_visibility, inputs=[state_visible], outputs=[state_visible, container])
    button_refresh.click(_js='getENActivePage', fn=en_refresh, inputs=[ui.search], outputs=ui.pages)
    return ui


def path_is_parent(parent_path, child_path):
    parent_path = os.path.abspath(parent_path)
    child_path = os.path.abspath(child_path)
    return child_path.startswith(parent_path)


def setup_ui(ui, gallery):

    def save_preview(pagename, index, images, filename):
        res = []
        for page in extra_pages:
            if pagename is None or pagename == '' or pagename == page.title or len(page.html) == 0:
                page.save_preview(index, images, filename)
                res.append(page.create_page(ui.tabname))
            else:
                res.append(page.html)
        return res


    ui.button_save_preview.click(
        fn=save_preview,
        _js="function(t, i, y, z) {return [getENActivePage(), selected_gallery_index(), y, z]}",
        inputs=[ui.search, ui.preview_target_filename, gallery, ui.preview_target_filename],
        outputs=ui.pages
    )

    def save_description(pagename, filename, desc):
        res = []
        for page in extra_pages:
            if pagename is None or pagename == '' or pagename == page.title or len(page.html) == 0:
                page.save_description(filename, desc)
                res.append(page.create_page(ui.tabname))
            else:
                res.append(page.html)
        return res

    ui.button_save_description.click(
        fn=save_description,
        _js="function(t, x, y) { return [getENActivePage(), x, y] }",
        inputs=[ui.search, ui.description_target_filename, ui.description],
        outputs=ui.pages
    )
