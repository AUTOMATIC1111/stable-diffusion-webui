import os
import re
import sys
import time
from collections import namedtuple
import gradio as gr
from modules import paths, script_callbacks, extensions, script_loading, scripts_postprocessing, errors
from installer import log


AlwaysVisible = object()
time_component = {}
time_setup = {}


class PostprocessImageArgs:
    def __init__(self, image):
        self.image = image


class Script:
    name = None
    filename = None
    args_from = None
    args_to = None
    alwayson = False
    is_txt2img = False
    is_img2img = False
    api_info = None
    group = None
    infotext_fields = None
    paste_field_names = None

    def title(self):
        """this function should return the title of the script. This is what will be displayed in the dropdown menu."""
        raise NotImplementedError()

    def ui(self, is_img2img):
        """this function should create gradio UI elements. See https://gradio.app/docs/#components
        The return value should be an array of all components that are used in processing.
        Values of those returned components will be passed to run() and process() functions.
        """
        pass # pylint: disable=unnecessary-pass

    def show(self, is_img2img): # pylint: disable=unused-argument
        """
        is_img2img is True if this function is called for the img2img interface, and Fasle otherwise
        This function should return:
         - False if the script should not be shown in UI at all
         - True if the script should be shown in UI if it's selected in the scripts dropdown
         - script.AlwaysVisible if the script should be shown in UI at all times
         """
        return True

    def run(self, p, *args):
        """
        This function is called if the script has been selected in the script dropdown.
        It must do all processing and return the Processed object with results, same as
        one returned by processing.process_images.
        Usually the processing is done by calling the processing.process_images function.
        args contains all values returned by components from ui()
        """
        pass # pylint: disable=unnecessary-pass

    def process(self, p, *args):
        """
        This function is called before processing begins for AlwaysVisible scripts.
        You can modify the processing object (p) here, inject hooks, etc.
        args contains all values returned by components from ui()
        """
        pass # pylint: disable=unnecessary-pass

    def before_process_batch(self, p, *args, **kwargs):
        """
        Called before extra networks are parsed from the prompt, so you can add
        new extra network keywords to the prompt with this callback.
        **kwargs will have those items:
          - batch_number - index of current batch, from 0 to number of batches-1
          - prompts - list of prompts for current batch; you can change contents of this list but changing the number of entries will likely break things
          - seeds - list of seeds for current batch
          - subseeds - list of subseeds for current batch
        """
        pass # pylint: disable=unnecessary-pass

    def process_batch(self, p, *args, **kwargs):
        """
        Same as process(), but called for every batch.
        **kwargs will have those items:
          - batch_number - index of current batch, from 0 to number of batches-1
          - prompts - list of prompts for current batch; you can change contents of this list but changing the number of entries will likely break things
          - seeds - list of seeds for current batch
          - subseeds - list of subseeds for current batch
        """
        pass # pylint: disable=unnecessary-pass

    def postprocess_batch(self, p, *args, **kwargs):
        """
        Same as process_batch(), but called for every batch after it has been generated.
        **kwargs will have same items as process_batch, and also:
          - batch_number - index of current batch, from 0 to number of batches-1
          - images - torch tensor with all generated images, with values ranging from 0 to 1;
        """
        pass # pylint: disable=unnecessary-pass

    def postprocess_image(self, p, pp: PostprocessImageArgs, *args):
        """
        Called for every image after it has been generated.
        """
        pass # pylint: disable=unnecessary-pass

    def postprocess(self, p, processed, *args):
        """
        This function is called after processing ends for AlwaysVisible scripts.
        args contains all values returned by components from ui()
        """
        pass # pylint: disable=unnecessary-pass

    def before_component(self, component, **kwargs):
        """
        Called before a component is created.
        Use elem_id/label fields of kwargs to figure out which component it is.
        This can be useful to inject your own components somewhere in the middle of vanilla UI.
        You can return created components in the ui() function to add them to the list of arguments for your processing functions
        """
        pass # pylint: disable=unnecessary-pass

    def after_component(self, component, **kwargs):
        """
        Called after a component is created. Same as above.
        """
        pass # pylint: disable=unnecessary-pass

    def describe(self):
        """unused"""
        return ""

    def elem_id(self, item_id):
        """helper function to generate id for a HTML element, constructs final id out of script name, tab and user-supplied item_id"""
        need_tabname = self.show(True) == self.show(False)
        tabkind = 'img2img' if self.is_img2img else 'txt2txt'
        tabname = f"{tabkind}_" if need_tabname else ""
        title = re.sub(r'[^a-z_0-9]', '', re.sub(r'\s', '_', self.title().lower()))
        return f'script_{tabname}{title}_{item_id}'


current_basedir = paths.script_path


def basedir():
    """returns the base directory for the current script. For scripts in the main scripts directory,
    this is the main directory (where webui.py resides), and for scripts in extensions directory
    (ie extensions/aesthetic/script/aesthetic.py), this is extension's directory (extensions/aesthetic)
    """
    return current_basedir


ScriptFile = namedtuple("ScriptFile", ["basedir", "filename", "path", "priority"])
scripts_data = []
postprocessing_scripts_data = []
ScriptClassData = namedtuple("ScriptClassData", ["script_class", "path", "basedir", "module"])


def list_scripts(scriptdirname, extension):
    tmp_list = []
    base = os.path.join(paths.script_path, scriptdirname)
    if os.path.exists(base):
        for filename in sorted(os.listdir(base)):
            tmp_list.append(ScriptFile(paths.script_path, filename, os.path.join(base, filename), '50'))
    for ext in extensions.active():
        tmp_list += ext.list_files(scriptdirname, extension)
    priority_list = []
    for script in tmp_list:
        if os.path.splitext(script.path)[1].lower() == extension and os.path.isfile(script.path):
            if script.basedir == paths.script_path:
                priority = '0'
            elif script.basedir.startswith(os.path.join(paths.script_path, 'scripts')):
                priority = '1'
            elif script.basedir.startswith(os.path.join(paths.script_path, 'extensions-builtin')):
                priority = '2'
            elif script.basedir.startswith(os.path.join(paths.script_path, 'extensions')):
                priority = '3'
            else:
                priority = '9'
            if os.path.isfile(os.path.join(base, "..", ".priority")):
                with open(os.path.join(base, "..", ".priority"), "r", encoding="utf-8") as f:
                    priority = priority + str(f.read().strip())
            else:
                priority = priority + script.priority
            priority_list.append(ScriptFile(script.basedir, script.filename, script.path, priority))
            # log.debug(f'Adding script: {script.basedir} {script.filename} {script.path} {priority}')
    priority_sort = sorted(priority_list, key=lambda item: item.priority + item.path.lower(), reverse=False)
    return priority_sort


def list_files_with_name(filename):
    res = []
    dirs = [paths.script_path] + [ext.path for ext in extensions.active()]
    for dirpath in dirs:
        if not os.path.isdir(dirpath):
            continue
        path = os.path.join(dirpath, filename)
        if os.path.isfile(path):
            res.append(path)
    return res


def load_scripts():
    global current_basedir # pylint: disable=global-statement
    scripts_data.clear()
    postprocessing_scripts_data.clear()
    script_callbacks.clear_callbacks()
    scripts_list = list_scripts("scripts", ".py")
    syspath = sys.path
    time_load = {}

    def register_scripts_from_module(module, scriptfile):
        for script_class in module.__dict__.values():
            if type(script_class) != type:
                continue
            # log.debug(f'Registering script: {scriptfile.path}')
            if issubclass(script_class, Script):
                scripts_data.append(ScriptClassData(script_class, scriptfile.path, scriptfile.basedir, module))
            elif issubclass(script_class, scripts_postprocessing.ScriptPostprocessing):
                postprocessing_scripts_data.append(ScriptClassData(script_class, scriptfile.path, scriptfile.basedir, module))

    for scriptfile in scripts_list:
        t0 = time.time()
        try:
            if scriptfile.basedir != paths.script_path:
                sys.path = [scriptfile.basedir] + sys.path
            current_basedir = scriptfile.basedir
            script_module = script_loading.load_module(scriptfile.path)
            register_scripts_from_module(script_module, scriptfile)
        except Exception as e:
            errors.display(e, f'Loading script: {scriptfile.filename}')
        finally:
            current_basedir = paths.script_path
            time_load[scriptfile.basedir] = time_load.get(scriptfile.basedir, 0) + (time.time()-t0)
            sys.path = syspath

    global scripts_txt2img, scripts_img2img, scripts_postproc # pylint: disable=global-statement
    scripts_txt2img = ScriptRunner()
    scripts_img2img = ScriptRunner()
    scripts_postproc = scripts_postprocessing.ScriptPostprocessingRunner()

    time_summary = [f'{os.path.basename(k)}:{round(v,3)}s' for (k,v) in time_load.items() if v > 0.05]
    log.debug(f'Scripts load: {time_summary}')


def wrap_call(func, filename, funcname, *args, default=None, **kwargs):
    try:
        res = func(*args, **kwargs)
        return res
    except Exception as e:
        errors.display(e, f'Calling script: {filename}/{funcname}')
    return default


class ScriptRunner:
    def __init__(self):
        self.scripts = []
        self.selectable_scripts = []
        self.alwayson_scripts = []
        self.titles = []
        self.infotext_fields = []
        self.paste_field_names = []
        self.script_load_ctr = 0
        self.is_img2img = False

    def initialize_scripts(self, is_img2img):
        from modules import scripts_auto_postprocessing

        self.scripts.clear()
        self.selectable_scripts.clear()
        self.alwayson_scripts.clear()
        self.titles.clear()
        self.infotext_fields.clear()
        self.paste_field_names.clear()
        self.script_load_ctr = 0
        self.is_img2img = is_img2img

        self.scripts.clear()
        self.alwayson_scripts.clear()
        self.selectable_scripts.clear()
        auto_processing_scripts = scripts_auto_postprocessing.create_auto_preprocessing_script_data()

        for script_class, path, _basedir, _script_module in auto_processing_scripts + scripts_data:
            try:
                script = script_class()
                script.filename = path
                script.is_txt2img = not is_img2img
                script.is_img2img = is_img2img
                visibility = script.show(script.is_img2img)
                if visibility == AlwaysVisible:
                    self.scripts.append(script)
                    self.alwayson_scripts.append(script)
                    script.alwayson = True
                elif visibility:
                    self.scripts.append(script)
                    self.selectable_scripts.append(script)
            except Exception as e:
                log.error(f'Script initialize: {path} {e}')

    def setup_ui(self):
        import modules.api.models as api_models
        self.titles = [wrap_call(script.title, script.filename, "title") or f"{script.filename} [error]" for script in self.selectable_scripts]
        inputs = [None]
        inputs_alwayson = [True]

        def create_script_ui(script, inputs, inputs_alwayson):
            script.args_from = len(inputs)
            script.args_to = len(inputs)
            controls = wrap_call(script.ui, script.filename, "ui", script.is_img2img)
            if controls is None:
                return
            script.name = wrap_call(script.title, script.filename, "title", default=script.filename).lower()
            api_args = []
            for control in controls:
                control.custom_script_source = os.path.basename(script.filename)
                arg_info = api_models.ScriptArg(label=control.label or "")
                for field in ("value", "minimum", "maximum", "step", "choices"):
                    v = getattr(control, field, None)
                    if v is not None:
                        setattr(arg_info, field, v)
                api_args.append(arg_info)

            script.api_info = api_models.ScriptInfo(
                name=script.name,
                is_img2img=script.is_img2img,
                is_alwayson=script.alwayson,
                args=api_args,
            )
            if script.infotext_fields is not None:
                self.infotext_fields += script.infotext_fields
            if script.paste_field_names is not None:
                self.paste_field_names += script.paste_field_names
            inputs += controls
            inputs_alwayson += [script.alwayson for _ in controls]
            script.args_to = len(inputs)

        with gr.Group(elem_id='scripts_alwayson_img2img' if self.is_img2img else 'scripts_alwayson_txt2img'):
            for script in self.alwayson_scripts:
                t0 = time.time()
                elem_id = f'script_{"txt2img" if script.is_txt2img else "img2img"}_{script.title().lower().replace(" ", "_")}'
                with gr.Group(elem_id=elem_id) as group:
                    create_script_ui(script, inputs, inputs_alwayson)
                script.group = group
                time_setup[script.title()] = time_setup.get(script.title(), 0) + (time.time()-t0)

        dropdown = gr.Dropdown(label="Script", elem_id="script_list", choices=["None"] + self.titles, value="None", type="index")
        inputs[0] = dropdown
        for script in self.selectable_scripts:
            with gr.Group(visible=False) as group:
                t0 = time.time()
                create_script_ui(script, inputs, inputs_alwayson)
                time_setup[script.title()] = time_setup.get(script.title(), 0) + (time.time()-t0)
            script.group = group

        def select_script(script_index):
            selected_script = self.selectable_scripts[script_index - 1] if script_index > 0 else None
            return [gr.update(visible=selected_script == s) for s in self.selectable_scripts]

        def init_field(title):
            """called when an initial value is set from ui-config.json to show script's UI components"""
            if title == 'None':
                return
            script_index = self.titles.index(title)
            self.selectable_scripts[script_index].group.visible = True

        dropdown.init_field = init_field
        dropdown.change(fn=select_script, inputs=[dropdown], outputs=[script.group for script in self.selectable_scripts])

        def onload_script_visibility(params):
            title = params.get('Script', None)
            if title:
                title_index = self.titles.index(title)
                visibility = title_index == self.script_load_ctr
                self.script_load_ctr = (self.script_load_ctr + 1) % len(self.titles)
                return gr.update(visible=visibility)
            else:
                return gr.update(visible=False)

        self.infotext_fields.append( (dropdown, lambda x: gr.update(value=x.get('Script', 'None'))) )
        self.infotext_fields.extend( [(script.group, onload_script_visibility) for script in self.selectable_scripts] )
        return inputs

    def run(self, p, *args):
        script_index = args[0]
        if script_index == 0:
            return None
        script = self.selectable_scripts[script_index-1]
        if script is None:
            return None
        parsed = p.per_script_args.get(script.title(), args[script.args_from:script.args_to])
        t0 = time.time()
        processed = script.run(p, *parsed)
        log.debug(f'Script run: {script.title()}:{round(time.time()-t0, 2)}s')
        return processed

    def process(self, p, **kwargs):
        s = []
        for script in self.alwayson_scripts:
            try:
                t0 = time.time()
                args = p.per_script_args.get(script.title(), p.script_args[script.args_from:script.args_to])
                script.process(p, *args, **kwargs)
                s.append(f'{script.title()}:{round(time.time()-t0, 2)}s')
            except Exception as e:
                errors.display(e, f'Running script process: {script.filename}')
        log.debug(f'Script process: {s}')

    def before_process_batch(self, p, **kwargs):
        s = []
        for script in self.alwayson_scripts:
            try:
                t0 = time.time()
                args = p.per_script_args.get(script.title(), p.script_args[script.args_from:script.args_to])
                script.before_process_batch(p, *args, **kwargs)
                s.append(f'{script.title()}:{round(time.time()-t0, 2)}s')
            except Exception as e:
                errors.display(e, f'Running script before process batch: {script.filename}')
        log.debug(f'Script before-process-batch: {s}')

    def process_batch(self, p, **kwargs):
        s = []
        for script in self.alwayson_scripts:
            try:
                t0 = time.time()
                args = p.per_script_args.get(script.title(), p.script_args[script.args_from:script.args_to])
                script.process_batch(p, *args, **kwargs)
                s.append(f'{script.title()}:{round(time.time()-t0, 2)}s')
            except Exception as e:
                errors.display(e, f'Running script process batch: {script.filename}')
        log.debug(f'Script process-batch: {s}')

    def postprocess(self, p, processed):
        s = []
        for script in self.alwayson_scripts:
            try:
                t0 = time.time()
                args = p.per_script_args.get(script.title(), p.script_args[script.args_from:script.args_to])
                script.postprocess(p, processed, *args)
                s.append(f'{script.title()}:{round(time.time()-t0, 2)}s')
            except Exception as e:
                errors.display(e, f'Running script postprocess: {script.filename}')
        log.debug(f'Script postprocess: {s}')

    def postprocess_batch(self, p, images, **kwargs):
        s = []
        for script in self.alwayson_scripts:
            try:
                t0 = time.time()
                args = p.per_script_args.get(script.title(), p.script_args[script.args_from:script.args_to])
                script.postprocess_batch(p, *args, images=images, **kwargs)
                s.append(f'{script.title()}:{round(time.time()-t0, 2)}s')
            except Exception as e:
                errors.display(e, f'Running script before postprocess batch: {script.filename}')
        log.debug(f'Script postprocess-batch: {s}')

    def postprocess_image(self, p, pp: PostprocessImageArgs):
        s = []
        for script in self.alwayson_scripts:
            try:
                t0 = time.time()
                args = p.per_script_args.get(script.title(), p.script_args[script.args_from:script.args_to])
                script.postprocess_image(p, pp, *args)
                s.append(f'{script.title()}:{round(time.time()-t0, 2)}s')
            except Exception as e:
                errors.display(e, f'Running script postprocess image: {script.filename}')
        log.debug(f'Script postprocess-image: {s}')

    def before_component(self, component, **kwargs):
        for script in self.scripts:
            try:
                t0 = time.time()
                script.before_component(component, **kwargs)
                time_component[script.title()] = time_component.get(script.title(), 0) + (time.time()-t0)
            except Exception as e:
                errors.display(e, f'Running script before component: {script.filename}')

    def after_component(self, component, **kwargs):
        for script in self.scripts:
            try:
                t0 = time.time()
                script.after_component(component, **kwargs)
                time_component[script.title()] = time_component.get(script.title(), 0) + (time.time()-t0)
            except Exception as e:
                errors.display(e, f'Running script after component: {script.filename}')

    def reload_sources(self, cache):
        s = []
        for si, script in list(enumerate(self.scripts)):
            t0 = time.time()
            args_from = script.args_from
            args_to = script.args_to
            filename = script.filename
            module = cache.get(filename, None)
            if module is None:
                module = script_loading.load_module(script.filename)
                cache[filename] = module
            for script_class in module.__dict__.values():
                if type(script_class) == type and issubclass(script_class, Script):
                    self.scripts[si] = script_class()
                    self.scripts[si].filename = filename
                    self.scripts[si].args_from = args_from
                    self.scripts[si].args_to = args_to
            s.append(f'{script.title()}:{round(time.time()-t0, 2)}s')
        log.debug(f'Script reload-sources: {s}')


scripts_txt2img: ScriptRunner = None
scripts_img2img: ScriptRunner = None
scripts_current: ScriptRunner = None
scripts_postproc: scripts_postprocessing.ScriptPostprocessingRunner = None
reload_scripts = load_scripts  # compatibility alias


def reload_script_body_only():
    cache = {}
    scripts_txt2img.reload_sources(cache)
    scripts_img2img.reload_sources(cache)


def add_classes_to_gradio_component(comp):
    """
    this adds gradio-* to the component for css styling (ie gradio-button to gr.Button), as well as some others
    """
    elem_classes = []
    if hasattr(comp, "elem_classes"):
        elem_classes = comp.elem_classes
    if elem_classes is None:
        elem_classes = []
    comp.elem_classes = [f"gradio-{comp.get_block_name()}", *(comp.elem_classes or [])]
    if getattr(comp, 'multiselect', False):
        comp.elem_classes.append('multiselect')


def IOComponent_init(self, *args, **kwargs):
    if scripts_current is not None:
        scripts_current.before_component(self, **kwargs)
    script_callbacks.before_component_callback(self, **kwargs)
    res = original_IOComponent_init(self, *args, **kwargs) # pylint: disable=assignment-from-no-return
    add_classes_to_gradio_component(self)
    script_callbacks.after_component_callback(self, **kwargs)
    if scripts_current is not None:
        scripts_current.after_component(self, **kwargs)
    return res


original_IOComponent_init = gr.components.IOComponent.__init__
gr.components.IOComponent.__init__ = IOComponent_init


def BlockContext_init(self, *args, **kwargs):
    res = original_BlockContext_init(self, *args, **kwargs) # pylint: disable=assignment-from-no-return
    add_classes_to_gradio_component(self)
    return res


original_BlockContext_init = gr.blocks.BlockContext.__init__
gr.blocks.BlockContext.__init__ = BlockContext_init
