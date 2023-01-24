import os
import re
import sys
import traceback
from collections import namedtuple

import gradio as gr

from modules.processing import StableDiffusionProcessing
from modules import shared, paths, script_callbacks, extensions, script_loading, scripts_postprocessing

AlwaysVisible = object()


class Script:
    filename = None
    args_from = None
    args_to = None
    alwayson = False

    is_txt2img = False
    is_img2img = False

    """A gr.Group component that has all script's UI inside it"""
    group = None

    infotext_fields = None
    """if set in ui(), this is a list of pairs of gradio component + text; the text will be used when
    parsing infotext to set the value for the component; see ui.py's txt2img_paste_fields for an example
    """

    def title(self):
        """this function should return the title of the script. This is what will be displayed in the dropdown menu."""

        raise NotImplementedError()

    def ui(self, is_img2img):
        """this function should create gradio UI elements. See https://gradio.app/docs/#components
        The return value should be an array of all components that are used in processing.
        Values of those returned components will be passed to run() and process() functions.
        """

        pass

    def show(self, is_img2img):
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

        raise NotImplementedError()

    def process(self, p, *args):
        """
        This function is called before processing begins for AlwaysVisible scripts.
        You can modify the processing object (p) here, inject hooks, etc.
        args contains all values returned by components from ui()
        """

        pass

    def process_batch(self, p, *args, **kwargs):
        """
        Same as process(), but called for every batch.

        **kwargs will have those items:
          - batch_number - index of current batch, from 0 to number of batches-1
          - prompts - list of prompts for current batch; you can change contents of this list but changing the number of entries will likely break things
          - seeds - list of seeds for current batch
          - subseeds - list of subseeds for current batch
        """

        pass

    def postprocess_batch(self, p, *args, **kwargs):
        """
        Same as process_batch(), but called for every batch after it has been generated.

        **kwargs will have same items as process_batch, and also:
          - batch_number - index of current batch, from 0 to number of batches-1
          - images - torch tensor with all generated images, with values ranging from 0 to 1;
        """

        pass

    def postprocess(self, p, processed, *args):
        """
        This function is called after processing ends for AlwaysVisible scripts.
        args contains all values returned by components from ui()
        """

        pass

    def before_component(self, component, **kwargs):
        """
        Called before a component is created.
        Use elem_id/label fields of kwargs to figure out which component it is.
        This can be useful to inject your own components somewhere in the middle of vanilla UI.
        You can return created components in the ui() function to add them to the list of arguments for your processing functions
        """

        pass

    def after_component(self, component, **kwargs):
        """
        Called after a component is created. Same as above.
        """

        pass

    def describe(self):
        """unused"""
        return ""

    def elem_id(self, item_id):
        """helper function to generate id for a HTML element, constructs final id out of script name, tab and user-supplied item_id"""

        need_tabname = self.show(True) == self.show(False)
        tabname = ('img2img' if self.is_img2img else 'txt2txt') + "_" if need_tabname else ""
        title = re.sub(r'[^a-z_0-9]', '', re.sub(r'\s', '_', self.title().lower()))

        return f'script_{tabname}{title}_{item_id}'


current_basedir = paths.script_path


def basedir():
    """returns the base directory for the current script. For scripts in the main scripts directory,
    this is the main directory (where webui.py resides), and for scripts in extensions directory
    (ie extensions/aesthetic/script/aesthetic.py), this is extension's directory (extensions/aesthetic)
    """
    return current_basedir


ScriptFile = namedtuple("ScriptFile", ["basedir", "filename", "path"])

scripts_data = []
postprocessing_scripts_data = []
ScriptClassData = namedtuple("ScriptClassData", ["script_class", "path", "basedir", "module"])


def list_scripts(scriptdirname, extension):
    scripts_list = []

    basedir = os.path.join(paths.script_path, scriptdirname)
    if os.path.exists(basedir):
        for filename in sorted(os.listdir(basedir)):
            scripts_list.append(ScriptFile(paths.script_path, filename, os.path.join(basedir, filename)))

    for ext in extensions.active():
        scripts_list += ext.list_files(scriptdirname, extension)

    scripts_list = [x for x in scripts_list if os.path.splitext(x.path)[1].lower() == extension and os.path.isfile(x.path)]

    return scripts_list


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
    global current_basedir
    scripts_data.clear()
    postprocessing_scripts_data.clear()
    script_callbacks.clear_callbacks()

    scripts_list = list_scripts("scripts", ".py")

    syspath = sys.path

    def register_scripts_from_module(module):
        for key, script_class in module.__dict__.items():
            if type(script_class) != type:
                continue

            if issubclass(script_class, Script):
                scripts_data.append(ScriptClassData(script_class, scriptfile.path, scriptfile.basedir, module))
            elif issubclass(script_class, scripts_postprocessing.ScriptPostprocessing):
                postprocessing_scripts_data.append(ScriptClassData(script_class, scriptfile.path, scriptfile.basedir, module))

    for scriptfile in sorted(scripts_list):
        try:
            if scriptfile.basedir != paths.script_path:
                sys.path = [scriptfile.basedir] + sys.path
            current_basedir = scriptfile.basedir

            script_module = script_loading.load_module(scriptfile.path)
            register_scripts_from_module(script_module)

        except Exception:
            print(f"Error loading script: {scriptfile.filename}", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)

        finally:
            sys.path = syspath
            current_basedir = paths.script_path


def wrap_call(func, filename, funcname, *args, default=None, **kwargs):
    try:
        res = func(*args, **kwargs)
        return res
    except Exception:
        print(f"Error calling: {filename}/{funcname}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)

    return default


class ScriptRunner:
    def __init__(self):
        self.scripts = []
        self.selectable_scripts = []
        self.alwayson_scripts = []
        self.titles = []
        self.infotext_fields = []

    def initialize_scripts(self, is_img2img):
        self.scripts.clear()
        self.alwayson_scripts.clear()
        self.selectable_scripts.clear()

        for script_class, path, basedir, script_module in scripts_data:
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

    def setup_ui(self):
        self.titles = [wrap_call(script.title, script.filename, "title") or f"{script.filename} [error]" for script in self.selectable_scripts]

        inputs = [None]
        inputs_alwayson = [True]

        def create_script_ui(script, inputs, inputs_alwayson):
            script.args_from = len(inputs)
            script.args_to = len(inputs)

            controls = wrap_call(script.ui, script.filename, "ui", script.is_img2img)

            if controls is None:
                return

            for control in controls:
                control.custom_script_source = os.path.basename(script.filename)

            if script.infotext_fields is not None:
                self.infotext_fields += script.infotext_fields

            inputs += controls
            inputs_alwayson += [script.alwayson for _ in controls]
            script.args_to = len(inputs)

        for script in self.alwayson_scripts:
            with gr.Group() as group:
                create_script_ui(script, inputs, inputs_alwayson)

            script.group = group

        dropdown = gr.Dropdown(label="Script", elem_id="script_list", choices=["None"] + self.titles, value="None", type="index")
        inputs[0] = dropdown

        for script in self.selectable_scripts:
            with gr.Group(visible=False) as group:
                create_script_ui(script, inputs, inputs_alwayson)

            script.group = group

        def select_script(script_index):
            selected_script = self.selectable_scripts[script_index - 1] if script_index>0 else None

            return [gr.update(visible=selected_script == s) for s in self.selectable_scripts]

        def init_field(title):
            """called when an initial value is set from ui-config.json to show script's UI components"""

            if title == 'None':
                return

            script_index = self.titles.index(title)
            self.selectable_scripts[script_index].group.visible = True

        dropdown.init_field = init_field

        dropdown.change(
            fn=select_script,
            inputs=[dropdown],
            outputs=[script.group for script in self.selectable_scripts]
        )

        return inputs

    def run(self, p: StableDiffusionProcessing, *args):
        script_index = args[0]

        if script_index == 0:
            return None

        script = self.selectable_scripts[script_index-1]

        if script is None:
            return None

        script_args = args[script.args_from:script.args_to]
        processed = script.run(p, *script_args)

        shared.total_tqdm.clear()

        return processed

    def process(self, p):
        for script in self.alwayson_scripts:
            try:
                script_args = p.script_args[script.args_from:script.args_to]
                script.process(p, *script_args)
            except Exception:
                print(f"Error running process: {script.filename}", file=sys.stderr)
                print(traceback.format_exc(), file=sys.stderr)

    def process_batch(self, p, **kwargs):
        for script in self.alwayson_scripts:
            try:
                script_args = p.script_args[script.args_from:script.args_to]
                script.process_batch(p, *script_args, **kwargs)
            except Exception:
                print(f"Error running process_batch: {script.filename}", file=sys.stderr)
                print(traceback.format_exc(), file=sys.stderr)

    def postprocess(self, p, processed):
        for script in self.alwayson_scripts:
            try:
                script_args = p.script_args[script.args_from:script.args_to]
                script.postprocess(p, processed, *script_args)
            except Exception:
                print(f"Error running postprocess: {script.filename}", file=sys.stderr)
                print(traceback.format_exc(), file=sys.stderr)

    def postprocess_batch(self, p, images, **kwargs):
        for script in self.alwayson_scripts:
            try:
                script_args = p.script_args[script.args_from:script.args_to]
                script.postprocess_batch(p, *script_args, images=images, **kwargs)
            except Exception:
                print(f"Error running postprocess_batch: {script.filename}", file=sys.stderr)
                print(traceback.format_exc(), file=sys.stderr)

    def before_component(self, component, **kwargs):
        for script in self.scripts:
            try:
                script.before_component(component, **kwargs)
            except Exception:
                print(f"Error running before_component: {script.filename}", file=sys.stderr)
                print(traceback.format_exc(), file=sys.stderr)

    def after_component(self, component, **kwargs):
        for script in self.scripts:
            try:
                script.after_component(component, **kwargs)
            except Exception:
                print(f"Error running after_component: {script.filename}", file=sys.stderr)
                print(traceback.format_exc(), file=sys.stderr)

    def reload_sources(self, cache):
        for si, script in list(enumerate(self.scripts)):
            args_from = script.args_from
            args_to = script.args_to
            filename = script.filename

            module = cache.get(filename, None)
            if module is None:
                module = script_loading.load_module(script.filename)
                cache[filename] = module

            for key, script_class in module.__dict__.items():
                if type(script_class) == type and issubclass(script_class, Script):
                    self.scripts[si] = script_class()
                    self.scripts[si].filename = filename
                    self.scripts[si].args_from = args_from
                    self.scripts[si].args_to = args_to


scripts_txt2img = ScriptRunner()
scripts_img2img = ScriptRunner()
scripts_postproc = scripts_postprocessing.ScriptPostprocessingRunner()
scripts_current: ScriptRunner = None


def reload_script_body_only():
    cache = {}
    scripts_txt2img.reload_sources(cache)
    scripts_img2img.reload_sources(cache)


def reload_scripts():
    global scripts_txt2img, scripts_img2img, scripts_postproc

    load_scripts()

    scripts_txt2img = ScriptRunner()
    scripts_img2img = ScriptRunner()
    scripts_postproc = scripts_postprocessing.ScriptPostprocessingRunner()


def IOComponent_init(self, *args, **kwargs):
    if scripts_current is not None:
        scripts_current.before_component(self, **kwargs)

    script_callbacks.before_component_callback(self, **kwargs)

    res = original_IOComponent_init(self, *args, **kwargs)

    script_callbacks.after_component_callback(self, **kwargs)

    if scripts_current is not None:
        scripts_current.after_component(self, **kwargs)

    return res


original_IOComponent_init = gr.components.IOComponent.__init__
gr.components.IOComponent.__init__ = IOComponent_init
