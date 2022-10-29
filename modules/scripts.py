import os
import sys
import traceback
from collections import namedtuple

import modules.ui as ui
import gradio as gr

from modules.processing import StableDiffusionProcessing
from modules import shared, paths, script_callbacks

AlwaysVisible = object()


class Script:
    filename = None
    args_from = None
    args_to = None
    alwayson = False

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
        Values of those returned componenbts will be passed to run() and process() functions.
        """

        pass

    def show(self, is_img2img):
        """
        is_img2img is True if this function is called for the img2img interface, and Fasle otherwise

        This function should return:
         - False if the script should not be shown in UI at all
         - True if the script should be shown in UI if it's scelected in the scripts drowpdown
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
        scripts. You can modify the processing object (p) here, inject hooks, etc.
        """

        pass

    def describe(self):
        """unused"""
        return ""


current_basedir = paths.script_path


def basedir():
    """returns the base directory for the current script. For scripts in the main scripts directory,
    this is the main directory (where webui.py resides), and for scripts in extensions directory
    (ie extensions/aesthetic/script/aesthetic.py), this is extension's directory (extensions/aesthetic)
    """
    return current_basedir


scripts_data = []
ScriptFile = namedtuple("ScriptFile", ["basedir", "filename", "path"])
ScriptClassData = namedtuple("ScriptClassData", ["script_class", "path", "basedir"])


def list_scripts(scriptdirname, extension):
    scripts_list = []

    basedir = os.path.join(paths.script_path, scriptdirname)
    if os.path.exists(basedir):
        for filename in sorted(os.listdir(basedir)):
            scripts_list.append(ScriptFile(paths.script_path, filename, os.path.join(basedir, filename)))

    extdir = os.path.join(paths.script_path, "extensions")
    if os.path.exists(extdir):
        for dirname in sorted(os.listdir(extdir)):
            dirpath = os.path.join(extdir, dirname)
            scriptdirpath = os.path.join(dirpath, scriptdirname)

            if not os.path.isdir(scriptdirpath):
                continue

            for filename in sorted(os.listdir(scriptdirpath)):
                scripts_list.append(ScriptFile(dirpath, filename, os.path.join(scriptdirpath, filename)))

    scripts_list = [x for x in scripts_list if os.path.splitext(x.path)[1].lower() == extension and os.path.isfile(x.path)]

    return scripts_list


def list_files_with_name(filename):
    res = []

    dirs = [paths.script_path]

    extdir = os.path.join(paths.script_path, "extensions")
    if os.path.exists(extdir):
        dirs += [os.path.join(extdir, d) for d in sorted(os.listdir(extdir))]

    for dirpath in dirs:
        if not os.path.isdir(dirpath):
            continue

        path = os.path.join(dirpath, filename)
        if os.path.isfile(filename):
            res.append(path)

    return res


def load_scripts():
    global current_basedir
    scripts_data.clear()
    script_callbacks.clear_callbacks()

    scripts_list = list_scripts("scripts", ".py")

    syspath = sys.path

    for scriptfile in sorted(scripts_list):
        try:
            if scriptfile.basedir != paths.script_path:
                sys.path = [scriptfile.basedir] + sys.path
            current_basedir = scriptfile.basedir

            with open(scriptfile.path, "r", encoding="utf8") as file:
                text = file.read()

            from types import ModuleType
            compiled = compile(text, scriptfile.path, 'exec')
            module = ModuleType(scriptfile.filename)
            exec(compiled, module.__dict__)

            for key, script_class in module.__dict__.items():
                if type(script_class) == type and issubclass(script_class, Script):
                    scripts_data.append(ScriptClassData(script_class, scriptfile.path, scriptfile.basedir))

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

    def setup_ui(self, is_img2img):
        for script_class, path, basedir in scripts_data:
            script = script_class()
            script.filename = path

            visibility = script.show(is_img2img)

            if visibility == AlwaysVisible:
                self.scripts.append(script)
                self.alwayson_scripts.append(script)
                script.alwayson = True

            elif visibility:
                self.scripts.append(script)
                self.selectable_scripts.append(script)

        self.titles = [wrap_call(script.title, script.filename, "title") or f"{script.filename} [error]" for script in self.selectable_scripts]

        inputs = [None]
        inputs_alwayson = [True]

        def create_script_ui(script, inputs, inputs_alwayson):
            script.args_from = len(inputs)
            script.args_to = len(inputs)

            controls = wrap_call(script.ui, script.filename, "ui", is_img2img)

            if controls is None:
                return

            for control in controls:
                control.custom_script_source = os.path.basename(script.filename)
                if not script.alwayson:
                    control.visible = False

            if script.infotext_fields is not None:
                self.infotext_fields += script.infotext_fields

            inputs += controls
            inputs_alwayson += [script.alwayson for _ in controls]
            script.args_to = len(inputs)

        for script in self.alwayson_scripts:
            with gr.Group():
                create_script_ui(script, inputs, inputs_alwayson)

        dropdown = gr.Dropdown(label="Script", elem_id="script_list", choices=["None"] + self.titles, value="None", type="index")
        dropdown.save_to_config = True
        inputs[0] = dropdown

        for script in self.selectable_scripts:
            create_script_ui(script, inputs, inputs_alwayson)

        def select_script(script_index):
            if 0 < script_index <= len(self.selectable_scripts):
                script = self.selectable_scripts[script_index-1]
                args_from = script.args_from
                args_to = script.args_to
            else:
                args_from = 0
                args_to = 0

            return [ui.gr_show(True if i == 0 else args_from <= i < args_to or is_alwayson) for i, is_alwayson in enumerate(inputs_alwayson)]

        def init_field(title):
            if title == 'None':
                return
            script_index = self.titles.index(title)
            script = self.selectable_scripts[script_index]
            for i in range(script.args_from, script.args_to):
                inputs[i].visible = True

        dropdown.init_field = init_field
        dropdown.change(
            fn=select_script,
            inputs=[dropdown],
            outputs=inputs
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

    def run_alwayson_scripts(self, p):
        for script in self.alwayson_scripts:
            try:
                script_args = p.script_args[script.args_from:script.args_to]
                script.process(p, *script_args)
            except Exception:
                print(f"Error running alwayson script: {script.filename}", file=sys.stderr)
                print(traceback.format_exc(), file=sys.stderr)

    def reload_sources(self, cache):
        for si, script in list(enumerate(self.scripts)):
            with open(script.filename, "r", encoding="utf8") as file:
                args_from = script.args_from
                args_to = script.args_to
                filename = script.filename
                text = file.read()

                from types import ModuleType

                module = cache.get(filename, None)
                if module is None:
                    compiled = compile(text, filename, 'exec')
                    module = ModuleType(script.filename)
                    exec(compiled, module.__dict__)
                    cache[filename] = module

                for key, script_class in module.__dict__.items():
                    if type(script_class) == type and issubclass(script_class, Script):
                        self.scripts[si] = script_class()
                        self.scripts[si].filename = filename
                        self.scripts[si].args_from = args_from
                        self.scripts[si].args_to = args_to


scripts_txt2img = ScriptRunner()
scripts_img2img = ScriptRunner()


def reload_script_body_only():
    cache = {}
    scripts_txt2img.reload_sources(cache)
    scripts_img2img.reload_sources(cache)


def reload_scripts():
    global scripts_txt2img, scripts_img2img

    load_scripts()

    scripts_txt2img = ScriptRunner()
    scripts_img2img = ScriptRunner()

