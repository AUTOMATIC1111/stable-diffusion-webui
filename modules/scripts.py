import os
import sys
import traceback

import modules.ui as ui
import gradio as gr

from modules.processing import StableDiffusionProcessing
from modules import shared

class Script:
    filename = None
    args_from = None
    args_to = None

    # The title of the script. This is what will be displayed in the dropdown menu.
    def title(self):
        raise NotImplementedError()

    # How the script is displayed in the UI. See https://gradio.app/docs/#components
    # for the different UI components you can use and how to create them.
    # Most UI components can return a value, such as a boolean for a checkbox.
    # The returned values are passed to the run method as parameters.
    def ui(self, is_img2img):
        pass

    # Determines when the script should be shown in the dropdown menu via the 
    # returned value. As an example:
    # is_img2img is True if the current tab is img2img, and False if it is txt2img.
    # Thus, return is_img2img to only show the script on the img2img tab.
    def show(self, is_img2img):
        return True

    # This is where the additional processing is implemented. The parameters include
    # self, the model object "p" (a StableDiffusionProcessing class, see
    # processing.py), and the parameters returned by the ui method.
    # Custom functions can be defined here, and additional libraries can be imported 
    # to be used in processing. The return value should be a Processed object, which is
    # what is returned by the process_images method.
    def run(self, *args):
        raise NotImplementedError()

    # The description method is currently unused.
    # To add a description that appears when hovering over the title, amend the "titles" 
    # dict in script.js to include the script title (returned by title) as a key, and 
    # your description as the value.
    def describe(self):
        return ""


scripts_data = []


def load_scripts(basedir):
    if not os.path.exists(basedir):
        return

    for filename in sorted(os.listdir(basedir)):
        path = os.path.join(basedir, filename)

        if not os.path.isfile(path):
            continue

        try:
            with open(path, "r", encoding="utf8") as file:
                text = file.read()

            from types import ModuleType
            compiled = compile(text, path, 'exec')
            module = ModuleType(filename)
            exec(compiled, module.__dict__)

            for key, script_class in module.__dict__.items():
                if type(script_class) == type and issubclass(script_class, Script):
                    scripts_data.append((script_class, path))

        except Exception:
            print(f"Error loading script: {filename}", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)


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

    def setup_ui(self, is_img2img):
        for script_class, path in scripts_data:
            script = script_class()
            script.filename = path

            if not script.show(is_img2img):
                continue

            self.scripts.append(script)

        titles = [wrap_call(script.title, script.filename, "title") or f"{script.filename} [error]" for script in self.scripts]

        dropdown = gr.Dropdown(label="Script", choices=["None"] + titles, value="None", type="index")
        inputs = [dropdown]

        for script in self.scripts:
            script.args_from = len(inputs)
            script.args_to = len(inputs)

            controls = wrap_call(script.ui, script.filename, "ui", is_img2img)

            if controls is None:
                continue

            for control in controls:
                control.custom_script_source = os.path.basename(script.filename)
                control.visible = False

            inputs += controls
            script.args_to = len(inputs)

        def select_script(script_index):
            if 0 < script_index <= len(self.scripts):
                script = self.scripts[script_index-1]
                args_from = script.args_from
                args_to = script.args_to
            else:
                args_from = 0
                args_to = 0

            return [ui.gr_show(True if i == 0 else args_from <= i < args_to) for i in range(len(inputs))]

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

        script = self.scripts[script_index-1]

        if script is None:
            return None

        script_args = args[script.args_from:script.args_to]
        processed = script.run(p, *script_args)

        shared.total_tqdm.clear()

        return processed

    def reload_sources(self):
        for si, script in list(enumerate(self.scripts)):
            with open(script.filename, "r", encoding="utf8") as file:
                args_from = script.args_from
                args_to = script.args_to
                filename = script.filename
                text = file.read()

                from types import ModuleType

                compiled = compile(text, filename, 'exec')
                module = ModuleType(script.filename)
                exec(compiled, module.__dict__)

                for key, script_class in module.__dict__.items():
                    if type(script_class) == type and issubclass(script_class, Script):
                        self.scripts[si] = script_class()
                        self.scripts[si].filename = filename
                        self.scripts[si].args_from = args_from
                        self.scripts[si].args_to = args_to

scripts_txt2img = ScriptRunner()
scripts_img2img = ScriptRunner()

def reload_script_body_only():
    scripts_txt2img.reload_sources()
    scripts_img2img.reload_sources()


def reload_scripts(basedir):
    global scripts_txt2img, scripts_img2img

    scripts_data.clear()
    load_scripts(basedir)

    scripts_txt2img = ScriptRunner()
    scripts_img2img = ScriptRunner()
