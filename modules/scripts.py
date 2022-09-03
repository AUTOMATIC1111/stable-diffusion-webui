import os
import sys
import traceback

import modules.ui as ui
import gradio as gr

from modules.processing import StableDiffusionProcessing

class Script:
    filename = None
    args_from = None
    args_to = None

    def title(self):
        raise NotImplementedError()

    def ui(self, is_img2img):
        pass

    def run(self, *args):
        raise NotImplementedError()

    def describe(self):
        return ""


scripts = []


def load_scripts(basedir):
    if not os.path.exists(basedir):
        return

    for filename in os.listdir(basedir):
        path = os.path.join(basedir, filename)

        if not os.path.isfile(path):
            continue

        with open(path, "r", encoding="utf8") as file:
            text = file.read()

        try:
            from types import ModuleType
            compiled = compile(text, path, 'exec')
            module = ModuleType(filename)
            exec(compiled, module.__dict__)

            for key, script_class in module.__dict__.items():
                if type(script_class) == type and issubclass(script_class, Script):
                    obj = script_class()
                    obj.filename = path

                    scripts.append(obj)
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


def setup_ui(is_img2img):
    titles = [wrap_call(script.title, script.filename, "title") or f"{script.filename} [error]" for script in scripts]

    dropdown = gr.Dropdown(label="Script", choices=["None"] + titles, value="None", type="index")

    inputs = [dropdown]

    for script in scripts:
        script.args_from = len(inputs)
        controls = script.ui(is_img2img)

        for control in controls:
            control.visible = False

        inputs += controls
        script.args_to = len(inputs)

    def select_script(index):
        if index > 0:
            script = scripts[index-1]
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


def run(p: StableDiffusionProcessing, *args):
    script_index = args[0] - 1

    if script_index < 0 or script_index >= len(scripts):
        return None

    script = scripts[script_index]

    script_args = args[script.args_from:script.args_to]
    processed = script.run(p, *script_args)

    return processed
