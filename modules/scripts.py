import os
import sys
import traceback

import gradio as gr

class Script:
    filename = None

    def title(self):
        raise NotImplementedError()


scripts = []


def load_scripts(basedir, globs):
    for filename in os.listdir(basedir):
        path = os.path.join(basedir, filename)

        if not os.path.isfile(path):
            continue

        with open(path, "r", encoding="utf8") as file:
            text = file.read()

        from types import ModuleType
        compiled = compile(text, path, 'exec')
        module = ModuleType(filename)
        module.__dict__.update(globs)
        exec(compiled, module.__dict__)

        for key, item in module.__dict__.items():
            if type(item) == type and issubclass(item, Script):
                item.filename = path

                scripts.append(item)


def wrap_call(func, filename, funcname, *args, default=None, **kwargs):
    try:
        res = func()
        return res
    except Exception:
        print(f"Error calling: {filename/funcname}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)

    return default

def setup_ui():
    titles = [wrap_call(script.title, script.filename, "title") for script in scripts]

    gr.Dropdown(options=[""] + titles, value="", type="index")
