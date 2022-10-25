import sys
import traceback
from collections import namedtuple
import inspect


def report_exception(c, job):
    print(f"Error executing callback {job} for {c.script}", file=sys.stderr)
    print(traceback.format_exc(), file=sys.stderr)


ScriptCallback = namedtuple("ScriptCallback", ["script", "callback"])
callbacks_model_loaded = []
callbacks_ui_tabs = []
callbacks_ui_settings = []
callbacks_image_saved = []

def clear_callbacks():
    callbacks_model_loaded.clear()
    callbacks_ui_tabs.clear()
    callbacks_image_saved.clear()


def model_loaded_callback(sd_model):
    for c in callbacks_model_loaded:
        try:
            c.callback(sd_model)
        except Exception:
            report_exception(c, 'model_loaded_callback')


def ui_tabs_callback():
    res = []
    
    for c in callbacks_ui_tabs:
        try:
            res += c.callback() or []
        except Exception:
            report_exception(c, 'ui_tabs_callback')

    return res


def ui_settings_callback():
    for c in callbacks_ui_settings:
        try:
            c.callback()
        except Exception:
            report_exception(c, 'ui_settings_callback')


def image_saved_callback(image, p, fullfn, txt_fullfn):
    for c in callbacks_image_saved:
        try:
            c.callback(image, p, fullfn, txt_fullfn)
        except Exception:
            report_exception(c, 'image_saved_callback')


def add_callback(callbacks, fun):
    stack = [x for x in inspect.stack() if x.filename != __file__]
    filename = stack[0].filename if len(stack) > 0 else 'unknown file'

    callbacks.append(ScriptCallback(filename, fun))



def on_model_loaded(callback):
    """register a function to be called when the stable diffusion model is created; the model is
    passed as an argument"""
    add_callback(callbacks_model_loaded, callback)


def on_ui_tabs(callback):
    """register a function to be called when the UI is creating new tabs.
    The function must either return a None, which means no new tabs to be added, or a list, where
    each element is a tuple:
        (gradio_component, title, elem_id)

    gradio_component is a gradio component to be used for contents of the tab (usually gr.Blocks)
    title is tab text displayed to user in the UI
    elem_id is HTML id for the tab
    """
    add_callback(callbacks_ui_tabs, callback)


def on_ui_settings(callback):
    """register a function to be called before UI settings are populated; add your settings
    by using shared.opts.add_option(shared.OptionInfo(...)) """
    add_callback(callbacks_ui_settings, callback)


def on_save_imaged(callback):
    """register a function to be called after modules.images.save_image is called.
    The callback is called with three arguments:
        - p - procesing object (or a dummy object with same fields if the image is saved using save button)
        - fullfn - image filename
        - txt_fullfn - text file with parameters; may be None
    """
    add_callback(callbacks_image_saved, callback)
