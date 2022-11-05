import sys
import traceback
from collections import namedtuple
import inspect
from typing import Optional

from fastapi import FastAPI
from gradio import Blocks

def report_exception(c, job):
    print(f"Error executing callback {job} for {c.script}", file=sys.stderr)
    print(traceback.format_exc(), file=sys.stderr)


class ImageSaveParams:
    def __init__(self, image, p, filename, pnginfo):
        self.image = image
        """the PIL image itself"""

        self.p = p
        """p object with processing parameters; either StableDiffusionProcessing or an object with same fields"""

        self.filename = filename
        """name of file that the image would be saved to"""

        self.pnginfo = pnginfo
        """dictionary with parameters for image's PNG info data; infotext will have the key 'parameters'"""


class CFGDenoiserParams:
    def __init__(self, x, image_cond, sigma, sampling_step, total_sampling_steps):
        self.x = x
        """Latent image representation in the process of being denoised"""
        
        self.image_cond = image_cond
        """Conditioning image"""
        
        self.sigma = sigma
        """Current sigma noise step value"""
        
        self.sampling_step = sampling_step
        """Current Sampling step number"""
        
        self.total_sampling_steps = total_sampling_steps
        """Total number of sampling steps planned"""


ScriptCallback = namedtuple("ScriptCallback", ["script", "callback"])
callback_map = dict(
    callbacks_app_started=[],
    callbacks_model_loaded=[],
    callbacks_ui_tabs=[],
    callbacks_ui_settings=[],
    callbacks_before_image_saved=[],
    callbacks_image_saved=[],
    callbacks_cfg_denoiser=[]
)


def clear_callbacks():
    for callback_list in callback_map.values():
        callback_list.clear()

def app_started_callback(demo: Optional[Blocks], app: FastAPI):
    for c in callback_map['callbacks_app_started']:
        try:
            c.callback(demo, app)
        except Exception:
            report_exception(c, 'app_started_callback')


def model_loaded_callback(sd_model):
    for c in callback_map['callbacks_model_loaded']:
        try:
            c.callback(sd_model)
        except Exception:
            report_exception(c, 'model_loaded_callback')


def ui_tabs_callback():
    res = []
    
    for c in callback_map['callbacks_ui_tabs']:
        try:
            res += c.callback() or []
        except Exception:
            report_exception(c, 'ui_tabs_callback')

    return res


def ui_settings_callback():
    for c in callback_map['callbacks_ui_settings']:
        try:
            c.callback()
        except Exception:
            report_exception(c, 'ui_settings_callback')


def before_image_saved_callback(params: ImageSaveParams):
    for c in callback_map['callbacks_before_image_saved']:
        try:
            c.callback(params)
        except Exception:
            report_exception(c, 'before_image_saved_callback')


def image_saved_callback(params: ImageSaveParams):
    for c in callback_map['callbacks_image_saved']:
        try:
            c.callback(params)
        except Exception:
            report_exception(c, 'image_saved_callback')


def cfg_denoiser_callback(params: CFGDenoiserParams):
    for c in callback_map['callbacks_cfg_denoiser']:
        try:
            c.callback(params)
        except Exception:
            report_exception(c, 'cfg_denoiser_callback')


def add_callback(callbacks, fun):
    stack = [x for x in inspect.stack() if x.filename != __file__]
    filename = stack[0].filename if len(stack) > 0 else 'unknown file'

    callbacks.append(ScriptCallback(filename, fun))

    
def remove_current_script_callbacks():
    stack = [x for x in inspect.stack() if x.filename != __file__]
    filename = stack[0].filename if len(stack) > 0 else 'unknown file'
    if filename == 'unknown file':
        return
    for callback_list in callback_map.values():
        for callback_to_remove in [cb for cb in callback_list if cb.script == filename]:
            callback_list.remove(callback_to_remove)


def remove_callbacks_for_function(callback_func):
    for callback_list in callback_map.values():
        for callback_to_remove in [cb for cb in callback_list if cb.callback == callback_func]:
            callback_list.remove(callback_to_remove)


def on_app_started(callback):
    """register a function to be called when the webui started, the gradio `Block` component and
    fastapi `FastAPI` object are passed as the arguments"""
    add_callback(callback_map['callbacks_app_started'], callback)


def on_model_loaded(callback):
    """register a function to be called when the stable diffusion model is created; the model is
    passed as an argument"""
    add_callback(callback_map['callbacks_model_loaded'], callback)


def on_ui_tabs(callback):
    """register a function to be called when the UI is creating new tabs.
    The function must either return a None, which means no new tabs to be added, or a list, where
    each element is a tuple:
        (gradio_component, title, elem_id)

    gradio_component is a gradio component to be used for contents of the tab (usually gr.Blocks)
    title is tab text displayed to user in the UI
    elem_id is HTML id for the tab
    """
    add_callback(callback_map['callbacks_ui_tabs'], callback)


def on_ui_settings(callback):
    """register a function to be called before UI settings are populated; add your settings
    by using shared.opts.add_option(shared.OptionInfo(...)) """
    add_callback(callback_map['callbacks_ui_settings'], callback)


def on_before_image_saved(callback):
    """register a function to be called before an image is saved to a file.
    The callback is called with one argument:
        - params: ImageSaveParams - parameters the image is to be saved with. You can change fields in this object.
    """
    add_callback(callback_map['callbacks_before_image_saved'], callback)


def on_image_saved(callback):
    """register a function to be called after an image is saved to a file.
    The callback is called with one argument:
        - params: ImageSaveParams - parameters the image was saved with. Changing fields in this object does nothing.
    """
    add_callback(callback_map['callbacks_image_saved'], callback)


def on_cfg_denoiser(callback):
    """register a function to be called in the kdiffussion cfg_denoiser method after building the inner model inputs.
    The callback is called with one argument:
        - params: CFGDenoiserParams - parameters to be passed to the inner model and sampling state details.
    """
    add_callback(callback_map['callbacks_cfg_denoiser'], callback)
