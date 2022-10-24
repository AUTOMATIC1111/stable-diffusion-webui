
callbacks_model_loaded = []
callbacks_ui_tabs = []
callbacks_ui_settings = []


def clear_callbacks():
    callbacks_model_loaded.clear()
    callbacks_ui_tabs.clear()


def model_loaded_callback(sd_model):
    for callback in callbacks_model_loaded:
        callback(sd_model)


def ui_tabs_callback():
    res = []
    
    for callback in callbacks_ui_tabs:
        res += callback() or []

    return res


def ui_settings_callback():
    for callback in callbacks_ui_settings:
        callback()


def on_model_loaded(callback):
    """register a function to be called when the stable diffusion model is created; the model is
    passed as an argument"""
    callbacks_model_loaded.append(callback)


def on_ui_tabs(callback):
    """register a function to be called when the UI is creating new tabs.
    The function must either return a None, which means no new tabs to be added, or a list, where
    each element is a tuple:
        (gradio_component, title, elem_id)

    gradio_component is a gradio component to be used for contents of the tab (usually gr.Blocks)
    title is tab text displayed to user in the UI
    elem_id is HTML id for the tab
    """
    callbacks_ui_tabs.append(callback)


def on_ui_settings(callback):
    """register a function to be called before UI settings are populated; add your settings
    by using shared.opts.add_option(shared.OptionInfo(...)) """
    callbacks_ui_settings.append(callback)
