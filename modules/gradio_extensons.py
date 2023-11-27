from inspect import signature
from functools import wraps
import gradio as gr

from modules import scripts, ui_tempdir, patches


def add_classes_to_gradio_component(comp):
    """
    this adds gradio-* to the component for css styling (ie gradio-button to gr.Button), as well as some others
    """

    comp.elem_classes = [f"gradio-{comp.get_block_name()}", *(comp.elem_classes or [])]

    if getattr(comp, 'multiselect', False):
        comp.elem_classes.append('multiselect')


def IOComponent_init(self, *args, **kwargs):
    self.webui_tooltip = kwargs.pop('tooltip', None)

    if scripts.scripts_current is not None:
        scripts.scripts_current.before_component(self, **kwargs)

    scripts.script_callbacks.before_component_callback(self, **kwargs)

    res = original_IOComponent_init(self, *args, **kwargs)

    add_classes_to_gradio_component(self)

    scripts.script_callbacks.after_component_callback(self, **kwargs)

    if scripts.scripts_current is not None:
        scripts.scripts_current.after_component(self, **kwargs)

    return res


def Block_get_config(self):
    config = original_Block_get_config(self)

    webui_tooltip = getattr(self, 'webui_tooltip', None)
    if webui_tooltip:
        config["webui_tooltip"] = webui_tooltip

    config.pop('example_inputs', None)

    return config


def BlockContext_init(self, *args, **kwargs):
    res = original_BlockContext_init(self, *args, **kwargs)

    add_classes_to_gradio_component(self)

    return res


def Blocks_get_config_file(self, *args, **kwargs):
    config = original_Blocks_get_config_file(self, *args, **kwargs)

    for comp_config in config["components"]:
        if "example_inputs" in comp_config:
            comp_config["example_inputs"] = {"serialized": []}

    return config


def gradio_component_compatibility_layer(component_function):
    @wraps(component_function)
    def patched_function(*args, **kwargs):
        original_signature = signature(component_function).parameters
        valid_kwargs = {k: v for k, v in kwargs.items() if k in original_signature}
        result = component_function(*args, **valid_kwargs)
        return result

    return patched_function


sub_events = ['then', 'success']


def gradio_component_events_compatibility_layer(component_function):
    @wraps(component_function)
    def patched_function(*args, **kwargs):
        kwargs['js'] = kwargs.get('js', kwargs.pop('_js', None))
        original_signature = signature(component_function).parameters
        valid_kwargs = {k: v for k, v in kwargs.items() if k in original_signature}

        result = component_function(*args, **valid_kwargs)

        for sub_event in sub_events:
            component_event_then_function = getattr(result, sub_event, None)
            if component_event_then_function:
                patched_component_event_then_function = gradio_component_sub_events_compatibility_layer(component_event_then_function)
                setattr(result, sub_event, patched_component_event_then_function)
        # original_component_event_then_function = patches.patch(f'{__name__}.', obj=result, field='then', replacement=patched_component_event_then_function)

        return result

    return patched_function


def gradio_component_sub_events_compatibility_layer(component_function):
    @wraps(component_function)
    def patched_function(*args, **kwargs):
        kwargs['js'] = kwargs.get('js', kwargs.pop('_js', None))
        original_signature = signature(component_function).parameters
        valid_kwargs = {k: v for k, v in kwargs.items() if k in original_signature}
        result = component_function(*args, **valid_kwargs)
        return result

    return patched_function


for component_name in set(gr.components.__all__ + gr.layouts.__all__):
    try:
        component = getattr(gr, component_name)
        component_init = getattr(component, '__init__')
        patched_component_init = gradio_component_compatibility_layer(component_init)
        original_IOComponent_init = patches.patch(f'{__name__}.{component_name}', obj=component, field="__init__", replacement=patched_component_init)

        component_events = set(getattr(component, 'EVENTS'))
        for component_event in component_events:
            component_event_function = getattr(component, component_event)
            patched_component_event_function = gradio_component_events_compatibility_layer(component_event_function)
            original_component_event_function = patches.patch(f'{__name__}.{component_name}.{component_event}', obj=component, field=component_event, replacement=patched_component_event_function)
    except Exception as e:
        print(e)
        pass

gr.Box = gr.Group


original_IOComponent_init = patches.patch(__name__, obj=gr.components.base.Component, field="__init__", replacement=IOComponent_init)
original_Block_get_config = patches.patch(__name__, obj=gr.blocks.Block, field="get_config", replacement=Block_get_config)
original_BlockContext_init = patches.patch(__name__, obj=gr.blocks.BlockContext, field="__init__", replacement=BlockContext_init)
original_Blocks_get_config_file = patches.patch(__name__, obj=gr.blocks.Blocks, field="get_config_file", replacement=Blocks_get_config_file)


ui_tempdir.install_ui_tempdir_override()

