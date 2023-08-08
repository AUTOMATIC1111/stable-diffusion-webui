import gradio as gr

from modules import scripts

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

    return config


def BlockContext_init(self, *args, **kwargs):
    res = original_BlockContext_init(self, *args, **kwargs)

    add_classes_to_gradio_component(self)

    return res


original_IOComponent_init = gr.components.IOComponent.__init__
original_Block_get_config = gr.blocks.Block.get_config
original_BlockContext_init = gr.blocks.BlockContext.__init__

gr.components.IOComponent.__init__ = IOComponent_init
gr.blocks.Block.get_config = Block_get_config
gr.blocks.BlockContext.__init__ = BlockContext_init
