import gradio as gr


class ToolButton(gr.Button, gr.components.FormComponent):
    """Small button with single emoji as text, fits inside gradio forms"""

    def __init__(self, **kwargs):
        super().__init__(variant="tool", **kwargs)

    def get_block_name(self):
        return "button"


class FormRow(gr.Row, gr.components.FormComponent):
    """Same as gr.Row but fits inside gradio forms"""

    def get_block_name(self):
        return "row"


class FormGroup(gr.Group, gr.components.FormComponent):
    """Same as gr.Row but fits inside gradio forms"""

    def get_block_name(self):
        return "group"
