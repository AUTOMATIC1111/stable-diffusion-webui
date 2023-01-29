import gradio as gr


class ToolButton(gr.Button, gr.components.FormComponent):
    """Small button with single emoji as text, fits inside gradio forms"""

    def __init__(self, **kwargs):
        super().__init__(variant="tool", **kwargs)

    def get_block_name(self):
        return "button"


class ToolButtonTop(gr.Button, gr.components.FormComponent):
    """Small button with single emoji as text, with extra margin at top, fits inside gradio forms"""

    def __init__(self, **kwargs):
        super().__init__(variant="tool-top", **kwargs)

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


class FormHTML(gr.HTML, gr.components.FormComponent):
    """Same as gr.HTML but fits inside gradio forms"""

    def get_block_name(self):
        return "html"


class FormColorPicker(gr.ColorPicker, gr.components.FormComponent):
    """Same as gr.ColorPicker but fits inside gradio forms"""

    def get_block_name(self):
        return "colorpicker"


class DropdownMulti(gr.Dropdown):
    """Same as gr.Dropdown but always multiselect"""
    def __init__(self, **kwargs):
        super().__init__(multiselect=True, **kwargs)

    def get_block_name(self):
        return "dropdown"
