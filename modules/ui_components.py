import gradio as gr


class FormComponent:
    def get_expected_parent(self):
        return gr.components.Form


gr.Dropdown.get_expected_parent = FormComponent.get_expected_parent


class ToolButton(FormComponent, gr.Button):
    """Small button with single emoji as text, fits inside gradio forms"""

    def __init__(self, *args, **kwargs):
        classes = kwargs.pop("elem_classes", [])
        super().__init__(*args, elem_classes=["tool", *classes], **kwargs)

    def get_block_name(self):
        return "button"


class FormRow(FormComponent, gr.Row):
    """Same as gr.Row but fits inside gradio forms"""

    def get_block_name(self):
        return "row"


class FormColumn(FormComponent, gr.Column):
    """Same as gr.Column but fits inside gradio forms"""

    def get_block_name(self):
        return "column"


class FormGroup(FormComponent, gr.Group):
    """Same as gr.Row but fits inside gradio forms"""

    def get_block_name(self):
        return "group"


class FormHTML(FormComponent, gr.HTML):
    """Same as gr.HTML but fits inside gradio forms"""

    def get_block_name(self):
        return "html"


class FormColorPicker(FormComponent, gr.ColorPicker):
    """Same as gr.ColorPicker but fits inside gradio forms"""

    def get_block_name(self):
        return "colorpicker"


class DropdownMulti(FormComponent, gr.Dropdown):
    """Same as gr.Dropdown but always multiselect"""
    def __init__(self, **kwargs):
        super().__init__(multiselect=True, **kwargs)

    def get_block_name(self):
        return "dropdown"
