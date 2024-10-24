from functools import wraps

import gradio as gr
from modules import gradio_extensions  # noqa: F401


class FormComponent:
    webui_do_not_create_gradio_pyi_thank_you = True

    def get_expected_parent(self):
        return gr.components.Form


gr.Dropdown.get_expected_parent = FormComponent.get_expected_parent


class ToolButton(gr.Button, FormComponent):
    """Small button with single emoji as text, fits inside gradio forms"""

    @wraps(gr.Button.__init__)
    def __init__(self, value="", *args, elem_classes=None, **kwargs):
        elem_classes = elem_classes or []
        super().__init__(*args, elem_classes=["tool", *elem_classes], value=value, **kwargs)

    def get_block_name(self):
        return "button"


class ResizeHandleRow(gr.Row):
    """Same as gr.Row but fits inside gradio forms"""
    webui_do_not_create_gradio_pyi_thank_you = True

    @wraps(gr.Row.__init__)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.elem_classes.append("resize-handle-row")

    def get_block_name(self):
        return "row"


class FormRow(gr.Row, FormComponent):
    """Same as gr.Row but fits inside gradio forms"""

    def get_block_name(self):
        return "row"


class FormColumn(gr.Column, FormComponent):
    """Same as gr.Column but fits inside gradio forms"""

    def get_block_name(self):
        return "column"


class FormGroup(gr.Group, FormComponent):
    """Same as gr.Group but fits inside gradio forms"""

    def get_block_name(self):
        return "group"


class FormHTML(gr.HTML, FormComponent):
    """Same as gr.HTML but fits inside gradio forms"""

    def get_block_name(self):
        return "html"


class FormColorPicker(gr.ColorPicker, FormComponent):
    """Same as gr.ColorPicker but fits inside gradio forms"""

    def get_block_name(self):
        return "colorpicker"


class DropdownMulti(gr.Dropdown, FormComponent):
    """Same as gr.Dropdown but always multiselect"""

    @wraps(gr.Dropdown.__init__)
    def __init__(self, **kwargs):
        kwargs['multiselect'] = True
        super().__init__(**kwargs)

    def get_block_name(self):
        return "dropdown"


class DropdownEditable(gr.Dropdown, FormComponent):
    """Same as gr.Dropdown but allows editing value"""

    @wraps(gr.Dropdown.__init__)
    def __init__(self, **kwargs):
        kwargs['allow_custom_value'] = True
        super().__init__(**kwargs)

    def get_block_name(self):
        return "dropdown"


class InputAccordionImpl(gr.Checkbox):
    """A gr.Accordion that can be used as an input - returns True if open, False if closed.

    Actually just a hidden checkbox, but creates an accordion that follows and is followed by the state of the checkbox.
    """

    webui_do_not_create_gradio_pyi_thank_you = True

    global_index = 0

    @wraps(gr.Checkbox.__init__)
    def __init__(self, value=None, setup=False, **kwargs):
        if not setup:
            super().__init__(value=value, **kwargs)
            return

        self.accordion_id = kwargs.get('elem_id')
        if self.accordion_id is None:
            self.accordion_id = f"input-accordion-{InputAccordionImpl.global_index}"
            InputAccordionImpl.global_index += 1

        kwargs_checkbox = {
            **kwargs,
            "elem_id": f"{self.accordion_id}-checkbox",
            "visible": False,
        }
        super().__init__(value=value, **kwargs_checkbox)

        self.change(fn=None, _js='function(checked){ inputAccordionChecked("' + self.accordion_id + '", checked); }', inputs=[self])

        kwargs_accordion = {
            **kwargs,
            "elem_id": self.accordion_id,
            "label": kwargs.get('label', 'Accordion'),
            "elem_classes": ['input-accordion'],
            "open": value,
        }

        self.accordion = gr.Accordion(**kwargs_accordion)

    def extra(self):
        """Allows you to put something into the label of the accordion.

        Use it like this:

        ```
        with InputAccordion(False, label="Accordion") as acc:
            with acc.extra():
                FormHTML(value="hello", min_width=0)

            ...
        ```
        """

        return gr.Column(elem_id=self.accordion_id + '-extra', elem_classes='input-accordion-extra', min_width=0)

    def __enter__(self):
        self.accordion.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.accordion.__exit__(exc_type, exc_val, exc_tb)

    def get_block_name(self):
        return "checkbox"


def InputAccordion(value=None, **kwargs):
    return InputAccordionImpl(value=value, setup=True, **kwargs)
