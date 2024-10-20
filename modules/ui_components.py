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


class ResizeHandleRow(gr.Row):
    """Same as gr.Row but fits inside gradio forms"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.elem_classes.append("resize-handle-row")

    def get_block_name(self):
        return "row"


class FormRow(FormComponent, gr.Row):
    """Same as gr.Row but fits inside gradio forms"""

    def get_block_name(self):
        return "row"


class FormColumn(FormComponent, gr.Column):
    """Same as gr.Column but fits inside gradio forms"""

    def get_block_name(self):
        return "column"


class FormGroup(FormComponent, gr.Group):
    """Same as gr.Group but fits inside gradio forms"""

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


class DropdownEditable(FormComponent, gr.Dropdown):
    """Same as gr.Dropdown but allows editing value"""
    def __init__(self, **kwargs):
        super().__init__(allow_custom_value=True, **kwargs)

    def get_block_name(self):
        return "dropdown"


class InputAccordion(gr.Checkbox):
    """A gr.Accordion that can be used as an input - returns True if open, False if closed.

    Actually just a hidden checkbox, but creates an accordion that follows and is followed by the state of the checkbox.
    """

    accordion_id_set = set()
    global_index = 0

    def __init__(self, value, **kwargs):
        self.accordion_id = kwargs.get('elem_id')
        if self.accordion_id is None:
            self.accordion_id = f"input-accordion-{InputAccordion.global_index}"
            InputAccordion.global_index += 1

        if not InputAccordion.accordion_id_set:
            from modules import script_callbacks
            script_callbacks.on_script_unloaded(InputAccordion.reset)

        if self.accordion_id in InputAccordion.accordion_id_set:
            count = 1
            while (unique_id := f'{self.accordion_id}-{count}') in InputAccordion.accordion_id_set:
                count += 1
            self.accordion_id = unique_id

        InputAccordion.accordion_id_set.add(self.accordion_id)

        kwargs_checkbox = {
            **kwargs,
            "elem_id": f"{self.accordion_id}-checkbox",
            "visible": False,
        }
        super().__init__(value, **kwargs_checkbox)

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

    @classmethod
    def reset(cls):
        cls.global_index = 0
        cls.accordion_id_set.clear()
