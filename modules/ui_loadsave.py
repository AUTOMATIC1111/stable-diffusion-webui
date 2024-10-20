import json
import os

import gradio as gr

from modules import errors
from modules.ui_components import ToolButton, InputAccordion


def radio_choices(comp):  # gradio 3.41 changes choices from list of values to list of pairs
    return [x[0] if isinstance(x, tuple) else x for x in getattr(comp, 'choices', [])]


class UiLoadsave:
    """allows saving and restoring default values for gradio components"""

    def __init__(self, filename):
        self.filename = filename
        self.ui_settings = {}
        self.component_mapping = {}
        self.error_loading = False
        self.finalized_ui = False

        self.ui_defaults_view = None
        self.ui_defaults_apply = None
        self.ui_defaults_review = None

        try:
            self.ui_settings = self.read_from_file()
        except FileNotFoundError:
            pass
        except Exception as e:
            self.error_loading = True
            errors.display(e, "loading settings")

    def add_component(self, path, x):
        """adds component to the registry of tracked components"""

        assert not self.finalized_ui

        def apply_field(obj, field, condition=None, init_field=None):
            key = f"{path}/{field}"

            if getattr(obj, 'custom_script_source', None) is not None:
                key = f"customscript/{obj.custom_script_source}/{key}"

            if getattr(obj, 'do_not_save_to_config', False):
                return

            saved_value = self.ui_settings.get(key, None)

            if isinstance(obj, gr.Accordion) and isinstance(x, InputAccordion) and field == 'value':
                field = 'open'

            if saved_value is None:
                self.ui_settings[key] = getattr(obj, field)
            elif condition and not condition(saved_value):
                pass
            else:
                if isinstance(obj, gr.Textbox) and field == 'value':  # due to an undesirable behavior of gr.Textbox, if you give it an int value instead of str, everything dies
                    saved_value = str(saved_value)
                elif isinstance(obj, gr.Number) and field == 'value':
                    try:
                        saved_value = float(saved_value)
                    except ValueError:
                        return

                setattr(obj, field, saved_value)
                if init_field is not None:
                    init_field(saved_value)

            if field == 'value' and key not in self.component_mapping:
                self.component_mapping[key] = obj

        if type(x) in [gr.Slider, gr.Radio, gr.Checkbox, gr.Textbox, gr.Number, gr.Dropdown, ToolButton, gr.Button] and x.visible:
            apply_field(x, 'visible')

        if type(x) == gr.Slider:
            apply_field(x, 'value')
            apply_field(x, 'minimum')
            apply_field(x, 'maximum')
            apply_field(x, 'step')

        if type(x) == gr.Radio:
            apply_field(x, 'value', lambda val: val in radio_choices(x))

        if type(x) == gr.Checkbox:
            apply_field(x, 'value')

        if type(x) == gr.Textbox:
            apply_field(x, 'value')

        if type(x) == gr.Number:
            apply_field(x, 'value')

        if type(x) == gr.Dropdown:
            def check_dropdown(val):
                choices = radio_choices(x)
                if getattr(x, 'multiselect', False):
                    return all(value in choices for value in val)
                else:
                    return val in choices

            apply_field(x, 'value', check_dropdown, getattr(x, 'init_field', None))

        if type(x) == InputAccordion:
            if hasattr(x, 'custom_script_source'):
                x.accordion.custom_script_source = x.custom_script_source
            if x.accordion.visible:
                apply_field(x.accordion, 'visible')
            apply_field(x, 'value')
            apply_field(x.accordion, 'value')

        def check_tab_id(tab_id):
            tab_items = list(filter(lambda e: isinstance(e, gr.TabItem), x.children))
            if type(tab_id) == str:
                tab_ids = [t.id for t in tab_items]
                return tab_id in tab_ids
            elif type(tab_id) == int:
                return 0 <= tab_id < len(tab_items)
            else:
                return False

        if type(x) == gr.Tabs:
            apply_field(x, 'selected', check_tab_id)

    def add_block(self, x, path=""):
        """adds all components inside a gradio block x to the registry of tracked components"""

        if hasattr(x, 'children'):
            if isinstance(x, gr.Tabs) and x.elem_id is not None:
                # Tabs element can't have a label, have to use elem_id instead
                self.add_component(f"{path}/Tabs@{x.elem_id}", x)
            for c in x.children:
                self.add_block(c, path)
        elif x.label is not None:
            self.add_component(f"{path}/{x.label}", x)
        elif isinstance(x, gr.Button) and x.value is not None:
            self.add_component(f"{path}/{x.value}", x)

    def read_from_file(self):
        with open(self.filename, "r", encoding="utf8") as file:
            return json.load(file)

    def write_to_file(self, current_ui_settings):
        with open(self.filename, "w", encoding="utf8") as file:
            json.dump(current_ui_settings, file, indent=4, ensure_ascii=False)

    def dump_defaults(self):
        """saves default values to a file unless the file is present and there was an error loading default values at start"""

        if self.error_loading and os.path.exists(self.filename):
            return

        self.write_to_file(self.ui_settings)

    def iter_changes(self, current_ui_settings, values):
        """
        given a dictionary with defaults from a file and current values from gradio elements, returns
        an iterator over tuples of values that are not the same between the file and the current;
        tuple contents are: path, old value, new value
        """

        for (path, component), new_value in zip(self.component_mapping.items(), values):
            old_value = current_ui_settings.get(path)

            choices = radio_choices(component)
            if isinstance(new_value, int) and choices:
                if new_value >= len(choices):
                    continue

                new_value = choices[new_value]
                if isinstance(new_value, tuple):
                    new_value = new_value[0]

            if new_value == old_value:
                continue

            if old_value is None and (new_value == '' or new_value == []):
                continue

            yield path, old_value, new_value

    def ui_view(self, *values):
        text = ["<table><thead><tr><th>Path</th><th>Old value</th><th>New value</th></thead><tbody>"]

        for path, old_value, new_value in self.iter_changes(self.read_from_file(), values):
            if old_value is None:
                old_value = "<span class='ui-defaults-none'>None</span>"

            text.append(f"<tr><td>{path}</td><td>{old_value}</td><td>{new_value}</td></tr>")

        if len(text) == 1:
            text.append("<tr><td colspan=3>No changes</td></tr>")

        text.append("</tbody>")
        return "".join(text)

    def ui_apply(self, *values):
        num_changed = 0

        current_ui_settings = self.read_from_file()

        for path, _, new_value in self.iter_changes(current_ui_settings.copy(), values):
            num_changed += 1
            current_ui_settings[path] = new_value

        if num_changed == 0:
            return "No changes."

        self.write_to_file(current_ui_settings)

        return f"Wrote {num_changed} changes."

    def create_ui(self):
        """creates ui elements for editing defaults UI, without adding any logic to them"""

        gr.HTML(
            f"This page allows you to change default values in UI elements on other tabs.<br />"
            f"Make your changes, press 'View changes' to review the changed default values,<br />"
            f"then press 'Apply' to write them to {self.filename}.<br />"
            f"New defaults will apply after you restart the UI.<br />"
        )

        with gr.Row():
            self.ui_defaults_view = gr.Button(value='View changes', elem_id="ui_defaults_view", variant="secondary")
            self.ui_defaults_apply = gr.Button(value='Apply', elem_id="ui_defaults_apply", variant="primary")

        self.ui_defaults_review = gr.HTML("")

    def setup_ui(self):
        """adds logic to elements created with create_ui; all add_block class must be made before this"""

        assert not self.finalized_ui
        self.finalized_ui = True

        self.ui_defaults_view.click(fn=self.ui_view, inputs=list(self.component_mapping.values()), outputs=[self.ui_defaults_review])
        self.ui_defaults_apply.click(fn=self.ui_apply, inputs=list(self.component_mapping.values()), outputs=[self.ui_defaults_review])
