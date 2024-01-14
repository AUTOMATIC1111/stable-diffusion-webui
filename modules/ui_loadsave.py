import os
import gradio as gr
from modules import errors
from modules.ui_components import ToolButton


class UiLoadsave:
    """allows saving and restorig default values for gradio components"""

    def __init__(self, filename):
        self.filename = filename
        self.component_mapping = {}
        self.finalized_ui = False
        self.ui_defaults_view = None # button
        self.ui_defaults_apply = None # button
        self.ui_defaults_review = None # button
        self.ui_defaults_restore = None # button
        self.ui_defaults_submenu = None # button
        self.component_open = {}
        self.ui_defaults = {}
        self.ui_settings = self.read_from_file()

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
            self.ui_defaults[key] = getattr(obj, field)
            if saved_value is None:
                # self.ui_settings[key] = getattr(obj, field)
                pass
            elif condition and not condition(saved_value):
                pass
            else:
                setattr(obj, field, saved_value)
                if init_field is not None:
                    init_field(saved_value)
            if field == 'value' and key not in self.component_mapping:
                self.component_mapping[key] = x
            if field == 'open' and key not in self.component_mapping:
                self.component_open[key] = x

        if type(x) in [gr.Slider, gr.Radio, gr.Checkbox, gr.Textbox, gr.Number, gr.Dropdown, ToolButton, gr.Button] and x.visible:
            apply_field(x, 'visible')
        if type(x) == gr.Accordion:
            apply_field(x, 'open')
        if type(x) == gr.Slider:
            apply_field(x, 'value')
            apply_field(x, 'minimum')
            apply_field(x, 'maximum')
            apply_field(x, 'step')
        if type(x) == gr.Radio:
            def check_choices(val):
                for choice in x.choices:
                    if type(choice) == tuple:
                        choice = choice[0]
                    if choice == val:
                        return True
                return False
            apply_field(x, 'value', check_choices)
        if type(x) == gr.Checkbox:
            apply_field(x, 'value')
        if type(x) == gr.Textbox:
            apply_field(x, 'value')
        if type(x) == gr.Number:
            apply_field(x, 'value')
        if type(x) == gr.Dropdown:
            def check_dropdown(val):
                if x.choices is None:
                    errors.log.warning(f'UI: path={path} value={getattr(x, "value", None)}, choices={getattr(x, "choices", None)}')
                    return False
                choices = [c[0] for c in x.choices] if type(x.choices) == list and len(x.choices) > 0 and type(x.choices[0]) == tuple else x.choices
                if getattr(x, 'multiselect', False):
                    return all(value in choices for value in val)
                else:
                    return val in choices
            apply_field(x, 'value', check_dropdown, getattr(x, 'init_field', None))

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
            if isinstance(x, gr.Accordion):
                self.add_component(f"{path}/{x.label}", x)
            if isinstance(x, gr.Tabs) and x.elem_id is not None:
                self.add_component(f"{path}/Tabs@{x.elem_id}", x) # Tabs element dont have a label, have to use elem_id instead
            for c in x.children:
                self.add_block(c, path)
        elif x.label is not None:
            self.add_component(f"{path}/{x.label}", x)
        elif isinstance(x, gr.Button) and x.value is not None:
            self.add_component(f"{path}/{x.value}", x)

    def read_from_file(self):
        from modules.shared import readfile
        return readfile(self.filename)

    def write_to_file(self, current_ui_settings):
        from modules.shared import writefile
        writefile(current_ui_settings, self.filename)

    def dump_defaults(self):
        """saves default values to a file unless the file is present and there was an error loading default values at start"""
        if os.path.exists(self.filename):
            return
        self.write_to_file(self.ui_settings)

    def iter_changes(self, values):
        for i, name in enumerate(self.component_mapping):
            component = self.component_mapping[name]
            choices = getattr(component, 'choices', None)
            if type(choices) is list and len(choices) > 0: # fix gradio radio button choices being tuples
                if type(choices[0]) is tuple:
                    choices = [c[0] for c in choices]
            new_value = values[i]
            if isinstance(new_value, int) and choices:
                if new_value >= len(choices):
                    continue
                new_value = choices[new_value]
            old_value = self.ui_settings.get(name, None)
            default_value = self.ui_defaults.get(name, '')
            if old_value == new_value:
                continue
            if old_value is None and (new_value == '' or new_value == []):
                continue
            if (new_value == default_value) and (old_value is None):
                continue
            yield name, old_value, new_value, default_value
        return []

    def iter_menus(self):
        for _i, name in enumerate(self.component_open):
            old_value = self.ui_settings.get(name, None)
            new_value = self.component_open[name].open
            default_value = self.ui_defaults.get(name, '')
            if old_value == new_value:
                continue
            if (new_value == default_value) and (old_value is None):
                continue
            yield name, old_value, new_value, default_value
        return []

    def ui_view(self, *values):
        text = """
            <table id="ui-defauls">
                <colgroup>
                    <col style="width: 20%; background: var(--table-border-color)">
                    <col style="width: 10%; background: var(--panel-background-fill)">
                    <col style="width: 10%; background: var(--panel-background-fill)">
                    <col style="width: 10%; background: var(--panel-background-fill)">
                </colgroup>
                <thead style="font-size: 110%; border-style: solid; border-bottom: 1px var(--button-primary-border-color) solid">
                <tr>
                    <th>Name</th>
                    <th>Saved value</th>
                    <th>New value</th>
                    <th>Default value</th>
                </tr>
                </thead>
            <tbody>"""
        changed = 0
        for name, old_value, new_value, default_value in self.iter_changes(values):
            changed += 1
            if old_value is None:
                old_value = "None"
            text += f"<tr><td>{name}</td><td>{old_value}</td><td>{new_value}</td><td>{default_value}</td></tr>"
        text += "</tbody></table>"
        if changed == 0:
            text = '<h2>No changes</h2>'
        else:
            text = f'<h2>Changed values: {changed}</h2>' + text
        return text

    def ui_apply(self, *values):
        from modules.shared import log
        num_changed = 0
        current_ui_settings = self.read_from_file()
        for name, old_value, new_value, default_value in self.iter_changes(values):
            component = self.component_mapping[name]
            log.debug(f'Settings: name={name} component={component} old={old_value} default={default_value} new={new_value}')
            num_changed += 1
            current_ui_settings[name] = new_value
        if num_changed == 0:
            return "No changes"
        self.write_to_file(current_ui_settings)
        errors.log.info(f'UI defaults saved: {self.filename}')
        return f"Wrote {num_changed} changes"

    def ui_submenu_apply(self, items):
        text = """
            <table id="ui-defauls">
                <colgroup>
                    <col style="width: 20%; background: var(--table-border-color)">
                    <col style="width: 10%; background: var(--panel-background-fill)">
                </colgroup>
                <thead style="font-size: 110%; border-style: solid; border-bottom: 1px var(--button-primary-border-color) solid">
                <tr>
                    <th>Menu</th>
                    <th>State</th>
                </tr>
                </thead>
            <tbody>"""
        for k in self.component_open.keys():
            opened = len([i for i, j in items.items() if j is True and i in k]) > 0
            self.component_open[k].open = opened
            text += f"<tr><td>{k}</td><td>{'open' if opened else 'closed'}</td></tr>"
        text += "</tbody></table>"

        from modules.shared import log
        num_changed = 0
        current_ui_settings = self.read_from_file()
        for name, _old_value, new_value, default_value in self.iter_menus():
            log.debug(f'Settings: name={name} default={default_value} new={new_value}')
            num_changed += 1
            current_ui_settings[name] = new_value
        if num_changed == 0:
            text += '<br>No changes'
        else:
            self.write_to_file(current_ui_settings)
            errors.log.info(f'UI defaults saved: {self.filename}')
            text += f'<br>Changes: {num_changed}'
        return text

    def ui_restore(self):
        if os.path.exists(self.filename):
            os.remove(self.filename)
        errors.log.info(f'UI defaults reset: {self.filename}')
        return "Restored system defaults for user interface"

    def create_ui(self):
        """creates ui elements for editing defaults UI, without adding any logic to them"""
        with gr.Row(elem_id="config_row"):
            self.ui_defaults_apply = gr.Button(value='Set new defaults', elem_id="ui_defaults_apply", variant="primary")
            self.ui_defaults_submenu = gr.Button(value='Set menu states', elem_id="ui_submenu_apply", variant="primary")
            self.ui_defaults_restore = gr.Button(value='Restore defaults', elem_id="ui_defaults_restore", variant="primary")
            self.ui_defaults_view = gr.Button(value='Refresh changes', elem_id="ui_defaults_view", variant="secondary")
        self.ui_defaults_review = gr.HTML("", elem_id="ui_defaults_review")

    def setup_ui(self):
        """adds logic to elements created with create_ui; all add_block class must be made before this"""
        assert not self.finalized_ui
        self.finalized_ui = True
        self.ui_defaults_view.click(fn=self.ui_view, inputs=list(self.component_mapping.values()), outputs=[self.ui_defaults_review])
        self.ui_defaults_apply.click(fn=self.ui_apply, inputs=list(self.component_mapping.values()), outputs=[self.ui_defaults_review])
        self.ui_defaults_restore.click(fn=self.ui_restore, inputs=[], outputs=[self.ui_defaults_review])
        self.ui_defaults_submenu.click(fn=self.ui_submenu_apply, _js='uiOpenSubmenus', inputs=[self.ui_defaults_review], outputs=[self.ui_defaults_review])
