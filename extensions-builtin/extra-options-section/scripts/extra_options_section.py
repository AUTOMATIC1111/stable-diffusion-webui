import gradio as gr
from modules import scripts, shared, ui_components, ui_settings
from modules.ui_components import FormColumn


class ExtraOptionsSection(scripts.Script):
    section = "extra_options"

    def __init__(self):
        self.comps = None
        self.setting_names = None

    def title(self):
        return "Extra options"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        self.comps = []
        self.setting_names = []

        with gr.Blocks() as interface:
            with gr.Accordion("Options", open=False) if shared.opts.extra_options_accordion and shared.opts.extra_options else gr.Group(), gr.Row():
                for setting_name in shared.opts.extra_options:
                    with FormColumn():
                        comp = ui_settings.create_setting_component(setting_name)

                    self.comps.append(comp)
                    self.setting_names.append(setting_name)

        def get_settings_values():
            return [ui_settings.get_value_for_setting(key) for key in self.setting_names]

        interface.load(fn=get_settings_values, inputs=[], outputs=self.comps, queue=False, show_progress=False)

        return self.comps

    def before_process(self, p, *args):
        for name, value in zip(self.setting_names, args):
            if name not in p.override_settings:
                p.override_settings[name] = value


shared.options_templates.update(shared.options_section(('ui', "User interface"), {
    "extra_options": shared.OptionInfo([], "Options in main UI", ui_components.DropdownMulti, lambda: {"choices": list(shared.opts.data_labels.keys())}).js("info", "settingsHintsShowQuicksettings").info("setting entries that also appear in txt2img/img2img interfaces").needs_restart(),
    "extra_options_accordion": shared.OptionInfo(False, "Place options in main UI into an accordion")
}))
