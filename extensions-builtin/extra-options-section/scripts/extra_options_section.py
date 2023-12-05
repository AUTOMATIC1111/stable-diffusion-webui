import math

import gradio as gr
from modules import scripts, shared, ui_components, ui_settings, generation_parameters_copypaste
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
        self.infotext_fields = []
        extra_options = shared.opts.extra_options_img2img if is_img2img else shared.opts.extra_options_txt2img

        mapping = {k: v for v, k in generation_parameters_copypaste.infotext_to_setting_name_mapping}

        with gr.Blocks() as interface:
            with gr.Accordion("Options", open=False) if shared.opts.extra_options_accordion and extra_options else gr.Group():

                row_count = math.ceil(len(extra_options) / shared.opts.extra_options_cols)

                for row in range(row_count):
                    with gr.Row():
                        for col in range(shared.opts.extra_options_cols):
                            index = row * shared.opts.extra_options_cols + col
                            if index >= len(extra_options):
                                break

                            setting_name = extra_options[index]

                            with FormColumn():
                                comp = ui_settings.create_setting_component(setting_name)

                            self.comps.append(comp)
                            self.setting_names.append(setting_name)

                            setting_infotext_name = mapping.get(setting_name)
                            if setting_infotext_name is not None:
                                self.infotext_fields.append((comp, setting_infotext_name))

        def get_settings_values():
            res = [ui_settings.get_value_for_setting(key) for key in self.setting_names]
            return res[0] if len(res) == 1 else res

        interface.load(fn=get_settings_values, inputs=[], outputs=self.comps, queue=False, show_progress=False)

        return self.comps

    def before_process(self, p, *args):
        for name, value in zip(self.setting_names, args):
            if name not in p.override_settings:
                p.override_settings[name] = value


shared.options_templates.update(shared.options_section(('settings_in_ui', "Settings in UI", "ui"), {
    "settings_in_ui": shared.OptionHTML("""
This page allows you to add some settings to the main interface of txt2img and img2img tabs.
"""),
    "extra_options_txt2img": shared.OptionInfo([], "Settings for txt2img", ui_components.DropdownMulti, lambda: {"choices": list(shared.opts.data_labels.keys())}).js("info", "settingsHintsShowQuicksettings").info("setting entries that also appear in txt2img interfaces").needs_reload_ui(),
    "extra_options_img2img": shared.OptionInfo([], "Settings for img2img", ui_components.DropdownMulti, lambda: {"choices": list(shared.opts.data_labels.keys())}).js("info", "settingsHintsShowQuicksettings").info("setting entries that also appear in img2img interfaces").needs_reload_ui(),
    "extra_options_cols": shared.OptionInfo(1, "Number of columns for added settings", gr.Number, {"precision": 0}).needs_reload_ui(),
    "extra_options_accordion": shared.OptionInfo(False, "Place added settings into an accordion").needs_reload_ui()
}))


