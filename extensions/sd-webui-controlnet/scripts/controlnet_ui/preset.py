import os
import gradio as gr

from typing import Dict, List

from modules import scripts
from scripts.infotext import parse_unit, serialize_unit
from scripts.controlnet_ui.tool_button import ToolButton
from scripts.logging import logger
from scripts.processor import preprocessor_filters
from scripts import external_code

save_symbol = "\U0001f4be"  # 💾
delete_symbol = "\U0001f5d1\ufe0f"  # 🗑️
refresh_symbol = "\U0001f504"  # 🔄
reset_symbol = "\U000021A9"  # ↩

NEW_PRESET = "New Preset"


def load_presets(preset_dir: str) -> Dict[str, str]:
    if not os.path.exists(preset_dir):
        os.makedirs(preset_dir)
        return {}

    presets = {}
    for filename in os.listdir(preset_dir):
        if filename.endswith(".txt"):
            with open(os.path.join(preset_dir, filename), "r") as f:
                name = filename.replace(".txt", "")
                if name == NEW_PRESET:
                    continue
                presets[name] = f.read()
    return presets


def infer_control_type(module: str, model: str) -> str:
    def matches_control_type(input_string: str, control_type: str) -> bool:
        return any(t.lower() in input_string for t in control_type.split("/"))

    control_types = preprocessor_filters.keys()
    control_type_candidates = [
        control_type
        for control_type in control_types
        if (
            matches_control_type(module, control_type)
            or matches_control_type(model, control_type)
        )
    ]
    if len(control_type_candidates) != 1:
        raise ValueError(
            f"Unable to infer control type from module {module} and model {model}"
        )
    return control_type_candidates[0]


class ControlNetPresetUI(object):
    preset_directory = os.path.join(scripts.basedir(), "presets")
    presets = load_presets(preset_directory)

    def __init__(self, id_prefix: str):
        with gr.Row():
            self.dropdown = gr.Dropdown(
                label="Presets",
                show_label=True,
                elem_classes=["cnet-preset-dropdown"],
                choices=ControlNetPresetUI.dropdown_choices(),
                value=NEW_PRESET,
            )
            self.reset_button = ToolButton(
                value=reset_symbol,
                elem_classes=["cnet-preset-reset"],
                tooltip="Reset preset",
                visible=False,
            )
            self.save_button = ToolButton(
                value=save_symbol,
                elem_classes=["cnet-preset-save"],
                tooltip="Save preset",
            )
            self.delete_button = ToolButton(
                value=delete_symbol,
                elem_classes=["cnet-preset-delete"],
                tooltip="Delete preset",
            )
            self.refresh_button = ToolButton(
                value=refresh_symbol,
                elem_classes=["cnet-preset-refresh"],
                tooltip="Refresh preset",
            )

        with gr.Box(
            elem_classes=["popup-dialog", "cnet-preset-enter-name"],
            elem_id=f"{id_prefix}_cnet_preset_enter_name",
        ) as self.name_dialog:
            with gr.Row():
                self.preset_name = gr.Textbox(
                    label="Preset name",
                    show_label=True,
                    lines=1,
                    elem_classes=["cnet-preset-name"],
                )
                self.confirm_preset_name = ToolButton(
                    value=save_symbol,
                    elem_classes=["cnet-preset-confirm-name"],
                    tooltip="Save preset",
                )

    def register_callbacks(
        self,
        uigroup,
        control_type: gr.Radio,
        *ui_states,
    ):
        def apply_preset(name: str, control_type: str, *ui_states):
            if name == NEW_PRESET:
                return (
                    gr.update(visible=False),
                    *(
                        (gr.skip(),)
                        * (len(vars(external_code.ControlNetUnit()).keys()) + 1)
                    ),
                )

            assert name in ControlNetPresetUI.presets

            infotext = ControlNetPresetUI.presets[name]
            preset_unit = parse_unit(infotext)
            current_unit = external_code.ControlNetUnit(*ui_states)
            preset_unit.image = None
            current_unit.image = None

            # Do not compare module param that are not used in preset.
            for module_param in ("processor_res", "threshold_a", "threshold_b"):
                if getattr(preset_unit, module_param) == -1:
                    setattr(current_unit, module_param, -1)

            # No update necessary.
            if vars(current_unit) == vars(preset_unit):
                return (
                    gr.update(visible=False),
                    *(
                        (gr.skip(),)
                        * (len(vars(external_code.ControlNetUnit()).keys()) + 1)
                    ),
                )

            unit = preset_unit

            try:
                new_control_type = infer_control_type(unit.module, unit.model)
            except ValueError as e:
                logger.error(e)
                new_control_type = control_type

            if new_control_type != control_type:
                uigroup.prevent_next_n_module_update += 1

            if preset_unit.module != current_unit.module:
                uigroup.prevent_next_n_slider_value_update += 1

            if preset_unit.pixel_perfect != current_unit.pixel_perfect:
                uigroup.prevent_next_n_slider_value_update += 1

            return (
                gr.update(visible=True),
                gr.update(value=new_control_type),
                *[
                    gr.update(value=value) if value is not None else gr.update()
                    for value in vars(unit).values()
                ],
            )

        for element, action in (
            (self.dropdown, "change"),
            (self.reset_button, "click"),
        ):
            getattr(element, action)(
                fn=apply_preset,
                inputs=[self.dropdown, control_type, *ui_states],
                outputs=[self.delete_button, control_type, *ui_states],
                show_progress="hidden",
            ).then(
                fn=lambda: gr.update(visible=False),
                inputs=None,
                outputs=[self.reset_button],
            )

        def save_preset(name: str, *ui_states):
            if name == NEW_PRESET:
                return gr.update(visible=True), gr.update(), gr.update()

            ControlNetPresetUI.save_preset(
                name, external_code.ControlNetUnit(*ui_states)
            )
            return (
                gr.update(),  # name dialog
                gr.update(choices=ControlNetPresetUI.dropdown_choices(), value=name),
                gr.update(visible=False),  # Reset button
            )

        self.save_button.click(
            fn=save_preset,
            inputs=[self.dropdown, *ui_states],
            outputs=[self.name_dialog, self.dropdown, self.reset_button],
            show_progress="hidden",
        ).then(
            fn=None,
            _js=f"""
            (name) => {{
                if (name === "{NEW_PRESET}")
                    popup(gradioApp().getElementById('{self.name_dialog.elem_id}'));
            }}""",
            inputs=[self.dropdown],
        )

        def delete_preset(name: str):
            ControlNetPresetUI.delete_preset(name)
            return gr.Dropdown.update(
                choices=ControlNetPresetUI.dropdown_choices(),
                value=NEW_PRESET,
            ), gr.update(visible=False)

        self.delete_button.click(
            fn=delete_preset,
            inputs=[self.dropdown],
            outputs=[self.dropdown, self.reset_button],
            show_progress="hidden",
        )

        self.name_dialog.visible = False

        def save_new_preset(new_name: str, *ui_states):
            if new_name == NEW_PRESET:
                logger.warn(f"Cannot save preset with reserved name '{NEW_PRESET}'")
                return gr.update(visible=False), gr.update()

            ControlNetPresetUI.save_preset(
                new_name, external_code.ControlNetUnit(*ui_states)
            )
            return gr.update(visible=False), gr.update(
                choices=ControlNetPresetUI.dropdown_choices(), value=new_name
            )

        self.confirm_preset_name.click(
            fn=save_new_preset,
            inputs=[self.preset_name, *ui_states],
            outputs=[self.name_dialog, self.dropdown],
            show_progress="hidden",
        ).then(fn=None, _js="closePopup")

        self.refresh_button.click(
            fn=ControlNetPresetUI.refresh_preset,
            inputs=None,
            outputs=[self.dropdown],
            show_progress="hidden",
        )

        def update_reset_button(preset_name: str, *ui_states):
            if preset_name == NEW_PRESET:
                return gr.update(visible=False)

            infotext = ControlNetPresetUI.presets[preset_name]
            preset_unit = parse_unit(infotext)
            current_unit = external_code.ControlNetUnit(*ui_states)
            preset_unit.image = None
            current_unit.image = None

            # Do not compare module param that are not used in preset.
            for module_param in ("processor_res", "threshold_a", "threshold_b"):
                if getattr(preset_unit, module_param) == -1:
                    setattr(current_unit, module_param, -1)

            return gr.update(visible=vars(current_unit) != vars(preset_unit))

        for ui_state in ui_states:
            if isinstance(ui_state, gr.Image):
                continue

            for action in ("edit", "click", "change", "clear", "release"):
                if action == "release" and not isinstance(ui_state, gr.Slider):
                    continue

                if hasattr(ui_state, action):
                    getattr(ui_state, action)(
                        fn=update_reset_button,
                        inputs=[self.dropdown, *ui_states],
                        outputs=[self.reset_button],
                    )

    @staticmethod
    def dropdown_choices() -> List[str]:
        return list(ControlNetPresetUI.presets.keys()) + [NEW_PRESET]

    @staticmethod
    def save_preset(name: str, unit: external_code.ControlNetUnit):
        infotext = serialize_unit(unit)
        with open(
            os.path.join(ControlNetPresetUI.preset_directory, f"{name}.txt"), "w"
        ) as f:
            f.write(infotext)

        ControlNetPresetUI.presets[name] = infotext

    @staticmethod
    def delete_preset(name: str):
        if name not in ControlNetPresetUI.presets:
            return

        del ControlNetPresetUI.presets[name]

        file = os.path.join(ControlNetPresetUI.preset_directory, f"{name}.txt")
        if os.path.exists(file):
            os.unlink(file)

    @staticmethod
    def refresh_preset():
        ControlNetPresetUI.presets = load_presets(ControlNetPresetUI.preset_directory)
        return gr.update(choices=ControlNetPresetUI.dropdown_choices())
