import json
import gradio as gr
import functools
from copy import copy
from typing import List, Optional, Union, Callable, Dict, Tuple, Literal
from dataclasses import dataclass
import numpy as np

from scripts.utils import svg_preprocess, read_image
from scripts import (
    global_state,
    external_code,
)
from scripts.processor import (
    preprocessor_sliders_config,
    no_control_mode_preprocessors,
    flag_preprocessor_resolution,
    model_free_preprocessors,
    preprocessor_filters,
    HWC3,
)
from scripts.logging import logger
from scripts.controlnet_ui.openpose_editor import OpenposeEditor
from scripts.controlnet_ui.preset import ControlNetPresetUI
from scripts.controlnet_ui.tool_button import ToolButton
from scripts.controlnet_ui.photopea import Photopea
from scripts.enums import InputMode
from modules import shared
from modules.ui_components import FormRow


@dataclass
class A1111Context:
    """Contains all components from A1111."""

    img2img_batch_input_dir: Optional[gr.components.IOComponent] = None
    img2img_batch_output_dir: Optional[gr.components.IOComponent] = None
    txt2img_submit_button: Optional[gr.components.IOComponent] = None
    img2img_submit_button: Optional[gr.components.IOComponent] = None

    # Slider controls from A1111 WebUI.
    txt2img_w_slider: Optional[gr.components.IOComponent] = None
    txt2img_h_slider: Optional[gr.components.IOComponent] = None
    img2img_w_slider: Optional[gr.components.IOComponent] = None
    img2img_h_slider: Optional[gr.components.IOComponent] = None

    img2img_img2img_tab: Optional[gr.components.IOComponent] = None
    img2img_img2img_sketch_tab: Optional[gr.components.IOComponent] = None
    img2img_batch_tab: Optional[gr.components.IOComponent] = None
    img2img_inpaint_tab: Optional[gr.components.IOComponent] = None
    img2img_inpaint_sketch_tab: Optional[gr.components.IOComponent] = None
    img2img_inpaint_upload_tab: Optional[gr.components.IOComponent] = None

    img2img_inpaint_area: Optional[gr.components.IOComponent] = None
    # txt2img_enable_hr is only available for A1111 > 1.7.0.
    txt2img_enable_hr: Optional[gr.components.IOComponent] = None
    setting_sd_model_checkpoint: Optional[gr.components.IOComponent] = None

    @property
    def img2img_inpaint_tabs(self) -> Tuple[gr.components.IOComponent]:
        return (
            self.img2img_inpaint_tab,
            self.img2img_inpaint_sketch_tab,
            self.img2img_inpaint_upload_tab,
        )

    @property
    def img2img_non_inpaint_tabs(self) -> List[gr.components.IOComponent]:
        return (
            self.img2img_img2img_tab,
            self.img2img_img2img_sketch_tab,
            self.img2img_batch_tab,
        )

    @property
    def ui_initialized(self) -> bool:
        optional_components = {
            # Optional components are only available after A1111 v1.7.0.
            "img2img_img2img_tab": "img2img_img2img_tab",
            "img2img_img2img_sketch_tab": "img2img_img2img_sketch_tab",
            "img2img_batch_tab": "img2img_batch_tab",
            "img2img_inpaint_tab": "img2img_inpaint_tab",
            "img2img_inpaint_sketch_tab": "img2img_inpaint_sketch_tab",
            "img2img_inpaint_upload_tab": "img2img_inpaint_upload_tab",
            # SDNext does not have this field. Temporarily disable the callback on
            # the checkpoint change until we find a way to register an event when
            # all A1111 UI components are ready.
            "setting_sd_model_checkpoint": "setting_sd_model_checkpoint",
        }
        return all(
            c
            for name, c in vars(self).items()
            if name not in optional_components.values()
        )

    def set_component(self, component: gr.components.IOComponent):
        id_mapping = {
            "img2img_batch_input_dir": "img2img_batch_input_dir",
            "img2img_batch_output_dir": "img2img_batch_output_dir",
            "txt2img_generate": "txt2img_submit_button",
            "img2img_generate": "img2img_submit_button",
            "txt2img_width": "txt2img_w_slider",
            "txt2img_height": "txt2img_h_slider",
            "img2img_width": "img2img_w_slider",
            "img2img_height": "img2img_h_slider",
            "img2img_img2img_tab": "img2img_img2img_tab",
            "img2img_img2img_sketch_tab": "img2img_img2img_sketch_tab",
            "img2img_batch_tab": "img2img_batch_tab",
            "img2img_inpaint_tab": "img2img_inpaint_tab",
            "img2img_inpaint_sketch_tab": "img2img_inpaint_sketch_tab",
            "img2img_inpaint_upload_tab": "img2img_inpaint_upload_tab",
            "img2img_inpaint_full_res": "img2img_inpaint_area",
            "txt2img_hr-checkbox": "txt2img_enable_hr",
            # setting_sd_model_checkpoint is expected to be initialized last.
            # "setting_sd_model_checkpoint": "setting_sd_model_checkpoint",
        }
        elem_id = getattr(component, "elem_id", None)
        # Do not set component if it has already been set.
        # https://github.com/Mikubill/sd-webui-controlnet/issues/2587
        if elem_id in id_mapping and getattr(self, id_mapping[elem_id]) is None:
            setattr(self, id_mapping[elem_id], component)
            logger.debug(f"Setting {elem_id}.")
            logger.debug(
                f"A1111 initialized {sum(c is not None for c in vars(self).values())}/{len(vars(self).keys())}."
            )


class UiControlNetUnit(external_code.ControlNetUnit):
    """The data class that stores all states of a ControlNetUnit."""

    def __init__(
        self,
        input_mode: InputMode = InputMode.SIMPLE,
        batch_images: Optional[Union[str, List[external_code.InputImage]]] = None,
        output_dir: str = "",
        loopback: bool = False,
        merge_gallery_files: List[
            Dict[Union[Literal["name"], Literal["data"]], str]
        ] = [],
        use_preview_as_input: bool = False,
        generated_image: Optional[np.ndarray] = None,
        mask_image: Optional[np.ndarray] = None,
        enabled: bool = True,
        module: Optional[str] = None,
        model: Optional[str] = None,
        weight: float = 1.0,
        image: Optional[Dict[str, np.ndarray]] = None,
        *args,
        **kwargs,
    ):
        if use_preview_as_input and generated_image is not None:
            input_image = generated_image
            module = "none"
        else:
            input_image = image

        # Prefer uploaded mask_image over hand-drawn mask.
        if input_image is not None and mask_image is not None:
            assert isinstance(input_image, dict)
            input_image["mask"] = mask_image

        if merge_gallery_files and input_mode == InputMode.MERGE:
            input_image = [
                {"image": read_image(file["name"])} for file in merge_gallery_files
            ]

        super().__init__(enabled, module, model, weight, input_image, *args, **kwargs)
        self.is_ui = True
        self.input_mode = input_mode
        self.batch_images = batch_images
        self.output_dir = output_dir
        self.loopback = loopback

    def unfold_merged(self) -> List[external_code.ControlNetUnit]:
        """Unfolds a merged unit to multiple units. Keeps the unit merged for
        preprocessors that can accept multiple input images.
        """
        if self.input_mode != InputMode.MERGE:
            return [copy(self)]

        if self.accepts_multiple_inputs():
            self.input_mode = InputMode.SIMPLE
            return [copy(self)]

        assert isinstance(self.image, list)
        result = []
        for image in self.image:
            unit = copy(self)
            unit.image = image["image"]
            unit.input_mode = InputMode.SIMPLE
            unit.weight = self.weight / len(self.image)
            result.append(unit)
        return result


class ControlNetUiGroup(object):
    refresh_symbol = "\U0001f504"  # 🔄
    switch_values_symbol = "\U000021C5"  # ⇅
    camera_symbol = "\U0001F4F7"  # 📷
    reverse_symbol = "\U000021C4"  # ⇄
    tossup_symbol = "\u2934"
    trigger_symbol = "\U0001F4A5"  # 💥
    open_symbol = "\U0001F4DD"  # 📝

    tooltips = {
        "🔄": "Refresh",
        "\u2934": "Send dimensions to stable diffusion",
        "💥": "Run preprocessor",
        "📝": "Open new canvas",
        "📷": "Enable webcam",
        "⇄": "Mirror webcam",
    }

    global_batch_input_dir = gr.Textbox(
        label="Controlnet input directory",
        placeholder="Leave empty to use input directory",
        **shared.hide_dirs,
        elem_id="controlnet_batch_input_dir",
    )
    a1111_context = A1111Context()
    # All ControlNetUiGroup instances created.
    all_ui_groups: List["ControlNetUiGroup"] = []

    def __init__(
        self,
        is_img2img: bool,
        default_unit: external_code.ControlNetUnit,
        preprocessors: List[Callable],
        photopea: Optional[Photopea],
    ):
        # Whether callbacks have been registered.
        self.callbacks_registered: bool = False
        # Whether the render method on this object has been called.
        self.ui_initialized: bool = False

        self.is_img2img = is_img2img
        self.default_unit = default_unit
        self.preprocessors = preprocessors
        self.photopea = photopea
        self.webcam_enabled = False
        self.webcam_mirrored = False

        # Note: All gradio elements declared in `render` will be defined as member variable.
        # Update counter to trigger a force update of UiControlNetUnit.
        # This is useful when a field with no event subscriber available changes.
        # e.g. gr.Gallery, gr.State, etc.
        self.update_unit_counter = None
        self.upload_tab = None
        self.image = None
        self.generated_image_group = None
        self.generated_image = None
        self.mask_image_group = None
        self.mask_image = None
        self.batch_tab = None
        self.batch_image_dir = None
        self.merge_tab = None
        self.merge_gallery = None
        self.merge_upload_button = None
        self.merge_clear_button = None
        self.create_canvas = None
        self.canvas_width = None
        self.canvas_height = None
        self.canvas_create_button = None
        self.canvas_cancel_button = None
        self.open_new_canvas_button = None
        self.webcam_enable = None
        self.webcam_mirror = None
        self.send_dimen_button = None
        self.enabled = None
        self.low_vram = None
        self.pixel_perfect = None
        self.preprocessor_preview = None
        self.mask_upload = None
        self.type_filter = None
        self.module = None
        self.trigger_preprocessor = None
        self.model = None
        self.refresh_models = None
        self.weight = None
        self.guidance_start = None
        self.guidance_end = None
        self.advanced = None
        self.processor_res = None
        self.threshold_a = None
        self.threshold_b = None
        self.control_mode = None
        self.resize_mode = None
        self.loopback = None
        self.use_preview_as_input = None
        self.openpose_editor = None
        self.preset_panel = None
        self.upload_independent_img_in_img2img = None
        self.image_upload_panel = None
        self.save_detected_map = None
        self.input_mode = gr.State(InputMode.SIMPLE)
        self.inpaint_crop_input_image = None
        self.hr_option = None
        self.batch_image_dir_state = None
        self.output_dir_state = None

        # Internal states for UI state pasting.
        self.prevent_next_n_module_update = 0
        self.prevent_next_n_slider_value_update = 0

        # API-only fields
        self.advanced_weighting = gr.State(None)

        ControlNetUiGroup.all_ui_groups.append(self)

    def render(self, tabname: str, elem_id_tabname: str) -> None:
        """The pure HTML structure of a single ControlNetUnit. Calling this
        function will populate `self` with all gradio element declared
        in local scope.

        Args:
            tabname:
            elem_id_tabname:

        Returns:
            None
        """
        self.update_unit_counter = gr.Number(value=0, visible=False)
        self.openpose_editor = OpenposeEditor()

        with gr.Group(visible=not self.is_img2img) as self.image_upload_panel:
            self.save_detected_map = gr.Checkbox(value=True, visible=False)
            with gr.Tabs():
                with gr.Tab(label="Single Image") as self.upload_tab:
                    with gr.Row(elem_classes=["cnet-image-row"], equal_height=True):
                        with gr.Group(elem_classes=["cnet-input-image-group"]):
                            self.image = gr.Image(
                                source="upload",
                                brush_radius=20,
                                mirror_webcam=False,
                                type="numpy",
                                tool="sketch",
                                elem_id=f"{elem_id_tabname}_{tabname}_input_image",
                                elem_classes=["cnet-image"],
                                brush_color=shared.opts.img2img_inpaint_mask_brush_color
                                if hasattr(
                                    shared.opts, "img2img_inpaint_mask_brush_color"
                                )
                                else None,
                            )
                            self.image.preprocess = functools.partial(
                                svg_preprocess, preprocess=self.image.preprocess
                            )
                            self.openpose_editor.render_upload()

                        with gr.Group(
                            visible=False, elem_classes=["cnet-generated-image-group"]
                        ) as self.generated_image_group:
                            self.generated_image = gr.Image(
                                value=None,
                                label="Preprocessor Preview",
                                elem_id=f"{elem_id_tabname}_{tabname}_generated_image",
                                elem_classes=["cnet-image"],
                                interactive=True,
                                height=242,
                            )  # Gradio's magic number. Only 242 works.

                            with gr.Group(
                                elem_classes=["cnet-generated-image-control-group"]
                            ):
                                if self.photopea:
                                    self.photopea.render_child_trigger()
                                self.openpose_editor.render_edit()
                                preview_check_elem_id = f"{elem_id_tabname}_{tabname}_controlnet_preprocessor_preview_checkbox"
                                preview_close_button_js = f"document.querySelector('#{preview_check_elem_id} input[type=\\'checkbox\\']').click();"
                                gr.HTML(
                                    value=f"""<a title="Close Preview" onclick="{preview_close_button_js}">Close</a>""",
                                    visible=True,
                                    elem_classes=["cnet-close-preview"],
                                )

                        with gr.Group(
                            visible=False, elem_classes=["cnet-mask-image-group"]
                        ) as self.mask_image_group:
                            self.mask_image = gr.Image(
                                value=None,
                                label="Upload Mask",
                                elem_id=f"{elem_id_tabname}_{tabname}_mask_image",
                                elem_classes=["cnet-mask-image"],
                                interactive=True,
                            )

                with gr.Tab(label="Batch") as self.batch_tab:
                    self.batch_image_dir = gr.Textbox(
                        label="Input Directory",
                        placeholder="Leave empty to use img2img batch controlnet input directory",
                        elem_id=f"{elem_id_tabname}_{tabname}_batch_image_dir",
                    )

                with gr.Tab(label="Multi-Inputs") as self.merge_tab:
                    self.merge_gallery = gr.Gallery(
                        columns=[4], rows=[2], object_fit="contain", height="auto"
                    )
                    with gr.Row():
                        self.merge_upload_button = gr.UploadButton(
                            "Upload Images",
                            file_types=["image"],
                            file_count="multiple",
                        )
                        self.merge_clear_button = gr.Button("Clear Images")

            if self.photopea:
                self.photopea.attach_photopea_output(self.generated_image)

            with gr.Accordion(
                label="Open New Canvas", visible=False
            ) as self.create_canvas:
                self.canvas_width = gr.Slider(
                    label="New Canvas Width",
                    minimum=256,
                    maximum=1024,
                    value=512,
                    step=64,
                    elem_id=f"{elem_id_tabname}_{tabname}_controlnet_canvas_width",
                )
                self.canvas_height = gr.Slider(
                    label="New Canvas Height",
                    minimum=256,
                    maximum=1024,
                    value=512,
                    step=64,
                    elem_id=f"{elem_id_tabname}_{tabname}_controlnet_canvas_height",
                )
                with gr.Row():
                    self.canvas_create_button = gr.Button(
                        value="Create New Canvas",
                        elem_id=f"{elem_id_tabname}_{tabname}_controlnet_canvas_create_button",
                    )
                    self.canvas_cancel_button = gr.Button(
                        value="Cancel",
                        elem_id=f"{elem_id_tabname}_{tabname}_controlnet_canvas_cancel_button",
                    )

            with gr.Row(elem_classes="controlnet_image_controls"):
                gr.HTML(
                    value="<p>Set the preprocessor to [invert] If your image has white background and black lines.</p>",
                    elem_classes="controlnet_invert_warning",
                )
                self.open_new_canvas_button = ToolButton(
                    value=ControlNetUiGroup.open_symbol,
                    elem_id=f"{elem_id_tabname}_{tabname}_controlnet_open_new_canvas_button",
                    tooltip=ControlNetUiGroup.tooltips[ControlNetUiGroup.open_symbol],
                )
                self.webcam_enable = ToolButton(
                    value=ControlNetUiGroup.camera_symbol,
                    elem_id=f"{elem_id_tabname}_{tabname}_controlnet_webcam_enable",
                    tooltip=ControlNetUiGroup.tooltips[ControlNetUiGroup.camera_symbol],
                )
                self.webcam_mirror = ToolButton(
                    value=ControlNetUiGroup.reverse_symbol,
                    elem_id=f"{elem_id_tabname}_{tabname}_controlnet_webcam_mirror",
                    tooltip=ControlNetUiGroup.tooltips[
                        ControlNetUiGroup.reverse_symbol
                    ],
                )
                self.send_dimen_button = ToolButton(
                    value=ControlNetUiGroup.tossup_symbol,
                    elem_id=f"{elem_id_tabname}_{tabname}_controlnet_send_dimen_button",
                    tooltip=ControlNetUiGroup.tooltips[ControlNetUiGroup.tossup_symbol],
                )

        with FormRow(elem_classes=["controlnet_main_options"]):
            self.enabled = gr.Checkbox(
                label="Enable",
                value=self.default_unit.enabled,
                elem_id=f"{elem_id_tabname}_{tabname}_controlnet_enable_checkbox",
                elem_classes=["cnet-unit-enabled"],
            )
            self.low_vram = gr.Checkbox(
                label="Low VRAM",
                value=self.default_unit.low_vram,
                elem_id=f"{elem_id_tabname}_{tabname}_controlnet_low_vram_checkbox",
            )
            self.pixel_perfect = gr.Checkbox(
                label="Pixel Perfect",
                value=self.default_unit.pixel_perfect,
                elem_id=f"{elem_id_tabname}_{tabname}_controlnet_pixel_perfect_checkbox",
            )
            self.preprocessor_preview = gr.Checkbox(
                label="Allow Preview",
                value=False,
                elem_classes=["cnet-allow-preview"],
                elem_id=preview_check_elem_id,
                visible=not self.is_img2img,
            )
            self.mask_upload = gr.Checkbox(
                label="Mask Upload",
                value=False,
                elem_classes=["cnet-mask-upload"],
                elem_id=f"{elem_id_tabname}_{tabname}_controlnet_mask_upload_checkbox",
                visible=not self.is_img2img,
            )
            self.use_preview_as_input = gr.Checkbox(
                label="Preview as Input",
                value=False,
                elem_classes=["cnet-preview-as-input"],
                visible=False,
            )

        with gr.Row(elem_classes="controlnet_img2img_options"):
            if self.is_img2img:
                self.upload_independent_img_in_img2img = gr.Checkbox(
                    label="Upload independent control image",
                    value=False,
                    elem_id=f"{elem_id_tabname}_{tabname}_controlnet_same_img2img_checkbox",
                    elem_classes=["cnet-unit-same_img2img"],
                )
            else:
                self.upload_independent_img_in_img2img = None

            # Note: The checkbox needs to exist for both img2img and txt2img as infotext
            # needs the checkbox value.
            self.inpaint_crop_input_image = gr.Checkbox(
                label="Crop input image based on A1111 mask",
                value=False,
                elem_classes=["cnet-crop-input-image"],
                visible=False,
            )

        with gr.Row(elem_classes=["controlnet_control_type", "controlnet_row"]):
            self.type_filter = gr.Radio(
                list(preprocessor_filters.keys()),
                label=f"Control Type",
                value="All",
                elem_id=f"{elem_id_tabname}_{tabname}_controlnet_type_filter_radio",
                elem_classes="controlnet_control_type_filter_group",
            )

        with gr.Row(elem_classes=["controlnet_preprocessor_model", "controlnet_row"]):
            self.module = gr.Dropdown(
                global_state.ui_preprocessor_keys,
                label=f"Preprocessor",
                value=self.default_unit.module,
                elem_id=f"{elem_id_tabname}_{tabname}_controlnet_preprocessor_dropdown",
            )
            self.trigger_preprocessor = ToolButton(
                value=ControlNetUiGroup.trigger_symbol,
                visible=not self.is_img2img,
                elem_id=f"{elem_id_tabname}_{tabname}_controlnet_trigger_preprocessor",
                elem_classes=["cnet-run-preprocessor"],
                tooltip=ControlNetUiGroup.tooltips[ControlNetUiGroup.trigger_symbol],
            )
            self.model = gr.Dropdown(
                list(global_state.cn_models.keys()),
                label=f"Model",
                value=self.default_unit.model,
                elem_id=f"{elem_id_tabname}_{tabname}_controlnet_model_dropdown",
            )
            self.refresh_models = ToolButton(
                value=ControlNetUiGroup.refresh_symbol,
                elem_id=f"{elem_id_tabname}_{tabname}_controlnet_refresh_models",
                tooltip=ControlNetUiGroup.tooltips[ControlNetUiGroup.refresh_symbol],
            )

        with gr.Row(elem_classes=["controlnet_weight_steps", "controlnet_row"]):
            self.weight = gr.Slider(
                label=f"Control Weight",
                value=self.default_unit.weight,
                minimum=0.0,
                maximum=2.0,
                step=0.05,
                elem_id=f"{elem_id_tabname}_{tabname}_controlnet_control_weight_slider",
                elem_classes="controlnet_control_weight_slider",
            )
            self.guidance_start = gr.Slider(
                label="Starting Control Step",
                value=self.default_unit.guidance_start,
                minimum=0.0,
                maximum=1.0,
                interactive=True,
                elem_id=f"{elem_id_tabname}_{tabname}_controlnet_start_control_step_slider",
                elem_classes="controlnet_start_control_step_slider",
            )
            self.guidance_end = gr.Slider(
                label="Ending Control Step",
                value=self.default_unit.guidance_end,
                minimum=0.0,
                maximum=1.0,
                interactive=True,
                elem_id=f"{elem_id_tabname}_{tabname}_controlnet_ending_control_step_slider",
                elem_classes="controlnet_ending_control_step_slider",
            )

        # advanced options
        with gr.Column(visible=False) as self.advanced:
            self.processor_res = gr.Slider(
                label="Preprocessor resolution",
                value=self.default_unit.processor_res,
                minimum=64,
                maximum=2048,
                visible=False,
                interactive=True,
                elem_id=f"{elem_id_tabname}_{tabname}_controlnet_preprocessor_resolution_slider",
            )
            self.threshold_a = gr.Slider(
                label="Threshold A",
                value=self.default_unit.threshold_a,
                minimum=64,
                maximum=1024,
                visible=False,
                interactive=True,
                elem_id=f"{elem_id_tabname}_{tabname}_controlnet_threshold_A_slider",
            )
            self.threshold_b = gr.Slider(
                label="Threshold B",
                value=self.default_unit.threshold_b,
                minimum=64,
                maximum=1024,
                visible=False,
                interactive=True,
                elem_id=f"{elem_id_tabname}_{tabname}_controlnet_threshold_B_slider",
            )

        self.control_mode = gr.Radio(
            choices=[e.value for e in external_code.ControlMode],
            value=self.default_unit.control_mode.value,
            label="Control Mode",
            elem_id=f"{elem_id_tabname}_{tabname}_controlnet_control_mode_radio",
            elem_classes="controlnet_control_mode_radio",
        )

        self.resize_mode = gr.Radio(
            choices=[e.value for e in external_code.ResizeMode],
            value=self.default_unit.resize_mode.value,
            label="Resize Mode",
            elem_id=f"{elem_id_tabname}_{tabname}_controlnet_resize_mode_radio",
            elem_classes="controlnet_resize_mode_radio",
            visible=not self.is_img2img,
        )

        self.hr_option = gr.Radio(
            choices=[e.value for e in external_code.HiResFixOption],
            value=self.default_unit.hr_option.value,
            label="Hires-Fix Option",
            elem_id=f"{elem_id_tabname}_{tabname}_controlnet_hr_option_radio",
            elem_classes="controlnet_hr_option_radio",
            visible=False,
        )

        self.loopback = gr.Checkbox(
            label="[Batch Loopback] Automatically send generated images to this ControlNet unit in batch generation",
            value=self.default_unit.loopback,
            elem_id=f"{elem_id_tabname}_{tabname}_controlnet_automatically_send_generated_images_checkbox",
            elem_classes="controlnet_loopback_checkbox",
            visible=False,
        )

        self.preset_panel = ControlNetPresetUI(
            id_prefix=f"{elem_id_tabname}_{tabname}_"
        )

        self.batch_image_dir_state = gr.State("")
        self.output_dir_state = gr.State("")
        unit_args = (
            self.input_mode,
            self.batch_image_dir_state,
            self.output_dir_state,
            self.loopback,
            # Non-persistent fields.
            # Following inputs will not be persistent on `ControlNetUnit`.
            # They are only used during object construction.
            self.merge_gallery,
            self.use_preview_as_input,
            self.generated_image,
            self.mask_image,
            # End of Non-persistent fields.
            self.enabled,
            self.module,
            self.model,
            self.weight,
            self.image,
            self.resize_mode,
            self.low_vram,
            self.processor_res,
            self.threshold_a,
            self.threshold_b,
            self.guidance_start,
            self.guidance_end,
            self.pixel_perfect,
            self.control_mode,
            self.inpaint_crop_input_image,
            self.hr_option,
        )

        unit = gr.State(self.default_unit)
        for comp in unit_args + (self.update_unit_counter,):
            event_subscribers = []
            if hasattr(comp, "edit"):
                event_subscribers.append(comp.edit)
            elif hasattr(comp, "click"):
                event_subscribers.append(comp.click)
            elif isinstance(comp, gr.Slider) and hasattr(comp, "release"):
                event_subscribers.append(comp.release)
            elif hasattr(comp, "change"):
                event_subscribers.append(comp.change)

            if hasattr(comp, "clear"):
                event_subscribers.append(comp.clear)

            for event_subscriber in event_subscribers:
                event_subscriber(
                    fn=UiControlNetUnit, inputs=list(unit_args), outputs=unit
                )

        (
            ControlNetUiGroup.a1111_context.img2img_submit_button
            if self.is_img2img
            else ControlNetUiGroup.a1111_context.txt2img_submit_button
        ).click(
            fn=UiControlNetUnit,
            inputs=list(unit_args),
            outputs=unit,
            queue=False,
        )
        self.register_core_callbacks()
        self.ui_initialized = True
        return unit

    def register_send_dimensions(self):
        """Register event handler for send dimension button."""

        def send_dimensions(image):
            def closesteight(num):
                rem = num % 8
                if rem <= 4:
                    return round(num - rem)
                else:
                    return round(num + (8 - rem))

            if image:
                interm = np.asarray(image.get("image"))
                return closesteight(interm.shape[1]), closesteight(interm.shape[0])
            else:
                return gr.Slider.update(), gr.Slider.update()

        outputs = (
            [
                ControlNetUiGroup.a1111_context.img2img_w_slider,
                ControlNetUiGroup.a1111_context.img2img_h_slider,
            ]
            if self.is_img2img
            else [
                ControlNetUiGroup.a1111_context.txt2img_w_slider,
                ControlNetUiGroup.a1111_context.txt2img_h_slider,
            ]
        )
        self.send_dimen_button.click(
            fn=send_dimensions,
            inputs=[self.image],
            outputs=outputs,
            show_progress=False,
        )

    def register_webcam_toggle(self):
        def webcam_toggle():
            self.webcam_enabled = not self.webcam_enabled
            return {
                "value": None,
                "source": "webcam" if self.webcam_enabled else "upload",
                "__type__": "update",
            }

        self.webcam_enable.click(
            webcam_toggle, inputs=None, outputs=self.image, show_progress=False
        )

    def register_webcam_mirror_toggle(self):
        def webcam_mirror_toggle():
            self.webcam_mirrored = not self.webcam_mirrored
            return {"mirror_webcam": self.webcam_mirrored, "__type__": "update"}

        self.webcam_mirror.click(
            webcam_mirror_toggle, inputs=None, outputs=self.image, show_progress=False
        )

    def register_refresh_all_models(self):
        def refresh_all_models(model: str):
            global_state.update_cn_models()
            choices = list(global_state.cn_models.keys())
            return gr.Dropdown.update(
                value=model if model in global_state.cn_models else "None",
                choices=choices,
            )

        self.refresh_models.click(
            refresh_all_models,
            inputs=[self.model],
            outputs=[self.model],
            show_progress=False,
        )

    def register_build_sliders(self):
        def build_sliders(module: str, pp: bool):
            logger.debug(
                f"Prevent update slider value: {self.prevent_next_n_slider_value_update}"
            )
            logger.debug(f"Build slider for module: {module} - {pp}")

            # Clear old slider values so that they do not cause confusion in
            # infotext.
            clear_slider_update = gr.update(
                visible=False,
                interactive=True,
                minimum=-1,
                maximum=-1,
                value=-1,
            )

            grs = []
            module = global_state.get_module_basename(module)
            if module not in preprocessor_sliders_config:
                default_res_slider_config = dict(
                    label=flag_preprocessor_resolution,
                    minimum=64,
                    maximum=2048,
                    step=1,
                )
                if self.prevent_next_n_slider_value_update == 0:
                    default_res_slider_config["value"] = 512

                grs += [
                    gr.update(
                        **default_res_slider_config,
                        visible=not pp,
                        interactive=True,
                    ),
                    copy(clear_slider_update),
                    copy(clear_slider_update),
                    gr.update(visible=True),
                ]
            else:
                for slider_config in preprocessor_sliders_config[module]:
                    if isinstance(slider_config, dict):
                        visible = True
                        if slider_config["name"] == flag_preprocessor_resolution:
                            visible = not pp
                        slider_update = gr.update(
                            label=slider_config["name"],
                            minimum=slider_config["min"],
                            maximum=slider_config["max"],
                            step=slider_config["step"]
                            if "step" in slider_config
                            else 1,
                            visible=visible,
                            interactive=True,
                        )
                        if self.prevent_next_n_slider_value_update == 0:
                            slider_update["value"] = slider_config["value"]

                        grs.append(slider_update)

                    else:
                        grs.append(copy(clear_slider_update))
                while len(grs) < 3:
                    grs.append(copy(clear_slider_update))
                grs.append(gr.update(visible=True))
            if module in model_free_preprocessors:
                grs += [
                    gr.update(visible=False, value="None"),
                    gr.update(visible=False),
                ]
            else:
                grs += [gr.update(visible=True), gr.update(visible=True)]

            self.prevent_next_n_slider_value_update = max(
                0, self.prevent_next_n_slider_value_update - 1
            )

            grs += [gr.update(visible=module not in no_control_mode_preprocessors)]

            return grs

        inputs = [
            self.module,
            self.pixel_perfect,
        ]
        outputs = [
            self.processor_res,
            self.threshold_a,
            self.threshold_b,
            self.advanced,
            self.model,
            self.refresh_models,
            self.control_mode,
        ]
        self.module.change(
            build_sliders, inputs=inputs, outputs=outputs, show_progress=False
        )
        self.pixel_perfect.change(
            build_sliders, inputs=inputs, outputs=outputs, show_progress=False
        )

        def filter_selected(k: str):
            logger.debug(f"Prevent update {self.prevent_next_n_module_update}")
            logger.debug(f"Switch to control type {k}")
            (
                filtered_preprocessor_list,
                filtered_model_list,
                default_option,
                default_model,
            ) = global_state.select_control_type(k, global_state.get_sd_version())

            if self.prevent_next_n_module_update > 0:
                self.prevent_next_n_module_update -= 1
                return [
                    gr.Dropdown.update(choices=filtered_preprocessor_list),
                    gr.Dropdown.update(choices=filtered_model_list),
                ]
            else:
                return [
                    gr.Dropdown.update(
                        value=default_option, choices=filtered_preprocessor_list
                    ),
                    gr.Dropdown.update(
                        value=default_model, choices=filtered_model_list
                    ),
                ]

        self.type_filter.change(
            fn=filter_selected,
            inputs=[self.type_filter],
            outputs=[self.module, self.model],
            show_progress=False,
        )

    def register_sd_version_changed(self):
        def sd_version_changed(type_filter: str, current_model: str):
            """When SD version changes, update model dropdown choices."""
            (
                filtered_preprocessor_list,
                filtered_model_list,
                default_option,
                default_model,
            ) = global_state.select_control_type(
                type_filter, global_state.get_sd_version()
            )

            if current_model in filtered_model_list:
                return gr.update()

            return gr.Dropdown.update(
                value=default_model,
                choices=filtered_model_list,
            )

        if ControlNetUiGroup.a1111_context.setting_sd_model_checkpoint:
            ControlNetUiGroup.a1111_context.setting_sd_model_checkpoint.change(
                fn=sd_version_changed,
                inputs=[self.type_filter, self.model],
                outputs=[self.model],
                show_progress=False,
            )

    def register_run_annotator(self):
        def run_annotator(image, module, pres, pthr_a, pthr_b, t2i_w, t2i_h, pp, rm):
            if image is None:
                return (
                    gr.update(value=None, visible=True),
                    gr.update(),
                    *self.openpose_editor.update(""),
                )

            img = HWC3(image["image"])
            has_mask = not (
                (image["mask"][:, :, 0] <= 5).all()
                or (image["mask"][:, :, 0] >= 250).all()
            )
            if "inpaint" in module:
                color = HWC3(image["image"])
                alpha = image["mask"][:, :, 0:1]
                img = np.concatenate([color, alpha], axis=2)
            elif has_mask and not shared.opts.data.get(
                "controlnet_ignore_noninpaint_mask", False
            ):
                img = HWC3(image["mask"][:, :, 0])

            module = global_state.get_module_basename(module)
            preprocessor = self.preprocessors[module]

            if pp:
                pres = external_code.pixel_perfect_resolution(
                    img,
                    target_H=t2i_h,
                    target_W=t2i_w,
                    resize_mode=external_code.resize_mode_from_value(rm),
                )

            class JsonAcceptor:
                def __init__(self) -> None:
                    self.value = ""

                def accept(self, json_dict: dict) -> None:
                    self.value = json.dumps(json_dict)

            json_acceptor = JsonAcceptor()

            logger.info(f"Preview Resolution = {pres}")

            def is_openpose(module: str):
                return "openpose" in module

            # Only openpose preprocessor returns a JSON output, pass json_acceptor
            # only when a JSON output is expected. This will make preprocessor cache
            # work for all other preprocessors other than openpose ones. JSON acceptor
            # instance are different every call, which means cache will never take
            # effect.
            # TODO: Maybe we should let `preprocessor` return a Dict to alleviate this issue?
            # This requires changing all callsites though.
            result, is_image = preprocessor(
                img,
                res=pres,
                thr_a=pthr_a,
                thr_b=pthr_b,
                low_vram=(
                    ("clip" in module or module == "ip-adapter_face_id_plus")
                    and shared.opts.data.get("controlnet_clip_detector_on_cpu", False)
                ),
                json_pose_callback=json_acceptor.accept
                if is_openpose(module)
                else None,
            )

            if not is_image:
                result = img
                is_image = True

            result = external_code.visualize_inpaint_mask(result)
            return (
                # Update to `generated_image`
                gr.update(value=result, visible=True, interactive=False),
                # preprocessor_preview
                gr.update(value=True),
                # openpose editor
                *self.openpose_editor.update(json_acceptor.value),
            )

        self.trigger_preprocessor.click(
            fn=run_annotator,
            inputs=[
                self.image,
                self.module,
                self.processor_res,
                self.threshold_a,
                self.threshold_b,
                ControlNetUiGroup.a1111_context.img2img_w_slider
                if self.is_img2img
                else ControlNetUiGroup.a1111_context.txt2img_w_slider,
                ControlNetUiGroup.a1111_context.img2img_h_slider
                if self.is_img2img
                else ControlNetUiGroup.a1111_context.txt2img_h_slider,
                self.pixel_perfect,
                self.resize_mode,
            ],
            outputs=[
                self.generated_image,
                self.preprocessor_preview,
                *self.openpose_editor.outputs(),
            ],
        )

    def register_shift_preview(self):
        def shift_preview(is_on):
            return (
                # generated_image
                gr.update() if is_on else gr.update(value=None),
                # generated_image_group
                gr.update(visible=is_on),
                # use_preview_as_input,
                gr.update(visible=False),  # Now this is automatically managed
                # download_pose_link
                gr.update() if is_on else gr.update(value=None),
                # modal edit button
                gr.update() if is_on else gr.update(visible=False),
            )

        self.preprocessor_preview.change(
            fn=shift_preview,
            inputs=[self.preprocessor_preview],
            outputs=[
                self.generated_image,
                self.generated_image_group,
                self.use_preview_as_input,
                self.openpose_editor.download_link,
                self.openpose_editor.modal,
            ],
            show_progress=False,
        )

    def register_create_canvas(self):
        self.open_new_canvas_button.click(
            lambda: gr.Accordion.update(visible=True),
            inputs=None,
            outputs=self.create_canvas,
            show_progress=False,
        )
        self.canvas_cancel_button.click(
            lambda: gr.Accordion.update(visible=False),
            inputs=None,
            outputs=self.create_canvas,
            show_progress=False,
        )

        def fn_canvas(h, w):
            return np.zeros(shape=(h, w, 3), dtype=np.uint8) + 255, gr.Accordion.update(
                visible=False
            )

        self.canvas_create_button.click(
            fn=fn_canvas,
            inputs=[self.canvas_height, self.canvas_width],
            outputs=[self.image, self.create_canvas],
            show_progress=False,
        )

    def register_img2img_same_input(self):
        def fn_same_checked(x):
            return [
                gr.update(value=None),
                gr.update(value=None),
                gr.update(value=False, visible=x),
            ] + [gr.update(visible=x)] * 4

        self.upload_independent_img_in_img2img.change(
            fn_same_checked,
            inputs=self.upload_independent_img_in_img2img,
            outputs=[
                self.image,
                self.batch_image_dir,
                self.preprocessor_preview,
                self.image_upload_panel,
                self.trigger_preprocessor,
                self.loopback,
                self.resize_mode,
            ],
            show_progress=False,
        )

    def register_shift_crop_input_image(self):
        # A1111 < 1.7.0 compatibility.
        if any(c is None for c in ControlNetUiGroup.a1111_context.img2img_inpaint_tabs):
            self.inpaint_crop_input_image.visible = True
            self.inpaint_crop_input_image.value = True
            return

        is_inpaint_tab = gr.State(False)

        def shift_crop_input_image(is_inpaint: bool, inpaint_area: int):
            # Note: inpaint_area (0: Whole picture, 1: Only masked)
            # By default set value to True, as most preprocessors need cropped result.
            return gr.update(value=True, visible=is_inpaint and inpaint_area == 1)

        gradio_kwargs = dict(
            fn=shift_crop_input_image,
            inputs=[
                is_inpaint_tab,
                ControlNetUiGroup.a1111_context.img2img_inpaint_area,
            ],
            outputs=[self.inpaint_crop_input_image],
            show_progress=False,
        )

        for elem in ControlNetUiGroup.a1111_context.img2img_inpaint_tabs:
            elem.select(fn=lambda: True, inputs=[], outputs=[is_inpaint_tab]).then(
                **gradio_kwargs
            )

        for elem in ControlNetUiGroup.a1111_context.img2img_non_inpaint_tabs:
            elem.select(fn=lambda: False, inputs=[], outputs=[is_inpaint_tab]).then(
                **gradio_kwargs
            )

        ControlNetUiGroup.a1111_context.img2img_inpaint_area.change(**gradio_kwargs)

    def register_shift_hr_options(self):
        # A1111 version < 1.6.0.
        if not ControlNetUiGroup.a1111_context.txt2img_enable_hr:
            return

        ControlNetUiGroup.a1111_context.txt2img_enable_hr.change(
            fn=lambda checked: gr.update(visible=checked),
            inputs=[ControlNetUiGroup.a1111_context.txt2img_enable_hr],
            outputs=[self.hr_option],
            show_progress=False,
        )

    def register_shift_upload_mask(self):
        """Controls whether the upload mask input should be visible."""
        self.mask_upload.change(
            fn=lambda checked: (
                # Clear mask_image if unchecked.
                (gr.update(visible=False), gr.update(value=None))
                if not checked
                else (gr.update(visible=True), gr.update())
            ),
            inputs=[self.mask_upload],
            outputs=[self.mask_image_group, self.mask_image],
            show_progress=False,
        )

        if self.upload_independent_img_in_img2img is not None:
            self.upload_independent_img_in_img2img.change(
                fn=lambda checked: (
                    # Uncheck `upload_mask` when not using independent input.
                    gr.update(visible=False, value=False)
                    if not checked
                    else gr.update(visible=True)
                ),
                inputs=[self.upload_independent_img_in_img2img],
                outputs=[self.mask_upload],
                show_progress=False,
            )

    def register_sync_batch_dir(self):
        def determine_batch_dir(batch_dir, fallback_dir, fallback_fallback_dir):
            if batch_dir:
                return batch_dir
            elif fallback_dir:
                return fallback_dir
            else:
                return fallback_fallback_dir

        batch_dirs = [
            self.batch_image_dir,
            ControlNetUiGroup.global_batch_input_dir,
            ControlNetUiGroup.a1111_context.img2img_batch_input_dir,
        ]
        for batch_dir_comp in batch_dirs:
            subscriber = getattr(batch_dir_comp, "blur", None)
            if subscriber is None:
                continue
            subscriber(
                fn=determine_batch_dir,
                inputs=batch_dirs,
                outputs=[self.batch_image_dir_state],
                queue=False,
            )

        ControlNetUiGroup.a1111_context.img2img_batch_output_dir.blur(
            fn=lambda a: a,
            inputs=[ControlNetUiGroup.a1111_context.img2img_batch_output_dir],
            outputs=[self.output_dir_state],
            queue=False,
        )

    def register_clear_preview(self):
        def clear_preview(x):
            if x:
                logger.info("Preview as input is cancelled.")
            return gr.update(value=False), gr.update(value=None)

        for comp in (
            self.pixel_perfect,
            self.module,
            self.image,
            self.processor_res,
            self.threshold_a,
            self.threshold_b,
            self.upload_independent_img_in_img2img,
        ):
            event_subscribers = []
            if hasattr(comp, "edit"):
                event_subscribers.append(comp.edit)
            elif hasattr(comp, "click"):
                event_subscribers.append(comp.click)
            elif isinstance(comp, gr.Slider) and hasattr(comp, "release"):
                event_subscribers.append(comp.release)
            elif hasattr(comp, "change"):
                event_subscribers.append(comp.change)
            if hasattr(comp, "clear"):
                event_subscribers.append(comp.clear)
            for event_subscriber in event_subscribers:
                event_subscriber(
                    fn=clear_preview,
                    inputs=self.use_preview_as_input,
                    outputs=[self.use_preview_as_input, self.generated_image],
                )

    def register_multi_images_upload(self):
        """Register callbacks on merge tab multiple images upload."""
        self.merge_clear_button.click(
            fn=lambda: [],
            inputs=[],
            outputs=[self.merge_gallery],
        ).then(
            fn=lambda x: gr.update(value=x + 1),
            inputs=[self.update_unit_counter],
            outputs=[self.update_unit_counter],
        )

        def upload_file(files, current_files):
            return {file_d["name"] for file_d in current_files} | {
                file.name for file in files
            }

        self.merge_upload_button.upload(
            upload_file,
            inputs=[self.merge_upload_button, self.merge_gallery],
            outputs=[self.merge_gallery],
            queue=False,
        ).then(
            fn=lambda x: gr.update(value=x + 1),
            inputs=[self.update_unit_counter],
            outputs=[self.update_unit_counter],
        )

    def register_core_callbacks(self):
        """Register core callbacks that only involves gradio components defined
        within this ui group."""
        self.register_webcam_toggle()
        self.register_webcam_mirror_toggle()
        self.register_refresh_all_models()
        self.register_build_sliders()
        self.register_shift_preview()
        self.register_shift_upload_mask()
        self.register_create_canvas()
        self.register_clear_preview()
        self.register_multi_images_upload()
        self.openpose_editor.register_callbacks(
            self.generated_image,
            self.use_preview_as_input,
            self.model,
        )
        assert self.type_filter is not None
        self.preset_panel.register_callbacks(
            self,
            self.type_filter,
            *[
                getattr(self, key)
                for key in vars(external_code.ControlNetUnit()).keys()
            ],
        )
        if self.is_img2img:
            self.register_img2img_same_input()

    def register_callbacks(self):
        """Register callbacks that involves A1111 context gradio components."""
        # Prevent infinite recursion.
        if self.callbacks_registered:
            return

        self.callbacks_registered = True
        self.register_sd_version_changed()
        self.register_send_dimensions()
        self.register_run_annotator()
        self.register_sync_batch_dir()
        if self.is_img2img:
            self.register_shift_crop_input_image()
        else:
            self.register_shift_hr_options()

    @staticmethod
    def register_input_mode_sync(ui_groups: List["ControlNetUiGroup"]):
        """
        - ui_group.input_mode should be updated when user switch tabs.
        - Loopback checkbox should only be visible if at least one ControlNet unit
        is set to batch mode.

        Argument:
            ui_groups: All ControlNetUiGroup instances defined in current Script context.

        Returns:
            None
        """
        if not ui_groups:
            return

        for ui_group in ui_groups:
            batch_fn = lambda: InputMode.BATCH
            simple_fn = lambda: InputMode.SIMPLE
            merge_fn = lambda: InputMode.MERGE
            for input_tab, fn in (
                (ui_group.upload_tab, simple_fn),
                (ui_group.batch_tab, batch_fn),
                (ui_group.merge_tab, merge_fn),
            ):
                # Sync input_mode.
                input_tab.select(
                    fn=fn,
                    inputs=[],
                    outputs=[ui_group.input_mode],
                    show_progress=False,
                ).then(
                    # Update visibility of loopback checkbox.
                    fn=lambda *mode_values: (
                        (
                            gr.update(
                                visible=any(m == InputMode.BATCH for m in mode_values)
                            ),
                        )
                        * len(ui_groups)
                    ),
                    inputs=[g.input_mode for g in ui_groups],
                    outputs=[g.loopback for g in ui_groups],
                    show_progress=False,
                )

    @staticmethod
    def reset():
        ControlNetUiGroup.a1111_context = A1111Context()
        ControlNetUiGroup.all_ui_groups = []

    @staticmethod
    def try_register_all_callbacks():
        unit_count = shared.opts.data.get("control_net_unit_count", 3)
        all_unit_count = unit_count * 2  # txt2img + img2img.
        if (
            # All A1111 components ControlNet units care about are all registered.
            ControlNetUiGroup.a1111_context.ui_initialized
            and all_unit_count == len(ControlNetUiGroup.all_ui_groups)
            and all(
                g.ui_initialized and (not g.callbacks_registered)
                for g in ControlNetUiGroup.all_ui_groups
            )
        ):
            for ui_group in ControlNetUiGroup.all_ui_groups:
                ui_group.register_callbacks()

            ControlNetUiGroup.register_input_mode_sync(
                [g for g in ControlNetUiGroup.all_ui_groups if g.is_img2img]
            )
            ControlNetUiGroup.register_input_mode_sync(
                [g for g in ControlNetUiGroup.all_ui_groups if not g.is_img2img]
            )
            logger.info("ControlNet UI callback registered.")

    @staticmethod
    def on_after_component(component, **_kwargs):
        """Register the A1111 component."""
        if getattr(component, "elem_id", None) == "img2img_batch_inpaint_mask_dir":
            ControlNetUiGroup.global_batch_input_dir.render()
            return

        ControlNetUiGroup.a1111_context.set_component(component)
        ControlNetUiGroup.try_register_all_callbacks()
