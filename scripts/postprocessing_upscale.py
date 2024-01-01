from PIL import Image
import numpy as np

from modules import scripts_postprocessing, shared
import gradio as gr

from modules.ui_components import FormRow, ToolButton
from modules.ui import switch_values_symbol

upscale_cache = {}


class ScriptPostprocessingUpscale(scripts_postprocessing.ScriptPostprocessing):
    name = "Upscale"
    order = 1000

    def ui(self):
        selected_tab = gr.State(value=0)

        with gr.Column():
            with FormRow():
                with gr.Tabs(elem_id="extras_resize_mode"):
                    with gr.TabItem('Scale by', elem_id="extras_scale_by_tab") as tab_scale_by:
                        upscaling_resize = gr.Slider(minimum=1.0, maximum=8.0, step=0.05, label="Resize", value=4, elem_id="extras_upscaling_resize")

                    with gr.TabItem('Scale to', elem_id="extras_scale_to_tab") as tab_scale_to:
                        with FormRow():
                            with gr.Column(elem_id="upscaling_column_size", scale=4):
                                upscaling_resize_w = gr.Slider(minimum=64, maximum=2048, step=8, label="Width", value=512, elem_id="extras_upscaling_resize_w")
                                upscaling_resize_h = gr.Slider(minimum=64, maximum=2048, step=8, label="Height", value=512, elem_id="extras_upscaling_resize_h")
                            with gr.Column(elem_id="upscaling_dimensions_row", scale=1, elem_classes="dimensions-tools"):
                                upscaling_res_switch_btn = ToolButton(value=switch_values_symbol, elem_id="upscaling_res_switch_btn", tooltip="Switch width/height")
                                upscaling_crop = gr.Checkbox(label='Crop to fit', value=True, elem_id="extras_upscaling_crop")

            with FormRow():
                extras_upscaler_1 = gr.Dropdown(label='Upscaler 1', elem_id="extras_upscaler_1", choices=[x.name for x in shared.sd_upscalers], value=shared.sd_upscalers[0].name)

            with FormRow():
                extras_upscaler_2 = gr.Dropdown(label='Upscaler 2', elem_id="extras_upscaler_2", choices=[x.name for x in shared.sd_upscalers], value=shared.sd_upscalers[0].name)
                extras_upscaler_2_visibility = gr.Slider(minimum=0.0, maximum=1.0, step=0.001, label="Upscaler 2 visibility", value=0.0, elem_id="extras_upscaler_2_visibility")

        upscaling_res_switch_btn.click(lambda w, h: (h, w), inputs=[upscaling_resize_w, upscaling_resize_h], outputs=[upscaling_resize_w, upscaling_resize_h], show_progress=False)
        tab_scale_by.select(fn=lambda: 0, inputs=[], outputs=[selected_tab])
        tab_scale_to.select(fn=lambda: 1, inputs=[], outputs=[selected_tab])

        return {
            "upscale_mode": selected_tab,
            "upscale_by": upscaling_resize,
            "upscale_to_width": upscaling_resize_w,
            "upscale_to_height": upscaling_resize_h,
            "upscale_crop": upscaling_crop,
            "upscaler_1_name": extras_upscaler_1,
            "upscaler_2_name": extras_upscaler_2,
            "upscaler_2_visibility": extras_upscaler_2_visibility,
        }

    def upscale(self, image, info, upscaler, upscale_mode, upscale_by,  upscale_to_width, upscale_to_height, upscale_crop):
        if upscale_mode == 1:
            upscale_by = max(upscale_to_width/image.width, upscale_to_height/image.height)
            info["Postprocess upscale to"] = f"{upscale_to_width}x{upscale_to_height}"
        else:
            info["Postprocess upscale by"] = upscale_by

        cache_key = (hash(np.array(image.getdata()).tobytes()), upscaler.name, upscale_mode, upscale_by,  upscale_to_width, upscale_to_height, upscale_crop)
        cached_image = upscale_cache.pop(cache_key, None)

        if cached_image is not None:
            image = cached_image
        else:
            image = upscaler.scaler.upscale(image, upscale_by, upscaler.data_path)

        upscale_cache[cache_key] = image
        if len(upscale_cache) > shared.opts.upscaling_max_images_in_cache:
            upscale_cache.pop(next(iter(upscale_cache), None), None)

        if upscale_mode == 1 and upscale_crop:
            cropped = Image.new("RGB", (upscale_to_width, upscale_to_height))
            cropped.paste(image, box=(upscale_to_width // 2 - image.width // 2, upscale_to_height // 2 - image.height // 2))
            image = cropped
            info["Postprocess crop to"] = f"{image.width}x{image.height}"

        return image

    def process_firstpass(self, pp: scripts_postprocessing.PostprocessedImage, upscale_mode=1, upscale_by=2.0, upscale_to_width=None, upscale_to_height=None, upscale_crop=False, upscaler_1_name=None, upscaler_2_name=None, upscaler_2_visibility=0.0):
        if upscale_mode == 1:
            pp.shared.target_width = upscale_to_width
            pp.shared.target_height = upscale_to_height
        else:
            pp.shared.target_width = int(pp.image.width * upscale_by)
            pp.shared.target_height = int(pp.image.height * upscale_by)

    def process(self, pp: scripts_postprocessing.PostprocessedImage, upscale_mode=1, upscale_by=2.0, upscale_to_width=None, upscale_to_height=None, upscale_crop=False, upscaler_1_name=None, upscaler_2_name=None, upscaler_2_visibility=0.0):
        if upscaler_1_name == "None":
            upscaler_1_name = None

        upscaler1 = next(iter([x for x in shared.sd_upscalers if x.name == upscaler_1_name]), None)
        assert upscaler1 or (upscaler_1_name is None), f'could not find upscaler named {upscaler_1_name}'

        if not upscaler1:
            return

        if upscaler_2_name == "None":
            upscaler_2_name = None

        upscaler2 = next(iter([x for x in shared.sd_upscalers if x.name == upscaler_2_name and x.name != "None"]), None)
        assert upscaler2 or (upscaler_2_name is None), f'could not find upscaler named {upscaler_2_name}'

        upscaled_image = self.upscale(pp.image, pp.info, upscaler1, upscale_mode, upscale_by, upscale_to_width, upscale_to_height, upscale_crop)
        pp.info["Postprocess upscaler"] = upscaler1.name

        if upscaler2 and upscaler_2_visibility > 0:
            second_upscale = self.upscale(pp.image, pp.info, upscaler2, upscale_mode, upscale_by, upscale_to_width, upscale_to_height, upscale_crop)
            upscaled_image = Image.blend(upscaled_image, second_upscale, upscaler_2_visibility)

            pp.info["Postprocess upscaler 2"] = upscaler2.name

        pp.image = upscaled_image

    def image_changed(self):
        upscale_cache.clear()


class ScriptPostprocessingUpscaleSimple(ScriptPostprocessingUpscale):
    name = "Simple Upscale"
    order = 900

    def ui(self):
        with FormRow():
            upscaler_name = gr.Dropdown(label='Upscaler', choices=[x.name for x in shared.sd_upscalers], value=shared.sd_upscalers[0].name)
            upscale_by = gr.Slider(minimum=0.05, maximum=8.0, step=0.05, label="Upscale by", value=2)

        return {
            "upscale_by": upscale_by,
            "upscaler_name": upscaler_name,
        }

    def process_firstpass(self, pp: scripts_postprocessing.PostprocessedImage, upscale_by=2.0, upscaler_name=None):
        pp.shared.target_width = int(pp.image.width * upscale_by)
        pp.shared.target_height = int(pp.image.height * upscale_by)

    def process(self, pp: scripts_postprocessing.PostprocessedImage, upscale_by=2.0, upscaler_name=None):
        if upscaler_name is None or upscaler_name == "None":
            return

        upscaler1 = next(iter([x for x in shared.sd_upscalers if x.name == upscaler_name]), None)
        assert upscaler1, f'could not find upscaler named {upscaler_name}'

        pp.image = self.upscale(pp.image, pp.info, upscaler1, 0, upscale_by, 0, 0, False)
        pp.info["Postprocess upscaler"] = upscaler1.name
