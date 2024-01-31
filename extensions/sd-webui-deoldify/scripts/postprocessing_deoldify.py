'''
Author: SpenserCai
Date: 2023-07-28 14:41:28
version: 
LastEditors: SpenserCai
LastEditTime: 2023-09-07 10:49:32
Description: file content
'''
# DeOldify UI & Processing
from modules import scripts_postprocessing, paths_internal
from modules.ui_components import FormRow
from scripts.deoldify_base import *
import gradio as gr

if hasattr(scripts_postprocessing.ScriptPostprocessing, 'process_firstpass'):  # webui >= 1.7
    from modules.ui_components import InputAccordion
else:
    InputAccordion = None


class ScriptPostprocessingUpscale(scripts_postprocessing.ScriptPostprocessing):
    name = "Deoldify"
    order = 20001

    def ui(self):
        with (
            InputAccordion(False, label="Deoldify") if InputAccordion
            else gr.Accordion("Deoldify", open=False)
            as is_enabled
        ):
            with FormRow():
                if not InputAccordion:
                    is_enabled = gr.Checkbox(label="enable", value=False)
                render_factor = gr.Slider(minimum=1, maximum=50, step=1, label="Render Factor")
                render_factor.value = 35
                # 一个名为artistic的复选框，初始值是False
                artistic = gr.Checkbox(label="Artistic")
                artistic.value = False
                pre_decolorization = gr.Checkbox(label="Pre-Decolorization",info="For yellowed photos, this option can be used to fade to black and white before coloring")
                pre_decolorization.value = False

        return {
            "is_enabled": is_enabled,
            "render_factor": render_factor,
            "artistic": artistic,
            "pre_decolorization": pre_decolorization
        }
    
    def process_image(self, image, render_factor, artistic, pre_decolorization):
        if pre_decolorization:
            image = Decolorization(image)
        vis = get_image_colorizer(root_folder=Path(paths_internal.models_path),render_factor=render_factor, artistic=artistic)
        outImg = vis.get_transformed_image_from_image(image, render_factor=render_factor)
        return outImg

    def process(self, pp: scripts_postprocessing.PostprocessedImage, is_enabled, render_factor, artistic, pre_decolorization):
        if not is_enabled or is_enabled is False:
            return

        pp.image = self.process_image(pp.image, render_factor, artistic, pre_decolorization)
        pp.info["deoldify"] = f"render_factor={render_factor}, artistic={artistic}, pre_decolorization={pre_decolorization}"