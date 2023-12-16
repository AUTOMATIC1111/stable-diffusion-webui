from PIL import ImageOps, Image

from modules import scripts_postprocessing, ui_components
import gradio as gr


class ScriptPostprocessingCreateFlippedCopies(scripts_postprocessing.ScriptPostprocessing):
    name = "Create flipped copies"
    order = 4000

    def ui(self):
        with ui_components.InputAccordion(False, label="Create flipped copies") as enable:
            with gr.Row():
                option = gr.CheckboxGroup(value=["Horizontal"], choices=["Horizontal", "Vertical", "Both"], show_label=False)

        return {
            "enable": enable,
            "option": option,
        }

    def process(self, pp: scripts_postprocessing.PostprocessedImage, enable, option):
        if not enable:
            return

        if "Horizontal" in option:
            pp.extra_images.append(ImageOps.mirror(pp.image))

        if "Vertical" in option:
            pp.extra_images.append(pp.image.transpose(Image.Transpose.FLIP_TOP_BOTTOM))

        if "Both" in option:
            pp.extra_images.append(pp.image.transpose(Image.Transpose.FLIP_TOP_BOTTOM).transpose(Image.Transpose.FLIP_LEFT_RIGHT))
