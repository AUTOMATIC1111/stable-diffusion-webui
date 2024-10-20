import math

from modules import scripts_postprocessing, ui_components
import gradio as gr


def split_pic(image, inverse_xy, width, height, overlap_ratio):
    if inverse_xy:
        from_w, from_h = image.height, image.width
        to_w, to_h = height, width
    else:
        from_w, from_h = image.width, image.height
        to_w, to_h = width, height
    h = from_h * to_w // from_w
    if inverse_xy:
        image = image.resize((h, to_w))
    else:
        image = image.resize((to_w, h))

    split_count = math.ceil((h - to_h * overlap_ratio) / (to_h * (1.0 - overlap_ratio)))
    y_step = (h - to_h) / (split_count - 1)
    for i in range(split_count):
        y = int(y_step * i)
        if inverse_xy:
            splitted = image.crop((y, 0, y + to_h, to_w))
        else:
            splitted = image.crop((0, y, to_w, y + to_h))
        yield splitted


class ScriptPostprocessingSplitOversized(scripts_postprocessing.ScriptPostprocessing):
    name = "Split oversized images"
    order = 4000

    def ui(self):
        with ui_components.InputAccordion(False, label="Split oversized images") as enable:
            with gr.Row():
                split_threshold = gr.Slider(label='Threshold', value=0.5, minimum=0.0, maximum=1.0, step=0.05, elem_id=self.elem_id_suffix("postprocess_split_threshold"))
                overlap_ratio = gr.Slider(label='Overlap ratio', value=0.2, minimum=0.0, maximum=0.9, step=0.05, elem_id=self.elem_id_suffix("postprocess_overlap_ratio"))

        return {
            "enable": enable,
            "split_threshold": split_threshold,
            "overlap_ratio": overlap_ratio,
        }

    def process(self, pp: scripts_postprocessing.PostprocessedImage, enable, split_threshold, overlap_ratio):
        if not enable:
            return

        width = pp.shared.target_width
        height = pp.shared.target_height

        if not width or not height:
            return

        if pp.image.height > pp.image.width:
            ratio = (pp.image.width * height) / (pp.image.height * width)
            inverse_xy = False
        else:
            ratio = (pp.image.height * width) / (pp.image.width * height)
            inverse_xy = True

        if ratio >= 1.0 or ratio > split_threshold:
            return

        result, *others = split_pic(pp.image, inverse_xy, width, height, overlap_ratio)

        pp.image = result
        pp.extra_images = [pp.create_copy(x) for x in others]

