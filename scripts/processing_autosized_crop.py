from PIL import Image

from modules import scripts_postprocessing, ui_components
import gradio as gr


def center_crop(image: Image, w: int, h: int):
    iw, ih = image.size
    if ih / h < iw / w:
        sw = w * ih / h
        box = (iw - sw) / 2, 0, iw - (iw - sw) / 2, ih
    else:
        sh = h * iw / w
        box = 0, (ih - sh) / 2, iw, ih - (ih - sh) / 2
    return image.resize((w, h), Image.Resampling.LANCZOS, box)


def multicrop_pic(image: Image, mindim, maxdim, minarea, maxarea, objective, threshold):
    iw, ih = image.size
    err = lambda w, h: 1 - (lambda x: x if x < 1 else 1 / x)(iw / ih / (w / h))
    wh = max(((w, h) for w in range(mindim, maxdim + 1, 64) for h in range(mindim, maxdim + 1, 64)
              if minarea <= w * h <= maxarea and err(w, h) <= threshold),
             key=lambda wh: (wh[0] * wh[1], -err(*wh))[::1 if objective == 'Maximize area' else -1],
             default=None
             )
    return wh and center_crop(image, *wh)


class ScriptPostprocessingAutosizedCrop(scripts_postprocessing.ScriptPostprocessing):
    name = "Auto-sized crop"
    order = 4000

    def ui(self):
        with ui_components.InputAccordion(False, label="Auto-sized crop") as enable:
            gr.Markdown('Each image is center-cropped with an automatically chosen width and height.')
            with gr.Row():
                mindim = gr.Slider(minimum=64, maximum=2048, step=8, label="Dimension lower bound", value=384, elem_id="postprocess_multicrop_mindim")
                maxdim = gr.Slider(minimum=64, maximum=2048, step=8, label="Dimension upper bound", value=768, elem_id="postprocess_multicrop_maxdim")
            with gr.Row():
                minarea = gr.Slider(minimum=64 * 64, maximum=2048 * 2048, step=1, label="Area lower bound", value=64 * 64, elem_id="postprocess_multicrop_minarea")
                maxarea = gr.Slider(minimum=64 * 64, maximum=2048 * 2048, step=1, label="Area upper bound", value=640 * 640, elem_id="postprocess_multicrop_maxarea")
            with gr.Row():
                objective = gr.Radio(["Maximize area", "Minimize error"], value="Maximize area", label="Resizing objective", elem_id="postprocess_multicrop_objective")
                threshold = gr.Slider(minimum=0, maximum=1, step=0.01, label="Error threshold", value=0.1, elem_id="postprocess_multicrop_threshold")

        return {
            "enable": enable,
            "mindim": mindim,
            "maxdim": maxdim,
            "minarea": minarea,
            "maxarea": maxarea,
            "objective": objective,
            "threshold": threshold,
        }

    def process(self, pp: scripts_postprocessing.PostprocessedImage, enable, mindim, maxdim, minarea, maxarea, objective, threshold):
        if not enable:
            return

        cropped = multicrop_pic(pp.image, mindim, maxdim, minarea, maxarea, objective, threshold)
        if cropped is not None:
            pp.image = cropped
        else:
            print(f"skipped {pp.image.width}x{pp.image.height} image (can't find suitable size within error threshold)")
