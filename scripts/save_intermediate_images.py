import os

from modules import scripts
from modules.processing import Processed, process_images, fix_seed
from modules.sd_samplers import KDiffusionSampler, sample_to_image
from modules.images import save_image

import gradio as gr

orig_callback_state = KDiffusionSampler.callback_state


class Script(scripts.Script):
    def title(self):
        return "Save intermediate images during the sampling process"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        is_active = gr.Checkbox(label="Save intermediate images", value=False)
        intermediate_type = gr.Radio(label="Should the intermediate images by denoised or noisy?", choices=["Denoised", "Noisy"], value="Denoised")
        every_n = gr.Number(label="Save every N images", value=5)
        return [is_active, intermediate_type, every_n]

    def run(self, p, is_active, intermediate_type, every_n):
        fix_seed(p)
        return Processed(p, images, p.seed)

    def process(self, p, is_active, intermediate_type, every_n):
        if is_active:
            def callback_state(self, d):
                """
                callback_state runs after each processing step
                """
                current_step = d["i"]

                if current_step % every_n == 0:
                    if intermediate_type == "Denoised":
                        image = sample_to_image(d["denoised"])
                    else:
                        image = sample_to_image(d["x"])

                    save_image(image, os.path.join(p.outpath_samples, "intermediates"), f"{current_step:02}", seed=p.seed, p=p)

                return orig_callback_state(self, d)

            setattr(KDiffusionSampler, "callback_state", callback_state)

    def postprocess(self, p, processed, is_active, intermediate_type, every_n):
        setattr(KDiffusionSampler, "callback_state", orig_callback_state)
