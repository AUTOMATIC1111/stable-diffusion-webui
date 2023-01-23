from PIL import Image
import numpy as np

from modules import scripts_postprocessing, gfpgan_model
import gradio as gr

from modules.ui_components import FormRow


class ScriptPostprocessingGfpGan(scripts_postprocessing.ScriptPostprocessing):
    name = "GFPGAN"
    order = 2000

    def ui(self):
        with FormRow():
            gfpgan_visibility = gr.Slider(minimum=0.0, maximum=1.0, step=0.001, label="GFPGAN visibility", value=0, elem_id="extras_gfpgan_visibility")

        return {
            "gfpgan_visibility": gfpgan_visibility,
        }

    def process(self, pp: scripts_postprocessing.PostprocessedImage, gfpgan_visibility):
        if gfpgan_visibility == 0:
            return

        restored_img = gfpgan_model.gfpgan_fix_faces(np.array(pp.image, dtype=np.uint8))
        res = Image.fromarray(restored_img)

        if gfpgan_visibility < 1.0:
            res = Image.blend(pp.image, res, gfpgan_visibility)

        pp.image = res
        pp.info["GFPGAN visibility"] = round(gfpgan_visibility, 3)
