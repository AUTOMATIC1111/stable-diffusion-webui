
from modules import scripts_postprocessing, ui_components, errors
import gradio as gr

from modules.textual_inversion import autocrop


class ScriptPostprocessingFocalCrop(scripts_postprocessing.ScriptPostprocessing):
    name = "Auto focal point crop"
    order = 4000

    def ui(self):
        with ui_components.InputAccordion(False, label="Auto focal point crop") as enable:
            face_weight = gr.Slider(label='Focal point face weight', value=0.9, minimum=0.0, maximum=1.0, step=0.05, elem_id="postprocess_focal_crop_face_weight")
            entropy_weight = gr.Slider(label='Focal point entropy weight', value=0.15, minimum=0.0, maximum=1.0, step=0.05, elem_id="postprocess_focal_crop_entropy_weight")
            edges_weight = gr.Slider(label='Focal point edges weight', value=0.5, minimum=0.0, maximum=1.0, step=0.05, elem_id="postprocess_focal_crop_edges_weight")
            debug = gr.Checkbox(label='Create debug image', elem_id="train_process_focal_crop_debug")

        return {
            "enable": enable,
            "face_weight": face_weight,
            "entropy_weight": entropy_weight,
            "edges_weight": edges_weight,
            "debug": debug,
        }

    def process(self, pp: scripts_postprocessing.PostprocessedImage, enable, face_weight, entropy_weight, edges_weight, debug):
        if not enable:
            return

        if not pp.shared.target_width or not pp.shared.target_height:
            return

        dnn_model_path = None
        try:
            dnn_model_path = autocrop.download_and_cache_models()
        except Exception:
            errors.report("Unable to load face detection model for auto crop selection. Falling back to lower quality haar method.", exc_info=True)

        autocrop_settings = autocrop.Settings(
            crop_width=pp.shared.target_width,
            crop_height=pp.shared.target_height,
            face_points_weight=face_weight,
            entropy_points_weight=entropy_weight,
            corner_points_weight=edges_weight,
            annotate_image=debug,
            dnn_model_path=dnn_model_path,
        )

        result, *others = autocrop.crop_image(pp.image, autocrop_settings)

        pp.image = result
        pp.extra_images = [pp.create_copy(x, nametags=["focal-crop-debug"], disable_processing=True) for x in others]

