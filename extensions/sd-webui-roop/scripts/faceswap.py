import os
import gradio as gr
import modules.scripts as scripts
from modules.upscaler import Upscaler, UpscalerData
from modules import scripts, shared, images, scripts_postprocessing
from modules.processing import (
    StableDiffusionProcessing,
    StableDiffusionProcessingImg2Img,
)
from modules.shared import cmd_opts, opts, state
from PIL import Image
import glob
from modules.face_restoration import FaceRestoration

from scripts.roop_logging import logger
from scripts.swapper import UpscaleOptions, swap_face, ImageResult
from scripts.roop_version import version_flag
import os


def get_models():
    models_path = os.path.join(scripts.basedir(), "models" + os.path.sep + "roop" + os.path.sep + "*")
    models = glob.glob(models_path)
    models = [x for x in models if x.endswith(".onnx") or x.endswith(".pth")]
    return models


class FaceSwapScript(scripts.Script):
    def title(self):
        return f"roop"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion(f"roop {version_flag}", open=False):
            with gr.Column():
                img = gr.inputs.Image(type="pil")
                enable = gr.Checkbox(False, placeholder="enable", label="Enable")
                faces_index = gr.Textbox(
                    value="0",
                    placeholder="Which face to swap (comma separated), start from 0",
                    label="Comma separated face number(s)",
                )
                with gr.Row():
                    face_restorer_name = gr.Radio(
                        label="Restore Face",
                        choices=["None"] + [x.name() for x in shared.face_restorers],
                        value=shared.face_restorers[0].name(),
                        type="value",
                    )
                    face_restorer_visibility = gr.Slider(
                        0, 1, 1, step=0.1, label="Restore visibility"
                    )
                upscaler_name = gr.inputs.Dropdown(
                    choices=[upscaler.name for upscaler in shared.sd_upscalers],
                    label="Upscaler",
                )
                upscaler_scale = gr.Slider(1, 8, 1, step=0.1, label="Upscaler scale")
                upscaler_visibility = gr.Slider(
                    0, 1, 1, step=0.1, label="Upscaler visibility (if scale = 1)"
                )

                models = get_models()
                if len(models) == 0:
                    logger.warning(
                        "You should at least have one model in models directory, please read the doc here : https://github.com/s0md3v/sd-webui-roop/"
                    )
                    model = gr.inputs.Dropdown(
                        choices=models,
                        label="Model not found, please download one and reload automatic 1111",
                    )
                else:
                    model = gr.inputs.Dropdown(
                        choices=models, label="Model", default=models[0]
                    )

                swap_in_source = gr.Checkbox(
                    False,
                    placeholder="Swap face in source image",
                    label="Swap in source image",
                    visible=is_img2img,
                )
                swap_in_generated = gr.Checkbox(
                    True,
                    placeholder="Swap face in generated image",
                    label="Swap in generated image",
                    visible=is_img2img,
                )

        return [
            img,
            enable,
            faces_index,
            model,
            face_restorer_name,
            face_restorer_visibility,
            upscaler_name,
            upscaler_scale,
            upscaler_visibility,
            swap_in_source,
            swap_in_generated,
        ]

    @property
    def upscaler(self) -> UpscalerData:
        for upscaler in shared.sd_upscalers:
            if upscaler.name == self.upscaler_name:
                return upscaler
        return None

    @property
    def face_restorer(self) -> FaceRestoration:
        for face_restorer in shared.face_restorers:
            if face_restorer.name() == self.face_restorer_name:
                return face_restorer
        return None

    @property
    def upscale_options(self) -> UpscaleOptions:
        return UpscaleOptions(
            scale=self.upscaler_scale,
            upscaler=self.upscaler,
            face_restorer=self.face_restorer,
            upscale_visibility=self.upscaler_visibility,
            restorer_visibility=self.face_restorer_visibility,
        )

    def process(
        self,
        p: StableDiffusionProcessing,
        img,
        enable,
        faces_index,
        model,
        face_restorer_name,
        face_restorer_visibility,
        upscaler_name,
        upscaler_scale,
        upscaler_visibility,
        swap_in_source,
        swap_in_generated,
    ):
        self.source = img
        self.face_restorer_name = face_restorer_name
        self.upscaler_scale = upscaler_scale
        self.upscaler_visibility = upscaler_visibility
        self.face_restorer_visibility = face_restorer_visibility
        self.enable = enable
        self.upscaler_name = upscaler_name
        self.swap_in_generated = swap_in_generated
        self.model = model
        self.faces_index = {
            int(x) for x in faces_index.strip(",").split(",") if x.isnumeric()
        }
        if len(self.faces_index) == 0:
            self.faces_index = {0}
        if self.enable:
            if self.source is not None:
                if isinstance(p, StableDiffusionProcessingImg2Img) and swap_in_source:
                    logger.info(f"roop enabled, face index %s", self.faces_index)

                    for i in range(len(p.init_images)):
                        logger.info(f"Swap in source %s", i)
                        result = swap_face(
                            self.source,
                            p.init_images[i],
                            faces_index=self.faces_index,
                            model=self.model,
                            upscale_options=self.upscale_options,
                        )
                        p.init_images[i] = result.image()
            else:
                logger.error(f"Please provide a source face")

    def postprocess_batch(self, *args, **kwargs):
        if self.enable:
            return images

    def postprocess_image(self, p, script_pp: scripts.PostprocessImageArgs, *args):
        if self.enable and self.swap_in_generated:
            if self.source is not None:
                image: Image.Image = script_pp.image
                result: ImageResult = swap_face(
                    self.source,
                    image,
                    faces_index=self.faces_index,
                    model=self.model,
                    upscale_options=self.upscale_options,
                )
                pp = scripts_postprocessing.PostprocessedImage(result.image())
                pp.info = {}
                p.extra_generation_params.update(pp.info)
                script_pp.image = pp.image
