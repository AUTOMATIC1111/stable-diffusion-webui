from krita import Extension

from .docker import script


class Hotkeys(Extension):
    def __init__(self, parent):
        super().__init__(parent)

    def setup(self):
        pass

    def createActions(self, window):
        txt2img_action = window.createAction(
            "txt2img", "Apply txt2img transform", "tools/scripts"
        )
        txt2img_action.triggered.connect(lambda: script.action_txt2img())
        img2img_action = window.createAction(
            "img2img", "Apply img2img transform", "tools/scripts"
        )
        img2img_action.triggered.connect(lambda: script.action_img2img())
        upscale_x_action = window.createAction(
            "img2img_upscale", "Apply img2img upscale transform", "tools/scripts"
        )
        upscale_x_action.triggered.connect(lambda: script.action_sd_upscale())
        upscale_x_action = window.createAction(
            "img2img_inpaint", "Apply img2img inpaint transform", "tools/scripts"
        )
        upscale_x_action.triggered.connect(lambda: script.action_inpaint())
        simple_upscale_action = window.createAction(
            "simple_upscale", "Apply ESRGAN upscaler", "tools/scripts"
        )
        simple_upscale_action.triggered.connect(lambda: script.action_simple_upscale())
