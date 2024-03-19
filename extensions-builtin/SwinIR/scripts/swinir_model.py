import logging
import sys

import torch
from PIL import Image

from modules import devices, modelloader, script_callbacks, shared, upscaler_utils
from modules.upscaler import Upscaler, UpscalerData

SWINIR_MODEL_URL = "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth"

logger = logging.getLogger(__name__)


class UpscalerSwinIR(Upscaler):
    def __init__(self, dirname):
        self._cached_model = None           # keep the model when SWIN_torch_compile is on to prevent re-compile every runs
        self._cached_model_config = None    # to clear '_cached_model' when changing model (v1/v2) or settings
        self.name = "SwinIR"
        self.model_url = SWINIR_MODEL_URL
        self.model_name = "SwinIR 4x"
        self.user_path = dirname
        super().__init__()
        scalers = []
        model_files = self.find_models(ext_filter=[".pt", ".pth"])
        for model in model_files:
            if model.startswith("http"):
                name = self.model_name
            else:
                name = modelloader.friendly_name(model)
            model_data = UpscalerData(name, model, self)
            scalers.append(model_data)
        self.scalers = scalers

    def do_upscale(self, img: Image.Image, model_file: str) -> Image.Image:
        current_config = (model_file, shared.opts.SWIN_tile)

        if self._cached_model_config == current_config:
            model = self._cached_model
        else:
            try:
                model = self.load_model(model_file)
            except Exception as e:
                print(f"Failed loading SwinIR model {model_file}: {e}", file=sys.stderr)
                return img
            self._cached_model = model
            self._cached_model_config = current_config

        img = upscaler_utils.upscale_2(
            img,
            model,
            tile_size=shared.opts.SWIN_tile,
            tile_overlap=shared.opts.SWIN_tile_overlap,
            scale=model.scale,
            desc="SwinIR",
        )
        devices.torch_gc()
        return img

    def load_model(self, path, scale=4):
        if path.startswith("http"):
            filename = modelloader.load_file_from_url(
                url=path,
                model_dir=self.model_download_path,
                file_name=f"{self.model_name.replace(' ', '_')}.pth",
            )
        else:
            filename = path

        model_descriptor = modelloader.load_spandrel_model(
            filename,
            device=self._get_device(),
            prefer_half=(devices.dtype == torch.float16),
            expected_architecture="SwinIR",
        )
        if getattr(shared.opts, 'SWIN_torch_compile', False):
            try:
                model_descriptor.model.compile()
            except Exception:
                logger.warning("Failed to compile SwinIR model, fallback to JIT", exc_info=True)
        return model_descriptor

    def _get_device(self):
        return devices.get_device_for('swinir')


def on_ui_settings():
    import gradio as gr

    shared.opts.add_option("SWIN_tile", shared.OptionInfo(192, "Tile size for all SwinIR.", gr.Slider, {"minimum": 16, "maximum": 512, "step": 16}, section=('upscaling', "Upscaling")))
    shared.opts.add_option("SWIN_tile_overlap", shared.OptionInfo(8, "Tile overlap, in pixels for SwinIR. Low values = visible seam.", gr.Slider, {"minimum": 0, "maximum": 48, "step": 1}, section=('upscaling', "Upscaling")))
    shared.opts.add_option("SWIN_torch_compile", shared.OptionInfo(False, "Use torch.compile to accelerate SwinIR.", gr.Checkbox, {"interactive": True}, section=('upscaling', "Upscaling")).info("Takes longer on first run"))


script_callbacks.on_ui_settings(on_ui_settings)
