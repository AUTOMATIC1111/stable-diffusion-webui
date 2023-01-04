import os
import sys
import traceback

from basicsr.utils.download_util import load_file_from_url

from modules.upscaler import Upscaler, UpscalerData
from ldsr_model_arch import LDSR
from modules import shared, script_callbacks
import sd_hijack_autoencoder, sd_hijack_ddpm_v1


class UpscalerLDSR(Upscaler):
    def __init__(self, user_path):
        self.name = "LDSR"
        self.user_path = user_path
        self.model_url = "https://heibox.uni-heidelberg.de/f/578df07c8fc04ffbadf3/?dl=1"
        self.yaml_url = "https://heibox.uni-heidelberg.de/f/31a76b13ea27482981b4/?dl=1"
        super().__init__()
        scaler_data = UpscalerData("LDSR", None, self)
        self.scalers = [scaler_data]

    def load_model(self, path: str):
        # Remove incorrect project.yaml file if too big
        yaml_path = os.path.join(self.model_path, "project.yaml")
        old_model_path = os.path.join(self.model_path, "model.pth")
        new_model_path = os.path.join(self.model_path, "model.ckpt")
        safetensors_model_path = os.path.join(self.model_path, "model.safetensors")
        if os.path.exists(yaml_path):
            statinfo = os.stat(yaml_path)
            if statinfo.st_size >= 10485760:
                print("Removing invalid LDSR YAML file.")
                os.remove(yaml_path)
        if os.path.exists(old_model_path):
            print("Renaming model from model.pth to model.ckpt")
            os.rename(old_model_path, new_model_path)
        if os.path.exists(safetensors_model_path):
            model = safetensors_model_path
        else:
            model = load_file_from_url(url=self.model_url, model_dir=self.model_path,
                                       file_name="model.ckpt", progress=True)
        yaml = load_file_from_url(url=self.yaml_url, model_dir=self.model_path,
                                  file_name="project.yaml", progress=True)

        try:
            return LDSR(model, yaml)

        except Exception:
            print("Error importing LDSR:", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)
        return None

    def do_upscale(self, img, path):
        ldsr = self.load_model(path)
        if ldsr is None:
            print("NO LDSR!")
            return img
        ddim_steps = shared.opts.ldsr_steps
        return ldsr.super_resolution(img, ddim_steps, self.scale)


def on_ui_settings():
    import gradio as gr

    shared.opts.add_option("ldsr_steps", shared.OptionInfo(100, "LDSR processing steps. Lower = faster", gr.Slider, {"minimum": 1, "maximum": 200, "step": 1}, section=('upscaling', "Upscaling")))
    shared.opts.add_option("ldsr_cached", shared.OptionInfo(False, "Cache LDSR model in memory", gr.Checkbox, {"interactive": True}, section=('upscaling', "Upscaling")))


script_callbacks.on_ui_settings(on_ui_settings)
