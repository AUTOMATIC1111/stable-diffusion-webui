import os
import sys
import traceback

from basicsr.utils.download_util import load_file_from_url

from modules.upscaler import Upscaler, UpscalerData
from modules.ldsr_model_arch import LDSR
from modules import shared
from modules.paths import models_path


class UpscalerLDSR(Upscaler):
    def __init__(self, user_path):
        self.name = "LDSR"
        self.model_path = os.path.join(models_path, self.name)
        self.user_path = user_path
        self.model_url = "https://heibox.uni-heidelberg.de/f/578df07c8fc04ffbadf3/?dl=1"
        self.yaml_url = "https://heibox.uni-heidelberg.de/f/31a76b13ea27482981b4/?dl=1"
        super().__init__()
        scaler_data = UpscalerData("LDSR", None, self)
        self.scalers = [scaler_data]

    def load_model(self, path: str):
        model = load_file_from_url(url=self.model_url, model_dir=self.model_path,
                                   file_name="model.pth", progress=True)
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
        pre_scale = shared.opts.ldsr_pre_down
        return ldsr.super_resolution(img, ddim_steps, self.scale)
