import os
import sys

from modules import modelloader, devices
from modules.shared import opts
from modules.upscaler import Upscaler, UpscalerData
from modules.upscaler_utils import upscale_with_model


class UpscalerHAT(Upscaler):
    def __init__(self, dirname):
        self.name = "HAT"
        self.scalers = []
        self.user_path = dirname
        super().__init__()
        for file in self.find_models(ext_filter=[".pt", ".pth"]):
            name = modelloader.friendly_name(file)
            scale = 4  # TODO: scale might not be 4, but we can't know without loading the model
            scaler_data = UpscalerData(name, file, upscaler=self, scale=scale)
            self.scalers.append(scaler_data)

    def do_upscale(self, img, selected_model):
        try:
            model = self.load_model(selected_model)
        except Exception as e:
            print(f"Unable to load HAT model {selected_model}: {e}", file=sys.stderr)
            return img
        model.to(devices.device_esrgan)  # TODO: should probably be device_hat
        return upscale_with_model(
            model,
            img,
            tile_size=opts.ESRGAN_tile,  # TODO: should probably be HAT_tile
            tile_overlap=opts.ESRGAN_tile_overlap,  # TODO: should probably be HAT_tile_overlap
        )

    def load_model(self, path: str):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Model file {path} not found")
        return modelloader.load_spandrel_model(
            path,
            device=devices.device_esrgan,  # TODO: should probably be device_hat
            expected_architecture='HAT',
        )
