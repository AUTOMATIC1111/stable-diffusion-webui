import os
from modules import modelloader, devices, errors
from modules.shared import opts, cmd_opts
from modules.upscaler import Upscaler, UpscalerData
from modules.upscaler_utils import upscale_with_model


class UpscalerCOMPACT(Upscaler):
    def __init__(self, dirname):
        self.name = "COMPACT"
        self.scalers = []
        self.user_path = dirname
        super().__init__()
        for file in self.find_models(ext_filter=[".pt", ".pth", ".safetensors"]):
            name = modelloader.friendly_name(file)
            scale = None
            scaler_data = UpscalerData(name, file, upscaler=self, scale=scale)
            self.scalers.append(scaler_data)

    def do_upscale(self, img, selected_model):
        try:
            model = self.load_model(selected_model)
        except Exception:
            errors.report(f"Unable to load COMPACT model {selected_model}", exc_info=True)
            return img
        model.to(devices.device_compact)
        return compact_upscale(model, img)

    def load_model(self, path: str):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Model file {path} not found")
        else:
            filename = path
        return modelloader.load_spandrel_model(
            filename,
            device=('cpu' if devices.device_compact.type == 'mps' else None),
            prefer_half=(not cmd_opts.no_half and not cmd_opts.upcast_sampling),
            expected_architecture='RealESRGAN Compact',
        )


def compact_upscale(model, img):
        return upscale_with_model(
            model,
            img,
            tile_size=opts.COMPACT_tile,
            tile_overlap=opts.COMPACT_tile_overlap,
        )
