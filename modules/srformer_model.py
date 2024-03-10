import os
from modules import modelloader, devices, errors
from modules.shared import opts, cmd_opts
from modules.upscaler import Upscaler, UpscalerData
from modules.upscaler_utils import upscale_with_model


class UpscalerSRFormer(Upscaler):
    def __init__(self, dirname):
        self.name = "SRFormer"
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
            errors.report(f"Unable to load SRFormer model {selected_model}", exc_info=True)
            return img
        model.to(devices.device_srformer)
        return srformer_upscale(model, img)

    def load_model(self, path: str):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Model file {path} not found")
        else:
            filename = path
        return modelloader.load_spandrel_model(
            filename,
            device=('cpu' if devices.device_srformer.type == 'mps' else None),
            prefer_half=(not cmd_opts.no_half and not cmd_opts.upcast_sampling),
            expected_architecture='SRFormer',
        )


def srformer_upscale(model, img):
        return upscale_with_model(
            model,
            img,
            tile_size=opts.SRFormer_tile,
            tile_overlap=opts.SRFormer_tile_overlap,
        )
