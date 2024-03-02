from modules import modelloader, devices, errors
from modules.shared import opts
from modules.upscaler import Upscaler, UpscalerData
from modules.upscaler_utils import upscale_with_model


class UpscalerESRGAN(Upscaler):
    def __init__(self, dirname):
        self.name = "ESRGAN"
        self.model_url = "https://github.com/cszn/KAIR/releases/download/v1.0/ESRGAN.pth"
        self.model_name = "ESRGAN_4x"
        self.scalers = []
        self.user_path = dirname
        super().__init__()
        model_paths = self.find_models(ext_filter=[".pt", ".pth"])
        scalers = []
        if len(model_paths) == 0:
            scaler_data = UpscalerData(self.model_name, self.model_url, self, 4)
            scalers.append(scaler_data)
        for file in model_paths:
            if file.startswith("http"):
                name = self.model_name
            else:
                name = modelloader.friendly_name(file)

            scaler_data = UpscalerData(name, file, self, 4)
            self.scalers.append(scaler_data)

    def do_upscale(self, img, selected_model):
        try:
            model = self.load_model(selected_model)
        except Exception:
            errors.report(f"Unable to load ESRGAN model {selected_model}", exc_info=True)
            return img
        model.to(devices.device_esrgan)
        return esrgan_upscale(model, img)

    def load_model(self, path: str):
        if path.startswith("http"):
            # TODO: this doesn't use `path` at all?
            filename = modelloader.load_file_from_url(
                url=self.model_url,
                model_dir=self.model_download_path,
                file_name=f"{self.model_name}.pth",
            )
        else:
            filename = path

        return modelloader.load_spandrel_model(
            filename,
            device=('cpu' if devices.device_esrgan.type == 'mps' else None),
            expected_architecture='ESRGAN',
        )


def esrgan_upscale(model, img):
    return upscale_with_model(
        model,
        img,
        tile_size=opts.ESRGAN_tile,
        tile_overlap=opts.ESRGAN_tile_overlap,
    )
