import os

from modules import modelloader, errors
from modules.shared import cmd_opts, opts
from modules.upscaler import Upscaler, UpscalerData
from modules.upscaler_utils import upscale_with_model


class UpscalerDAT(Upscaler):
    def __init__(self, user_path):
        self.name = "DAT"
        self.user_path = user_path
        self.scalers = []
        super().__init__()

        for file in self.find_models(ext_filter=[".pt", ".pth"]):
            name = modelloader.friendly_name(file)
            scaler_data = UpscalerData(name, file, upscaler=self, scale=None)
            self.scalers.append(scaler_data)

        for model in get_dat_models(self):
            if model.name in opts.dat_enabled_models:
                self.scalers.append(model)

    def do_upscale(self, img, path):
        try:
            info = self.load_model(path)
        except Exception:
            errors.report(f"Unable to load DAT model {path}", exc_info=True)
            return img

        model_descriptor = modelloader.load_spandrel_model(
            info.local_data_path,
            device=self.device,
            prefer_half=(not cmd_opts.no_half and not cmd_opts.upcast_sampling),
            expected_architecture="DAT",
        )
        return upscale_with_model(
            model_descriptor,
            img,
            tile_size=opts.DAT_tile,
            tile_overlap=opts.DAT_tile_overlap,
        )

    def load_model(self, path):
        for scaler in self.scalers:
            if scaler.data_path == path:
                if scaler.local_data_path.startswith("http"):
                    scaler.local_data_path = modelloader.load_file_from_url(
                        scaler.data_path,
                        model_dir=self.model_download_path,
                    )
                if not os.path.exists(scaler.local_data_path):
                    raise FileNotFoundError(f"DAT data missing: {scaler.local_data_path}")
                return scaler
        raise ValueError(f"Unable to find model info: {path}")


def get_dat_models(scaler):
    return [
        UpscalerData(
            name="DAT x2",
            path="https://github.com/n0kovo/dat_upscaler_models/raw/main/DAT/DAT_x2.pth",
            scale=2,
            upscaler=scaler,
        ),
        UpscalerData(
            name="DAT x3",
            path="https://github.com/n0kovo/dat_upscaler_models/raw/main/DAT/DAT_x3.pth",
            scale=3,
            upscaler=scaler,
        ),
        UpscalerData(
            name="DAT x4",
            path="https://github.com/n0kovo/dat_upscaler_models/raw/main/DAT/DAT_x4.pth",
            scale=4,
            upscaler=scaler,
        ),
    ]
