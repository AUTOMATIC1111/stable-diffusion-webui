import sys

import PIL.Image

import modules.upscaler
from modules import devices, errors, modelloader, script_callbacks, shared, upscaler_utils


class UpscalerScuNET(modules.upscaler.Upscaler):
    def __init__(self, dirname):
        self.name = "ScuNET"
        self.model_name = "ScuNET GAN"
        self.model_name2 = "ScuNET PSNR"
        self.model_url = "https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_real_gan.pth"
        self.model_url2 = "https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_real_psnr.pth"
        self.user_path = dirname
        super().__init__()
        model_paths = self.find_models(ext_filter=[".pth"])
        scalers = []
        add_model2 = True
        for file in model_paths:
            if file.startswith("http"):
                name = self.model_name
            else:
                name = modelloader.friendly_name(file)
            if name == self.model_name2 or file == self.model_url2:
                add_model2 = False
            try:
                scaler_data = modules.upscaler.UpscalerData(name, file, self, 4)
                scalers.append(scaler_data)
            except Exception:
                errors.report(f"Error loading ScuNET model: {file}", exc_info=True)
        if add_model2:
            scaler_data2 = modules.upscaler.UpscalerData(self.model_name2, self.model_url2, self)
            scalers.append(scaler_data2)
        self.scalers = scalers

    def do_upscale(self, img: PIL.Image.Image, selected_file):
        devices.torch_gc()
        try:
            model = self.load_model(selected_file)
        except Exception as e:
            print(f"ScuNET: Unable to load model from {selected_file}: {e}", file=sys.stderr)
            return img

        img = upscaler_utils.upscale_2(
            img,
            model,
            tile_size=shared.opts.SCUNET_tile,
            tile_overlap=shared.opts.SCUNET_tile_overlap,
            scale=1,  # ScuNET is a denoising model, not an upscaler
            desc='ScuNET',
        )
        devices.torch_gc()
        return img

    def load_model(self, path: str):
        device = devices.get_device_for('scunet')
        if path.startswith("http"):
            # TODO: this doesn't use `path` at all?
            filename = modelloader.load_file_from_url(self.model_url, model_dir=self.model_download_path, file_name=f"{self.name}.pth")
        else:
            filename = path
        return modelloader.load_spandrel_model(filename, device=device, expected_architecture='SCUNet')


def on_ui_settings():
    import gradio as gr

    shared.opts.add_option("SCUNET_tile", shared.OptionInfo(256, "Tile size for SCUNET upscalers.", gr.Slider, {"minimum": 0, "maximum": 512, "step": 16}, section=('upscaling', "Upscaling")).info("0 = no tiling"))
    shared.opts.add_option("SCUNET_tile_overlap", shared.OptionInfo(8, "Tile overlap for SCUNET upscalers.", gr.Slider, {"minimum": 0, "maximum": 64, "step": 1}, section=('upscaling', "Upscaling")).info("Low values = visible seam"))


script_callbacks.on_ui_settings(on_ui_settings)
