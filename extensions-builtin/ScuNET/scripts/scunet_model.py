import sys

import PIL.Image
import numpy as np
import torch
from tqdm import tqdm

import modules.upscaler
from modules import devices, modelloader, script_callbacks, errors
from scunet_model_arch import SCUNet

from modules.modelloader import load_file_from_url
from modules.shared import opts


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

    @staticmethod
    @torch.no_grad()
    def tiled_inference(img, model):
        # test the image tile by tile
        h, w = img.shape[2:]
        tile = opts.SCUNET_tile
        tile_overlap = opts.SCUNET_tile_overlap
        if tile == 0:
            return model(img)

        device = devices.get_device_for('scunet')
        assert tile % 8 == 0, "tile size should be a multiple of window_size"
        sf = 1

        stride = tile - tile_overlap
        h_idx_list = list(range(0, h - tile, stride)) + [h - tile]
        w_idx_list = list(range(0, w - tile, stride)) + [w - tile]
        E = torch.zeros(1, 3, h * sf, w * sf, dtype=img.dtype, device=device)
        W = torch.zeros_like(E, dtype=devices.dtype, device=device)

        with tqdm(total=len(h_idx_list) * len(w_idx_list), desc="ScuNET tiles") as pbar:
            for h_idx in h_idx_list:

                for w_idx in w_idx_list:

                    in_patch = img[..., h_idx: h_idx + tile, w_idx: w_idx + tile]

                    out_patch = model(in_patch)
                    out_patch_mask = torch.ones_like(out_patch)

                    E[
                        ..., h_idx * sf: (h_idx + tile) * sf, w_idx * sf: (w_idx + tile) * sf
                    ].add_(out_patch)
                    W[
                        ..., h_idx * sf: (h_idx + tile) * sf, w_idx * sf: (w_idx + tile) * sf
                    ].add_(out_patch_mask)
                    pbar.update(1)
        output = E.div_(W)

        return output

    def do_upscale(self, img: PIL.Image.Image, selected_file):

        devices.torch_gc()

        try:
            model = self.load_model(selected_file)
        except Exception as e:
            print(f"ScuNET: Unable to load model from {selected_file}: {e}", file=sys.stderr)
            return img

        device = devices.get_device_for('scunet')
        tile = opts.SCUNET_tile
        h, w = img.height, img.width
        np_img = np.array(img)
        np_img = np_img[:, :, ::-1]  # RGB to BGR
        np_img = np_img.transpose((2, 0, 1)) / 255  # HWC to CHW
        torch_img = torch.from_numpy(np_img).float().unsqueeze(0).to(device)  # type: ignore

        if tile > h or tile > w:
            _img = torch.zeros(1, 3, max(h, tile), max(w, tile), dtype=torch_img.dtype, device=torch_img.device)
            _img[:, :, :h, :w] = torch_img # pad image
            torch_img = _img

        torch_output = self.tiled_inference(torch_img, model).squeeze(0)
        torch_output = torch_output[:, :h * 1, :w * 1] # remove padding, if any
        np_output: np.ndarray = torch_output.float().cpu().clamp_(0, 1).numpy()
        del torch_img, torch_output
        devices.torch_gc()

        output = np_output.transpose((1, 2, 0))  # CHW to HWC
        output = output[:, :, ::-1]  # BGR to RGB
        return PIL.Image.fromarray((output * 255).astype(np.uint8))

    def load_model(self, path: str):
        device = devices.get_device_for('scunet')
        if path.startswith("http"):
            # TODO: this doesn't use `path` at all?
            filename = load_file_from_url(self.model_url, model_dir=self.model_download_path, file_name=f"{self.name}.pth")
        else:
            filename = path
        model = SCUNet(in_nc=3, config=[4, 4, 4, 4, 4, 4, 4], dim=64)
        model.load_state_dict(torch.load(filename), strict=True)
        model.eval()
        for _, v in model.named_parameters():
            v.requires_grad = False
        model = model.to(device)

        return model


def on_ui_settings():
    import gradio as gr
    from modules import shared

    shared.opts.add_option("SCUNET_tile", shared.OptionInfo(256, "Tile size for SCUNET upscalers.", gr.Slider, {"minimum": 0, "maximum": 512, "step": 16}, section=('upscaling', "Upscaling")).info("0 = no tiling"))
    shared.opts.add_option("SCUNET_tile_overlap", shared.OptionInfo(8, "Tile overlap for SCUNET upscalers.", gr.Slider, {"minimum": 0, "maximum": 64, "step": 1}, section=('upscaling', "Upscaling")).info("Low values = visible seam"))


script_callbacks.on_ui_settings(on_ui_settings)
