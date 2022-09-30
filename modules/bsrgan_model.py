import os.path
import sys
import traceback

import PIL.Image
import numpy as np
import torch
from basicsr.utils.download_util import load_file_from_url

import modules.upscaler
from modules import shared, modelloader
from modules.bsrgan_model_arch import RRDBNet
from modules.paths import models_path


class UpscalerBSRGAN(modules.upscaler.Upscaler):
    def __init__(self, dirname):
        self.name = "BSRGAN"
        self.model_path = os.path.join(models_path, self.name)
        self.model_name = "BSRGAN 4x"
        self.model_url = "https://github.com/cszn/KAIR/releases/download/v1.0/BSRGAN.pth"
        self.user_path = dirname
        super().__init__()
        model_paths = self.find_models(ext_filter=[".pt", ".pth"])
        scalers = []
        if len(model_paths) == 0:
            scaler_data = modules.upscaler.UpscalerData(self.model_name, self.model_url, self, 4)
            scalers.append(scaler_data)
        for file in model_paths:
            if "http" in file:
                name = self.model_name
            else:
                name = modelloader.friendly_name(file)
            try:
                scaler_data = modules.upscaler.UpscalerData(name, file, self, 4)
                scalers.append(scaler_data)
            except Exception:
                print(f"Error loading BSRGAN model: {file}", file=sys.stderr)
                print(traceback.format_exc(), file=sys.stderr)
        self.scalers = scalers

    def do_upscale(self, img: PIL.Image, selected_file):
        torch.cuda.empty_cache()
        model = self.load_model(selected_file)
        if model is None:
            return img
        model.to(shared.device)
        torch.cuda.empty_cache()
        img = np.array(img)
        img = img[:, :, ::-1]
        img = np.moveaxis(img, 2, 0) / 255
        img = torch.from_numpy(img).float()
        img = img.unsqueeze(0).to(shared.device)
        with torch.no_grad():
            output = model(img)
        output = output.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = 255. * np.moveaxis(output, 0, 2)
        output = output.astype(np.uint8)
        output = output[:, :, ::-1]
        torch.cuda.empty_cache()
        return PIL.Image.fromarray(output, 'RGB')

    def load_model(self, path: str):
        if "http" in path:
            filename = load_file_from_url(url=self.model_url, model_dir=self.model_path, file_name="%s.pth" % self.name,
                                          progress=True)
        else:
            filename = path
        if not os.path.exists(filename) or filename is None:
            print(f"BSRGAN: Unable to load model from {filename}", file=sys.stderr)
            return None
        model = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, sf=4)  # define network
        model.load_state_dict(torch.load(filename), strict=True)
        model.eval()
        for k, v in model.named_parameters():
            v.requires_grad = False
        return model

