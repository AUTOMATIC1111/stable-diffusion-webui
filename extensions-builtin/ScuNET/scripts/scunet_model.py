import os.path
import sys
import traceback

import PIL.Image
import numpy as np
import torch
from basicsr.utils.download_util import load_file_from_url

import modules.upscaler
from modules import devices, modelloader
from scunet_model_arch import SCUNet as net


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
            if "http" in file:
                name = self.model_name
            else:
                name = modelloader.friendly_name(file)
            if name == self.model_name2 or file == self.model_url2:
                add_model2 = False
            try:
                scaler_data = modules.upscaler.UpscalerData(name, file, self, 4)
                scalers.append(scaler_data)
            except Exception:
                print(f"Error loading ScuNET model: {file}", file=sys.stderr)
                print(traceback.format_exc(), file=sys.stderr)
        if add_model2:
            scaler_data2 = modules.upscaler.UpscalerData(self.model_name2, self.model_url2, self)
            scalers.append(scaler_data2)
        self.scalers = scalers

    def do_upscale(self, img: PIL.Image, selected_file):
        torch.cuda.empty_cache()

        model = self.load_model(selected_file)
        if model is None:
            return img

        device = devices.get_device_for('scunet')
        img = np.array(img)
        img = img[:, :, ::-1]
        img = np.moveaxis(img, 2, 0) / 255
        img = torch.from_numpy(img).float()
        img = img.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img)
        output = output.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = 255. * np.moveaxis(output, 0, 2)
        output = output.astype(np.uint8)
        output = output[:, :, ::-1]
        torch.cuda.empty_cache()
        return PIL.Image.fromarray(output, 'RGB')

    def load_model(self, path: str):
        device = devices.get_device_for('scunet')
        if "http" in path:
            filename = load_file_from_url(url=self.model_url, model_dir=self.model_path, file_name="%s.pth" % self.name,
                                          progress=True)
        else:
            filename = path
        if not os.path.exists(os.path.join(self.model_path, filename)) or filename is None:
            print(f"ScuNET: Unable to load model from {filename}", file=sys.stderr)
            return None

        model = net(in_nc=3, config=[4, 4, 4, 4, 4, 4, 4], dim=64)
        model.load_state_dict(torch.load(filename), strict=True)
        model.eval()
        for k, v in model.named_parameters():
            v.requires_grad = False
        model = model.to(device)

        return model

