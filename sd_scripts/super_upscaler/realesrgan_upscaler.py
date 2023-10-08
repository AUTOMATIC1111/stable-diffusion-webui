"""
图像放大：
a)	三次元：R-ESRGAN + ULTRASHARP（0.3权重）
b)	二次元：R-ESRGAN ANIME + ULTRASHARP（0.3权重）
"""

import os
import time

import numpy as np
from PIL import Image
import PIL
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
# from modules.paths_internal import models_path

LANCZOS = (Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS)


class RealESRGANUpscaler:
    def __init__(self, style, models_path):
        self.path = ""
        if style:
            self.path = os.path.join(models_path,'RealESRGAN/RealESRGAN_x4plus.pth') 
            self.models = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        else:
            self.path = os.path.join(models_path,'RealESRGAN/RealESRGAN_x4plus_anime_6B.pth')
            self.models = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
           

    def do_upscale(self, img):
        path = self.path
        if not os.path.exists(path):
            print("Unable to load RealESRGAN model: %s" + path)
            return img
        upsampler = RealESRGANer(
            scale=4,
            model_path=path,
            model=self.models,
            half=False,
            tile=192,
            tile_pad=8,
        )
        upsampled = upsampler.enhance(np.array(img), outscale=4)[0]
        image = Image.fromarray(upsampled)
        return image

    def upscale(self, img: PIL.Image, scale):
        dest_w = int(img.width * scale)
        dest_h = int(img.height * scale)
        for i in range(3):
            shape = (img.width, img.height)
            img = self.do_upscale(img)
            if shape == (img.width, img.height):
                break
            if img.width >= dest_w and img.height >= dest_h:
                break
        if img.width != dest_w or img.height != dest_h:
            img = img.resize((int(dest_w), int(dest_h)), resample=LANCZOS)
        return img
