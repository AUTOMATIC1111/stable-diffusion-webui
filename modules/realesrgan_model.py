import sys
import traceback
from collections import namedtuple
import numpy as np
from PIL import Image

import modules.images
from modules.shared import cmd_opts, opts

RealesrganModelInfo = namedtuple("RealesrganModelInfo", ["name", "location", "model", "netscale"])

realesrgan_models = []
have_realesrgan = False
RealESRGANer_constructor = None


class UpscalerRealESRGAN(modules.images.Upscaler):
    def __init__(self, upscaling, model_index):
        self.upscaling = upscaling
        self.model_index = model_index
        self.name = realesrgan_models[model_index].name

    def do_upscale(self, img):
        return upscale_with_realesrgan(img, self.upscaling, self.model_index)


def setup_realesrgan():
    global realesrgan_models
    global have_realesrgan
    global RealESRGANer_constructor

    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
        from realesrgan.archs.srvgg_arch import SRVGGNetCompact

        realesrgan_models = [
            RealesrganModelInfo(
                name="Real-ESRGAN 4x plus",
                location="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
                netscale=4, model=lambda: RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            ),
            RealesrganModelInfo(
                name="Real-ESRGAN 4x plus anime 6B",
                location="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
                netscale=4, model=lambda: RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
            ),
            RealesrganModelInfo(
                name="Real-ESRGAN 2x plus",
                location="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
                netscale=2, model=lambda: RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
            ),
        ]
        have_realesrgan = True
        RealESRGANer_constructor = RealESRGANer

        for i, model in enumerate(realesrgan_models):
            modules.shared.sd_upscalers.append(UpscalerRealESRGAN(model.netscale, i))

    except Exception:
        print("Error importing Real-ESRGAN:", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)

        realesrgan_models = [RealesrganModelInfo('None', '', 0, None)]
        have_realesrgan = False


def upscale_with_realesrgan(image, RealESRGAN_upscaling, RealESRGAN_model_index):
    if not have_realesrgan or RealESRGANer_constructor is None:
        return image

    info = realesrgan_models[RealESRGAN_model_index]

    model = info.model()
    upsampler = RealESRGANer_constructor(
        scale=info.netscale,
        model_path=info.location,
        model=model,
        half=not cmd_opts.no_half,
        tile=opts.GAN_tile,
        tile_pad=opts.ESRGAN_tile_overlap,
    )

    upsampled = upsampler.enhance(np.array(image), outscale=RealESRGAN_upscaling)[0]

    image = Image.fromarray(upsampled)
    return image
