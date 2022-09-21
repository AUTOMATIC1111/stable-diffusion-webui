import sys
import traceback
from collections import namedtuple

import numpy as np
from PIL import Image
from realesrgan import RealESRGANer

import modules.images
from modules import realesrgan_model_loader
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

        realesrgan_models = realesrgan_model_loader.get_realesrgan_models()
        have_realesrgan = True
        RealESRGANer_constructor = RealESRGANer

        for i, model in enumerate(realesrgan_models):
            if model.name in opts.realesrgan_enabled_models:
                modules.shared.sd_upscalers.append(UpscalerRealESRGAN(model.netscale, i))

    except Exception:
        print("Error importing Real-ESRGAN:", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)

        realesrgan_models = [RealesrganModelInfo('None', '', 0, None)]
        have_realesrgan = False


def upscale_with_realesrgan(image, RealESRGAN_upscaling, RealESRGAN_model_index):
    if not have_realesrgan:
        return image

    info = realesrgan_models[RealESRGAN_model_index]

    model = info.model()
    upsampler = RealESRGANer(
        scale=info.netscale,
        model_path=info.location,
        model=model,
        half=not cmd_opts.no_half,
        tile=opts.ESRGAN_tile,
        tile_pad=opts.ESRGAN_tile_overlap,
    )

    upsampled = upsampler.enhance(np.array(image), outscale=RealESRGAN_upscaling)[0]

    image = Image.fromarray(upsampled)
    return image
