import sys
import traceback
from collections import namedtuple

import numpy as np
from PIL import Image
from realesrgan import RealESRGANer

import modules.images
from modules.shared import cmd_opts, opts

RealesrganModelInfo = namedtuple("RealesrganModelInfo", ["name", "location", "model", "netscale"])
realesrgan_models = []
have_realesrgan = False


def get_realesrgan_models():
    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
        from realesrgan.archs.srvgg_arch import SRVGGNetCompact
        models = [
            RealesrganModelInfo(
                name="Real-ESRGAN General x4x3",
                location="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
                netscale=4,
                model=lambda: SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
            ),
            RealesrganModelInfo(
                name="Real-ESRGAN General WDN x4x3",
                location="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth",
                netscale=4,
                model=lambda: SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
            ),
            RealesrganModelInfo(
                name="Real-ESRGAN AnimeVideo",
                location="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth",
                netscale=4,
                model=lambda: SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
            ),
            RealesrganModelInfo(
                name="Real-ESRGAN 4x plus",
                location="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
                netscale=4,
                model=lambda: RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            ),
            RealesrganModelInfo(
                name="Real-ESRGAN 4x plus anime 6B",
                location="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
                netscale=4,
                model=lambda: RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
            ),
            RealesrganModelInfo(
                name="Real-ESRGAN 2x plus",
                location="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
                netscale=2,
                model=lambda: RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
            ),
        ]
        return models
    except Exception as e:
        print("Error makeing Real-ESRGAN midels list:", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)


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

    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
        from realesrgan.archs.srvgg_arch import SRVGGNetCompact

        realesrgan_models = get_realesrgan_models()
        have_realesrgan = True

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
