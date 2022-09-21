from collections import namedtuple

RealesrganModelInfo = namedtuple("RealesrganModelInfo", ["name", "location", "model", "netscale"])


def get_realesrgan_model_names():
    names = set()
    models = get_realesrgan_models()
    for model in models:
        names.add(model.name)
    return sorted(names)


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
                model=lambda: SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4,
                                              act_type='prelu')
            ),
            RealesrganModelInfo(
                name="Real-ESRGAN General WDN x4x3",
                location="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth",
                netscale=4,
                model=lambda: SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4,
                                              act_type='prelu')
            ),
            RealesrganModelInfo(
                name="Real-ESRGAN AnimeVideo",
                location="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth",
                netscale=4,
                model=lambda: SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4,
                                              act_type='prelu')
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
        print("Exception loading models: %s" % e)
