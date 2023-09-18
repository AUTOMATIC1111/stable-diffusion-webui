import os
import numpy as np
from PIL import Image
from basicsr.utils.download_util import load_file_from_url
from modules.upscaler import Upscaler, UpscalerData
from modules.shared import opts, device, log
from modules import modelloader


class UpscalerRealESRGAN(Upscaler):
    def __init__(self, path):
        self.name = "RealESRGAN"
        self.model_path = path
        super().__init__()
        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet # pylint: disable=unused-import
            from realesrgan import RealESRGANer # pylint: disable=unused-import
            from realesrgan.archs.srvgg_arch import SRVGGNetCompact # pylint: disable=unused-import
            self.enable = True
            self.scalers = []
            scalers = self.load_models(path)
            local_model_paths = self.find_models(ext_filter=[".pth"])
            for scaler in scalers:
                if scaler.local_data_path.startswith("http"):
                    filename = modelloader.friendly_name(scaler.local_data_path)
                    local_model_candidates = [local_model for local_model in local_model_paths if local_model.endswith(f"{filename}.pth")]
                    if local_model_candidates:
                        scaler.local_data_path = local_model_candidates[0]
                if scaler.name in opts.realesrgan_enabled_models:
                    self.scalers.append(scaler)
        except Exception as e:
            log.error(f"Error loading Real-ESRGAN: model={path} {e}")
            self.enable = False
            self.scalers = []

    def do_upscale(self, img, selected_model):
        if not self.enable:
            return img

        try:
            from realesrgan import RealESRGANer
        except Exception:
            log.error("Error importing Real-ESRGAN:")
            return img

        info = self.load_model(selected_model)
        if info is None or not os.path.exists(info.local_data_path):
            return img

        upsampler = RealESRGANer(
            scale=info.scale,
            model_path=info.local_data_path,
            model=info.model(),
            half=not opts.no_half and not opts.upcast_sampling,
            tile=opts.ESRGAN_tile,
            tile_pad=opts.ESRGAN_tile_overlap,
            device=device,
        )

        upsampled = upsampler.enhance(np.array(img), outscale=info.scale)[0]

        image = Image.fromarray(upsampled)
        return image

    def load_model(self, path):
        try:
            info = next(iter([scaler for scaler in self.scalers if scaler.data_path == path]), None)
            if info is None:
                log.error(f"Model failed loading: type=R-ESRGAN model={info.name}")
                return None
            if info.local_data_path.startswith("http"):
                info.local_data_path = load_file_from_url(url=info.data_path, model_dir=self.model_download_path, progress=True)
            log.info(f"Model loaded: type=R-ESRGAN model={info.name}")
            return info
        except Exception as e:
            log.error(f"Model failed loading: type=R-ESRGAN model={info.name} {e}")
        return None

    def load_models(self, _):
        return get_realesrgan_models(self)


def get_realesrgan_models(scaler):
    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan.archs.srvgg_arch import SRVGGNetCompact
        models = [
            UpscalerData(
                name="R-ESRGAN General 4xV3",
                path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
                scale=4,
                upscaler=scaler,
                model=lambda: SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
            ),
            UpscalerData(
                name="R-ESRGAN General WDN 4xV3",
                path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth",
                scale=4,
                upscaler=scaler,
                model=lambda: SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
            ),
            UpscalerData(
                name="R-ESRGAN AnimeVideo",
                path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth",
                scale=4,
                upscaler=scaler,
                model=lambda: SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
            ),
            UpscalerData(
                name="R-ESRGAN 4x+",
                path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
                scale=4,
                upscaler=scaler,
                model=lambda: RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            ),
            UpscalerData(
                name="R-ESRGAN 4x+ Anime6B",
                path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
                scale=4,
                upscaler=scaler,
                model=lambda: RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
            ),
            UpscalerData(
                name="R-ESRGAN 2x+",
                path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
                scale=2,
                upscaler=scaler,
                model=lambda: RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
            ),
        ]
        return models
    except Exception as e:
        log.error(f'Error creating Real-ESRGAN models list: {e}')
        return []
