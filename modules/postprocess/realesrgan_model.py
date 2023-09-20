import os
import numpy as np
from PIL import Image
from basicsr.archs.rrdbnet_arch import RRDBNet
from modules.postprocess.realesrgan_model_arch import SRVGGNetCompact
from modules.upscaler import Upscaler
from modules.shared import opts, device, log


class UpscalerRealESRGAN(Upscaler):
    def __init__(self, dirname):
        self.name = "RealESRGAN"
        self.user_path = dirname
        super().__init__()
        self.scalers = self.find_scalers()
        for scaler in self.scalers:
            if scaler.name == 'RealESRGAN 2x+':
                scaler.model = lambda: RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
                scaler.scale = 2
            elif scaler.name == 'RealESRGAN 4x+ Anime6B':
                scaler.model=lambda: RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
            elif scaler.name == 'RealESRGAN 4x General V3':
                scaler.model = lambda: SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
            elif scaler.name == 'RealESRGAN 4x General WDN V3':
                scaler.model = lambda: SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
            elif scaler.name == 'RealESRGAN AnimeVideo V3':
                scaler.model=lambda: SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
            elif scaler.name == 'RealESRGAN 4x+':
                scaler.model = lambda: RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            else:
                log.error(f"Upscaler unrecognized model: type={self.name} model={scaler.name}")

    def load_model(self, path): # pylint: disable=unused-argument
        pass

    def do_upscale(self, img, selected_model):
        if not self.enable:
            return img
        try:
            from modules.postprocess.realesrgan_model_arch import RealESRGANer
        except Exception:
            log.error("Error importing Real-ESRGAN:")
            return img

        info = self.find_model(selected_model)
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
