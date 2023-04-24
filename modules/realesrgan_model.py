import os
import sys

import numpy as np
from PIL import Image
from basicsr.utils.download_util import load_file_from_url

from modules.upscaler import Upscaler, UpscalerData
from modules.shared import cmd_opts, opts, device
import modules.errors as errors


# DML ISSUE: Some tensors turn 0 after Extended Slices.
def realesrgan_tile_process_dml_fix(self):
    import math
    import torch
    batch, channel, height, width = self.img.shape
    output_height = height * self.scale
    output_width = width * self.scale
    output_shape = (batch, channel, output_height, output_width)

    # start with black image
    self.output = self.img.new_zeros(output_shape, device='cpu' if self.device.type == 'privateuseone' else self.device)
    tiles_x = math.ceil(width / self.tile_size)
    tiles_y = math.ceil(height / self.tile_size)

    # loop over all tiles
    for y in range(tiles_y):
        for x in range(tiles_x):
            # extract tile from input image
            ofs_x = x * self.tile_size
            ofs_y = y * self.tile_size
            # input tile area on total image
            input_start_x = ofs_x
            input_end_x = min(ofs_x + self.tile_size, width)
            input_start_y = ofs_y
            input_end_y = min(ofs_y + self.tile_size, height)

            # input tile area on total image with padding
            input_start_x_pad = max(input_start_x - self.tile_pad, 0)
            input_end_x_pad = min(input_end_x + self.tile_pad, width)
            input_start_y_pad = max(input_start_y - self.tile_pad, 0)
            input_end_y_pad = min(input_end_y + self.tile_pad, height)

            # input tile dimensions
            input_tile_width = input_end_x - input_start_x
            input_tile_height = input_end_y - input_start_y
            tile_idx = y * tiles_x + x + 1
            input_tile = self.img[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]

            # upscale tile
            try:
                with torch.no_grad():
                    output_tile = self.model(input_tile)
                    output_tile = output_tile.cpu()
            except RuntimeError as error:
                print('Error', error)
            print(f'\tTile {tile_idx}/{tiles_x * tiles_y}')

            # output tile area on total image
            output_start_x = input_start_x * self.scale
            output_end_x = input_end_x * self.scale
            output_start_y = input_start_y * self.scale
            output_end_y = input_end_y * self.scale

            # output tile area without padding
            output_start_x_tile = (input_start_x - input_start_x_pad) * self.scale
            output_end_x_tile = output_start_x_tile + input_tile_width * self.scale
            output_start_y_tile = (input_start_y - input_start_y_pad) * self.scale
            output_end_y_tile = output_start_y_tile + input_tile_height * self.scale

            # put tile into output image
            self.output[:, :, output_start_y:output_end_y,
                        output_start_x:output_end_x] = output_tile[:, :, output_start_y_tile:output_end_y_tile,
                                                                    output_start_x_tile:output_end_x_tile]
    self.output = self.output.to(self.device)


class UpscalerRealESRGAN(Upscaler):
    def __init__(self, path):
        self.name = "RealESRGAN"
        self.user_path = path
        super().__init__()
        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
            from realesrgan.archs.srvgg_arch import SRVGGNetCompact
            self.enable = True
            self.scalers = []
            scalers = self.load_models(path)
            for scaler in scalers:
                if scaler.name in opts.realesrgan_enabled_models:
                    self.scalers.append(scaler)

        except Exception as e:
            errors.display(e, 'real-esrgan')
            self.enable = False
            self.scalers = []

    def do_upscale(self, img, path):
        if not self.enable:
            return img

        try:
            from realesrgan import RealESRGANer
            if device.type == 'privateuseone':
                RealESRGANer.tile_process = realesrgan_tile_process_dml_fix
        except:
            print("Error importing Real-ESRGAN:", file=sys.stderr)
            return img

        info = self.load_model(path)
        if not os.path.exists(info.local_data_path):
            print("Unable to load RealESRGAN model: %s" % info.name)
            return img

        upsampler = RealESRGANer(
            scale=info.scale,
            model_path=info.local_data_path,
            model=info.model(),
            half=not cmd_opts.no_half and not opts.upcast_sampling,
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
                print(f"Unable to find model info: {path}")
                return None

            info.local_data_path = load_file_from_url(url=info.data_path, model_dir=self.model_path, progress=True)
            return info
        except Exception as e:
            errors.display(e, 'real-esrgan model list')
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
        print("Error creating Real-ESRGAN models list", file=sys.stderr)
        return []
