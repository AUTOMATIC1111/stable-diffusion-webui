# 外部から簡単にupscalerを呼ぶためのスクリプト
# 単体で動くようにモデル定義も含めている

import argparse
import glob
import os
import cv2
from diffusers import AutoencoderKL

from typing import Dict, List
import numpy as np

import torch
from torch import nn
from tqdm import tqdm
from PIL import Image


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock, self).__init__()

        if out_channels is None:
            out_channels = in_channels

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu2 = nn.ReLU(inplace=True)  # このReLUはresidualに足す前にかけるほうがいいかも

        # initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual

        out = self.relu2(out)

        return out


class Upscaler(nn.Module):
    def __init__(self):
        super(Upscaler, self).__init__()

        # define layers
        # latent has 4 channels

        self.conv1 = nn.Conv2d(4, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu1 = nn.ReLU(inplace=True)

        # resblocks
        # 数の暴力で20個：次元数を増やすよりもブロックを増やしたほうがreceptive fieldが広がるはずだぞ
        self.resblock1 = ResidualBlock(128)
        self.resblock2 = ResidualBlock(128)
        self.resblock3 = ResidualBlock(128)
        self.resblock4 = ResidualBlock(128)
        self.resblock5 = ResidualBlock(128)
        self.resblock6 = ResidualBlock(128)
        self.resblock7 = ResidualBlock(128)
        self.resblock8 = ResidualBlock(128)
        self.resblock9 = ResidualBlock(128)
        self.resblock10 = ResidualBlock(128)
        self.resblock11 = ResidualBlock(128)
        self.resblock12 = ResidualBlock(128)
        self.resblock13 = ResidualBlock(128)
        self.resblock14 = ResidualBlock(128)
        self.resblock15 = ResidualBlock(128)
        self.resblock16 = ResidualBlock(128)
        self.resblock17 = ResidualBlock(128)
        self.resblock18 = ResidualBlock(128)
        self.resblock19 = ResidualBlock(128)
        self.resblock20 = ResidualBlock(128)

        # last convs
        self.conv2 = nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)

        # final conv: output 4 channels
        self.conv_final = nn.Conv2d(64, 4, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))

        # initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

        # initialize final conv weights to 0: 流行りのzero conv
        nn.init.constant_(self.conv_final.weight, 0)

    def forward(self, x):
        inp = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        # いくつかのresblockを通した後に、residualを足すことで精度向上と学習速度向上が見込めるはず
        residual = x
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)
        x = x + residual
        residual = x
        x = self.resblock5(x)
        x = self.resblock6(x)
        x = self.resblock7(x)
        x = self.resblock8(x)
        x = x + residual
        residual = x
        x = self.resblock9(x)
        x = self.resblock10(x)
        x = self.resblock11(x)
        x = self.resblock12(x)
        x = x + residual
        residual = x
        x = self.resblock13(x)
        x = self.resblock14(x)
        x = self.resblock15(x)
        x = self.resblock16(x)
        x = x + residual
        residual = x
        x = self.resblock17(x)
        x = self.resblock18(x)
        x = self.resblock19(x)
        x = self.resblock20(x)
        x = x + residual

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)

        # ここにreluを入れないほうがいい気がする

        x = self.conv_final(x)

        # network estimates the difference between the input and the output
        x = x + inp

        return x

    def support_latents(self) -> bool:
        return False

    def upscale(
        self,
        vae: AutoencoderKL,
        lowreso_images: List[Image.Image],
        lowreso_latents: torch.Tensor,
        dtype: torch.dtype,
        width: int,
        height: int,
        batch_size: int = 1,
        vae_batch_size: int = 1,
    ):
        # assertion
        assert lowreso_images is not None, "Upscaler requires lowreso image"

        # make upsampled image with lanczos4
        upsampled_images = []
        for lowreso_image in lowreso_images:
            upsampled_image = np.array(lowreso_image.resize((width, height), Image.LANCZOS))
            upsampled_images.append(upsampled_image)

        # convert to tensor: this tensor is too large to be converted to cuda
        upsampled_images = [torch.from_numpy(upsampled_image).permute(2, 0, 1).float() for upsampled_image in upsampled_images]
        upsampled_images = torch.stack(upsampled_images, dim=0)
        upsampled_images = upsampled_images.to(dtype)

        # normalize to [-1, 1]
        upsampled_images = upsampled_images / 127.5 - 1.0

        # convert upsample images to latents with batch size
        # print("Encoding upsampled (LANCZOS4) images...")
        upsampled_latents = []
        for i in tqdm(range(0, upsampled_images.shape[0], vae_batch_size)):
            batch = upsampled_images[i : i + vae_batch_size].to(vae.device)
            with torch.no_grad():
                batch = vae.encode(batch).latent_dist.sample()
            upsampled_latents.append(batch)

        upsampled_latents = torch.cat(upsampled_latents, dim=0)

        # upscale (refine) latents with this model with batch size
        print("Upscaling latents...")
        upscaled_latents = []
        for i in range(0, upsampled_latents.shape[0], batch_size):
            with torch.no_grad():
                upscaled_latents.append(self.forward(upsampled_latents[i : i + batch_size]))
        upscaled_latents = torch.cat(upscaled_latents, dim=0)

        return upscaled_latents * 0.18215


# external interface: returns a model
def create_upscaler(**kwargs):
    weights = kwargs["weights"]
    model = Upscaler()

    print(f"Loading weights from {weights}...")
    if os.path.splitext(weights)[1] == ".safetensors":
        from safetensors.torch import load_file

        sd = load_file(weights)
    else:
        sd = torch.load(weights, map_location=torch.device("cpu"))
    model.load_state_dict(sd)
    return model


# another interface: upscale images with a model for given images from command line
def upscale_images(args: argparse.Namespace):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    us_dtype = torch.float16  # TODO: support fp32/bf16
    os.makedirs(args.output_dir, exist_ok=True)

    # load VAE with Diffusers
    assert args.vae_path is not None, "VAE path is required"
    print(f"Loading VAE from {args.vae_path}...")
    vae = AutoencoderKL.from_pretrained(args.vae_path, subfolder="vae")
    vae.to(DEVICE, dtype=us_dtype)

    # prepare model
    print("Preparing model...")
    upscaler: Upscaler = create_upscaler(weights=args.weights)
    # print("Loading weights from", args.weights)
    # upscaler.load_state_dict(torch.load(args.weights))
    upscaler.eval()
    upscaler.to(DEVICE, dtype=us_dtype)

    # load images
    image_paths = glob.glob(args.image_pattern)
    images = []
    for image_path in image_paths:
        image = Image.open(image_path)
        image = image.convert("RGB")

        # make divisible by 8
        width = image.width
        height = image.height
        if width % 8 != 0:
            width = width - (width % 8)
        if height % 8 != 0:
            height = height - (height % 8)
        if width != image.width or height != image.height:
            image = image.crop((0, 0, width, height))

        images.append(image)

    # debug output
    if args.debug:
        for image, image_path in zip(images, image_paths):
            image_debug = image.resize((image.width * 2, image.height * 2), Image.LANCZOS)

            basename = os.path.basename(image_path)
            basename_wo_ext, ext = os.path.splitext(basename)
            dest_file_name = os.path.join(args.output_dir, f"{basename_wo_ext}_lanczos4{ext}")
            image_debug.save(dest_file_name)

    # upscale
    print("Upscaling...")
    upscaled_latents = upscaler.upscale(
        vae, images, None, us_dtype, width * 2, height * 2, batch_size=args.batch_size, vae_batch_size=args.vae_batch_size
    )
    upscaled_latents /= 0.18215

    # decode with batch
    print("Decoding...")
    upscaled_images = []
    for i in tqdm(range(0, upscaled_latents.shape[0], args.vae_batch_size)):
        with torch.no_grad():
            batch = vae.decode(upscaled_latents[i : i + args.vae_batch_size]).sample
        batch = batch.to("cpu")
        upscaled_images.append(batch)
    upscaled_images = torch.cat(upscaled_images, dim=0)

    # tensor to numpy
    upscaled_images = upscaled_images.permute(0, 2, 3, 1).numpy()
    upscaled_images = (upscaled_images + 1.0) * 127.5
    upscaled_images = upscaled_images.clip(0, 255).astype(np.uint8)

    upscaled_images = upscaled_images[..., ::-1]

    # save images
    for i, image in enumerate(upscaled_images):
        basename = os.path.basename(image_paths[i])
        basename_wo_ext, ext = os.path.splitext(basename)
        dest_file_name = os.path.join(args.output_dir, f"{basename_wo_ext}_upscaled{ext}")
        cv2.imwrite(dest_file_name, image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae_path", type=str, default=None, help="VAE path")
    parser.add_argument("--weights", type=str, default=None, help="Weights path")
    parser.add_argument("--image_pattern", type=str, default=None, help="Image pattern")
    parser.add_argument("--output_dir", type=str, default=".", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--vae_batch_size", type=int, default=1, help="VAE batch size")
    parser.add_argument("--debug", action="store_true", help="Debug mode")

    args = parser.parse_args()
    upscale_images(args)
