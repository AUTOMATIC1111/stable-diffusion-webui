# Copyright 2023 Bingxin Ke, ETH Zurich. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------------------------
# If you find this code useful, we kindly ask you to cite our paper in your work.
# Please find bibtex at: https://github.com/prs-eth/Marigold#-citation
# More information about the method can be found at https://marigoldmonodepth.github.io
# --------------------------------------------------------------------------


from typing import Dict, Union

import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm.auto import tqdm
from PIL import Image

from diffusers import (
    DiffusionPipeline,
    DDIMScheduler,
    UNet2DConditionModel,
    AutoencoderKL,
)
from diffusers.utils import BaseOutput
from transformers import CLIPTextModel, CLIPTokenizer

from .util.image_util import chw2hwc, colorize_depth_maps, resize_max_res
from .util.batchsize import find_batch_size
from .util.ensemble import ensemble_depths


class MarigoldDepthOutput(BaseOutput):
    """
    Output class for Marigold monocular depth prediction pipeline.

    Args:
        depth_np (`np.ndarray`):
            Predicted depth map, with depth values in the range of [0, 1].
        depth_colored (`PIL.Image.Image`):
            Colorized depth map, with the shape of [3, H, W] and values in [0, 1].
        uncertainty (`None` or `np.ndarray`):
            Uncalibrated uncertainty(MAD, median absolute deviation) coming from ensembling.
    """

    depth_np: np.ndarray
    depth_colored: Image.Image
    uncertainty: Union[None, np.ndarray]


class MarigoldPipeline(DiffusionPipeline):
    """
    Pipeline for monocular depth estimation using Marigold: https://marigoldmonodepth.github.io.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        unet (`UNet2DConditionModel`):
            Conditional U-Net to denoise the depth latent, conditioned on image latent.
        vae (`AutoencoderKL`):
            Variational Auto-Encoder (VAE) Model to encode and decode images and depth maps
            to and from latent representations.
        scheduler (`DDIMScheduler`):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents.
        text_encoder (`CLIPTextModel`):
            Text-encoder, for empty text embedding.
        tokenizer (`CLIPTokenizer`):
            CLIP tokenizer.
    """

    rgb_latent_scale_factor = 0.18215
    depth_latent_scale_factor = 0.18215

    def __init__(
        self,
        unet: UNet2DConditionModel,
        vae: AutoencoderKL,
        scheduler: DDIMScheduler,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
    ):
        super().__init__()

        self.register_modules(
            unet=unet,
            vae=vae,
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
        )

        self.empty_text_embed = None

    @torch.no_grad()
    def __call__(
        self,
        input_image: Image,
        denoising_steps: int = 10,
        ensemble_size: int = 10,
        processing_res: int = 768,
        match_input_res: bool = True,
        batch_size: int = 0,
        color_map: str = "Spectral",
        show_progress_bar: bool = True,
        ensemble_kwargs: Dict = None,
    ) -> MarigoldDepthOutput:
        """
        Function invoked when calling the pipeline.

        Args:
            input_image (`Image`):
                Input RGB (or gray-scale) image.
            processing_res (`int`, *optional*, defaults to `768`):
                Maximum resolution of processing.
                If set to 0: will not resize at all.
            match_input_res (`bool`, *optional*, defaults to `True`):
                Resize depth prediction to match input resolution.
                Only valid if `limit_input_res` is not None.
            denoising_steps (`int`, *optional*, defaults to `10`):
                Number of diffusion denoising steps (DDIM) during inference.
            ensemble_size (`int`, *optional*, defaults to `10`):
                Number of predictions to be ensembled.
            batch_size (`int`, *optional*, defaults to `0`):
                Inference batch size, no bigger than `num_ensemble`.
                If set to 0, the script will automatically decide the proper batch size.
            show_progress_bar (`bool`, *optional*, defaults to `True`):
                Display a progress bar of diffusion denoising.
            color_map (`str`, *optional*, defaults to `"Spectral"`):
                Colormap used to colorize the depth map.
            ensemble_kwargs (`dict`, *optional*, defaults to `None`):
                Arguments for detailed ensembling settings.
        Returns:
            `MarigoldDepthOutput`: Output class for Marigold monocular depth prediction pipeline, including:
            - **depth_np** (`np.ndarray`) Predicted depth map, with depth values in the range of [0, 1]
            - **depth_colored** (`PIL.Image.Image`) Colorized depth map, with the shape of [3, H, W] and values in [0, 1]
            - **uncertainty** (`None` or `np.ndarray`) Uncalibrated uncertainty(MAD, median absolute deviation)
                    coming from ensembling. None if `ensemble_size = 1`
        """

        device = self.device
        input_size = input_image.size

        if not match_input_res:
            assert (
                processing_res is not None
            ), "Value error: `resize_output_back` is only valid with "
        assert processing_res >= 0
        assert denoising_steps >= 1
        assert ensemble_size >= 1

        # ----------------- Image Preprocess -----------------
        # Resize image
        if processing_res > 0:
            input_image = resize_max_res(
                input_image, max_edge_resolution=processing_res
            )
        # Convert the image to RGB, to 1.remove the alpha channel 2.convert B&W to 3-channel
        input_image = input_image.convert("RGB")
        image = np.asarray(input_image)

        # Normalize rgb values
        rgb = np.transpose(image, (2, 0, 1))  # [H, W, rgb] -> [rgb, H, W]
        rgb_norm = rgb / 255.0
        rgb_norm = torch.from_numpy(rgb_norm).to(self.dtype)
        rgb_norm = rgb_norm.to(device)
        assert rgb_norm.min() >= 0.0 and rgb_norm.max() <= 1.0

        # ----------------- Predicting depth -----------------
        # Batch repeated input image
        duplicated_rgb = torch.stack([rgb_norm] * ensemble_size)
        single_rgb_dataset = TensorDataset(duplicated_rgb)
        if batch_size > 0:
            _bs = batch_size
        else:
            _bs = find_batch_size(
                ensemble_size=ensemble_size,
                input_res=max(rgb_norm.shape[1:]),
                dtype=self.dtype,
            )

        single_rgb_loader = DataLoader(
            single_rgb_dataset, batch_size=_bs, shuffle=False
        )

        # Predict depth maps (batched)
        depth_pred_ls = []
        if show_progress_bar:
            iterable = tqdm(
                single_rgb_loader, desc=" " * 2 + "Inference batches", leave=False
            )
        else:
            iterable = single_rgb_loader
        for batch in iterable:
            (batched_img,) = batch
            depth_pred_raw = self.single_infer(
                rgb_in=batched_img,
                num_inference_steps=denoising_steps,
                show_pbar=show_progress_bar,
            )
            depth_pred_ls.append(depth_pred_raw.detach().clone())
        depth_preds = torch.concat(depth_pred_ls, axis=0).squeeze()
        torch.cuda.empty_cache()  # clear vram cache for ensembling

        # ----------------- Test-time ensembling -----------------
        if ensemble_size > 1:
            depth_pred, pred_uncert = ensemble_depths(
                depth_preds, **(ensemble_kwargs or {})
            )
        else:
            depth_pred = depth_preds
            pred_uncert = None

        # ----------------- Post processing -----------------
        # Scale prediction to [0, 1]
        min_d = torch.min(depth_pred)
        max_d = torch.max(depth_pred)
        depth_pred = (depth_pred - min_d) / (max_d - min_d)

        # Convert to numpy
        depth_pred = depth_pred.cpu().numpy().astype(np.float32)

        # Resize back to original resolution
        if match_input_res:
            pred_img = Image.fromarray(depth_pred)
            pred_img = pred_img.resize(input_size)
            depth_pred = np.asarray(pred_img)

        # Clip output range
        depth_pred = depth_pred.clip(0, 1)

        # Colorize
        depth_colored = colorize_depth_maps(
            depth_pred, 0, 1, cmap=color_map
        ).squeeze()  # [3, H, W], value in (0, 1)
        depth_colored = (depth_colored * 255).astype(np.uint8)
        depth_colored_hwc = chw2hwc(depth_colored)
        depth_colored_img = Image.fromarray(depth_colored_hwc)
        return MarigoldDepthOutput(
            depth_np=depth_pred,
            depth_colored=depth_colored_img,
            uncertainty=pred_uncert,
        )

    def __encode_empty_text(self):
        """
        Encode text embedding for empty prompt
        """
        prompt = ""
        text_inputs = self.tokenizer(
            prompt,
            padding="do_not_pad",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.text_encoder.device)
        self.empty_text_embed = self.text_encoder(text_input_ids)[0].to(self.dtype)

    @torch.no_grad()
    def single_infer(
        self, rgb_in: torch.Tensor, num_inference_steps: int, show_pbar: bool
    ) -> torch.Tensor:
        """
        Perform an individual depth prediction without ensembling.

        Args:
            rgb_in (`torch.Tensor`):
                Input RGB image.
            num_inference_steps (`int`):
                Number of diffusion denoisign steps (DDIM) during inference.
            show_pbar (`bool`):
                Display a progress bar of diffusion denoising.
        Returns:
            `torch.Tensor`: Predicted depth map.
        """
        device = rgb_in.device

        # Set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps  # [T]

        # Encode image
        rgb_latent = self.encode_rgb(rgb_in)

        # Initial depth map (noise)
        depth_latent = torch.randn(
            rgb_latent.shape, device=device, dtype=self.dtype
        )  # [B, 4, h, w]

        # Batched empty text embedding
        if self.empty_text_embed is None:
            self.__encode_empty_text()
        batch_empty_text_embed = self.empty_text_embed.repeat(
            (rgb_latent.shape[0], 1, 1)
        )  # [B, 2, 1024]

        # Denoising loop
        if show_pbar:
            iterable = tqdm(
                enumerate(timesteps),
                total=len(timesteps),
                leave=False,
                desc=" " * 4 + "Diffusion denoising",
            )
        else:
            iterable = enumerate(timesteps)

        for _i, t in iterable:
            unet_input = torch.cat(
                [rgb_latent, depth_latent], dim=1
            )  # this order is important

            # predict the noise residual
            noise_pred = self.unet(
                unet_input, t, encoder_hidden_states=batch_empty_text_embed
            ).sample  # [B, 4, h, w]

            # compute the previous noisy sample x_t -> x_t-1
            depth_latent = self.scheduler.step(noise_pred, t, depth_latent).prev_sample
        torch.cuda.empty_cache()
        depth = self.decode_depth(depth_latent)

        # clip prediction
        depth = torch.clip(depth, -1.0, 1.0)
        # shift to [0, 1]
        depth = (depth + 1.0) / 2.0

        return depth

    def encode_rgb(self, rgb_in: torch.Tensor) -> torch.Tensor:
        """
        Encode RGB image into latent.

        Args:
            rgb_in (`torch.Tensor`):
                Input RGB image to be encoded.

        Returns:
            `torch.Tensor`: Image latent.
        """
        # encode
        h = self.vae.encoder(rgb_in)
        moments = self.vae.quant_conv(h)
        mean, _logvar = torch.chunk(moments, 2, dim=1)
        # scale latent
        rgb_latent = mean * self.rgb_latent_scale_factor
        return rgb_latent

    def decode_depth(self, depth_latent: torch.Tensor) -> torch.Tensor:
        """
        Decode depth latent into depth map.

        Args:
            depth_latent (`torch.Tensor`):
                Depth latent to be decoded.

        Returns:
            `torch.Tensor`: Decoded depth map.
        """
        # scale latent
        depth_latent = depth_latent / self.depth_latent_scale_factor
        # decode
        z = self.vae.post_quant_conv(depth_latent)
        stacked = self.vae.decoder(z)
        # mean of output channels
        depth_mean = stacked.mean(dim=1, keepdim=True)
        return depth_mean
