from __future__ import annotations

from typing import Any

from pydantic import BaseModel

# NOTE: I changed some field names from krita_config.yaml to match the API endpoints


class BaseOptions(BaseModel):
    sample_path: str = "outputs/krita-out"
    """Where to save generated images to."""


class GenerationOptions(BaseModel):
    prompt: Any = "dog"
    """Requested prompt."""
    negative_prompt: Any = ""
    """Requested negative prompt."""
    seed: int = -1
    """Seed used for noise generation. Incremented by 1 for each image rendered."""

    sampler_name: str = "k_euler_a"
    """Exact name of sampler to use. Name should follow exact spelling and capitalization as in the WebUI."""
    steps: int = 20
    """Number of steps for diffusion."""
    cfg_scale: float = 12.0
    """Guidance scale for diffusion."""

    batch_count: int = 1
    """Number of batches to render."""
    batch_size: int = 1
    """Number of images per batch to render."""

    base_size: int = 512
    """Native/base resolution of model used."""
    max_size: int = 768
    """Max input resolution allowed to prevent image artifacts."""
    tiling: bool = False
    """Whether to generate a tileable image."""


class FaceRestorationOptions(BaseModel):
    use_gfpgan: bool = False
    """Whether to use GFPGAN for face restoration."""
    face_restorer: str = "CodeFormer"
    """Exact name of face restorer to use."""
    codeformer_weight: float = 0.5
    """Strength of face restoration if using CodeFormer. 0.0 is the strongest and 1.0 is the weakest."""
    use_realesrgan: bool = False
    """Whether to use RealESRGAN models for face restoration."""
    realesrgan_model_name: str = "RealESRGAN_x4plus_anime_6B"
    """Name of RealESRGAN model to use."""


class InpaintingOptions(BaseModel):
    inpainting_fill: int = 0
    """What to fill inpainted region with. 0 is blur, 1 is empty, 2 is latent noise, and 3 is latent empty."""
    inpaint_full_res: bool = False
    """Whether to use the full resolution for inpainting."""
    mask_blur: int = 0
    """Size of blur at boundaries of mask."""
    invert_mask: bool = False
    """Whether to invert the mask."""


class Txt2ImgOptions(BaseOptions, GenerationOptions, FaceRestorationOptions):
    pass


class Img2ImgOptions(
    BaseOptions, GenerationOptions, InpaintingOptions, FaceRestorationOptions
):
    mode: int = 0
    """Img2Img mode. 0 is normal img2img on the selected region, 1 is inpainting, and 2 (unsupported) is batch processing."""
    denoising_strength: float = 0.35
    """Strength of denoising from 0.0 to 1.0."""
    resize_mode: int = 1

    # upscale_overlap: int = 64
    # """Size of overlap in pixels for upscaling."""
    # upscaler_name: str = "None"
    # """Exact name of upscaler to use."""

    steps: int = 50


class UpscaleOptions(BaseOptions):
    upscaler_name: str = "None"
    """Exact name of upscaler to use."""
    downscale_first: bool = False
    """Whether to downscale the image by x0.5 first."""


class PluginOptions(BaseOptions):
    sample_path: str = "outputs/krita-in"


class MainConfig(BaseModel):
    txt2img: Txt2ImgOptions = Txt2ImgOptions()
    img2img: Img2ImgOptions = Img2ImgOptions()
    upscale: UpscaleOptions = UpscaleOptions()
    plugin: PluginOptions = PluginOptions()
