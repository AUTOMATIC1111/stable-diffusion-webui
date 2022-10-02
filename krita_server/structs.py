from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class CommonOptions:
    """Options that are shared between Txt2Img and Img2Img."""

    prompt: Optional[str]
    """Requested prompt."""
    negative_prompt: Optional[str]
    """Requested negative prompt."""
    sampler_name: Optional[str]
    """Exact name of sampler to use. Name should follow exact spelling and capitalization as in the WebUI."""
    steps: Optional[int]
    """Number of steps for diffusion."""
    cfg_scale: Optional[float]
    """Guidance scale for diffusion."""

    batch_count: Optional[int]
    """Number of batches to render."""
    batch_size: Optional[int]
    """Number of images per batch to render."""
    base_size: Optional[int]
    """Native/base resolution of model used."""
    max_size: Optional[int]
    """Max input resolution allowed to prevent image artifacts."""
    seed: Optional[int]
    """Seed used for noise generation. Incremented by 1 for each image rendered."""
    tiling: Optional[bool]
    """Whether to generate a tileable image."""

    use_gfpgan: Optional[bool]
    """Whether to use GFPGAN for face restoration."""
    face_restorer: Optional[str]
    """Exact name of face restorer to use."""
    codeformer_weight: Optional[float]
    """Strength of face restoration if using CodeFormer. 0.0 is the strongest and 1.0 is the weakest."""


class Txt2ImgRequest(CommonOptions, BaseModel):
    """Text2Img API request. If optional attributes aren't set, the defaults
    from `krita_config.yaml` will be used.
    """

    orig_width: int
    """Requested image width."""
    orig_height: int
    """Requested image height."""


class Img2ImgRequest(CommonOptions, BaseModel):
    """Img2Img API request. If optional attributes aren't set, the defaults from
    `krita_config.yaml` will be used.
    """

    src_path: str
    """Path to image being used."""
    mask_path: Optional[str]
    """Path to image mask being used."""

    mode: Optional[int]
    """Img2Img mode. 0 is normal img2img on the selected region, 1 is inpainting, and 2 (unsupported) is batch processing."""

    inpainting_fill: Optional[int]
    """What to fill inpainted region with. 0 is blur, 1 is empty, 2 is latent noise, and 3 is latent empty."""
    inpaint_full_res: Optional[bool]
    """Whether to use the full resolution for inpainting."""
    mask_blur: Optional[int]
    """Size of blur at boundaries of mask."""
    invert_mask: Optional[bool]
    """Whether to invert the mask."""

    denoising_strength: Optional[float]
    """Strength of denoising from 0.0 to 1.0."""

    upscale_overlap: Optional[int]
    """Size of overlap in pixels for upscaling."""
    upscaler_name: Optional[str]
    """Exact name of upscaler to use."""


class UpscaleRequest(BaseModel):
    """Upscale API request. If optional attributes aren't set, the defaults from
    `krita_config.yaml` will be used.
    """

    src_path: str
    """Path to image being used."""
    upscaler_name: Optional[str]
    """Exact name of upscaler to use."""
    downscale_first: Optional[bool]
    """Whether to downscale the image by x0.5 first."""
