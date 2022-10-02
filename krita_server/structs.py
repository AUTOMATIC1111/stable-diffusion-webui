from __future__ import annotations

from typing import Optional

from .config import Img2ImgOptions, Txt2ImgOptions, UpscaleOptions
from .utils import optional


@optional
class DefaultTxt2ImgOptions(Txt2ImgOptions):
    pass


class Txt2ImgRequest(DefaultTxt2ImgOptions):
    """Text2Img API request. If optional attributes aren't set, the defaults
    from `krita_config.yaml` will be used.
    """

    orig_width: int
    """Requested image width."""
    orig_height: int
    """Requested image height."""


@optional
class DefaultImg2ImgOptions(Img2ImgOptions):
    pass


class Img2ImgRequest(DefaultImg2ImgOptions):
    """Img2Img API request. If optional attributes aren't set, the defaults from
    `krita_config.yaml` will be used.
    """

    src_path: str
    """Path to image being used."""
    mask_path: Optional[str]
    """Path to image mask being used."""


@optional
class DefaultUpscaleOptions(UpscaleOptions):
    pass


class UpscaleRequest(DefaultUpscaleOptions):
    """Upscale API request. If optional attributes aren't set, the defaults from
    `krita_config.yaml` will be used.
    """

    src_path: str
    """Path to image being used."""
