"""Do stuff to images to prepare them.
"""
from __future__ import annotations

import warnings

from deprecation import deprecated
from PIL import Image


@deprecated(deprecated_in="2021.1", removed_in="", details="use renderWAlphaOffset")
def rasterImageOA(  # pylint:disable=missing-function-docstring
	image: Image.Image, size: tuple[int, int], alpha: float = 1.0, offsets: tuple[int, int] = (0, 0)
) -> Image.Image:
	warnings.warn(
		"Call to deprecated function rasterImageOA.", category=DeprecationWarning, stacklevel=2
	)
	return renderWAlphaOffset(image, size, alpha, offsets)


@deprecated(deprecated_in="2021.1", removed_in="", details="use renderWAlphaOffset")
def rasterImageOffset(  # pylint:disable=missing-function-docstring
	image: Image.Image, size: tuple[int, int], offsets: tuple[int, int] = (0, 0)
) -> Image.Image:
	warnings.warn(
		"Call to deprecated function rasterImageOffset.", category=DeprecationWarning, stacklevel=2
	)
	return renderWAlphaOffset(image, size, 1, offsets)


def renderWAlphaOffset(
	image: Image.Image, size: tuple[int, int], alpha: float = 1.0, offsets: tuple[int, int] = (0, 0)
) -> Image.Image:
	"""Render an image with offset and alpha to a given size.

	Args:
		image (Image.Image): pil image to draw
		size (tuple[int, int]): width, height as a tuple
		alpha (float, optional): alpha transparency. Defaults to 1.0.
		offsets (tuple[int, int], optional): x, y offsets as a tuple.
		Defaults to (0, 0).

	Returns:
		Image.Image: new image
	"""
	imageOffset = Image.new("RGBA", size)
	imageOffset.paste(image.convert("RGBA"), offsets, image.convert("RGBA"))
	return Image.blend(Image.new("RGBA", size), imageOffset, alpha)
