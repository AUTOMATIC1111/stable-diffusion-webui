"""Provide blending functions and types.

Adapted from https://github.com/addisonElliott/pypdn/blob/master/pypdn/reader.py
and https://gitlab.com/inklabapp/pyora/-/blob/master/pyora/BlendNonSep.py
MIT License Copyright (c) 2020 FredHappyface

Credits to:

MIT License Copyright (c) 2019 Paul Jewell
For implementing blending from the Open Raster Image Spec

MIT License Copyright (c) 2018 Addison Elliott
For implementing blending from Paint.NET

MIT License Copyright (c) 2017 pashango
For implementing a number of blending functions used by other popular image
editors
"""

from __future__ import annotations

import warnings

import numpy as np
from PIL import Image

from .blendtype import BlendType


def normal(background: np.ndarray, foreground: np.ndarray) -> np.ndarray:
	"""BlendType.NORMAL."""
	del background  # we don't care about this
	return foreground


def multiply(background: np.ndarray, foreground: np.ndarray) -> np.ndarray:
	"""BlendType.MULTIPLY."""
	return np.clip(foreground * background, 0.0, 1.0)


def additive(background: np.ndarray, foreground: np.ndarray) -> np.ndarray:
	"""BlendType.ADDITIVE."""
	return np.minimum(background + foreground, 1.0)


def colourburn(background: np.ndarray, foreground: np.ndarray) -> np.ndarray:
	"""BlendType.COLOURBURN."""
	with np.errstate(divide="ignore"):
		return np.where(
			foreground != 0.0, np.maximum(1.0 - ((1.0 - background) / foreground), 0.0), 0.0
		)


def colourdodge(background: np.ndarray, foreground: np.ndarray) -> np.ndarray:
	"""BlendType.COLOURDODGE."""
	with np.errstate(divide="ignore"):
		return np.where(foreground != 1.0, np.minimum(background / (1.0 - foreground), 1.0), 1.0)


def reflect(background: np.ndarray, foreground: np.ndarray) -> np.ndarray:
	"""BlendType.REFLECT."""
	with np.errstate(divide="ignore"):
		return np.where(
			foreground != 1.0, np.minimum((background ** 2) / (1.0 - foreground), 1.0), 1.0
		)


def glow(background: np.ndarray, foreground: np.ndarray) -> np.ndarray:
	"""BlendType.GLOW."""
	with np.errstate(divide="ignore"):
		return np.where(
			background != 1.0, np.minimum((foreground ** 2) / (1.0 - background), 1.0), 1.0
		)


def overlay(background: np.ndarray, foreground: np.ndarray) -> np.ndarray:
	"""BlendType.OVERLAY."""
	return np.where(
		background < 0.5,
		2 * background * foreground,
		1.0 - (2 * (1.0 - background) * (1.0 - foreground)),
	)


def difference(background: np.ndarray, foreground: np.ndarray) -> np.ndarray:
	"""BlendType.DIFFERENCE."""
	return np.abs(background - foreground)


def negation(background: np.ndarray, foreground: np.ndarray) -> np.ndarray:
	"""BlendType.NEGATION."""
	return np.maximum(background - foreground, 0.0)


def lighten(background: np.ndarray, foreground: np.ndarray) -> np.ndarray:
	"""BlendType.LIGHTEN."""
	return np.maximum(background, foreground)


def darken(background: np.ndarray, foreground: np.ndarray) -> np.ndarray:
	"""BlendType.DARKEN."""
	return np.minimum(background, foreground)


def screen(background: np.ndarray, foreground: np.ndarray) -> np.ndarray:
	"""BlendType.SCREEN."""
	return background + foreground - background * foreground


def xor(background: np.ndarray, foreground: np.ndarray) -> np.ndarray:
	"""BlendType.XOR."""
	# XOR requires int values so convert to uint8
	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		return imageIntToFloat(imageFloatToInt(background) ^ imageFloatToInt(foreground))


def softlight(background: np.ndarray, foreground: np.ndarray) -> np.ndarray:
	"""BlendType.SOFTLIGHT."""
	return (1.0 - background) * background * foreground + background * (
		1.0 - (1.0 - background) * (1.0 - foreground)
	)


def hardlight(background: np.ndarray, foreground: np.ndarray) -> np.ndarray:
	"""BlendType.HARDLIGHT."""
	return np.where(
		foreground < 0.5,
		np.minimum(background * 2 * foreground, 1.0),
		np.minimum(1.0 - ((1.0 - background) * (1.0 - (foreground - 0.5) * 2.0)), 1.0),
	)


def grainextract(background: np.ndarray, foreground: np.ndarray) -> np.ndarray:
	"""BlendType.GRAINEXTRACT."""
	return np.clip(background - foreground + 0.5, 0.0, 1.0)


def grainmerge(background: np.ndarray, foreground: np.ndarray) -> np.ndarray:
	"""BlendType.GRAINMERGE."""
	return np.clip(background + foreground - 0.5, 0.0, 1.0)


def divide(background: np.ndarray, foreground: np.ndarray) -> np.ndarray:
	"""BlendType.DIVIDE."""
	return np.minimum((256.0 / 255.0 * background) / (1.0 / 255.0 + foreground), 1.0)


def pinlight(background: np.ndarray, foreground: np.ndarray) -> np.ndarray:
	"""BlendType.PINLIGHT."""
	return np.minimum(background, 2 * foreground) * (foreground < 0.5) + np.maximum(
		background, 2 * (foreground - 0.5)
	) * (foreground >= 0.5)


def vividlight(background: np.ndarray, foreground: np.ndarray) -> np.ndarray:
	"""BlendType.VIVIDLIGHT."""
	return colourburn(background, foreground * 2) * (foreground < 0.5) + colourdodge(
		background, 2 * (foreground - 0.5)
	) * (foreground >= 0.5)


def exclusion(background: np.ndarray, foreground: np.ndarray) -> np.ndarray:
	"""BlendType.EXCLUSION."""
	return background + foreground - (2.0 * background * foreground)


def _lum(colours: np.ndarray) -> np.ndarray:
	"""Luminosity.

	:param colours: x by x by 3 matrix of rgb color components of pixels
	:return: x by x by 3 matrix of luminosity of pixels
	"""
	return (colours[:, :, 0] * 0.299) + (colours[:, :, 1] * 0.587) + (colours[:, :, 2] * 0.114)


def _setLum(originalColours: np.ndarray, newLuminosity: np.ndarray) -> np.ndarray:
	"""Set a new luminosity value for the matrix of color."""
	_colours = originalColours.copy()
	_luminosity = _lum(_colours)
	deltaLum = newLuminosity - _luminosity
	_colours[:, :, 0] += deltaLum
	_colours[:, :, 1] += deltaLum
	_colours[:, :, 2] += deltaLum
	_luminosity = _lum(_colours)
	_minColours = np.min(_colours, axis=2)
	_MaxColours = np.max(_colours, axis=2)
	for i in range(_colours.shape[0]):
		for j in range(_colours.shape[1]):
			_colour = _colours[i][j]
			newLuminosity = _luminosity[i, j]
			minColour = _minColours[i, j]
			maxColour = _MaxColours[i, j]
			if minColour < 0:
				_colours[i][j] = newLuminosity + (
					((_colour - newLuminosity) * newLuminosity) / (newLuminosity - minColour)
				)
			if maxColour > 1:
				_colours[i][j] = newLuminosity + (
					((_colour - newLuminosity) * (1 - newLuminosity)) / (maxColour - newLuminosity)
				)
	return _colours


def _sat(colours: np.ndarray) -> np.ndarray:
	"""Saturation.

	:param colours: x by x by 3 matrix of rgb color components of pixels
	:return: int of saturation of pixels
	"""
	return np.max(colours, axis=2) - np.min(colours, axis=2)


def _setSat(originalColours: np.ndarray, newSaturation: np.ndarray) -> np.ndarray:
	"""Set a new saturation value for the matrix of color.

	The current implementation cannot be vectorized in an efficient manner,
	so it is very slow,
	O(m*n) at least. This might be able to be improved with openCL if that is
	the direction that the lib takes.
	:param c: x by x by 3 matrix of rgb color components of pixels
	:param s: int of the new saturation value for the matrix
	:return: x by x by 3 matrix of luminosity of pixels
	"""
	_colours = originalColours.copy()
	for i in range(_colours.shape[0]):
		for j in range(_colours.shape[1]):
			_colour = _colours[i][j]
			minI = 0
			midI = 1
			maxI = 2
			if _colour[midI] < _colour[minI]:
				minI, midI = midI, minI
			if _colour[maxI] < _colour[midI]:
				midI, maxI = maxI, midI
			if _colour[midI] < _colour[minI]:
				minI, midI = midI, minI
			if _colour[maxI] - _colour[minI] > 0.0:
				_colours[i][j][midI] = ((_colour[midI] - _colour[minI]) * newSaturation[i, j]) / (
					_colour[maxI] - _colour[minI]
				)
				_colours[i][j][maxI] = newSaturation[i, j]
			else:
				_colours[i][j][midI] = 0
				_colours[i][j][maxI] = 0
			_colours[i][j][minI] = 0
	return _colours


def hue(background: np.ndarray, foreground: np.ndarray) -> np.ndarray:
	"""BlendType.HUE."""
	return _setLum(_setSat(foreground, _sat(background)), _lum(background))


def saturation(background: np.ndarray, foreground: np.ndarray) -> np.ndarray:
	"""BlendType.SATURATION."""
	return _setLum(_setSat(background, _sat(foreground)), _lum(background))


def colour(background: np.ndarray, foreground: np.ndarray) -> np.ndarray:
	"""BlendType.COLOUR."""
	return _setLum(foreground, _lum(background))


def luminosity(background: np.ndarray, foreground: np.ndarray) -> np.ndarray:
	"""BlendType.LUMINOSITY."""
	return _setLum(background, _lum(foreground))


def destin(
	backgroundAlpha: np.ndarray,
	foregroundAlpha: np.ndarray,
	backgroundColour: np.ndarray,
	foregroundColour: np.ndarray,
):
	"""'clip' composite mode.

	All parts of 'layer above' which are alpha in 'layer below' will be made
	also alpha in 'layer above'
	(to whatever degree of alpha they were)

	Destination which overlaps the source, replaces the source.

	Fa = 0; Fb = αs
	co = αb x Cb x αs
	αo = αb x αs
	"""
	del foregroundColour  # Not used by function
	outAlpha = backgroundAlpha * foregroundAlpha
	with np.errstate(divide="ignore", invalid="ignore"):
		outRGB = np.divide(
			np.multiply((backgroundAlpha * foregroundAlpha)[:, :, None], backgroundColour),
			outAlpha[:, :, None],
		)
	return outRGB, outAlpha


def destout(
	backgroundAlpha: np.ndarray,
	foregroundAlpha: np.ndarray,
	backgroundColour: np.ndarray,
	foregroundColour: np.ndarray,
):
	"""Reverse 'Clip' composite mode.

	All parts of 'layer below' which are alpha in 'layer above' will be made
	also alpha in 'layer below'
	(to whatever degree of alpha they were)

	"""
	del foregroundColour  # Not used by function
	outAlpha = backgroundAlpha * (1 - foregroundAlpha)
	with np.errstate(divide="ignore", invalid="ignore"):
		outRGB = np.divide(
			np.multiply((backgroundAlpha * (1 - foregroundAlpha))[:, :, None], backgroundColour),
			outAlpha[:, :, None],
		)
	return outRGB, outAlpha


def destatop(
	backgroundAlpha: np.ndarray,
	foregroundAlpha: np.ndarray,
	backgroundColour: np.ndarray,
	foregroundColour: np.ndarray,
):
	"""Place the layer below above the 'layer above' in places where the 'layer above' exists...

	where 'layer below' does not exist, but 'layer above' does, place 'layer-above'

	"""
	outAlpha = (foregroundAlpha * (1 - backgroundAlpha)) + (backgroundAlpha * foregroundAlpha)
	with np.errstate(divide="ignore", invalid="ignore"):
		outRGB = np.divide(
			np.multiply((foregroundAlpha * (1 - backgroundAlpha))[:, :, None], foregroundColour)
			+ np.multiply((backgroundAlpha * foregroundAlpha)[:, :, None], backgroundColour),
			outAlpha[:, :, None],
		)
	return outRGB, outAlpha


def srcatop(
	backgroundAlpha: np.ndarray,
	foregroundAlpha: np.ndarray,
	backgroundColour: np.ndarray,
	foregroundColour: np.ndarray,
):
	"""Place the layer below above the 'layer above' in places where the 'layer above' exists."""
	outAlpha = (foregroundAlpha * backgroundAlpha) + (backgroundAlpha * (1 - foregroundAlpha))
	with np.errstate(divide="ignore", invalid="ignore"):
		outRGB = np.divide(
			np.multiply((foregroundAlpha * backgroundAlpha)[:, :, None], foregroundColour)
			+ np.multiply((backgroundAlpha * (1 - foregroundAlpha))[:, :, None], backgroundColour),
			outAlpha[:, :, None],
		)

	return outRGB, outAlpha


def imageIntToFloat(image: np.ndarray) -> np.ndarray:
	"""Convert a numpy array representing an image to an array of floats.

	Args:
		image (np.ndarray): numpy array of ints

	Returns:
		np.ndarray: numpy array of floats
	"""
	return image / 255


def imageFloatToInt(image: np.ndarray) -> np.ndarray:
	"""Convert a numpy array representing an image to an array of ints.

	Args:
		image (np.ndarray): numpy array of floats

	Returns:
		np.ndarray: numpy array of ints
	"""
	return (image * 255).astype(np.uint8)


def blend(background: np.ndarray, foreground: np.ndarray, blendType: BlendType) -> np.ndarray:
	"""Blend pixels.

	Args:
		background (np.ndarray): background
		foreground (np.ndarray): foreground
		blendType (BlendType): the blend type

	Returns:
		np.ndarray: new array representing the image

	background: np.ndarray,
	foreground: np.ndarray and the return are in the form

		[[[0. 0. 0.]
		[0. 0. 0.]
		[0. 0. 0.]
		...
		[0. 0. 0.]
		[0. 0. 0.]
		[0. 0. 0.]]

		...

		[[0. 0. 0.]
		[0. 0. 0.]
		[0. 0. 0.]
		...
		[0. 0. 0.]
		[0. 0. 0.]
		[0. 0. 0.]]]
	"""
	blendLookup = {
		BlendType.NORMAL: normal,
		BlendType.MULTIPLY: multiply,
		BlendType.COLOURBURN: colourburn,
		BlendType.COLOURDODGE: colourdodge,
		BlendType.REFLECT: reflect,
		BlendType.OVERLAY: overlay,
		BlendType.DIFFERENCE: difference,
		BlendType.LIGHTEN: lighten,
		BlendType.DARKEN: darken,
		BlendType.SCREEN: screen,
		BlendType.SOFTLIGHT: softlight,
		BlendType.HARDLIGHT: hardlight,
		BlendType.GRAINEXTRACT: grainextract,
		BlendType.GRAINMERGE: grainmerge,
		BlendType.DIVIDE: divide,
		BlendType.HUE: hue,
		BlendType.SATURATION: saturation,
		BlendType.COLOUR: colour,
		BlendType.LUMINOSITY: luminosity,
		BlendType.XOR: xor,
		BlendType.NEGATION: negation,
		BlendType.PINLIGHT: pinlight,
		BlendType.VIVIDLIGHT: vividlight,
		BlendType.EXCLUSION: exclusion,
	}

	if blendType not in blendLookup:
		return normal(background, foreground)
	return blendLookup[blendType](background, foreground)


def blendLayers(
	background: Image.Image,
	foreground: Image.Image,
	blendType: BlendType | tuple[str, ...],
	opacity: float = 1.0,
) -> Image.Image:
	"""Blend layers using numpy array.

	Args:
		background (Image.Image): background layer
		foreground (Image.Image): foreground layer (must be same size as background)
		blendType (BlendType): The blendtype
		opacity (float): The opacity of the foreground image

	Returns:
		Image.Image: combined image
	"""
	# Convert the Image.Image to a numpy array
	npForeground: np.ndarray = imageIntToFloat(np.array(foreground.convert("RGBA")))
	npBackground: np.ndarray = imageIntToFloat(np.array(background.convert("RGBA")))

	# Get the alpha from the layers
	backgroundAlpha = npBackground[:, :, 3]
	foregroundAlpha = npForeground[:, :, 3] * opacity
	combinedAlpha = backgroundAlpha * foregroundAlpha

	# Get the colour from the layers
	backgroundColor = npBackground[:, :, 0:3]
	foregroundColor = npForeground[:, :, 0:3]

	# Some effects require alpha
	alphaFunc = {
		BlendType.DESTIN: destin,
		BlendType.DESTOUT: destout,
		BlendType.SRCATOP: srcatop,
		BlendType.DESTATOP: destatop,
	}

	if blendType in alphaFunc:
		return Image.fromarray(
			imageFloatToInt(
				np.clip(
					np.dstack(
						alphaFunc[blendType](
							backgroundAlpha, foregroundAlpha, backgroundColor, foregroundColor
						)
					),
					a_min=0,
					a_max=1,
				)
			)
		)

	# Get the colours and the alpha for the new image
	colorComponents = (
		(backgroundAlpha - combinedAlpha)[:, :, None] * backgroundColor
		+ (foregroundAlpha - combinedAlpha)[:, :, None] * foregroundColor
		+ combinedAlpha[:, :, None] * blend(backgroundColor, foregroundColor, blendType)
	)
	alphaComponent = backgroundAlpha + foregroundAlpha - combinedAlpha

	return Image.fromarray(
		imageFloatToInt(np.clip(np.dstack((colorComponents, alphaComponent)), a_min=0, a_max=1))
	)
