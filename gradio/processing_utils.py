from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
import os
import secrets
import shutil
import subprocess
import tempfile
import urllib.request
import warnings
from io import BytesIO
from pathlib import Path
from typing import Dict, Set, Tuple

import aiofiles
import numpy as np
import requests
from fastapi import UploadFile
from ffmpy import FFmpeg, FFprobe, FFRuntimeError
from PIL import Image, ImageOps, PngImagePlugin

from gradio import utils

with warnings.catch_warnings():
    warnings.simplefilter("ignore")  # Ignore pydub warning if ffmpeg is not installed
    from pydub import AudioSegment


#########################
# GENERAL
#########################


def to_binary(x: str | Dict) -> bytes:
    """Converts a base64 string or dictionary to a binary string that can be sent in a POST."""
    if isinstance(x, dict):
        if x.get("data"):
            base64str = x["data"]
        else:
            base64str = encode_url_or_file_to_base64(x["name"])
    else:
        base64str = x
    return base64.b64decode(base64str.split(",")[1])


#########################
# IMAGE PRE-PROCESSING
#########################


def decode_base64_to_image(encoding: str) -> Image.Image:
    content = encoding.split(";")[1]
    image_encoded = content.split(",")[1]
    img = Image.open(BytesIO(base64.b64decode(image_encoded)))
    exif = img.getexif()
    # 274 is the code for image rotation and 1 means "correct orientation"
    if exif.get(274, 1) != 1 and hasattr(ImageOps, "exif_transpose"):
        img = ImageOps.exif_transpose(img)
    return img


def encode_url_or_file_to_base64(path: str | Path):
    path = str(path)
    if utils.validate_url(path):
        return encode_url_to_base64(path)
    else:
        return encode_file_to_base64(path)


def get_mimetype(filename: str) -> str | None:
    mimetype = mimetypes.guess_type(filename)[0]
    if mimetype is not None:
        mimetype = mimetype.replace("x-wav", "wav").replace("x-flac", "flac")
    return mimetype


def get_extension(encoding: str) -> str | None:
    encoding = encoding.replace("audio/wav", "audio/x-wav")
    type = mimetypes.guess_type(encoding)[0]
    if type == "audio/flac":  # flac is not supported by mimetypes
        return "flac"
    elif type is None:
        return None
    extension = mimetypes.guess_extension(type)
    if extension is not None and extension.startswith("."):
        extension = extension[1:]
    return extension


def encode_file_to_base64(f):
    with open(f, "rb") as file:
        encoded_string = base64.b64encode(file.read())
        base64_str = str(encoded_string, "utf-8")
        mimetype = get_mimetype(f)
        return (
            "data:"
            + (mimetype if mimetype is not None else "")
            + ";base64,"
            + base64_str
        )


def encode_url_to_base64(url):
    encoded_string = base64.b64encode(requests.get(url).content)
    base64_str = str(encoded_string, "utf-8")
    mimetype = get_mimetype(url)
    return (
        "data:" + (mimetype if mimetype is not None else "") + ";base64," + base64_str
    )


def encode_plot_to_base64(plt):
    with BytesIO() as output_bytes:
        plt.savefig(output_bytes, format="png")
        bytes_data = output_bytes.getvalue()
    base64_str = str(base64.b64encode(bytes_data), "utf-8")
    return "data:image/png;base64," + base64_str


def save_array_to_file(image_array, dir=None):
    pil_image = Image.fromarray(_convert(image_array, np.uint8, force_copy=False))
    file_obj = tempfile.NamedTemporaryFile(delete=False, suffix=".png", dir=dir)
    pil_image.save(file_obj)
    return file_obj


def get_pil_metadata(pil_image):
    # Copy any text-only metadata
    metadata = PngImagePlugin.PngInfo()
    for key, value in pil_image.info.items():
        if isinstance(key, str) and isinstance(value, str):
            metadata.add_text(key, value)

    return metadata


def save_pil_to_file(pil_image, dir=None):
    file_obj = tempfile.NamedTemporaryFile(delete=False, suffix=".png", dir=dir)
    pil_image.save(file_obj, pnginfo=get_pil_metadata(pil_image))
    return file_obj


def encode_pil_to_base64(pil_image):
    with BytesIO() as output_bytes:
        pil_image.save(output_bytes, "PNG", pnginfo=get_pil_metadata(pil_image))
        bytes_data = output_bytes.getvalue()
    base64_str = str(base64.b64encode(bytes_data), "utf-8")
    return "data:image/png;base64," + base64_str


def encode_array_to_base64(image_array):
    with BytesIO() as output_bytes:
        pil_image = Image.fromarray(_convert(image_array, np.uint8, force_copy=False))
        pil_image.save(output_bytes, "PNG")
        bytes_data = output_bytes.getvalue()
    base64_str = str(base64.b64encode(bytes_data), "utf-8")
    return "data:image/png;base64," + base64_str


def resize_and_crop(img, size, crop_type="center"):
    """
    Resize and crop an image to fit the specified size.
    args:
        size: `(width, height)` tuple. Pass `None` for either width or height
        to only crop and resize the other.
        crop_type: can be 'top', 'middle' or 'bottom', depending on this
            value, the image will cropped getting the 'top/left', 'middle' or
            'bottom/right' of the image to fit the size.
    raises:
        ValueError: if an invalid `crop_type` is provided.
    """
    if crop_type == "top":
        center = (0, 0)
    elif crop_type == "center":
        center = (0.5, 0.5)
    else:
        raise ValueError

    resize = list(size)
    if size[0] is None:
        resize[0] = img.size[0]
    if size[1] is None:
        resize[1] = img.size[1]
    return ImageOps.fit(img, resize, centering=center)  # type: ignore


##################
# Audio
##################


def audio_from_file(filename, crop_min=0, crop_max=100):
    try:
        audio = AudioSegment.from_file(filename)
    except FileNotFoundError as e:
        isfile = Path(filename).is_file()
        msg = (
            f"Cannot load audio from file: `{'ffprobe' if isfile else filename}` not found."
            + " Please install `ffmpeg` in your system to use non-WAV audio file formats"
            " and make sure `ffprobe` is in your PATH."
            if isfile
            else ""
        )
        raise RuntimeError(msg) from e
    if crop_min != 0 or crop_max != 100:
        audio_start = len(audio) * crop_min / 100
        audio_end = len(audio) * crop_max / 100
        audio = audio[audio_start:audio_end]
    data = np.array(audio.get_array_of_samples())
    if audio.channels > 1:
        data = data.reshape(-1, audio.channels)
    return audio.frame_rate, data


def audio_to_file(sample_rate, data, filename):
    data = convert_to_16_bit_wav(data)
    audio = AudioSegment(
        data.tobytes(),
        frame_rate=sample_rate,
        sample_width=data.dtype.itemsize,
        channels=(1 if len(data.shape) == 1 else data.shape[1]),
    )
    file = audio.export(filename, format="wav")
    file.close()  # type: ignore


def convert_to_16_bit_wav(data):
    # Based on: https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.write.html
    warning = "Trying to convert audio automatically from {} to 16-bit int format."
    if data.dtype in [np.float64, np.float32, np.float16]:
        warnings.warn(warning.format(data.dtype))
        data = data / np.abs(data).max()
        data = data * 32767
        data = data.astype(np.int16)
    elif data.dtype == np.int32:
        warnings.warn(warning.format(data.dtype))
        data = data / 65538
        data = data.astype(np.int16)
    elif data.dtype == np.int16:
        pass
    elif data.dtype == np.uint16:
        warnings.warn(warning.format(data.dtype))
        data = data - 32768
        data = data.astype(np.int16)
    elif data.dtype == np.uint8:
        warnings.warn(warning.format(data.dtype))
        data = data * 257 - 32768
        data = data.astype(np.int16)
    else:
        raise ValueError(
            "Audio data cannot be converted automatically from "
            f"{data.dtype} to 16-bit int format."
        )
    return data


##################
# OUTPUT
##################


def decode_base64_to_binary(encoding) -> Tuple[bytes, str | None]:
    extension = get_extension(encoding)
    try:
        data = encoding.split(",")[1]
    except IndexError:
        data = ""
    return base64.b64decode(data), extension


def decode_base64_to_file(encoding, file_path=None, dir=None, prefix=None):
    if dir is not None:
        os.makedirs(dir, exist_ok=True)
    data, extension = decode_base64_to_binary(encoding)
    if file_path is not None and prefix is None:
        filename = Path(file_path).name
        prefix = filename
        if "." in filename:
            prefix = filename[0 : filename.index(".")]
            extension = filename[filename.index(".") + 1 :]

    if prefix is not None:
        prefix = utils.strip_invalid_filename_characters(prefix)

    if extension is None:
        file_obj = tempfile.NamedTemporaryFile(delete=False, prefix=prefix, dir=dir)
    else:
        file_obj = tempfile.NamedTemporaryFile(
            delete=False,
            prefix=prefix,
            suffix="." + extension,
            dir=dir,
        )
    file_obj.write(data)
    file_obj.flush()
    return file_obj


def dict_or_str_to_json_file(jsn, dir=None):
    if dir is not None:
        os.makedirs(dir, exist_ok=True)

    file_obj = tempfile.NamedTemporaryFile(
        delete=False, suffix=".json", dir=dir, mode="w+"
    )
    if isinstance(jsn, str):
        jsn = json.loads(jsn)
    json.dump(jsn, file_obj)
    file_obj.flush()
    return file_obj


def file_to_json(file_path: str | Path) -> Dict:
    with open(file_path) as f:
        return json.load(f)


class TempFileManager:
    """
    A class that should be inherited by any Component that needs to manage temporary files.
    It should be instantiated in the __init__ method of the component.
    """

    def __init__(self) -> None:
        # Set stores all the temporary files created by this component.
        self.temp_files: Set[str] = set()
        self.DEFAULT_TEMP_DIR = tempfile.gettempdir()

    def hash_file(self, file_path: str, chunk_num_blocks: int = 128) -> str:
        sha1 = hashlib.sha1()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_num_blocks * sha1.block_size), b""):
                sha1.update(chunk)
        return sha1.hexdigest()

    def hash_url(self, url: str, chunk_num_blocks: int = 128) -> str:
        sha1 = hashlib.sha1()
        remote = urllib.request.urlopen(url)
        max_file_size = 100 * 1024 * 1024  # 100MB
        total_read = 0
        while True:
            data = remote.read(chunk_num_blocks * sha1.block_size)
            total_read += chunk_num_blocks * sha1.block_size
            if not data or total_read > max_file_size:
                break
            sha1.update(data)
        return sha1.hexdigest()

    def hash_base64(self, base64_encoding: str, chunk_num_blocks: int = 128) -> str:
        sha1 = hashlib.sha1()
        for i in range(0, len(base64_encoding), chunk_num_blocks * sha1.block_size):
            data = base64_encoding[i : i + chunk_num_blocks * sha1.block_size]
            sha1.update(data.encode("utf-8"))
        return sha1.hexdigest()

    def make_temp_copy_if_needed(self, file_path: str) -> str:
        """Returns a temporary file path for a copy of the given file path if it does
        not already exist. Otherwise returns the path to the existing temp file."""
        temp_dir = self.hash_file(file_path)
        temp_dir = Path(self.DEFAULT_TEMP_DIR) / temp_dir
        temp_dir.mkdir(exist_ok=True, parents=True)

        f = tempfile.NamedTemporaryFile(delete=False, dir=temp_dir)
        f.name = utils.strip_invalid_filename_characters(Path(file_path).name)
        full_temp_file_path = str(utils.abspath(temp_dir / f.name))

        if not Path(full_temp_file_path).exists():
            shutil.copy2(file_path, full_temp_file_path)

        self.temp_files.add(full_temp_file_path)
        return full_temp_file_path

    async def save_uploaded_file(self, file: UploadFile, upload_dir: str) -> str:
        temp_dir = secrets.token_hex(
            20
        )  # Since the full file is being uploaded anyways, there is no benefit to hashing the file.
        temp_dir = Path(upload_dir) / temp_dir
        temp_dir.mkdir(exist_ok=True, parents=True)
        output_file_obj = tempfile.NamedTemporaryFile(delete=False, dir=temp_dir)

        if file.filename:
            file_name = Path(file.filename).name
            output_file_obj.name = utils.strip_invalid_filename_characters(file_name)

        full_temp_file_path = str(utils.abspath(temp_dir / output_file_obj.name))

        async with aiofiles.open(full_temp_file_path, "wb") as output_file:
            while True:
                content = await file.read(100 * 1024 * 1024)
                if not content:
                    break
                await output_file.write(content)

        return full_temp_file_path

    def download_temp_copy_if_needed(self, url: str) -> str:
        """Downloads a file and makes a temporary file path for a copy if does not already
        exist. Otherwise returns the path to the existing temp file."""
        temp_dir = self.hash_url(url)
        temp_dir = Path(self.DEFAULT_TEMP_DIR) / temp_dir
        temp_dir.mkdir(exist_ok=True, parents=True)
        f = tempfile.NamedTemporaryFile(delete=False, dir=temp_dir)

        f.name = utils.strip_invalid_filename_characters(Path(url).name)
        full_temp_file_path = str(utils.abspath(temp_dir / f.name))

        if not Path(full_temp_file_path).exists():
            with requests.get(url, stream=True) as r:
                with open(full_temp_file_path, "wb") as f:
                    shutil.copyfileobj(r.raw, f)

        self.temp_files.add(full_temp_file_path)
        return full_temp_file_path

    def base64_to_temp_file_if_needed(
        self, base64_encoding: str, file_name: str | None = None
    ) -> str:
        """Converts a base64 encoding to a file and returns the path to the file if
        the file doesn't already exist. Otherwise returns the path to the existing file."""
        temp_dir = self.hash_base64(base64_encoding)
        temp_dir = Path(self.DEFAULT_TEMP_DIR) / temp_dir
        temp_dir.mkdir(exist_ok=True, parents=True)

        guess_extension = get_extension(base64_encoding)
        if file_name:
            file_name = utils.strip_invalid_filename_characters(file_name)
        elif guess_extension:
            file_name = "file." + guess_extension
        else:
            file_name = "file"
        f = tempfile.NamedTemporaryFile(delete=False, dir=temp_dir)
        f.name = file_name
        full_temp_file_path = str(utils.abspath(temp_dir / f.name))

        if not Path(full_temp_file_path).exists():
            data, _ = decode_base64_to_binary(base64_encoding)
            with open(full_temp_file_path, "wb") as fb:
                fb.write(data)

        self.temp_files.add(full_temp_file_path)
        return full_temp_file_path


def download_tmp_copy_of_file(
    url_path: str, access_token: str | None = None, dir: str | None = None
) -> tempfile._TemporaryFileWrapper:
    if dir is not None:
        os.makedirs(dir, exist_ok=True)
    headers = {"Authorization": "Bearer " + access_token} if access_token else {}
    prefix = Path(url_path).stem
    suffix = Path(url_path).suffix
    file_obj = tempfile.NamedTemporaryFile(
        delete=False,
        prefix=prefix,
        suffix=suffix,
        dir=dir,
    )
    with requests.get(url_path, headers=headers, stream=True) as r:
        with open(file_obj.name, "wb") as f:
            shutil.copyfileobj(r.raw, f)
    return file_obj


def create_tmp_copy_of_file(
    file_path: str, dir: str | None = None
) -> tempfile._TemporaryFileWrapper:
    if dir is not None:
        os.makedirs(dir, exist_ok=True)
    prefix = Path(file_path).stem
    suffix = Path(file_path).suffix
    file_obj = tempfile.NamedTemporaryFile(
        delete=False,
        prefix=prefix,
        suffix=suffix,
        dir=dir,
    )
    shutil.copy2(file_path, file_obj.name)
    return file_obj


def _convert(image, dtype, force_copy=False, uniform=False):
    """
    Adapted from: https://github.com/scikit-image/scikit-image/blob/main/skimage/util/dtype.py#L510-L531

    Convert an image to the requested data-type.
    Warnings are issued in case of precision loss, or when negative values
    are clipped during conversion to unsigned integer types (sign loss).
    Floating point values are expected to be normalized and will be clipped
    to the range [0.0, 1.0] or [-1.0, 1.0] when converting to unsigned or
    signed integers respectively.
    Numbers are not shifted to the negative side when converting from
    unsigned to signed integer types. Negative values will be clipped when
    converting to unsigned integers.
    Parameters
    ----------
    image : ndarray
        Input image.
    dtype : dtype
        Target data-type.
    force_copy : bool, optional
        Force a copy of the data, irrespective of its current dtype.
    uniform : bool, optional
        Uniformly quantize the floating point range to the integer range.
        By default (uniform=False) floating point values are scaled and
        rounded to the nearest integers, which minimizes back and forth
        conversion errors.
    .. versionchanged :: 0.15
        ``_convert`` no longer warns about possible precision or sign
        information loss. See discussions on these warnings at:
        https://github.com/scikit-image/scikit-image/issues/2602
        https://github.com/scikit-image/scikit-image/issues/543#issuecomment-208202228
        https://github.com/scikit-image/scikit-image/pull/3575
    References
    ----------
    .. [1] DirectX data conversion rules.
           https://msdn.microsoft.com/en-us/library/windows/desktop/dd607323%28v=vs.85%29.aspx
    .. [2] Data Conversions. In "OpenGL ES 2.0 Specification v2.0.25",
           pp 7-8. Khronos Group, 2010.
    .. [3] Proper treatment of pixels as integers. A.W. Paeth.
           In "Graphics Gems I", pp 249-256. Morgan Kaufmann, 1990.
    .. [4] Dirty Pixels. J. Blinn. In "Jim Blinn's corner: Dirty Pixels",
           pp 47-57. Morgan Kaufmann, 1998.
    """
    dtype_range = {
        bool: (False, True),
        np.bool_: (False, True),
        np.bool8: (False, True),
        float: (-1, 1),
        np.float_: (-1, 1),
        np.float16: (-1, 1),
        np.float32: (-1, 1),
        np.float64: (-1, 1),
    }

    def _dtype_itemsize(itemsize, *dtypes):
        """Return first of `dtypes` with itemsize greater than `itemsize`
        Parameters
        ----------
        itemsize: int
            The data type object element size.
        Other Parameters
        ----------------
        *dtypes:
            Any Object accepted by `np.dtype` to be converted to a data
            type object
        Returns
        -------
        dtype: data type object
            First of `dtypes` with itemsize greater than `itemsize`.
        """
        return next(dt for dt in dtypes if np.dtype(dt).itemsize >= itemsize)

    def _dtype_bits(kind, bits, itemsize=1):
        """Return dtype of `kind` that can store a `bits` wide unsigned int
        Parameters:
        kind: str
            Data type kind.
        bits: int
            Desired number of bits.
        itemsize: int
            The data type object element size.
        Returns
        -------
        dtype: data type object
            Data type of `kind` that can store a `bits` wide unsigned int
        """

        s = next(
            i
            for i in (itemsize,) + (2, 4, 8)
            if bits < (i * 8) or (bits == (i * 8) and kind == "u")
        )

        return np.dtype(kind + str(s))

    def _scale(a, n, m, copy=True):
        """Scale an array of unsigned/positive integers from `n` to `m` bits.
        Numbers can be represented exactly only if `m` is a multiple of `n`.
        Parameters
        ----------
        a : ndarray
            Input image array.
        n : int
            Number of bits currently used to encode the values in `a`.
        m : int
            Desired number of bits to encode the values in `out`.
        copy : bool, optional
            If True, allocates and returns new array. Otherwise, modifies
            `a` in place.
        Returns
        -------
        out : array
            Output image array. Has the same kind as `a`.
        """
        kind = a.dtype.kind
        if n > m and a.max() < 2**m:
            return a.astype(_dtype_bits(kind, m))
        elif n == m:
            return a.copy() if copy else a
        elif n > m:
            # downscale with precision loss
            if copy:
                b = np.empty(a.shape, _dtype_bits(kind, m))
                np.floor_divide(a, 2 ** (n - m), out=b, dtype=a.dtype, casting="unsafe")
                return b
            else:
                a //= 2 ** (n - m)
                return a
        elif m % n == 0:
            # exact upscale to a multiple of `n` bits
            if copy:
                b = np.empty(a.shape, _dtype_bits(kind, m))
                np.multiply(a, (2**m - 1) // (2**n - 1), out=b, dtype=b.dtype)
                return b
            else:
                a = a.astype(_dtype_bits(kind, m, a.dtype.itemsize), copy=False)
                a *= (2**m - 1) // (2**n - 1)
                return a
        else:
            # upscale to a multiple of `n` bits,
            # then downscale with precision loss
            o = (m // n + 1) * n
            if copy:
                b = np.empty(a.shape, _dtype_bits(kind, o))
                np.multiply(a, (2**o - 1) // (2**n - 1), out=b, dtype=b.dtype)
                b //= 2 ** (o - m)
                return b
            else:
                a = a.astype(_dtype_bits(kind, o, a.dtype.itemsize), copy=False)
                a *= (2**o - 1) // (2**n - 1)
                a //= 2 ** (o - m)
                return a

    image = np.asarray(image)
    dtypeobj_in = image.dtype
    if dtype is np.floating:
        dtypeobj_out = np.dtype("float64")
    else:
        dtypeobj_out = np.dtype(dtype)
    dtype_in = dtypeobj_in.type
    dtype_out = dtypeobj_out.type
    kind_in = dtypeobj_in.kind
    kind_out = dtypeobj_out.kind
    itemsize_in = dtypeobj_in.itemsize
    itemsize_out = dtypeobj_out.itemsize

    # Below, we do an `issubdtype` check.  Its purpose is to find out
    # whether we can get away without doing any image conversion.  This happens
    # when:
    #
    # - the output and input dtypes are the same or
    # - when the output is specified as a type, and the input dtype
    #   is a subclass of that type (e.g. `np.floating` will allow
    #   `float32` and `float64` arrays through)

    if np.issubdtype(dtype_in, np.obj2sctype(dtype)):
        if force_copy:
            image = image.copy()
        return image

    if kind_in in "ui":
        imin_in = np.iinfo(dtype_in).min
        imax_in = np.iinfo(dtype_in).max
    if kind_out in "ui":
        imin_out = np.iinfo(dtype_out).min  # type: ignore
        imax_out = np.iinfo(dtype_out).max  # type: ignore

    # any -> binary
    if kind_out == "b":
        return image > dtype_in(dtype_range[dtype_in][1] / 2)

    # binary -> any
    if kind_in == "b":
        result = image.astype(dtype_out)
        if kind_out != "f":
            result *= dtype_out(dtype_range[dtype_out][1])
        return result

    # float -> any
    if kind_in == "f":
        if kind_out == "f":
            # float -> float
            return image.astype(dtype_out)

        if np.min(image) < -1.0 or np.max(image) > 1.0:
            raise ValueError("Images of type float must be between -1 and 1.")
        # floating point -> integer
        # use float type that can represent output integer type
        computation_type = _dtype_itemsize(
            itemsize_out, dtype_in, np.float32, np.float64
        )

        if not uniform:
            if kind_out == "u":
                image_out = np.multiply(image, imax_out, dtype=computation_type)  # type: ignore
            else:
                image_out = np.multiply(
                    image, (imax_out - imin_out) / 2, dtype=computation_type  # type: ignore
                )
                image_out -= 1.0 / 2.0
            np.rint(image_out, out=image_out)
            np.clip(image_out, imin_out, imax_out, out=image_out)  # type: ignore
        elif kind_out == "u":
            image_out = np.multiply(image, imax_out + 1, dtype=computation_type)  # type: ignore
            np.clip(image_out, 0, imax_out, out=image_out)  # type: ignore
        else:
            image_out = np.multiply(
                image, (imax_out - imin_out + 1.0) / 2.0, dtype=computation_type  # type: ignore
            )
            np.floor(image_out, out=image_out)
            np.clip(image_out, imin_out, imax_out, out=image_out)  # type: ignore
        return image_out.astype(dtype_out)

    # signed/unsigned int -> float
    if kind_out == "f":
        # use float type that can exactly represent input integers
        computation_type = _dtype_itemsize(
            itemsize_in, dtype_out, np.float32, np.float64
        )

        if kind_in == "u":
            # using np.divide or np.multiply doesn't copy the data
            # until the computation time
            image = np.multiply(image, 1.0 / imax_in, dtype=computation_type)  # type: ignore
            # DirectX uses this conversion also for signed ints
            # if imin_in:
            #     np.maximum(image, -1.0, out=image)
        else:
            image = np.add(image, 0.5, dtype=computation_type)
            image *= 2 / (imax_in - imin_in)  # type: ignore

        return np.asarray(image, dtype_out)

    # unsigned int -> signed/unsigned int
    if kind_in == "u":
        if kind_out == "i":
            # unsigned int -> signed int
            image = _scale(image, 8 * itemsize_in, 8 * itemsize_out - 1)
            return image.view(dtype_out)
        else:
            # unsigned int -> unsigned int
            return _scale(image, 8 * itemsize_in, 8 * itemsize_out)

    # signed int -> unsigned int
    if kind_out == "u":
        image = _scale(image, 8 * itemsize_in - 1, 8 * itemsize_out)
        result = np.empty(image.shape, dtype_out)
        np.maximum(image, 0, out=result, dtype=image.dtype, casting="unsafe")
        return result

    # signed int -> signed int
    if itemsize_in > itemsize_out:
        return _scale(image, 8 * itemsize_in - 1, 8 * itemsize_out - 1)

    image = image.astype(_dtype_bits("i", itemsize_out * 8))
    image -= imin_in  # type: ignore
    image = _scale(image, 8 * itemsize_in, 8 * itemsize_out, copy=False)
    image += imin_out  # type: ignore
    return image.astype(dtype_out)


def ffmpeg_installed() -> bool:
    return shutil.which("ffmpeg") is not None


def video_is_playable(video_filepath: str) -> bool:
    """Determines if a video is playable in the browser.

    A video is playable if it has a playable container and codec.
        .mp4 -> h264
        .webm -> vp9
        .ogg -> theora
    """
    try:
        container = Path(video_filepath).suffix.lower()
        probe = FFprobe(
            global_options="-show_format -show_streams -select_streams v -print_format json",
            inputs={video_filepath: None},
        )
        output = probe.run(stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        output = json.loads(output[0])
        video_codec = output["streams"][0]["codec_name"]
        return (container, video_codec) in [
            (".mp4", "h264"),
            (".ogg", "theora"),
            (".webm", "vp9"),
        ]
    # If anything goes wrong, assume the video can be played to not convert downstream
    except (FFRuntimeError, IndexError, KeyError):
        return True


def convert_video_to_playable_mp4(video_path: str) -> str:
    """Convert the video to mp4. If something goes wrong return the original video."""
    try:
        output_path = Path(video_path).with_suffix(".mp4")
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            shutil.copy2(video_path, tmp_file.name)
            # ffmpeg will automatically use h264 codec (playable in browser) when converting to mp4
            ff = FFmpeg(
                inputs={str(tmp_file.name): None},
                outputs={str(output_path): None},
                global_options="-y -loglevel quiet",
            )
            ff.run()
    except FFRuntimeError as e:
        print(f"Error converting video to browser-playable format {str(e)}")
        output_path = video_path
    return str(output_path)
