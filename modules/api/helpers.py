import io
import base64
from PIL import Image, PngImagePlugin
import piexif
import piexif.helper
from fastapi.exceptions import HTTPException
from modules import shared, sd_samplers


def validate_sampler_name(name):
    config = sd_samplers.all_samplers_map.get(name, None)
    if config is None:
        raise HTTPException(status_code=404, detail="Sampler not found")
    return name


def decode_base64_to_image(encoding):
    if encoding.startswith("data:image/"):
        encoding = encoding.split(";")[1].split(",")[1]
    try:
        image = Image.open(io.BytesIO(base64.b64decode(encoding)))
        return image
    except Exception as e:
        shared.log.warning(f'API cannot decode image: {e}')
        raise HTTPException(status_code=500, detail="Invalid encoded image") from e


def encode_pil_to_base64(image):
    """
    with io.BytesIO() as output_bytes:
        images.save_image(image, output_bytes, shared.opts.samples_format)
        bytes_data = output_bytes.getvalue()
    return base64.b64encode(bytes_data)
    """
    if not isinstance(image, Image.Image):
        shared.log.error('API cannot encode image: not a PIL image')
        return ''
    buffered = io.BytesIO()
    image_format = Image.registered_extensions()[f'.{shared.opts.samples_format}']
    image.save(buffered, format=image_format)
    b64 = base64.b64encode(buffered.getvalue())
    return b64


def upscaler_to_index(name: str):
    try:
        return [x.name.lower() for x in shared.sd_upscalers].index(name.lower())
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid upscaler, needs to be one of these: {' , '.join([x.name for x in shared.sd_upscalers])}") from e

def save_image(image, fn, ext):
    # actual save
    parameters = image.info.get('parameters', None)
    image_format = Image.registered_extensions()[f'.{ext}']
    if image_format == 'PNG':
        pnginfo_data = PngImagePlugin.PngInfo()
        for k, v in image.info.items():
            pnginfo_data.add_text(k, str(v))
        image.save(fn, format=image_format, quality=shared.opts.jpeg_quality, pnginfo=pnginfo_data)
    elif image_format == 'JPEG':
        if image.mode == 'RGBA':
            shared.log.warning('Saving RGBA image as JPEG: Alpha channel will be lost')
            image = image.convert("RGB")
        elif image.mode == 'I;16':
            image = image.point(lambda p: p * 0.0038910505836576).convert("L")
        exif_bytes = piexif.dump({ "Exif": { piexif.ExifIFD.UserComment: piexif.helper.UserComment.dump(parameters or "", encoding="unicode") } })
        image.save(fn, format=image_format, quality=shared.opts.jpeg_quality, exif=exif_bytes)
    elif image_format == 'WEBP':
        if image.mode == 'I;16':
            image = image.point(lambda p: p * 0.0038910505836576).convert("RGB")
        exif_bytes = piexif.dump({ "Exif": { piexif.ExifIFD.UserComment: piexif.helper.UserComment.dump(parameters or "", encoding="unicode") } })
        image.save(fn, format=image_format, quality=shared.opts.jpeg_quality, lossless=shared.opts.webp_lossless, exif=exif_bytes)
    else:
        # shared.log.warning(f'Unrecognized image format: {extension} attempting save as {image_format}')
        image.save(fn, format=image_format, quality=shared.opts.jpeg_quality)
