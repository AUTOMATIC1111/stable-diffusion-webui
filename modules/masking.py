from PIL import Image, ImageFilter, ImageOps


def get_crop_region_v2(mask, pad=0):
    """
    Finds a rectangular region that contains all masked ares in a mask.
    Returns None if mask is completely black mask (all 0)

    Parameters:
    mask: PIL.Image.Image L mode or numpy 1d array
    pad: int number of pixels that the region will be extended on all sides
    Returns: (x1, y1, x2, y2) | None

    Introduced post 1.9.0
    """
    mask = mask if isinstance(mask, Image.Image) else Image.fromarray(mask)
    if box := mask.getbbox():
        x1, y1, x2, y2 = box
        return (max(x1 - pad, 0), max(y1 - pad, 0), min(x2 + pad, mask.size[0]), min(y2 + pad, mask.size[1])) if pad else box


def get_crop_region(mask, pad=0):
    """
    Same function as get_crop_region_v2 but handles completely black mask (all 0) differently
    when mask all black still return coordinates but the coordinates may be invalid ie x2>x1 or y2>y1
    Notes: it is possible for the coordinates to be "valid" again if pad size is sufficiently large
    (mask_size.x-pad, mask_size.y-pad, pad, pad)

    Extension developer should use get_crop_region_v2 instead unless for compatibility considerations.
    """
    mask = mask if isinstance(mask, Image.Image) else Image.fromarray(mask)
    if box := get_crop_region_v2(mask, pad):
        return box
    x1, y1 = mask.size
    x2 = y2 = 0
    return max(x1 - pad, 0), max(y1 - pad, 0), min(x2 + pad, mask.size[0]), min(y2 + pad, mask.size[1])


def expand_crop_region(crop_region, processing_width, processing_height, image_width, image_height):
    """expands crop region get_crop_region() to match the ratio of the image the region will processed in; returns expanded region
    for example, if user drew mask in a 128x32 region, and the dimensions for processing are 512x512, the region will be expanded to 128x128."""

    x1, y1, x2, y2 = crop_region

    ratio_crop_region = (x2 - x1) / (y2 - y1)
    ratio_processing = processing_width / processing_height

    if ratio_crop_region > ratio_processing:
        desired_height = (x2 - x1) / ratio_processing
        desired_height_diff = int(desired_height - (y2-y1))
        y1 -= desired_height_diff//2
        y2 += desired_height_diff - desired_height_diff//2
        if y2 >= image_height:
            diff = y2 - image_height
            y2 -= diff
            y1 -= diff
        if y1 < 0:
            y2 -= y1
            y1 -= y1
        if y2 >= image_height:
            y2 = image_height
    else:
        desired_width = (y2 - y1) * ratio_processing
        desired_width_diff = int(desired_width - (x2-x1))
        x1 -= desired_width_diff//2
        x2 += desired_width_diff - desired_width_diff//2
        if x2 >= image_width:
            diff = x2 - image_width
            x2 -= diff
            x1 -= diff
        if x1 < 0:
            x2 -= x1
            x1 -= x1
        if x2 >= image_width:
            x2 = image_width

    return x1, y1, x2, y2


def expand_too_small_crop_region(crop_region, processing_width, processing_height, image_width, image_height):
    """expands a crop_region to not have dimensions smaller than processing_dimensions"""

    def _expand_segment(c1, c2, desirable_length, maximal_coordinate):
        """expands the segment given by c1 c2 to the desired_dimension but not exceeding the boundaries of the maximal_coordinate"""
        if (diff := desirable_length + c1 - c2) > 0:
            # if the region is smaller than desirable_length, extend both sides equally
            diff_l = diff // 2
            c1 -= diff_l
            c2 += diff - diff_l
            if c1 < 0:  # shift the region to the right by c1
                c2 = min(c2 - c1, maximal_coordinate)  # ensure c2 is within maximal_coordinate
                c1 = 0
            elif c2 >= maximal_coordinate:  # shift the region to the left by (c2 - maximal_coordinate)
                c1 = max(c1 - c2 + maximal_coordinate, 0)  # ensure c1 is not below 0
                c2 = maximal_coordinate
        return c1, c2

    x1, y1, x2, y2 = crop_region
    x1, x2 = _expand_segment(x1, x2, processing_width, image_width)
    y1, y2 = _expand_segment(y1, y2, processing_height, image_height)
    new_crop_region = x1, y1, x2, y2
    if new_crop_region != crop_region:
        print("Crop region was smaller then resolution and has been corrected")
    return new_crop_region


def fill(image, mask):
    """fills masked regions with colors from image using blur. Not extremely effective."""

    image_mod = Image.new('RGBA', (image.width, image.height))

    image_masked = Image.new('RGBa', (image.width, image.height))
    image_masked.paste(image.convert("RGBA").convert("RGBa"), mask=ImageOps.invert(mask.convert('L')))

    image_masked = image_masked.convert('RGBa')

    for radius, repeats in [(256, 1), (64, 1), (16, 2), (4, 4), (2, 2), (0, 1)]:
        blurred = image_masked.filter(ImageFilter.GaussianBlur(radius)).convert('RGBA')
        for _ in range(repeats):
            image_mod.alpha_composite(blurred)

    return image_mod.convert("RGB")

