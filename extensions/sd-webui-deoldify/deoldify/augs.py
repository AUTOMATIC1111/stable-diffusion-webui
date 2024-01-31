import random

from fastai.vision.image import TfmPixel

# Contributed by Rani Horev. Thank you!
def _noisify(
    x, pct_pixels_min: float = 0.001, pct_pixels_max: float = 0.4, noise_range: int = 30
):
    if noise_range > 255 or noise_range < 0:
        raise Exception("noise_range must be between 0 and 255, inclusively.")

    h, w = x.shape[1:]
    img_size = h * w
    mult = 10000.0
    pct_pixels = (
        random.randrange(int(pct_pixels_min * mult), int(pct_pixels_max * mult)) / mult
    )
    noise_count = int(img_size * pct_pixels)

    for ii in range(noise_count):
        yy = random.randrange(h)
        xx = random.randrange(w)
        noise = random.randrange(-noise_range, noise_range) / 255.0
        x[:, yy, xx].add_(noise)

    return x


noisify = TfmPixel(_noisify)
