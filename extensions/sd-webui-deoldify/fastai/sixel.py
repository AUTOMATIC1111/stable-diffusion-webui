from .core import *

libsixel = try_import('libsixel')

def _sixel_encode(data, width, height):
    s = io.BytesIO()
    output = libsixel.sixel_output_new(lambda data, s: s.write(data), s)
    dither = libsixel.sixel_dither_new(256)
    w,h = int(width),int(height)
    libsixel.sixel_dither_initialize(dither, data, w, h, libsixel.SIXEL_PIXELFORMAT_RGBA8888)
    libsixel.sixel_encode(data, w, h, 1, dither, output)
    return s.getvalue().decode('ascii')

def plot_sixel(fig=None):
    if not libsixel:
        warn("You could see this plot with `libsixel`. See https://github.com/saitoha/libsixel")
        return
    if fig is None: fig = plt.gcf()
    fig.canvas.draw()
    dpi = fig.get_dpi()
    res = _sixel_encode(fig.canvas.buffer_rgba(), fig.get_figwidth()* dpi, fig.get_figheight() * dpi)
    print(res)

