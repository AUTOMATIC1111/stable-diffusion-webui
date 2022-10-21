from math import ceil

from krita import Document, QBuffer, QByteArray, QImage, QIODevice


def fix_prompt(prompt: str):
    """Multiline tokens -> comma-separated tokens. Replace empty prompts with None."""
    joined = ", ".join(filter(bool, [x.strip() for x in prompt.splitlines()]))
    return joined if joined != "" else None


def find_fixed_aspect_ratio(
    base_size: int, max_size: int, orig_width: int, orig_height: int
):
    """Copy of `krita_server.utils.sddebz_highres_fix()`.

    This is used by `find_optimal_selection_region()` below to adjust the selected region.
    """

    def rnd(r, x, z=64):
        """Scale dimension x with stride z while attempting to preserve aspect ratio r."""
        return z * ceil(r * x / z)

    ratio = orig_width / orig_height

    # height is smaller dimension
    if orig_width > orig_height:
        width, height = rnd(ratio, base_size), base_size
        if width > max_size:
            width, height = max_size, rnd(1 / ratio, max_size)
    # width is smaller dimension
    else:
        width, height = base_size, rnd(1 / ratio, base_size)
        if height > max_size:
            width, height = rnd(ratio, max_size), max_size

    return width / height


def find_optimal_selection_region(
    base_size: int,
    max_size: int,
    orig_x: int,
    orig_y: int,
    orig_width: int,
    orig_height: int,
    canvas_width: int,
    canvas_height: int,
):
    """Adjusts the selected region in order to attempt to preserve the original
    aspect ratio of the selection. This prevents the image from being stretched
    after being scaled and strided.

    After grasping what @sddebz intended to do, I fixed some logical errors &
    made it clearer.

    Iterating the padding is naive, but easier to understand & verify then figuring
    out how to grow the rectangle using the fixed aspect ratio alone while accounting
    for the canvas boundary. Also, it only grows the selection, not shrink, to
    prevent clipping what the user selected.

    Args:
        base_size (int): Native/base input size of the model.
        max_size (int): Max input size to accept.
        orig_x (int): Original left position of selection.
        orig_y (int): Original top position of selection.
        orig_width (int): Original width of selection.
        orig_height (int): Original height of selection.
        canvas_width (int): Canvas width.
        canvas_height (int): Canvas height.

    Returns:
        Tuple[int, int, int, int]: Best x, y, width, height to use.
    """
    orig_ratio = orig_width / orig_height
    fix_ratio = find_fixed_aspect_ratio(base_size, max_size, orig_width, orig_height)

    # h * (w/h - w/h) = w
    xpad_limit = ceil(abs(fix_ratio - orig_ratio) * orig_height)
    # w * (h/w - h/w) = h
    ypad_limit = ceil(abs(1 / fix_ratio - 1 / orig_ratio) * orig_width)

    best_x = orig_x
    best_y = orig_y
    best_width = orig_width
    best_height = orig_height
    best_delta = abs(fix_ratio - orig_ratio)
    for x in range(1, xpad_limit + 1):
        for y in range(1, ypad_limit + 1):
            # account for boundary of canvas
            # padding is on both sides i.e the selection grows while center anchored
            x1 = max(0, orig_x - x // 2)
            x2 = min(canvas_width, x1 + orig_width + x)
            y1 = max(0, orig_y - y // 2)
            y2 = min(canvas_height, y1 + orig_height + y)

            new_width = x2 - x1
            new_height = y2 - y1
            new_ratio = new_width / new_height
            new_delta = abs(fix_ratio - new_ratio)
            if new_delta < best_delta:
                best_delta = new_delta
                best_x = x1
                best_y = y1
                best_width = new_width
                best_height = new_height

    return best_x, best_y, best_width, best_height


def create_layer(doc: Document, name: str):
    """Create new layer in document"""
    root = doc.rootNode()
    layer = doc.createNode(name, "paintLayer")
    root.addChildNode(layer, None)
    return layer


def save_img(img: QImage, path: str):
    """Expects QImage"""
    # png is lossless; setting compression to max (0) won't affect quality
    img.save(path, "PNG", 0)


def img_to_ba(img: QImage):
    """Converts QImage to QByteArray"""
    ptr = img.bits()
    ptr.setsize(img.byteCount())
    return QByteArray(ptr.asstring())


def img_to_b64(img: QImage):
    """Converts QImage to base64-encoded string"""
    ba = QByteArray()
    buffer = QBuffer(ba)
    buffer.open(QIODevice.WriteOnly)
    img.save(buffer, "PNG", 0)
    return ba.toBase64().data().decode("utf-8")


def b64_to_img(enc: str):
    """Converts base64-encoded string to QImage"""
    ba = QByteArray.fromBase64(enc.encode("utf-8"))
    return QImage.fromData(ba, "PNG")
