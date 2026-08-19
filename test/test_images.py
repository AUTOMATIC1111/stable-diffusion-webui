import pytest
from PIL import Image, ImageDraw


@pytest.mark.usefixtures("initialize")
def test_draw_grid_annotations_without_multiline_textsize(monkeypatch):
    from modules.images import GridAnnotation, draw_grid_annotations

    monkeypatch.delattr(ImageDraw.ImageDraw, "multiline_textsize", raising=False)

    image = Image.new("RGB", (64, 64), "white")
    result = draw_grid_annotations(
        image,
        64,
        64,
        [[GridAnnotation("a long grid annotation")]],
        [[GridAnnotation("")]],
    )

    assert result.width == image.width
    assert result.height > image.height
