from gradio.themes.base import Base, ThemeClass
from gradio.themes.default import Default
from gradio.themes.glass import Glass
from gradio.themes.monochrome import Monochrome
from gradio.themes.soft import Soft
from gradio.themes.utils import colors, sizes
from gradio.themes.utils.colors import Color
from gradio.themes.utils.fonts import Font, GoogleFont
from gradio.themes.utils.sizes import Size

__all__ = [
    "Base",
    "Color",
    "Default",
    "Font",
    "Glass",
    "GoogleFont",
    "Monochrome",
    "Size",
    "Soft",
    "ThemeClass",
    "colors",
    "sizes",
]


def builder(*args, **kwargs):
    from gradio.themes.builder_app import demo

    return demo.launch(*args, **kwargs)
