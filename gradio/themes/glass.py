from __future__ import annotations

from typing import Iterable

from gradio.themes.base import Base
from gradio.themes.utils import colors, fonts, sizes


class Glass(Base):
    def __init__(
        self,
        *,
        primary_hue: colors.Color | str = colors.stone,
        secondary_hue: colors.Color | str = colors.stone,
        neutral_hue: colors.Color | str = colors.stone,
        spacing_size: sizes.Size | str = sizes.spacing_sm,
        radius_size: sizes.Size | str = sizes.radius_sm,
        text_size: sizes.Size | str = sizes.text_sm,
        font: fonts.Font
        | str
        | Iterable[fonts.Font | str] = (
            "Optima",
            "Candara",
            "Noto Sans",
            "source-sans-pro",
            "sans-serif",
        ),
        font_mono: fonts.Font
        | str
        | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("IBM Plex Mono"),
            "ui-monospace",
            "Consolas",
            "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            spacing_size=spacing_size,
            radius_size=radius_size,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )
        self.name = "glass"
        super().set(
            body_background_fill_dark="*primary_800",
            background_fill_secondary_dark="*primary_800",
            block_background_fill_dark="*primary_800",
            button_primary_background_fill="linear-gradient(180deg, *primary_50 0%, *primary_200 50%, *primary_300 50%, *primary_200 100%)",
            button_primary_background_fill_hover="linear-gradient(180deg, *primary_100 0%, *primary_200 50%, *primary_300 50%, *primary_200 100%)",
            button_primary_background_fill_dark="linear-gradient(180deg, *primary_400 0%, *primary_500 50%, *primary_600 50%, *primary_500 100%)",
            button_primary_background_fill_hover_dark="linear-gradient(180deg, *primary_400 0%, *primary_500 50%, *primary_600 50%, *primary_500 100%)",
            button_secondary_background_fill="*button_primary_background_fill",
            button_secondary_background_fill_hover="*button_primary_background_fill_hover",
            button_secondary_background_fill_dark="*button_primary_background_fill",
            button_secondary_background_fill_hover_dark="*button_primary_background_fill_hover",
            button_cancel_background_fill="*button_primary_background_fill",
            button_cancel_background_fill_hover="*button_primary_background_fill_hover",
            button_cancel_background_fill_dark="*button_primary_background_fill",
            button_cancel_background_fill_hover_dark="*button_primary_background_fill_hover",
            button_cancel_border_color="*button_secondary_border_color",
            button_cancel_border_color_dark="*button_secondary_border_color",
            button_cancel_text_color="*button_secondary_text_color",
            checkbox_border_width="0px",
            checkbox_label_background_fill="*button_secondary_background_fill",
            checkbox_label_background_fill_dark="*button_secondary_background_fill",
            checkbox_label_background_fill_hover="*button_secondary_background_fill_hover",
            checkbox_label_background_fill_hover_dark="*button_secondary_background_fill_hover",
            checkbox_label_border_width="1px",
            checkbox_background_color_dark="*primary_600",
            button_border_width="1px",
            button_shadow_active="*shadow_inset",
            input_background_fill="linear-gradient(0deg, *secondary_50 0%, white 100%)",
            input_background_fill_dark="*secondary_600",
            input_border_color_focus_dark="*primary_400",
            input_border_width="1px",
            slider_color="*primary_400",
            block_label_text_color="*primary_500",
            block_title_text_color="*primary_500",
            block_label_text_weight="600",
            block_title_text_weight="600",
            block_label_text_size="*text_md",
            block_title_text_size="*text_md",
            block_label_background_fill="*primary_200",
            block_label_background_fill_dark="*primary_700",
            block_border_width="0px",
            block_border_width_dark="1px",
            panel_border_width="1px",
            border_color_primary_dark="*primary_500",
            background_fill_primary_dark="*neutral_700",
            background_fill_secondary="*primary_100",
            block_background_fill="*primary_50",
            block_shadow="*primary_400 0px 0px 3px 0px",
            table_even_background_fill_dark="*neutral_700",
            table_odd_background_fill_dark="*neutral_700",
        )
