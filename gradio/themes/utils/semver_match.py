from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import huggingface_hub
import semantic_version
import semantic_version as semver


@dataclass
class ThemeAsset:
    filename: str
    version: semver.Version = field(init=False)

    def __post_init__(self):
        self.version = semver.Version(self.filename.split("@")[1].replace(".json", ""))


def get_theme_assets(space_info: huggingface_hub.hf_api.SpaceInfo) -> List[ThemeAsset]:
    if "gradio-theme" not in getattr(space_info, "tags", []):
        raise ValueError(f"{space_info.id} is not a valid gradio-theme space!")

    return [
        ThemeAsset(filename.rfilename)
        for filename in space_info.siblings
        if filename.rfilename.startswith("themes/")
    ]


def get_matching_version(
    assets: List[ThemeAsset], expression: str | None
) -> ThemeAsset | None:

    expression = expression or "*"

    # Return most recent version that matches
    matching_version = semantic_version.SimpleSpec(expression).select(
        [a.version for a in assets]
    )

    return next((a for a in assets if a.version == matching_version), None)
