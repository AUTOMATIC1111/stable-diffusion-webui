# We need this so Python doesn't complain about the unknown StableDiffusionProcessing-typehint at runtime
from __future__ import annotations

import csv
import os
import os.path
import typing
import collections.abc as abc
import tempfile
import shutil

if typing.TYPE_CHECKING:
    # Only import this when code is being type-checked, it doesn't have any effect at runtime
    from .processing import StableDiffusionProcessing


class PromptArtists(typing.NamedTuple):
    name: str
    prompt: str
    negative_prompt: str


def load_artists(path: str) -> dict[str, PromptDescriptor]:
    artists = {"None": PromptArtists("None", "", "")}

    if os.path.exists(path):
        with open(path, "r", encoding="utf8", newline='') as file:
            reader = csv.DictReader(file)
            for row in reader:
                # Support loading old CSV format with "name, text"-columns
                prompt = row["score"] if "score" in row else row["text"]
                negative_prompt = row.get("category", "")
                artists[row["artist"]] = PromptArtists(row["artist"], prompt, negative_prompt)

    return artists