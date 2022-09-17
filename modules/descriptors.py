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


class PromptDescriptor(typing.NamedTuple):
    name: str
    prompt: str
    negative_prompt: str


def load_descriptors(path: str) -> dict[str, PromptDescriptor]:
    descriptors = {"None": PromptDescriptor("None", "", "")}

    if os.path.exists(path):
        with open(path, "r", encoding="utf8", newline='') as file:
            reader = csv.DictReader(file)
            for row in reader:
                # Support loading old CSV format with "name, text"-columns
                prompt = row["desc_prompt"] if "desc_prompt" in row else row["text"]
                negative_prompt = row.get("desc_negative_prompt", "")
                descriptors[row["desc_name"]] = PromptDescriptor(row["desc_name"], prompt, negative_prompt)

    return descriptors
