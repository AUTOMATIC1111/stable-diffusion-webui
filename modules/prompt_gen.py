# We need this so Python doesn't complain about the unknown StableDiffusionProcessing-typehint at runtime
from __future__ import annotations

import csv
import os
import os.path
import typing
import collections.abc as abc
import tempfile
import shutil
from modules.paths import script_path

if typing.TYPE_CHECKING:
    # Only import this when code is being type-checked, it doesn't have any effect at runtime
    from .processing import StableDiffusionProcessing


class PromptGen(typing.NamedTuple):
    name: str
    blank: str
    blank2: str

def load_promptgen(path: str) -> dict[str, PromptGen]:
    gendata = {"None": PromptGen("None", "", "")}

    if os.path.exists(path):
        with open(path, "r", encoding="utf8", newline='') as file:
            reader = csv.DictReader(file)
            for row in reader:
                # Support loading old CSV format with "name, text"-columns
                blank = row["blank"] if "blank" in row else row["text"]
                blank2 = row.get("blank2", "")
                gendata[row["name"]] = PromptGen(row["name"], blank, blank2)

    return gendata

def load_promptgen_text(path: str):
    if os.path.exists(path):
        promptgen_dict = {}
        with open(path, "r", encoding="utf8") as file:
            count = 1
            for line in file:
 
                value = line.split()
 
                promptgen_dict[int(count)] = value
                count += 1
    return promptgen_dict
