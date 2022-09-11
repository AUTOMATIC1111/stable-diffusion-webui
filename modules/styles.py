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


class PromptStyle(typing.NamedTuple):
    name: str
    prompt: str
    negative_prompt: str


def load_styles(path: str) -> dict[str, PromptStyle]:
    styles = {"None": PromptStyle("None", "", "")}

    if os.path.exists(path):
        with open(path, "r", encoding="utf8", newline='') as file:
            reader = csv.DictReader(file)
            for row in reader:
                # Support loading old CSV format with "name, text"-columns
                prompt = row["prompt"] if "prompt" in row else row["text"]
                negative_prompt = row.get("negative_prompt", "")
                styles[row["name"]] = PromptStyle(row["name"], prompt, negative_prompt)

    return styles


def merge_prompts(style_prompt: str, prompt: str) -> str:
    parts = filter(None, (prompt.strip(), style_prompt.strip()))
    return ", ".join(parts)


def apply_style(processing: StableDiffusionProcessing, style: PromptStyle) -> None:
    if isinstance(processing.prompt, list):
        processing.prompt = [merge_prompts(style.prompt, p) for p in processing.prompt]
    else:
        processing.prompt = merge_prompts(style.prompt, processing.prompt)

    if isinstance(processing.negative_prompt, list):
        processing.negative_prompt = [merge_prompts(style.negative_prompt, p) for p in processing.negative_prompt]
    else:
        processing.negative_prompt = merge_prompts(style.negative_prompt, processing.negative_prompt)


def save_styles(path: str, styles: abc.Iterable[PromptStyle]) -> None:
    # Write to temporary file first, so we don't nuke the file if something goes wrong
    fd, temp_path = tempfile.mkstemp(".csv")
    with os.fdopen(fd, "w", encoding="utf8", newline='') as file:
        # _fields is actually part of the public API: typing.NamedTuple is a replacement for collections.NamedTuple,
        # and collections.NamedTuple has explicit documentation for accessing _fields. Same goes for _asdict()
        writer = csv.DictWriter(file, fieldnames=PromptStyle._fields)
        writer.writeheader()
        writer.writerows(style._asdict() for style in styles)

    # Always keep a backup file around
    if os.path.exists(path):
        shutil.move(path, path + ".bak")
    shutil.move(temp_path, path)
