from __future__ import annotations

import json
from typing import Iterable


class FontEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Font):
            return {
                "__gradio_font__": True,
                "name": obj.name,
                "class": "google" if isinstance(obj, GoogleFont) else "font",
            }
        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)


def as_font(dct):
    if "__gradio_font__" in dct:
        name = dct["name"]
        return GoogleFont(name) if dct["class"] == "google" else Font(name)
    return dct


class Font:
    def __init__(self, name: str):
        self.name = name

    def __str__(self) -> str:
        return (
            self.name
            if self.name in ["sans-serif", "serif", "monospace", "cursive", "fantasy"]
            else f"'{self.name}'"
        )

    def stylesheet(self) -> str:
        return None

    def __eq__(self, other: Font) -> bool:
        return self.name == other.name and self.stylesheet() == other.stylesheet()


class GoogleFont(Font):
    def __init__(self, name: str, weights: Iterable[int] = (400, 600)):
        self.name = name
        self.weights = weights

    def stylesheet(self) -> str:
        return f'https://fonts.googleapis.com/css2?family={self.name.replace(" ", "+")}:wght@{";".join(str(weight) for weight in self.weights)}&display=swap'
