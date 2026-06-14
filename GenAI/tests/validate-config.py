#!/usr/bin/env python3
"""Validate GenAI local-paths and settings configuration."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_PATH_KEYS = {
    "comfyui": ["cloneDir", "envDir", "modelsDir", "inputDir", "outputDir", "logDir"],
    "facefusion": ["cloneDir", "envDir", "modelsDir", "inputDir", "outputDir", "tempDir", "logDir"],
}


def validate_paths(path: Path) -> list[str]:
    errors: list[str] = []
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schemaVersion") != 1:
        errors.append("local-paths schemaVersion must be 1")
    for section, keys in REQUIRED_PATH_KEYS.items():
        if section not in data:
            errors.append(f"missing section: {section}")
            continue
        for key in keys:
            if key not in data[section]:
                errors.append(f"missing {section}.{key}")
    return errors


def validate_settings_example(path: Path) -> list[str]:
    errors: list[str] = []
    text = path.read_text(encoding="utf-8")
    required = ["GENAI_HOST", "COMFYUI_PORT", "FACEFUSION_PORT"]
    for key in required:
        if not re.search(rf"^{key}=", text, re.MULTILINE):
            errors.append(f"settings.example.env missing {key}")
    if "127.0.0.1" not in text:
        errors.append("settings.example.env should default GENAI_HOST to 127.0.0.1")
    return errors


def main() -> int:
    errors: list[str] = []
    example_paths = ROOT / "config" / "local-paths.example.json"
    example_settings = ROOT / "config" / "settings.example.env"
    errors.extend(validate_paths(example_paths))
    errors.extend(validate_settings_example(example_settings))
    if errors:
        for e in errors:
            print(f"FAIL: {e}")
        return 1
    print("PASS: configuration templates valid")
    return 0


if __name__ == "__main__":
    sys.exit(main())
