#!/usr/bin/env python3
"""Validate upstreams.lock.json structure and commit pin format."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "upstreams.lock.json"
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def main() -> int:
    data = json.loads(LOCK.read_text(encoding="utf-8"))
    errors: list[str] = []
    if data.get("schemaVersion") != 1:
        errors.append("schemaVersion must be 1")
    for name in ("comfyui", "facefusion"):
        upstream = data.get("upstreams", {}).get(name)
        if not upstream:
            errors.append(f"missing upstream {name}")
            continue
        for field in ("repository", "release", "commit", "license", "purpose"):
            if field not in upstream:
                errors.append(f"{name} missing {field}")
        repo = upstream.get("repository", "")
        if not repo.startswith("https://github.com/"):
            errors.append(f"{name} repository must use HTTPS GitHub URL")
        commit = upstream.get("commit", "")
        if not COMMIT_RE.match(commit):
            errors.append(f"{name} commit must be 40-char SHA")
    if errors:
        for e in errors:
            print(f"FAIL: {e}")
        return 1
    print("PASS: upstreams.lock.json valid")
    print(f"  ComfyUI:    {data['upstreams']['comfyui']['release']} @ {data['upstreams']['comfyui']['commit'][:7]}")
    print(f"  FaceFusion: {data['upstreams']['facefusion']['release']} @ {data['upstreams']['facefusion']['commit'][:7]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
