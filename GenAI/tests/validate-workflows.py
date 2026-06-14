#!/usr/bin/env python3
"""Validate committed ComfyUI workflow JSON files."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_DIR = ROOT / "workflows" / "comfyui"


def validate_workflow(path: Path) -> list[str]:
    errors: list[str] = []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"{path.name}: invalid JSON ({exc})"]
    if "nodes" not in data or not isinstance(data["nodes"], list):
        errors.append(f"{path.name}: missing nodes array")
        return errors
    if len(data["nodes"]) == 0:
        errors.append(f"{path.name}: empty workflow")
    node_ids = {n.get("id") for n in data["nodes"]}
    if None in node_ids:
        errors.append(f"{path.name}: node missing id")
    for node in data["nodes"]:
        if "type" not in node:
            errors.append(f"{path.name}: node {node.get('id')} missing type")
    return errors


def main() -> int:
    if not WORKFLOW_DIR.exists():
        print("FAIL: workflows/comfyui directory missing")
        return 1
    files = sorted(WORKFLOW_DIR.glob("*.json"))
    if not files:
        print("FAIL: no workflow JSON files found")
        return 1
    all_errors: list[str] = []
    for wf in files:
        all_errors.extend(validate_workflow(wf))
    if all_errors:
        for e in all_errors:
            print(f"FAIL: {e}")
        return 1
    print(f"PASS: {len(files)} workflow(s) valid")
    for wf in files:
        print(f"  - {wf.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
