#!/usr/bin/env python3
"""GenAI smoke and static validation orchestrator."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS = [
    "validate-config.py",
    "validate-upstreams.py",
    "validate-workflows.py",
]


def run_test(name: str) -> int:
    path = ROOT / "tests" / name
    print(f"\n--- {name} ---")
    result = subprocess.run([sys.executable, str(path)], cwd=ROOT)
    return result.returncode


def check_git_hygiene() -> int:
    print("\n--- git-hygiene ---")
    result = subprocess.run(
        ["git", "status", "--porcelain", "GenAI"],
        cwd=ROOT.parent,
        capture_output=True,
        text=True,
    )
    blocked_ext = (".ckpt", ".safetensors", ".onnx", ".pth", ".pt")
    for line in result.stdout.splitlines():
        path = line[3:].strip() if len(line) > 3 else line
        if path.lower().endswith(blocked_ext):
            print(f"FAIL: tracked model-like file: {path}")
            return 1
    print("PASS: no model extensions staged under GenAI")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--static-only", action="store_true", help="Skip runtime startup tests")
    args = parser.parse_args()
    rc = 0
    for test in TESTS:
        rc |= run_test(test)
    rc |= check_git_hygiene()
    if args.static_only:
        print("\nSKIP: runtime startup tests (--static-only)")
    else:
        print("\nNOT TESTED: ComfyUI/FaceFusion startup requires local setup and models")
    return 1 if rc else 0


if __name__ == "__main__":
    sys.exit(main())
