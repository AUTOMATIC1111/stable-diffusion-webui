#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"
python3 tests/validate-config.py
python3 tests/validate-upstreams.py
python3 tests/validate-workflows.py
python3 tests/smoke_test.py --static-only
echo "Smoke tests passed (static)."
