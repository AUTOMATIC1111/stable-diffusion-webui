#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$SCRIPT_DIR/setup-comfyui.sh"
"$SCRIPT_DIR/setup-facefusion.sh"
echo "Repair complete."
