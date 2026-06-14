#!/usr/bin/env bash
set -euo pipefail
[[ "${1:-}" == "--confirm" ]] || { echo "Usage: $0 --confirm"; exit 1; }
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../lib/common.sh"
genai_load_config

for key in comfyui.cloneDir facefusion.cloneDir comfyui.envDir facefusion.envDir facefusion.tempDir; do
  section="${key%%.*}"; field="${key#*.}"
  rel="$(python3 -c "import json;d=json.load(open('$PATHS_FILE'));print(d['$section']['$field'])")"
  full="$(genai_resolve_path "$rel")"
  [[ -d "$full" ]] && rm -rf "$full" && echo "Removed $full"
done
echo "Reset complete. Run setup-all.sh"
