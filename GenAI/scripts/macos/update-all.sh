#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../lib/common.sh"
genai_load_config
echo "Pinned ComfyUI:    $(python3 -c "import json;d=json.load(open('$LOCK_FILE'));u=d['upstreams']['comfyui'];print(u['release'], u['commit'])")"
echo "Pinned FaceFusion: $(python3 -c "import json;d=json.load(open('$LOCK_FILE'));u=d['upstreams']['facefusion'];print(u['release'], u['commit'])")"
echo "Edit upstreams.lock.json and re-run setup-all.sh to update."
