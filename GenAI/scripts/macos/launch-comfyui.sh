#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../lib/common.sh"
genai_load_config

CLONE_DIR="$(genai_resolve_path "$(python3 -c "import json;print(json.load(open('$PATHS_FILE'))['comfyui']['cloneDir'])")")"
ENV_DIR="$(genai_resolve_path "$(python3 -c "import json;print(json.load(open('$PATHS_FILE'))['comfyui']['envDir'])")")"
HOST="${GENAI_HOST:-127.0.0.1}"
PORT="${COMFYUI_PORT:-8188}"

[[ -x "$ENV_DIR/bin/python" ]] || { echo "Run setup-comfyui.sh first"; exit 1; }

BACKEND="$("$ENV_DIR/bin/python" -c "import torch; print('MPS' if torch.backends.mps.is_available() else 'CPU')")"

echo "=== ComfyUI Launch ==="
echo "Env:     $ENV_DIR"
echo "Backend: $BACKEND"
echo "URL:     http://${HOST}:${PORT}"
echo "Models:  $(genai_resolve_path "$(python3 -c "import json;print(json.load(open('$PATHS_FILE'))['comfyui']['modelsDir'])")")"
echo "======================"

cd "$CLONE_DIR"
exec "$ENV_DIR/bin/python" main.py --listen "$HOST" --port "$PORT"
