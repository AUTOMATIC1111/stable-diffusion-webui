#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../lib/common.sh"
genai_load_config
genai_ensure_dirs

LOG="$(genai_resolve_path logs/facefusion/setup.log)"
CLONE_DIR="$(genai_resolve_path "$(python3 -c "import json;print(json.load(open('$PATHS_FILE'))['facefusion']['cloneDir'])")")"
ENV_DIR="$(genai_resolve_path "$(python3 -c "import json;print(json.load(open('$PATHS_FILE'))['facefusion']['envDir'])")")"
COMMIT="$(python3 -c "import json;print(json.load(open('$LOCK_FILE'))['upstreams']['facefusion']['commit'])")"
REPO="$(python3 -c "import json;print(json.load(open('$LOCK_FILE'))['upstreams']['facefusion']['repository'])")"

PROVIDER="${FACEFUSION_EXECUTION_PROVIDER:-coreml}"
genai_log "$LOG" "FaceFusion setup provider=$PROVIDER"

command -v git >/dev/null || { echo "Git required"; exit 1; }
command -v ffmpeg >/dev/null || echo "WARN: ffmpeg not found — install via brew install ffmpeg"

PY="$(genai_python_candidate)" || exit 1
genai_git_clone_pinned "$REPO" "$COMMIT" "$CLONE_DIR" "$LOG"

[[ -x "$ENV_DIR/bin/python" ]] || "$PY" -m venv "$ENV_DIR"
PIP="$ENV_DIR/bin/pip"
"$PIP" install --upgrade pip wheel setuptools
"$PIP" install -r "$CLONE_DIR/requirements.txt"

if [[ "$PROVIDER" == "coreml" ]]; then
  "$PIP" install onnxruntime-silicon 2>/dev/null || "$PIP" install onnxruntime 2>/dev/null || true
fi

"$ENV_DIR/bin/python" -c "import onnxruntime as ort; print('providers:', ort.get_available_providers())"
echo "FaceFusion setup complete."
