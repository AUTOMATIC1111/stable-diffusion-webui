#!/usr/bin/env bash
# Bootstrap ComfyUI on macOS with isolated venv and MPS-capable PyTorch
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/common.sh
source "$SCRIPT_DIR/../lib/common.sh"
genai_load_config
genai_ensure_dirs

LOG="$(genai_resolve_path "${PATHS_FILE%/*}")"
LOG="$(genai_resolve_path logs/comfyui/setup.log)"
CLONE_DIR="$(genai_resolve_path "$(python3 -c "import json;print(json.load(open('$PATHS_FILE'))['comfyui']['cloneDir'])")")"
ENV_DIR="$(genai_resolve_path "$(python3 -c "import json;print(json.load(open('$PATHS_FILE'))['comfyui']['envDir'])")")"

COMMIT="$(python3 -c "import json;print(json.load(open('$LOCK_FILE'))['upstreams']['comfyui']['commit'])")"
REPO="$(python3 -c "import json;print(json.load(open('$LOCK_FILE'))['upstreams']['comfyui']['repository'])")"

genai_log "$LOG" "Starting ComfyUI macOS setup"
command -v git >/dev/null || { echo "Git required"; exit 1; }
PY="$(genai_python_candidate)" || { echo "Python 3.10+ required"; exit 1; }

genai_git_clone_pinned "$REPO" "$COMMIT" "$CLONE_DIR" "$LOG"

if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  "$PY" -m venv "$ENV_DIR"
fi

PIP="$ENV_DIR/bin/pip"
PYTHON="$ENV_DIR/bin/python"
"$PIP" install --upgrade pip wheel setuptools
"$PIP" install torch torchvision torchaudio
"$PIP" install -r "$CLONE_DIR/requirements.txt"

MPS="$("$PYTHON" -c "import torch; print(torch.backends.mps.is_available())")"
genai_log "$LOG" "MPS available: $MPS"

MODELS="$(genai_resolve_path "$(python3 -c "import json;print(json.load(open('$PATHS_FILE'))['comfyui']['modelsDir'])")")"
cat > "$CLONE_DIR/extra_model_paths.yaml" <<EOF
comfyui:
  base_path: ${MODELS}
  checkpoints: checkpoints
  vae: vae
  loras: loras
  embeddings: embeddings
  controlnet: controlnet
  upscale_models: upscale_models
EOF

echo "ComfyUI setup complete. MPS=$MPS"
