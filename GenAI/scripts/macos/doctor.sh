#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../lib/common.sh"
genai_load_config

FAIL=0
check() {
  local name="$1" status="$2" detail="${3:-}"
  genai_diag_line "$name" "$status" "$detail"
  [[ "$status" == "FAIL" ]] && FAIL=1
}

check "Repository" "PASS" "$GENAI_ROOT"
[[ -f "$PATHS_FILE" ]] && check "local-paths.json" "PASS" || check "local-paths.json" "FAIL" "Run setup-all.sh"
[[ -f "$SETTINGS_FILE" ]] && check "settings.env" "PASS" || check "settings.env" "FAIL"
command -v git >/dev/null && check "Git" "PASS" "$(git --version)" || check "Git" "FAIL"
command -v ffmpeg >/dev/null && check "FFmpeg" "PASS" || check "FFmpeg" "WARN" "brew install ffmpeg"

ARCH="$(uname -m)"
check "Architecture" "PASS" "$ARCH"
sw_vers 2>/dev/null | while read -r line; do check "macOS" "PASS" "$line"; done

ENV_C="$(genai_resolve_path "$(python3 -c "import json;print(json.load(open('$PATHS_FILE'))['comfyui']['envDir'])")")"
if [[ -x "$ENV_C/bin/python" ]]; then
  check "ComfyUI venv" "PASS" "$ENV_C"
  MPS="$("$ENV_C/bin/python" -c "import torch; print(torch.backends.mps.is_available())")"
  check "PyTorch MPS" "$([[ "$MPS" == "True" ]] && echo PASS || echo WARN)" "mps=$MPS"
else
  check "ComfyUI venv" "FAIL" "Run setup-comfyui.sh"
fi

ENV_F="$(genai_resolve_path "$(python3 -c "import json;print(json.load(open('$PATHS_FILE'))['facefusion']['envDir'])")")"
if [[ -x "$ENV_F/bin/python" ]]; then
  check "FaceFusion venv" "PASS" "$ENV_F"
  PROV="$("$ENV_F/bin/python" -c "import onnxruntime as ort; print(','.join(ort.get_available_providers()))")"
  check "ONNX providers" "PASS" "$PROV"
else
  check "FaceFusion venv" "FAIL" "Run setup-facefusion.sh"
fi

exit $FAIL
