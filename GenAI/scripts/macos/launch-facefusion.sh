#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../lib/common.sh"
genai_load_config

CLONE_DIR="$(genai_resolve_path "$(python3 -c "import json;print(json.load(open('$PATHS_FILE'))['facefusion']['cloneDir'])")")"
ENV_DIR="$(genai_resolve_path "$(python3 -c "import json;print(json.load(open('$PATHS_FILE'))['facefusion']['envDir'])")")"
PROVIDER="${FACEFUSION_EXECUTION_PROVIDER:-coreml}"

[[ -x "$ENV_DIR/bin/python" ]] || { echo "Run setup-facefusion.sh first"; exit 1; }

case "$PROVIDER" in
  coreml) PROV_ARG=coreml ;;
  *) PROV_ARG=cpu ;;
esac

echo "=== FaceFusion Launch ==="
echo "Provider: $PROV_ARG"
"$ENV_DIR/bin/python" -c "import onnxruntime as ort; print('Available:', ort.get_available_providers())"
echo "========================="

cd "$CLONE_DIR"
exec "$ENV_DIR/bin/python" facefusion.py run --execution-providers "$PROV_ARG"
