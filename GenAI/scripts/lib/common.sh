#!/usr/bin/env bash
# GenAI shared bash helpers for macOS scripts
set -euo pipefail

genai_root() {
  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
  echo "$script_dir"
}

genai_load_config() {
  GENAI_ROOT="$(genai_root)"
  PATHS_FILE="$GENAI_ROOT/config/local-paths.json"
  SETTINGS_FILE="$GENAI_ROOT/config/settings.env"
  LOCK_FILE="$GENAI_ROOT/upstreams.lock.json"

  if [[ ! -f "$PATHS_FILE" ]]; then
    cp "$GENAI_ROOT/config/local-paths.example.json" "$PATHS_FILE"
    echo "Created config/local-paths.json from example."
  fi
  if [[ ! -f "$SETTINGS_FILE" ]]; then
    cp "$GENAI_ROOT/config/settings.example.env" "$SETTINGS_FILE"
    echo "Created config/settings.env from example."
  fi

  export GENAI_ROOT PATHS_FILE SETTINGS_FILE LOCK_FILE
  # shellcheck disable=SC1090
  set -a; source "$SETTINGS_FILE"; set +a
}

genai_json_field() {
  local section="$1" field="$2"
  GENAI_JSON_SECTION="$section" GENAI_JSON_FIELD="$field" python3 - "$PATHS_FILE" <<'PY'
import json, os, sys
path = sys.argv[1]
data = json.load(open(path, encoding="utf-8"))
section = os.environ["GENAI_JSON_SECTION"]
field = os.environ["GENAI_JSON_FIELD"]
print(data[section][field])
PY
}

genai_resolve_config_path() {
  genai_resolve_path "$(genai_json_field "$1" "$2")"
}

genai_log() {
  local logfile="$1"; shift
  local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
  echo "$msg"
  mkdir -p "$(dirname "$logfile")"
  echo "$msg" >> "$logfile"
}

genai_ensure_dirs() {
  local dirs=(
    runtime/comfyui runtime/environments/comfyui
    runtime/facefusion runtime/environments/facefusion runtime/facefusion-temp
    models/comfyui/checkpoints models/comfyui/vae models/comfyui/loras
    models/comfyui/embeddings models/comfyui/controlnet models/comfyui/upscale_models
    models/facefusion inputs/comfyui inputs/facefusion
    outputs/comfyui outputs/facefusion logs/comfyui logs/facefusion
  )
  for d in "${dirs[@]}"; do
    mkdir -p "$(genai_resolve_path "$d")"
  done
}

genai_git_clone_pinned() {
  local repo="$1" commit="$2" target="$3" logfile="$4"
  if [[ -d "$target/.git" ]]; then
    git -C "$target" fetch --tags --quiet origin
    git -C "$target" checkout "$commit"
    genai_log "$logfile" "Checked out pinned commit $commit"
  else
    rm -rf "$target"
    mkdir -p "$(dirname "$target")"
    git clone "$repo" "$target"
    git -C "$target" checkout "$commit"
    genai_log "$logfile" "Cloned and checked out $commit"
  fi
}

genai_python_candidate() {
  for cmd in python3.12 python3.11 python3.10 python3 python; do
    if command -v "$cmd" >/dev/null 2>&1; then
      echo "$cmd"
      return 0
    fi
  done
  return 1
}

genai_diag_line() {
  printf '[%s] %s\n' "$2" "$1"
  [[ -n "${3:-}" ]] && printf '      %s\n' "$3"
}
