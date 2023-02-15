#!/bin/env bash

TF_CPP_MIN_LOG_LEVEL=2
FORCE_CUDA="1"
ATTN_PRECISION=fp16
PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.9,max_split_size_mb:512
# TORCH_CUDA_ARCH_LIST="8.6"
CUDA_LAUNCH_BLOCKING=0
CUDA_CACHE_DISABLE=0
CUDA_AUTO_BOOST=1
CUDA_DEVICE_DEFAULT_PERSISTING_L2_CACHE_PERCENTAGE_LIMIT=0

if [ "$PYTHON" == "" ]; then
  PYTHON=$(which python)
fi

CMD="launch.py --api --xformers --disable-console-progressbars --gradio-queue --skip-version-check --cors-allow-origins=http://127.0.0.1:7860"
# --opt-channelslast
MODE=optimized

if [[ $(id -u) -eq 0 ]]; then
    echo "Running as root, aborting"
    exit 1
fi

for i in "$@"; do
  case $i in
    install)
      MODE=install
      ;;
    public)
      MODE=public
      ;;
    clean)
      MODE=clean
      ;;
    *)
      CMD="$CMD $i"
      ;;
  esac
  shift
done

echo "SD server: $MODE"

VER=$(git log -1 --pretty=format:"%h %ad")
URL=$(git remote get-url origin)
LSB=$(lsb_release -ds 2>/dev/null)
UNAME=$(uname -rm 2>/dev/null)
MERGE=$(git log --pretty=format:"%ad %s" | grep "Merge pull" | head -1)
echo "Version: $VER"
echo "Repository: $URL"
echo "Last Merge: $MERGE"
echo "Platform: $LSB $UNAME"
"$PYTHON" -c 'import torch; import platform; print("Python:", platform.python_version(), "Torch:", torch.__version__, "CUDA:", torch.version.cuda, "cuDNN:", torch.backends.cudnn.version(), "GPU:", torch.cuda.get_device_name(torch.cuda.current_device()), "Arch:", torch.cuda.get_device_capability());'

if [ "$MODE" == install ]; then
  "$PYTHON" -m pip --version

  echo "Installing general requirements"
  "$PYTHON" -m pip install --disable-pip-version-check --quiet --no-warn-conflicts --requirement requirements.txt

  echo "Installing versioned requirements"
  "$PYTHON" -m pip install --disable-pip-version-check --quiet --no-warn-conflicts --requirement requirements_versions.txt

  echo "Updating submodules"
  git submodule update --init --recursive
  git submodule update --rebase --remote
  echo "Modules:"
  git submodule foreach --quiet 'VER=$(git log -1 --pretty=format:"%h %ad"); URL=$(git remote get-url origin); echo "- $VER $URL"'

  echo "Updating extensions"
  echo "Extensions:"
  ls extensions/ | while read LINE; do
    pushd extensions/$LINE >/dev/null
    git pull --quiet
    VER=$(git log -1 --pretty=format:"%h %ad")
    URL=$(git remote get-url origin)
    popd >/dev/null
    echo "- $VER $URL"
  done

  echo "Local changes"
  git status --untracked=no --ignore-submodules=all --short

  exit 0
fi

if [ "$MODE" == clean ]; then
  CMD="--disable-opt-split-attention --disable-console-progressbars --api"
  "$PYTHON" launch.py $CMD
  exit 0
fi

if [ $MODE == public ]; then
  CMD="$CMD --port 7860 --gradio-auth admin:pwd --listen --enable-insecure-extension-access"
fi

if [ $MODE == optimized ]; then
  CMD="$CMD"
fi

exec accelerate launch --no_python --quiet --num_cpu_threads_per_process=6 "$PYTHON" $CMD
