#!/bin/env bash

export TF_CPP_MIN_LOG_LEVEL=2
export ACCELERATE="True"
export FORCE_CUDA="1"
export ATTN_PRECISION=fp16
export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.9,max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=0
export CUDA_CACHE_DISABLE=0
export CUDA_AUTO_BOOST=1
export CUDA_MODULE_LOADING="LAZY"
export CUDA_DEVICE_DEFAULT_PERSISTING_L2_CACHE_PERCENTAGE_LIMIT=0
export GRADIO_ANALYTICS_ENABLED="False"

if [ "$PYTHON" == "" ]; then
  PYTHON=$(which python)
fi

# Note: Some defaults are changed in shared.py
CMD="launch.py --xformers --skip-install --skip-torch-cuda-test --cors-allow-origins=http://127.0.0.1:7860"

MODE=optimized

if [[ $(id -u) -eq 0 ]]; then
    echo "Running as root, aborting"
    exit 1
fi

"$PYTHON" -m pip --quiet show torch
if [ $? -ne 0 ]; then
  echo "Torch not installed, aborting"
  exit 1
fi

for i in "$@"; do
  case $i in
    update)
      MODE=update
      ;;
    install)
      MODE=install
      ;;
    public)
      MODE=public
      ;;
    clean)
      MODE=clean
      ;;
    help)
      MODE=help
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
SMI=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader --id=0 2>/dev/null)
echo "Version: $VER"
echo "Repository: $URL"
echo "Last Merge: $MERGE"
echo "System"
echo "- Platform: $LSB $UNAME"
echo "- nVIDIA: $SMI"
"$PYTHON" -c 'import torch; import platform; print("- Python:", platform.python_version(), "Torch:", torch.__version__, "CUDA:", torch.version.cuda, "cuDNN:", torch.backends.cudnn.version(), "GPU:", torch.cuda.get_device_name(torch.cuda.current_device()), "Arch:", torch.cuda.get_device_capability());'

git-version () {
    pushd $1 >/dev/null
    BRANCH=$(git branch | grep -E 'main|master' | tail -1 | awk '{print $NF}')
    VER=$(git log -1 --pretty=format:"%h %ad")
    URL=$(git remote get-url origin)
    popd >/dev/null
    echo "- $VER $BRANCH $URL"
}

git-update () {
    pushd $1 >/dev/null
    BRANCH=$(git branch | grep -E 'main|master' | tail -1 | awk '{print $NF}')
    git checkout --quiet $BRANCH
    git pull --quiet --rebase --autostash
    popd >/dev/null
    git-version $1
}

if [ "$MODE" == update ]; then
  echo "Updating main repository"
  git-update .
  "$PYTHON" launch.py --exit
  echo "Local changes"
  git status --untracked=no --ignore-submodules=all --short
  echo "Note: To update any new dependencies or submodules, run 'automatic.sh install'"
  
  exit 0
fi

if [ "$MODE" == help ]; then
  "$PYTHON" webui.py --help
  exit 0
fi

if [ "$MODE" == install ]; then
  "$PYTHON" -m pip --version

  echo "Installing general requirements"
  "$PYTHON" -m pip install --disable-pip-version-check --quiet --no-warn-conflicts --requirement requirements.txt

  echo "Installing versioned requirements"
  "$PYTHON" -m pip install --disable-pip-version-check --quiet --no-warn-conflicts --requirement requirements_versions.txt

  echo "Updating submodules"
  git submodule --quiet update --init --recursive
  git submodule --quiet foreach 'echo $sm_path' | while read LINE; do git-update $LINE ; done
  # git submodule --quiet update --rebase --remote
  # git submodule foreach --quiet 'VER=$(git log -1 --pretty=format:"%h %ad"); BRANCH=$(git branch); URL=$(git remote get-url origin); echo "- $VER $BRANCH $URL"'

  echo "Updating extensions"
  ls extensions/ | while read LINE; do git-update extensions/$LINE ; done

  echo "Updating wiki"
  git-update wiki
  git-update wiki/origin-wiki

  echo "Detached repos"
  ls repositories/ | while read LINE; do git-version repositories/$LINE ; done

  "$PYTHON" launch.py --exit

  echo "Local changes"
  git status --untracked=no --ignore-submodules=all --short
  
  exit 0
fi

if [ "$MODE" == clean ]; then
  CMD="--disable-opt-split-attention"
  "$PYTHON" launch.py $CMD
  exit 0
fi

if [ $MODE == public ]; then
  CMD="$CMD --port 7860 --gradio-auth admin:pwd --listen --enable-insecure-extension-access"
fi

if [ $MODE == optimized ]; then
  CMD="$CMD"
fi

# exec accelerate launch --no_python --quiet --num_cpu_threads_per_process=6 "$PYTHON" $CMD
exec "$PYTHON" $CMD

# export LD_PRELOAD=libtcmalloc.so
# TORCH_CUDA_ARCH_LIST="8.6"
# --opt-channelslast
# --opt-sdp-attention
