#/bin/env bash

TF_CPP_MIN_LOG_LEVEL=2
FORCE_CUDA="1"
ATTN_PRECISION=fp16
PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.9,max_split_size_mb:512
# TORCH_CUDA_ARCH_LIST="8.6"
ARGS="--num_cpu_threads_per_process=6 launch.py --api --xformers --disable-console-progressbars"
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
    env)
      MODE=env
      ;;
    *)
      ARGS="$ARGS $i"
      ;;
  esac
  shift
done

echo "SD server: $MODE"

if [ $MODE == install ]; then
  python --version
  pip3 --version
  echo "Installing general requirements"
  pip3 install --disable-pip-version-check --quiet --no-warn-conflicts --requirement requirements.txt
  echo "Installing versioned requirements"
  pip3 install --disable-pip-version-check --quiet --no-warn-conflicts --requirement requirements_versions.txt
  exit 0
fi

if [ $MODE == env ]; then
  VER=`git log -1 --pretty=format:"%h %ad"`
  LSB=`lsb_release -ds 2>/dev/null`
  UN=`uname -rm 2>/dev/null`
  echo "Version: $VER"
  echo "Platform: $LSB $UN"
  python --version
  python -c 'import torch; print("Torch:", torch.__version__, "CUDA:", torch.version.cuda, "cuDNN:", torch.backends.cudnn.version(), "GPU:", torch.cuda.get_device_name(torch.cuda.current_device()), "Arch:", torch.cuda.get_device_capability());'
  exit 0
fi

if [ $MODE == clean ]; then
  ARGS="$ARGS ----disable-opt-split-attention"
  python launch.py "$ARGS"
  exit 0
fi

if [ $MODE == public ]; then
  ARGS="$ARGS --port 8000 --gradio-auth admin:pwd --listen --enable-insecure-extension-access"
fi

if [ $MODE == optimized ]; then
  ARGS="$ARGS --xformers"
fi

exec accelerate launch "$ARGS"
