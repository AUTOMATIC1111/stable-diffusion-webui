#/bin/env bash

export TF_CPP_MIN_LOG_LEVEL=2
export FORCE_CUDA="1"
export ATTN_PRECISION=fp16
export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.9,max_split_size_mb:512
exec accelerate launch --num_cpu_threads_per_process=6 launch.py --api --xformers --disable-console-progressbars --port 8000 --gradio-auth admin:pwd --listen --enable-insecure-extension-access "$@"
