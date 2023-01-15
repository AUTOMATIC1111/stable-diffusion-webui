#/bin/env bash

export TF_CPP_MIN_LOG_LEVEL=2
export FORCE_CUDA="1"
export ACCELERATE="True"
export ATTN_PRECISION=fp16export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.9,max_split_size_mb:512
python launch.py --api --xformers --disable-console-progressbars
# python launch.py --api  --disable-console-progressbars --opt-channelslast
