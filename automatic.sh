#/bin/env bash

export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.9,max_split_size_mb:512
python launch.py --api --disable-console-progressbars 
# python launch.py --api --xformers --disable-console-progressbars --opt-channelslast
