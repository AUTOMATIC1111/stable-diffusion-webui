#!/bin/bash

gpu_count=$(nvidia-smi -L | wc -l);
port=7860

for i in $(seq 0 "$gpu_count")
do
  python3 launch.py --api --nowebui --listen --device-id "$i" --port "$port" "$@";
  port=$((port + 1));
done