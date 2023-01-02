#!/bin/bash

gpu_count=$(nvidia-smi -L | wc -l);
port=7860

python3 launch.py --prepare-only --api --nowebui --listen "$@"

for i in $(seq 0 $((gpu_count - 1)))
do
  screen -d -m python3 launch.py --no-prepare --api --nowebui --listen --device-id "$i" --port "$port" "$@";
  port=$((port + 1));
done

sleep infinity