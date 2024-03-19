#!/bin/bash

# Name of your node application file
NODE_APP="webui.sh"

# Check if the monitor script is already running



MONITOR_COUNT=$(pgrep -f "webuidetection.sh" | grep -v "^$$$" | wc -l)
if [[ $MONITOR_COUNT -gt 2 ]]; then
    echo "Another instance of monitor.sh is running. Exiting. $MONITOR_COUNT"
    exit 1
fi

# Function to check if the Node app is running
is_app_running() {
    pgrep -f "$NODE_APP" > /dev/null
    return $?
}

# Main loop to check and start the Node app if not running
while true; do
    if ! is_app_running; then
        echo "webui app is not running. Starting it now..."
        sudo nohup ./webui.sh --listen --xformers --upcast-sampling  --skip-torch-cuda-test --enable-insecure-extension-access --no-half-vae --api --port 7890 &
    fi
    sleep 10 # Check every 10 seconds
done
