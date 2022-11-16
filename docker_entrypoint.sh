#!/bin/bash

# A Docker entrypoint to download the dependencies and models required
# by SD. Supports mounting a cache at /var/sd.

set -ex

CACHED_FOLDERS=(
    .xdg_cache    # Transformers models   
    .mpl_config   # MatPlotLib config files
    models        # SD models and some others e.g. GFPGAN
    repositories  # Codeformer and some others
    venv          # Python dependencies
)

# If the cache is being initialized, wait up to $max_wait_mins minutes
max_wait_mins=15
loop_counter=0
while [[ -f /var/sd/.container_initializing ]] && [[ $loop_counter -lt $max_wait_mins ]]; do
    min=$[$max_wait_mins - $loop_counter]
    echo "Cache is initializing, waiting up to ${min} minutes"
    sleep 1m
    loop_counter=$[$loop_counter + 1]
done

# If there is no initialized folder at /var/sd, create it
if [[ ! -f /var/sd/.container_initialized ]]; then

    # Delete the .container_initializing marker on exit
    function cleanup {
        rm -r /var/sd/.container_initializing
    }
    trap cleanup EXIT

    # Create cache directory and check write access
    mkdir -p /var/sd
    touch /var/sd/.container_initializing

    # Download models and dependencies
    source docker_warmup.sh

    # Move everything to the cache
    # Folders are linked to main /sd folder below
    for folder in "${CACHED_FOLDERS[@]}"; do
        mkdir -p "${folder}"
        mv "${folder}" "/var/sd/${folder}"
    done

    # Mark cache as initialized
    touch /var/sd/.container_initialized
    rm /var/sd/.container_initializing

fi

# Use the cached dependencies and models
for folder in "${CACHED_FOLDERS[@]}"; do
    ln -s "/var/sd/${folder}" "/sd/${folder}"
done

# Load venv
source venv/bin/activate

# Start service
exec python launch.py --api --listen --xformers
