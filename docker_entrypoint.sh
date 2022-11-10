# A Docker entrypoint to support mounting a cache into the image

set -ex

# If cached packages and models are mounted, use them
if [ -d /var/sd ]; then
    ln -s /var/sd/.mpl_config /sd/.mpl_config
    ln -s /var/sd/.xdg_cache /sd/.xdg_cache
    ln -s /var/sd/models /sd/models
    ln -s /var/sd/repositories /sd/repositories
    ln -s /var/sd/venv /sd/venv
fi

# Create venv if not exists, then activate it
if [ ! -d "venv" ]; then
    python -m venv venv
fi

source venv/bin/activate

# Ensure xformers installed
if [ ! -d "venv/lib/python3.10/site-packages/xformers" ]; then
    pip install /xformers-0.0.14.dev0-cp310-cp310-linux_x86_64.whl --no-deps
fi

# Ensure dependencies installed
if [ ! -d "venv/lib/python3.10/site-packages/opencv-python-headless" ]; then
    # Installs dependencies
    python launch.py --exit --skip-torch-cuda-test

    # Replace opencv-python (installed as a side effect of `python launch.py) with
    # opencv-python-headless, to remove dependency on missing libGL.so.1.
    pip install opencv-python-headless
fi

# Start service
python launch.py --api --listen --xformers
