# Installing and running on Linux with AMD GPUs
If your AMD GPU is compatible with ROCm, you can try running: `TORCH_COMMAND='pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/rocm5.1.1' python launch.py --precision full --no-half`. Make sure to do this in a new virtual environment, or activate your existing environment and `pip uninstall torch torchvision` beforehand.

The first generation after starting the WebUI might take very long, and you might see a message similar to this: 
> MIOpen(HIP): Warning [SQLiteBase] Missing system database file: gfx1030_40.kdb Performance may degrade. Please follow instructions to install: https://github.com/ROCmSoftwarePlatform/MIOpen#installing-miopen-kernels-package

Subsequent generations should work with regular performance. You can follow the link in the message, and if you happen to use the same operating system, follow the steps there to fix this issue. If there is no clear way to compile or install the MIOpen kernels for your operating system, consider following the Docker guide below.

# Installing and running using Docker
This is only tested using a Linux host!

Pull the latest `rocm/pytorch` Docker image, start the image and attach to the container (taken from the `rocm/pytorch` documentation): `docker run -it --network=host --device=/dev/kfd --device=/dev/dri --group-add=video --ipc=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -v $HOME/dockerx:/dockerx rocm/pytorch`

Execute the following inside the container:
```bash
cd /dockerx
git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui
cd stable-diffusion-webui
python -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip wheel
TORCH_COMMAND='pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/rocm5.1.1' REQS_FILE='requirements.txt' python launch.py --precision full --no-half
```

Subsequent runs will only require you to restart the container, attach to it again and execute the following inside the container: Find the container name from this listing: `docker container ls --all`, select the one matching the `rocm/pytorch` image, restart it: `docker container restart <container-id>` then attach to it: `docker exec -it <container-id> bash`.

```bash
cd /dockerx/stable-diffusion-webui
# Optional: "git pull" to update the repository
source venv/bin/activate
TORCH_COMMAND='pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/rocm5.1.1' REQS_FILE='requirements.txt' python launch.py --precision full --no-half
```

The `/dockerx` folder inside the container should be accessible in your home directory under the same name.