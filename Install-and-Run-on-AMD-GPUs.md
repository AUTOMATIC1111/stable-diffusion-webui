# Installing and running on Linux with AMD GPUs
If your AMD GPU is compatible with ROCm, you can try running: 

```bash
# It's possible that you don't need "--precision full", dropping "--no-half" however crashes my drivers
TORCH_COMMAND='pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/rocm5.1.1' python launch.py --precision full --no-half
```

Make sure to do this in a new virtual environment, or activate your existing environment and
`pip uninstall torch torchvision` beforehand.

The first generation after starting the WebUI might take very long, and you might see a message similar to this: 
> MIOpen(HIP): Warning [SQLiteBase] Missing system database file: gfx1030_40.kdb Performance may degrade. Please follow
> instructions to install: https://github.com/ROCmSoftwarePlatform/MIOpen#installing-miopen-kernels-package

The next generations should work with regular performance. You can follow the link in the message, and if you happen
to use the same operating system, follow the steps there to fix this issue. If there is no clear way to compile or
install the MIOpen kernels for your operating system, consider following the Docker guide below.

# Installing and running on Linux with AMD GPUs (Docker)
Pull the latest `rocm/pytorch` Docker image, start the image and attach to the container (taken from the `rocm/pytorch`
documentation): `docker run -it --network=host --device=/dev/kfd --device=/dev/dri --group-add=video --ipc=host
--cap-add=SYS_PTRACE --security-opt seccomp=unconfined -v $HOME/dockerx:/dockerx rocm/pytorch`

Execute the following inside the container:
```bash
cd /dockerx
git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui
cd stable-diffusion-webui
python -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip wheel

# It's possible that you don't need "--precision full", dropping "--no-half" however crashes my drivers
TORCH_COMMAND='pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/rocm5.1.1' REQS_FILE='requirements.txt' python launch.py --precision full --no-half
```

Following runs will only require you to restart the container, attach to it again and execute the following inside the
container: Find the container name from this listing: `docker container ls --all`, select the one matching the
`rocm/pytorch` image, restart it: `docker container restart <container-id>` then attach to it: `docker exec -it
<container-id> bash`.

```bash
cd /dockerx/stable-diffusion-webui
# Optional: "git pull" to update the repository
source venv/bin/activate

# It's possible that you don't need "--precision full", dropping "--no-half" however crashes my drivers
TORCH_COMMAND='pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/rocm5.1.1' REQS_FILE='requirements.txt' python launch.py --precision full --no-half
```

The `/dockerx` folder inside the container should be accessible in your home directory under the same name.

## Updating Python version inside Docker
If the web UI becomes incompatible with the pre-installed Python 3.7 version inside the Docker image, here are
instructions on how to update it (assuming you have successfully followed "Installing and running using Docker"):

Execute the following inside the container:
```bash
apt install python3.9-full # Confirm every prompt
update-alternatives --install /usr/local/bin/python python /usr/bin/python3.9 1
echo 'PATH=/usr/local/bin:$PATH' >> ~/.bashrc
```

Then restart the container and attach again. If you check `python --version` it should now say `Python 3.9.5` or newer.

Run `rm -rf /dockerx/stable-diffusion-webui/venv` inside the container and then follow the steps in "Installing and
running using Docker" again, skipping the `git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui` and using
the modified launch-command below instead:

```bash
# It's possible that you don't need "--precision full", dropping "--no-half" however crashes my drivers
TORCH_COMMAND='pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/rocm5.1.1' python launch.py --precision full --no-half
```

Always use this new launch-command from now on, also when restarting the web UI in following runs.