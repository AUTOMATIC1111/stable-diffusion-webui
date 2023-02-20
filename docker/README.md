# Docker and compose support for Stable DIffusion web UI

This folder contains files necessairy to build and run Stable Diffusion web UI as a docker container.
As for now, only the Nvidia accelerarion on Linux is supported.

## Requirements
* A [docker](https://docs.docker.com/engine/install/) of course.
* [Nvidia docker runtime](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker).


# HOWTO

1. Make sure you have [Nvidia docker runtime](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) installled and configured properly, i.e. `docker run --rm --runtime=nvidia --gpus all nvidia/cuda:11.6.2-base-ubuntu20.04 nvidia-smi` outputs the information about your GPU.
1. Enter this very directory `stable-diffusion-webui/docker`.
1. Build the container using `./build.sh` or `docker compose build`.
1. Run the container with `./run.sh`
1. Wait for the UI to start then open the URL displayed on the screen:
```
 ./run.sh
[+] Running 1/1
 ⠿ Container docker-stable-diffusion-webui-1  Started                                                                                                                                                                                                                              11.2s

Wait for the UI to start then point your browser to: http://localhost:xxxxx   <-----


...
