# Docker image and compose file for Stable DIffusion web UI

This folder contains files necessairy to build and run Stable Diffusion web UI in docker container.
As for now, only the Nvidia accelerarion on Linux is supported.

## Requirements
* A [docker](https://docs.docker.com/engine/install/) of course
* [Nvidia docker runtime](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)


# HOWTO

1. Make sure you have [Nvidia docker runtime](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) installled and configured properly, i.e. `docker run --rm --runtime=nvidia --gpus all nvidia/cuda:11.6.2-base-ubuntu20.04 nvidia-smi` outputs the information about your GPU.
1. Enter this very directory `stable-diffusion-webui/docker`.
1. Build the container using `./build.sh` or `docker compose build`.
1. Run the container with `./run.sh`
1. Examine the port assigned to the container:
```
$ docker compose ps
NAME                              IMAGE                                 COMMAND                  SERVICE                  CREATED             STATUS              PORTS
docker-stable-diffusion-webui-1   automatic111/stable-diffusion-webui   "/opt/nvidia/nvidia_…"   stable-diffusion-webui   2 seconds ago       Up 1 second         0.0.0.0:32787->7860/tcp, :::32787->7860/tcp
```
1. Point your browkes to `localhost:<PORT>` where `<PORT>` is the port number assigned to the container (in above example it's 32787)
