## Important notes

The Old installation guide below may be outdated, please try running `webui-user.sh` to install. See [this Pull Request](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/5286) for reference.
 
## Automatic installation

First, you need to install the required dependencies using [Homebrew](https://brew.sh).

`brew install cmake protobuf rust python git wget`

The script can be downloaded from [here](https://github.com/dylancl/stable-diffusion-webui-mps/blob/master/setup_mac.sh), or follow the instructions below.

1. Open Terminal.app
2. Run the following commands:

```
$ cd ~/Documents/
$ curl https://raw.githubusercontent.com/dylancl/stable-diffusion-webui-mps/master/setup_mac.sh -o setup_mac.sh
$ chmod +x setup_mac.sh
$ ./setup_mac.sh
```

3. Follow the instructions in the terminal window.

#### Usage

After installation, you'll now find `run_webui_mac.sh` in the `stable-diffusion-webui` directory. Run this script to start the web UI using `./run_webui_mac.sh`.
This script automatically activates the conda environment, pulls the latest changes from the repository, and starts the web UI. On exit, the conda environment is deactivated.