# Xformers Library (Optional)
The Xformers library provides an optional method to accelerate image generation. This enhancement is exclusively available for NVIDIA GPUs, optimizing image generation and reducing VRAM usage. Older versions below 0.0.20 will produce [non-deterministic](https://github.com/AUTOMATIC1111/stable-diffusion-webui/discussions/2705#discussioncomment-4024378) results.

## Important Notice - No Need for Manual Installation
As of January 23, 2023, neither Windows nor Linux users are required to manually build the Xformers library. This change was implemented when WebUI transitioned from a user-built wheel to an [official wheel](https://pypi.org/project/xformers/0.0.16rc425/#history). You can view the package upgrades and other details of this update in [this PR](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/5939/commits/c091cf1b4acd2047644d3571bcbfd81c81b4c3af).

## Usage
If you are using a Pascal, Turing, Ampere, Lovelace, or Hopper card with Python 3.10, simply launch the repository using the --xformers flag. The compatible wheel will be automatically installed.

## Building xformers on Windows by [@duckness](https://github.com/duckness)

1. [Install VS Build Tools 2022](https://visualstudio.microsoft.com/downloads/?q=build+tools#build-tools-for-visual-studio-2022), you only need `Desktop development with C++`

![setup_COFbK0AJAZ](https://user-images.githubusercontent.com/6380270/194767872-232136a1-9204-4b16-ae21-3e01f6f526ea.png)

2. [Install CUDA 11.3](https://developer.nvidia.com/cuda-11.3.0-download-archive) (later versions are not tested), select custom, you only need the following (VS integration is probably unecessary):

![setup_QwCdsQ28FM](https://user-images.githubusercontent.com/6380270/194767963-6df7ce14-e6eb-4718-8e93-a11abf172f14.png)

3. Clone the [xFormers repo](https://github.com/facebookresearch/xformers), create a `venv` and activate it

```sh
git clone https://github.com/facebookresearch/xformers.git
cd xformers
git submodule update --init --recursive
python -m venv venv
./venv/scripts/activate
```

4. To avoid issues with getting the CPU version, [install pyTorch seperately](https://pytorch.org/get-started/locally/):

```sh
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113
```

5. Then install the rest of the dependencies:

```sh
pip install -r requirements.txt
pip install wheel
```

6. As CUDA 11.3 is rather old, you need to force enable it to be built on MS Build Tools 2022. Do `$env:NVCC_FLAGS = "-allow-unsupported-compiler"` if on `powershell`, or `set NVCC_FLAGS=-allow-unsupported-compiler` if on `cmd`


7. You can finally build xFormers, note that the build will take a long time (probably 10-20minutes), it may initially complain of some errors but it should still compile correctly. 

> OPTIONAL tip: To further speed up on multi-core CPU Windows systems, install ninja https://github.com/ninja-build/ninja.
> Steps to install:
> 1. download ninja-win.zip from https://github.com/ninja-build/ninja/releases and unzip
> 2. place ninja.exe under C:\Windows OR add the full path to the extracted ninja.exe into system PATH
> 3. Run ninja -h in cmd and verify if you see a help message printed
> 4. Run the follow commands to start building. It should automatically use Ninja, no extra config is needed. You should see significantly higher CPU usage (40%+).
> ```
> python setup.py build
> python setup.py bdist_wheel
> ```
> This has reduced build time on a windows PC with a AMD 5800X CPU from 1.5hr to 10min.
> Ninja is also supported on Linux and MacOS but I do not have these OS to test thus can not provide step-by-step tutorial.



8. Run the following:
 ```sh
python setup.py build
python setup.py bdist_wheel
```

9. In `xformers` directory, navigate to the `dist` folder and copy the `.whl` file to the base directory of `stable-diffusion-webui`

10. In `stable-diffusion-webui` directory, install the `.whl`, change the name of the file in the command below if the name is different:

```sh
./venv/scripts/activate
pip install xformers-0.0.14.dev0-cp310-cp310-win_amd64.whl
```

11. Ensure that `xformers` is activated by launching `stable-diffusion-webui` with `--force-enable-xformers`

## Building xformers on Linux (from anonymous user)

1. go to the webui directory
2. `source ./venv/bin/activate`
3. `cd repositories`
3. `git clone https://github.com/facebookresearch/xformers.git`
4. `cd xformers`
5. `git submodule update --init --recursive`
6. `pip install -r requirements.txt`
7. `pip install -e .`
