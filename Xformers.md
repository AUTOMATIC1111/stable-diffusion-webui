Xformers library is an optional way to speedup your image generation.

There are no binaries for Windows except for one specific configuration, but you can build it yourself.

A guide from an anonymous user, although I think it is for building on Linux:

GUIDES ON HOW TO BUILD XFORMERS
also includes how to uncuck yourself from sm86 restriction on voldy's new commit

1. go to the webui directory
2. `source ./venv/bin/activate`
3. `cd repositories`
3. `git clone https://github.com/facebookresearch/xformers.git`
4. `cd xformers`
5. `git submodule update --init --recursive`
6. `pip install -r requirements.txt`
7. `pip install -e .`

## Building xFormers on Windows by [@duckness](https://github.com/duckness)

***


### If you use a Pascal, Turing, Ampere, Lovelace or Hopper card with Python 3.10, you shouldn't need to build manually anymore. Uninstall your existing xformers and launch the repo with `--xformers`. A compatible wheel will be installed.




***

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
