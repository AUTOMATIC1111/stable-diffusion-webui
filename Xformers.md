Xformers library is an optional way to speedup your image generation.

There are no binaries for Windows except for one specific configuration, but you can build it yourself.

A guide from an anonymous user, although I think it is for building on Linux:

GUIDE ON HOW TO BUILD XFORMERS
also includes how to uncuck yourself from sm86 restriction on voldy's new commit

1. go to the webui directory
2. `source ./venv/bin/activate`
3. `cd repositories`
3. `git clone https://github.com/facebookresearch/xformers.git`
4. `cd xformers`
5. `git submodule update --init --recursive`
6. `pip install -r requirements.txt`
7. `pip install -e .`

    If you encounter some error about torch not being built with your cuda version blah blah, then try:
    `pip install setuptools==49.6.0`

8. wait like 30 minutes for everything to build
9. launch with `--force-enable-xformers` commandline argument
