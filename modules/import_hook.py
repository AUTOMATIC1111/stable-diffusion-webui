import sys

# this will break any attempt to import xformers which will prevent stability diffusion repo from trying to use it
if "--xformers" not in "".join(sys.argv):
    sys.modules["xformers"] = None
    print('Note: not using xformers. To enable xformers install it and pass --xformers flag')
