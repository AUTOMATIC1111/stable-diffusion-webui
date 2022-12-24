import sys


if "xformers" not in "".join(sys.argv):
    sys.modules["xformers"] = None
