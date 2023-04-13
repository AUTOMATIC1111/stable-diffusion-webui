import sys
from modules.shared import opts

# this will break any attempt to import xformers which will prevent stability diffusion repo from trying to use it
if opts.cross_attention_optimization != "xFormers":
    sys.modules["xformers"] = None
