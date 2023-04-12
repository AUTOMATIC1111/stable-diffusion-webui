import sys
from modules import shared

# this will break any attempt to import xformers which will prevent stability diffusion repo from trying to use it
if shared.opts.cross_attention_optimization != "xFormers":
    sys.modules["xformers"] = None
