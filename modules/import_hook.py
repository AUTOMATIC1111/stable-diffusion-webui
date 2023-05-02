import sys
from modules.shared import opts, log

# this will break any attempt to import xformers which will prevent stability diffusion repo from trying to use it
try:
    import xformers # pylint: disable=unused-import, import-error
    import xformers.ops # pylint: disable=unused-import, import-error
except:
    pass

if opts.cross_attention_optimization != "xFormers":
    if sys.modules.get("xformers", None) is not None:
        log.info('Unloading xFormers')
    sys.modules["xformers"] = None
    sys.modules["xformers.ops"] = None
