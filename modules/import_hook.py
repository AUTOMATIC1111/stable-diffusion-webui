import builtins
import sys

old_import = builtins.__import__
IMPORT_BLACKLIST = []


if "xformers" not in "".join(sys.argv):
    IMPORT_BLACKLIST.append("xformers")


def import_hook(*args, **kwargs):
    if args[0] in IMPORT_BLACKLIST:
        raise ImportError("Import of %s is blacklisted" % args[0])
    return old_import(*args, **kwargs)


builtins.__import__ = import_hook
