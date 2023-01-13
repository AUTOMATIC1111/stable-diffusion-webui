# this code is adapted from the script contributed by anon from /h/

import io
import pickle
import collections
import sys
import traceback

import torch
import numpy
import _codecs
import zipfile
import re


# PyTorch 1.13 and later have _TypedStorage renamed to TypedStorage
TypedStorage = torch.storage.TypedStorage if hasattr(torch.storage, 'TypedStorage') else torch.storage._TypedStorage


def encode(*args):
    out = _codecs.encode(*args)
    return out


class RestrictedUnpickler(pickle.Unpickler):
    extra_handler = None

    def persistent_load(self, saved_id):
        assert saved_id[0] == 'storage'
        return TypedStorage()

    def find_class(self, module, name):
        if self.extra_handler is not None:
            res = self.extra_handler(module, name)
            if res is not None:
                return res

        if module == 'collections' and name == 'OrderedDict':
            return getattr(collections, name)
        if module == 'torch._utils' and name in ['_rebuild_tensor_v2', '_rebuild_parameter', '_rebuild_device_tensor_from_numpy']:
            return getattr(torch._utils, name)
        if module == 'torch' and name in ['FloatStorage', 'HalfStorage', 'IntStorage', 'LongStorage', 'DoubleStorage', 'ByteStorage', 'float32']:
            return getattr(torch, name)
        if module == 'torch.nn.modules.container' and name in ['ParameterDict']:
            return getattr(torch.nn.modules.container, name)
        if module == 'numpy.core.multiarray' and name in ['scalar', '_reconstruct']:
            return getattr(numpy.core.multiarray, name)
        if module == 'numpy' and name in ['dtype', 'ndarray']:
            return getattr(numpy, name)
        if module == '_codecs' and name == 'encode':
            return encode
        if module == "pytorch_lightning.callbacks" and name == 'model_checkpoint':
            import pytorch_lightning.callbacks
            return pytorch_lightning.callbacks.model_checkpoint
        if module == "pytorch_lightning.callbacks.model_checkpoint" and name == 'ModelCheckpoint':
            import pytorch_lightning.callbacks.model_checkpoint
            return pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint
        if module == "__builtin__" and name == 'set':
            return set

        # Forbid everything else.
        raise Exception(f"global '{module}/{name}' is forbidden")


# Regular expression that accepts 'dirname/version', 'dirname/data.pkl', and 'dirname/data/<number>'
allowed_zip_names_re = re.compile(r"^([^/]+)/((data/\d+)|version|(data\.pkl))$")
data_pkl_re = re.compile(r"^([^/]+)/data\.pkl$")

def check_zip_filenames(filename, names):
    for name in names:
        if allowed_zip_names_re.match(name):
            continue

        raise Exception(f"bad file inside {filename}: {name}")


def check_pt(filename, extra_handler):
    try:

        # new pytorch format is a zip file
        with zipfile.ZipFile(filename) as z:
            check_zip_filenames(filename, z.namelist())

            # find filename of data.pkl in zip file: '<directory name>/data.pkl'
            data_pkl_filenames = [f for f in z.namelist() if data_pkl_re.match(f)]
            if len(data_pkl_filenames) == 0:
                raise Exception(f"data.pkl not found in {filename}")
            if len(data_pkl_filenames) > 1:
                raise Exception(f"Multiple data.pkl found in {filename}")
            with z.open(data_pkl_filenames[0]) as file:
                unpickler = RestrictedUnpickler(file)
                unpickler.extra_handler = extra_handler
                unpickler.load()

    except zipfile.BadZipfile:

        # if it's not a zip file, it's an olf pytorch format, with five objects written to pickle
        with open(filename, "rb") as file:
            unpickler = RestrictedUnpickler(file)
            unpickler.extra_handler = extra_handler
            for i in range(5):
                unpickler.load()


def load(filename, *args, **kwargs):
    return load_with_extra(filename, extra_handler=global_extra_handler, *args, **kwargs)


def load_with_extra(filename, extra_handler=None, *args, **kwargs):
    """
    this function is intended to be used by extensions that want to load models with
    some extra classes in them that the usual unpickler would find suspicious.

    Use the extra_handler argument to specify a function that takes module and field name as text,
    and returns that field's value:

    ```python
    def extra(module, name):
        if module == 'collections' and name == 'OrderedDict':
            return collections.OrderedDict

        return None

    safe.load_with_extra('model.pt', extra_handler=extra)
    ```

    The alternative to this is just to use safe.unsafe_torch_load('model.pt'), which as the name implies is
    definitely unsafe.
    """

    from modules import shared

    try:
        if not shared.cmd_opts.disable_safe_unpickle:
            check_pt(filename, extra_handler)

    except pickle.UnpicklingError:
        print(f"Error verifying pickled file from {filename}:", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        print("-----> !!!! The file is most likely corrupted !!!! <-----", file=sys.stderr)
        print("You can skip this check with --disable-safe-unpickle commandline argument, but that is not going to help you.\n\n", file=sys.stderr)
        return None

    except Exception:
        print(f"Error verifying pickled file from {filename}:", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        print("\nThe file may be malicious, so the program is not going to read it.", file=sys.stderr)
        print("You can skip this check with --disable-safe-unpickle commandline argument.\n\n", file=sys.stderr)
        return None

    return unsafe_torch_load(filename, *args, **kwargs)


class Extra:
    """
    A class for temporarily setting the global handler for when you can't explicitly call load_with_extra
    (because it's not your code making the torch.load call). The intended use is like this:

```
import torch
from modules import safe

def handler(module, name):
    if module == 'torch' and name in ['float64', 'float16']:
        return getattr(torch, name)

    return None

with safe.Extra(handler):
    x = torch.load('model.pt')
```
    """

    def __init__(self, handler):
        self.handler = handler

    def __enter__(self):
        global global_extra_handler

        assert global_extra_handler is None, 'already inside an Extra() block'
        global_extra_handler = self.handler

    def __exit__(self, exc_type, exc_val, exc_tb):
        global global_extra_handler

        global_extra_handler = None


unsafe_torch_load = torch.load
torch.load = load
global_extra_handler = None

