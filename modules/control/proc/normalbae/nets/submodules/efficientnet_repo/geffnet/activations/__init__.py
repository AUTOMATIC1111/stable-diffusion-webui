from geffnet import config
from geffnet.activations.activations_me import *
from geffnet.activations.activations_jit import *
from geffnet.activations.activations import *
import torch

_has_silu = 'silu' in dir(torch.nn.functional)

_ACT_FN_DEFAULT = dict(
    silu=F.silu if _has_silu else swish,
    swish=F.silu if _has_silu else swish,
    mish=mish,
    relu=F.relu,
    relu6=F.relu6,
    sigmoid=sigmoid,
    tanh=tanh,
    hard_sigmoid=hard_sigmoid,
    hard_swish=hard_swish,
)

_ACT_FN_JIT = dict(
    silu=F.silu if _has_silu else swish_jit,
    swish=F.silu if _has_silu else swish_jit,
    mish=mish_jit,
)

_ACT_FN_ME = dict(
    silu=F.silu if _has_silu else swish_me,
    swish=F.silu if _has_silu else swish_me,
    mish=mish_me,
    hard_swish=hard_swish_me,
    hard_sigmoid_jit=hard_sigmoid_me,
)

_ACT_LAYER_DEFAULT = dict(
    silu=nn.SiLU if _has_silu else Swish,
    swish=nn.SiLU if _has_silu else Swish,
    mish=Mish,
    relu=nn.ReLU,
    relu6=nn.ReLU6,
    sigmoid=Sigmoid,
    tanh=Tanh,
    hard_sigmoid=HardSigmoid,
    hard_swish=HardSwish,
)

_ACT_LAYER_JIT = dict(
    silu=nn.SiLU if _has_silu else SwishJit,
    swish=nn.SiLU if _has_silu else SwishJit,
    mish=MishJit,
)

_ACT_LAYER_ME = dict(
    silu=nn.SiLU if _has_silu else SwishMe,
    swish=nn.SiLU if _has_silu else SwishMe,
    mish=MishMe,
    hard_swish=HardSwishMe,
    hard_sigmoid=HardSigmoidMe
)

_OVERRIDE_FN = dict()
_OVERRIDE_LAYER = dict()


def add_override_act_fn(name, fn):
    global _OVERRIDE_FN
    _OVERRIDE_FN[name] = fn


def update_override_act_fn(overrides):
    assert isinstance(overrides, dict)
    global _OVERRIDE_FN
    _OVERRIDE_FN.update(overrides)


def clear_override_act_fn():
    global _OVERRIDE_FN
    _OVERRIDE_FN = dict()


def add_override_act_layer(name, fn):
    _OVERRIDE_LAYER[name] = fn


def update_override_act_layer(overrides):
    assert isinstance(overrides, dict)
    global _OVERRIDE_LAYER
    _OVERRIDE_LAYER.update(overrides)


def clear_override_act_layer():
    global _OVERRIDE_LAYER
    _OVERRIDE_LAYER = dict()


def get_act_fn(name='relu'):
    """ Activation Function Factory
    Fetching activation fns by name with this function allows export or torch script friendly
    functions to be returned dynamically based on current config.
    """
    if name in _OVERRIDE_FN:
        return _OVERRIDE_FN[name]
    use_me = not (config.is_exportable() or config.is_scriptable() or config.is_no_jit())
    if use_me and name in _ACT_FN_ME:
        # If not exporting or scripting the model, first look for a memory optimized version
        # activation with custom autograd, then fallback to jit scripted, then a Python or Torch builtin
        return _ACT_FN_ME[name]
    if config.is_exportable() and name in ('silu', 'swish'):
        # FIXME PyTorch SiLU doesn't ONNX export, this is a temp hack
        return swish
    use_jit = not (config.is_exportable() or config.is_no_jit())
    # NOTE: export tracing should work with jit scripted components, but I keep running into issues
    if use_jit and name in _ACT_FN_JIT:  # jit scripted models should be okay for export/scripting
        return _ACT_FN_JIT[name]
    return _ACT_FN_DEFAULT[name]


def get_act_layer(name='relu'):
    """ Activation Layer Factory
    Fetching activation layers by name with this function allows export or torch script friendly
    functions to be returned dynamically based on current config.
    """
    if name in _OVERRIDE_LAYER:
        return _OVERRIDE_LAYER[name]
    use_me = not (config.is_exportable() or config.is_scriptable() or config.is_no_jit())
    if use_me and name in _ACT_LAYER_ME:
        return _ACT_LAYER_ME[name]
    if config.is_exportable() and name in ('silu', 'swish'):
        # FIXME PyTorch SiLU doesn't ONNX export, this is a temp hack
        return Swish
    use_jit = not (config.is_exportable() or config.is_no_jit())
    # NOTE: export tracing should work with jit scripted components, but I keep running into issues
    if use_jit and name in _ACT_FN_JIT:  # jit scripted models should be okay for export/scripting
        return _ACT_LAYER_JIT[name]
    return _ACT_LAYER_DEFAULT[name]


