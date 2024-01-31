# Copyright (c) Facebook, Inc. and its affiliates.
import io
import numpy as np
import os
import re
import tempfile
import unittest
from typing import Callable
import torch
import torch.onnx.symbolic_helper as sym_help
from packaging import version
from torch._C import ListType
from torch.onnx import register_custom_op_symbolic

from annotator.oneformer.detectron2 import model_zoo
from annotator.oneformer.detectron2.config import CfgNode, LazyConfig, instantiate
from annotator.oneformer.detectron2.data import DatasetCatalog
from annotator.oneformer.detectron2.data.detection_utils import read_image
from annotator.oneformer.detectron2.modeling import build_model
from annotator.oneformer.detectron2.structures import Boxes, Instances, ROIMasks
from annotator.oneformer.detectron2.utils.file_io import PathManager


"""
Internal utilities for tests. Don't use except for writing tests.
"""


def get_model_no_weights(config_path):
    """
    Like model_zoo.get, but do not load any weights (even pretrained)
    """
    cfg = model_zoo.get_config(config_path)
    if isinstance(cfg, CfgNode):
        if not torch.cuda.is_available():
            cfg.MODEL.DEVICE = "cpu"
        return build_model(cfg)
    else:
        return instantiate(cfg.model)


def random_boxes(num_boxes, max_coord=100, device="cpu"):
    """
    Create a random Nx4 boxes tensor, with coordinates < max_coord.
    """
    boxes = torch.rand(num_boxes, 4, device=device) * (max_coord * 0.5)
    boxes.clamp_(min=1.0)  # tiny boxes cause numerical instability in box regression
    # Note: the implementation of this function in torchvision is:
    # boxes[:, 2:] += torch.rand(N, 2) * 100
    # but it does not guarantee non-negative widths/heights constraints:
    # boxes[:, 2] >= boxes[:, 0] and boxes[:, 3] >= boxes[:, 1]:
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def get_sample_coco_image(tensor=True):
    """
    Args:
        tensor (bool): if True, returns 3xHxW tensor.
            else, returns a HxWx3 numpy array.

    Returns:
        an image, in BGR color.
    """
    try:
        file_name = DatasetCatalog.get("coco_2017_val_100")[0]["file_name"]
        if not PathManager.exists(file_name):
            raise FileNotFoundError()
    except IOError:
        # for public CI to run
        file_name = PathManager.get_local_path(
            "http://images.cocodataset.org/train2017/000000000009.jpg"
        )
    ret = read_image(file_name, format="BGR")
    if tensor:
        ret = torch.from_numpy(np.ascontiguousarray(ret.transpose(2, 0, 1)))
    return ret


def convert_scripted_instances(instances):
    """
    Convert a scripted Instances object to a regular :class:`Instances` object
    """
    assert hasattr(
        instances, "image_size"
    ), f"Expect an Instances object, but got {type(instances)}!"
    ret = Instances(instances.image_size)
    for name in instances._field_names:
        val = getattr(instances, "_" + name, None)
        if val is not None:
            ret.set(name, val)
    return ret


def assert_instances_allclose(input, other, *, rtol=1e-5, msg="", size_as_tensor=False):
    """
    Args:
        input, other (Instances):
        size_as_tensor: compare image_size of the Instances as tensors (instead of tuples).
             Useful for comparing outputs of tracing.
    """
    if not isinstance(input, Instances):
        input = convert_scripted_instances(input)
    if not isinstance(other, Instances):
        other = convert_scripted_instances(other)

    if not msg:
        msg = "Two Instances are different! "
    else:
        msg = msg.rstrip() + " "

    size_error_msg = msg + f"image_size is {input.image_size} vs. {other.image_size}!"
    if size_as_tensor:
        assert torch.equal(
            torch.tensor(input.image_size), torch.tensor(other.image_size)
        ), size_error_msg
    else:
        assert input.image_size == other.image_size, size_error_msg
    fields = sorted(input.get_fields().keys())
    fields_other = sorted(other.get_fields().keys())
    assert fields == fields_other, msg + f"Fields are {fields} vs {fields_other}!"

    for f in fields:
        val1, val2 = input.get(f), other.get(f)
        if isinstance(val1, (Boxes, ROIMasks)):
            # boxes in the range of O(100) and can have a larger tolerance
            assert torch.allclose(val1.tensor, val2.tensor, atol=100 * rtol), (
                msg + f"Field {f} differs too much!"
            )
        elif isinstance(val1, torch.Tensor):
            if val1.dtype.is_floating_point:
                mag = torch.abs(val1).max().cpu().item()
                assert torch.allclose(val1, val2, atol=mag * rtol), (
                    msg + f"Field {f} differs too much!"
                )
            else:
                assert torch.equal(val1, val2), msg + f"Field {f} is different!"
        else:
            raise ValueError(f"Don't know how to compare type {type(val1)}")


def reload_script_model(module):
    """
    Save a jit module and load it back.
    Similar to the `getExportImportCopy` function in torch/testing/
    """
    buffer = io.BytesIO()
    torch.jit.save(module, buffer)
    buffer.seek(0)
    return torch.jit.load(buffer)


def reload_lazy_config(cfg):
    """
    Save an object by LazyConfig.save and load it back.
    This is used to test that a config still works the same after
    serialization/deserialization.
    """
    with tempfile.TemporaryDirectory(prefix="detectron2") as d:
        fname = os.path.join(d, "d2_cfg_test.yaml")
        LazyConfig.save(cfg, fname)
        return LazyConfig.load(fname)


def min_torch_version(min_version: str) -> bool:
    """
    Returns True when torch's  version is at least `min_version`.
    """
    try:
        import torch
    except ImportError:
        return False

    installed_version = version.parse(torch.__version__.split("+")[0])
    min_version = version.parse(min_version)
    return installed_version >= min_version


def has_dynamic_axes(onnx_model):
    """
    Return True when all ONNX input/output have only dynamic axes for all ranks
    """
    return all(
        not dim.dim_param.isnumeric()
        for inp in onnx_model.graph.input
        for dim in inp.type.tensor_type.shape.dim
    ) and all(
        not dim.dim_param.isnumeric()
        for out in onnx_model.graph.output
        for dim in out.type.tensor_type.shape.dim
    )


def register_custom_op_onnx_export(
    opname: str, symbolic_fn: Callable, opset_version: int, min_version: str
) -> None:
    """
    Register `symbolic_fn` as PyTorch's symbolic `opname`-`opset_version` for ONNX export.
    The registration is performed only when current PyTorch's version is < `min_version.`
    IMPORTANT: symbolic must be manually unregistered after the caller function returns
    """
    if min_torch_version(min_version):
        return
    register_custom_op_symbolic(opname, symbolic_fn, opset_version)
    print(f"_register_custom_op_onnx_export({opname}, {opset_version}) succeeded.")


def unregister_custom_op_onnx_export(opname: str, opset_version: int, min_version: str) -> None:
    """
    Unregister PyTorch's symbolic `opname`-`opset_version` for ONNX export.
    The un-registration is performed only when PyTorch's version is < `min_version`
    IMPORTANT: The symbolic must have been manually registered by the caller, otherwise
               the incorrect symbolic may be unregistered instead.
    """

    # TODO: _unregister_custom_op_symbolic is introduced PyTorch>=1.10
    #       Remove after PyTorch 1.10+ is used by ALL detectron2's CI
    try:
        from torch.onnx import unregister_custom_op_symbolic as _unregister_custom_op_symbolic
    except ImportError:

        def _unregister_custom_op_symbolic(symbolic_name, opset_version):
            import torch.onnx.symbolic_registry as sym_registry
            from torch.onnx.symbolic_helper import _onnx_main_opset, _onnx_stable_opsets

            def _get_ns_op_name_from_custom_op(symbolic_name):
                try:
                    from torch.onnx.utils import get_ns_op_name_from_custom_op

                    ns, op_name = get_ns_op_name_from_custom_op(symbolic_name)
                except ImportError as import_error:
                    if not bool(
                        re.match(r"^[a-zA-Z0-9-_]*::[a-zA-Z-_]+[a-zA-Z0-9-_]*$", symbolic_name)
                    ):
                        raise ValueError(
                            f"Invalid symbolic name {symbolic_name}. Must be `domain::name`"
                        ) from import_error

                    ns, op_name = symbolic_name.split("::")
                    if ns == "onnx":
                        raise ValueError(f"{ns} domain cannot be modified.") from import_error

                    if ns == "aten":
                        ns = ""

                return ns, op_name

            def _unregister_op(opname: str, domain: str, version: int):
                try:
                    sym_registry.unregister_op(op_name, ns, ver)
                except AttributeError as attribute_error:
                    if sym_registry.is_registered_op(opname, domain, version):
                        del sym_registry._registry[(domain, version)][opname]
                        if not sym_registry._registry[(domain, version)]:
                            del sym_registry._registry[(domain, version)]
                    else:
                        raise RuntimeError(
                            f"The opname {opname} is not registered."
                        ) from attribute_error

            ns, op_name = _get_ns_op_name_from_custom_op(symbolic_name)
            for ver in _onnx_stable_opsets + [_onnx_main_opset]:
                if ver >= opset_version:
                    _unregister_op(op_name, ns, ver)

    if min_torch_version(min_version):
        return
    _unregister_custom_op_symbolic(opname, opset_version)
    print(f"_unregister_custom_op_onnx_export({opname}, {opset_version}) succeeded.")


skipIfOnCPUCI = unittest.skipIf(
    os.environ.get("CI") and not torch.cuda.is_available(),
    "The test is too slow on CPUs and will be executed on CircleCI's GPU jobs.",
)


def skipIfUnsupportedMinOpsetVersion(min_opset_version, current_opset_version=None):
    """
    Skips tests for ONNX Opset versions older than min_opset_version.
    """

    def skip_dec(func):
        def wrapper(self):
            try:
                opset_version = self.opset_version
            except AttributeError:
                opset_version = current_opset_version
            if opset_version < min_opset_version:
                raise unittest.SkipTest(
                    f"Unsupported opset_version {opset_version}"
                    f", required is {min_opset_version}"
                )
            return func(self)

        return wrapper

    return skip_dec


def skipIfUnsupportedMinTorchVersion(min_version):
    """
    Skips tests for PyTorch versions older than min_version.
    """
    reason = f"module 'torch' has __version__ {torch.__version__}" f", required is: {min_version}"
    return unittest.skipIf(not min_torch_version(min_version), reason)


# TODO: Remove after PyTorch 1.11.1+ is used by detectron2's CI
def _pytorch1111_symbolic_opset9_to(g, self, *args):
    """aten::to() symbolic that must be used for testing with PyTorch < 1.11.1."""

    def is_aten_to_device_only(args):
        if len(args) == 4:
            # aten::to(Tensor, Device, bool, bool, memory_format)
            return (
                args[0].node().kind() == "prim::device"
                or args[0].type().isSubtypeOf(ListType.ofInts())
                or (
                    sym_help._is_value(args[0])
                    and args[0].node().kind() == "onnx::Constant"
                    and isinstance(args[0].node()["value"], str)
                )
            )
        elif len(args) == 5:
            # aten::to(Tensor, Device, ScalarType, bool, bool, memory_format)
            # When dtype is None, this is a aten::to(device) call
            dtype = sym_help._get_const(args[1], "i", "dtype")
            return dtype is None
        elif len(args) in (6, 7):
            # aten::to(Tensor, ScalarType, Layout, Device, bool, bool, memory_format)
            # aten::to(Tensor, ScalarType, Layout, Device, bool, bool, bool, memory_format)
            # When dtype is None, this is a aten::to(device) call
            dtype = sym_help._get_const(args[0], "i", "dtype")
            return dtype is None
        return False

    # ONNX doesn't have a concept of a device, so we ignore device-only casts
    if is_aten_to_device_only(args):
        return self

    if len(args) == 4:
        # TestONNXRuntime::test_ones_bool shows args[0] of aten::to can be onnx::Constant[Tensor]
        # In this case, the constant value is a tensor not int,
        # so sym_help._maybe_get_const(args[0], 'i') would not work.
        dtype = args[0]
        if sym_help._is_value(args[0]) and args[0].node().kind() == "onnx::Constant":
            tval = args[0].node()["value"]
            if isinstance(tval, torch.Tensor):
                if len(tval.shape) == 0:
                    tval = tval.item()
                    dtype = int(tval)
                else:
                    dtype = tval

        if sym_help._is_value(dtype) or isinstance(dtype, torch.Tensor):
            # aten::to(Tensor, Tensor, bool, bool, memory_format)
            dtype = args[0].type().scalarType()
            return g.op("Cast", self, to_i=sym_help.cast_pytorch_to_onnx[dtype])
        else:
            # aten::to(Tensor, ScalarType, bool, bool, memory_format)
            # memory_format is ignored
            return g.op("Cast", self, to_i=sym_help.scalar_type_to_onnx[dtype])
    elif len(args) == 5:
        # aten::to(Tensor, Device, ScalarType, bool, bool, memory_format)
        dtype = sym_help._get_const(args[1], "i", "dtype")
        # memory_format is ignored
        return g.op("Cast", self, to_i=sym_help.scalar_type_to_onnx[dtype])
    elif len(args) == 6:
        # aten::to(Tensor, ScalarType, Layout, Device, bool, bool, memory_format)
        dtype = sym_help._get_const(args[0], "i", "dtype")
        # Layout, device and memory_format are ignored
        return g.op("Cast", self, to_i=sym_help.scalar_type_to_onnx[dtype])
    elif len(args) == 7:
        # aten::to(Tensor, ScalarType, Layout, Device, bool, bool, bool, memory_format)
        dtype = sym_help._get_const(args[0], "i", "dtype")
        # Layout, device and memory_format are ignored
        return g.op("Cast", self, to_i=sym_help.scalar_type_to_onnx[dtype])
    else:
        return sym_help._onnx_unsupported("Unknown aten::to signature")


# TODO: Remove after PyTorch 1.11.1+ is used by detectron2's CI
def _pytorch1111_symbolic_opset9_repeat_interleave(g, self, repeats, dim=None, output_size=None):

    # from torch.onnx.symbolic_helper import ScalarType
    from torch.onnx.symbolic_opset9 import expand, unsqueeze

    input = self
    # if dim is None flatten
    # By default, use the flattened input array, and return a flat output array
    if sym_help._is_none(dim):
        input = sym_help._reshape_helper(g, self, g.op("Constant", value_t=torch.tensor([-1])))
        dim = 0
    else:
        dim = sym_help._maybe_get_scalar(dim)

    repeats_dim = sym_help._get_tensor_rank(repeats)
    repeats_sizes = sym_help._get_tensor_sizes(repeats)
    input_sizes = sym_help._get_tensor_sizes(input)
    if repeats_dim is None:
        raise RuntimeError(
            "Unsupported: ONNX export of repeat_interleave for unknown " "repeats rank."
        )
    if repeats_sizes is None:
        raise RuntimeError(
            "Unsupported: ONNX export of repeat_interleave for unknown " "repeats size."
        )
    if input_sizes is None:
        raise RuntimeError(
            "Unsupported: ONNX export of repeat_interleave for unknown " "input size."
        )

    input_sizes_temp = input_sizes.copy()
    for idx, input_size in enumerate(input_sizes):
        if input_size is None:
            input_sizes[idx], input_sizes_temp[idx] = 0, -1

    # Cases where repeats is an int or single value tensor
    if repeats_dim == 0 or (repeats_dim == 1 and repeats_sizes[0] == 1):
        if not sym_help._is_tensor(repeats):
            repeats = g.op("Constant", value_t=torch.LongTensor(repeats))
        if input_sizes[dim] == 0:
            return sym_help._onnx_opset_unsupported_detailed(
                "repeat_interleave",
                9,
                13,
                "Unsupported along dimension with unknown input size",
            )
        else:
            reps = input_sizes[dim]
            repeats = expand(g, repeats, g.op("Constant", value_t=torch.tensor([reps])), None)

    # Cases where repeats is a 1 dim Tensor
    elif repeats_dim == 1:
        if input_sizes[dim] == 0:
            return sym_help._onnx_opset_unsupported_detailed(
                "repeat_interleave",
                9,
                13,
                "Unsupported along dimension with unknown input size",
            )
        if repeats_sizes[0] is None:
            return sym_help._onnx_opset_unsupported_detailed(
                "repeat_interleave", 9, 13, "Unsupported for cases with dynamic repeats"
            )
        assert (
            repeats_sizes[0] == input_sizes[dim]
        ), "repeats must have the same size as input along dim"
        reps = repeats_sizes[0]
    else:
        raise RuntimeError("repeats must be 0-dim or 1-dim tensor")

    final_splits = list()
    r_splits = sym_help._repeat_interleave_split_helper(g, repeats, reps, 0)
    if isinstance(r_splits, torch._C.Value):
        r_splits = [r_splits]
    i_splits = sym_help._repeat_interleave_split_helper(g, input, reps, dim)
    if isinstance(i_splits, torch._C.Value):
        i_splits = [i_splits]
    input_sizes[dim], input_sizes_temp[dim] = -1, 1
    for idx, r_split in enumerate(r_splits):
        i_split = unsqueeze(g, i_splits[idx], dim + 1)
        r_concat = [
            g.op("Constant", value_t=torch.LongTensor(input_sizes_temp[: dim + 1])),
            r_split,
            g.op("Constant", value_t=torch.LongTensor(input_sizes_temp[dim + 1 :])),
        ]
        r_concat = g.op("Concat", *r_concat, axis_i=0)
        i_split = expand(g, i_split, r_concat, None)
        i_split = sym_help._reshape_helper(
            g,
            i_split,
            g.op("Constant", value_t=torch.LongTensor(input_sizes)),
            allowzero=0,
        )
        final_splits.append(i_split)
    return g.op("Concat", *final_splits, axis_i=dim)
