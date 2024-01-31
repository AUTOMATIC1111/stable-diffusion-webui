# Copyright (c) Facebook, Inc. and its affiliates.

import collections
import copy
import functools
import logging
import numpy as np
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from unittest import mock
import caffe2.python.utils as putils
import torch
import torch.nn.functional as F
from caffe2.proto import caffe2_pb2
from caffe2.python import core, net_drawer, workspace
from torch.nn.functional import interpolate as interp

logger = logging.getLogger(__name__)


# ==== torch/utils_toffee/cast.py =======================================


def to_device(t, device_str):
    """
    This function is a replacement of .to(another_device) such that it allows the
    casting to be traced properly by explicitly calling the underlying copy ops.
    It also avoids introducing unncessary op when casting to the same device.
    """
    src = t.device
    dst = torch.device(device_str)

    if src == dst:
        return t
    elif src.type == "cuda" and dst.type == "cpu":
        return torch.ops._caffe2.CopyGPUToCPU(t)
    elif src.type == "cpu" and dst.type == "cuda":
        return torch.ops._caffe2.CopyCPUToGPU(t)
    else:
        raise RuntimeError("Can't cast tensor from device {} to device {}".format(src, dst))


# ==== torch/utils_toffee/interpolate.py =======================================


# Note: borrowed from vision/detection/fair/detectron/detectron/modeling/detector.py
def BilinearInterpolation(tensor_in, up_scale):
    assert up_scale % 2 == 0, "Scale should be even"

    def upsample_filt(size):
        factor = (size + 1) // 2
        if size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5

        og = np.ogrid[:size, :size]
        return (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)

    kernel_size = int(up_scale) * 2
    bil_filt = upsample_filt(kernel_size)

    dim = int(tensor_in.shape[1])
    kernel = np.zeros((dim, dim, kernel_size, kernel_size), dtype=np.float32)
    kernel[range(dim), range(dim), :, :] = bil_filt

    tensor_out = F.conv_transpose2d(
        tensor_in,
        weight=to_device(torch.Tensor(kernel), tensor_in.device),
        bias=None,
        stride=int(up_scale),
        padding=int(up_scale / 2),
    )

    return tensor_out


# NOTE: ONNX is incompatible with traced torch.nn.functional.interpolate if
# using dynamic `scale_factor` rather than static `size`. (T43166860)
# NOTE: Caffe2 Int8 conversion might not be able to quantize `size` properly.
def onnx_compatibale_interpolate(
    input, size=None, scale_factor=None, mode="nearest", align_corners=None
):
    # NOTE: The input dimensions are interpreted in the form:
    # `mini-batch x channels x [optional depth] x [optional height] x width`.
    if size is None and scale_factor is not None:
        if input.dim() == 4:
            if isinstance(scale_factor, (int, float)):
                height_scale, width_scale = (scale_factor, scale_factor)
            else:
                assert isinstance(scale_factor, (tuple, list))
                assert len(scale_factor) == 2
                height_scale, width_scale = scale_factor

            assert not align_corners, "No matching C2 op for align_corners == True"
            if mode == "nearest":
                return torch.ops._caffe2.ResizeNearest(
                    input, order="NCHW", width_scale=width_scale, height_scale=height_scale
                )
            elif mode == "bilinear":
                logger.warning(
                    "Use F.conv_transpose2d for bilinear interpolate"
                    " because there's no such C2 op, this may cause significant"
                    " slowdown and the boundary pixels won't be as same as"
                    " using F.interpolate due to padding."
                )
                assert height_scale == width_scale
                return BilinearInterpolation(input, up_scale=height_scale)
        logger.warning("Output size is not static, it might cause ONNX conversion issue")

    return interp(input, size, scale_factor, mode, align_corners)


def mock_torch_nn_functional_interpolate():
    def decorator(func):
        @functools.wraps(func)
        def _mock_torch_nn_functional_interpolate(*args, **kwargs):
            if torch.onnx.is_in_onnx_export():
                with mock.patch(
                    "torch.nn.functional.interpolate", side_effect=onnx_compatibale_interpolate
                ):
                    return func(*args, **kwargs)
            else:
                return func(*args, **kwargs)

        return _mock_torch_nn_functional_interpolate

    return decorator


# ==== torch/utils_caffe2/ws_utils.py ==========================================


class ScopedWS(object):
    def __init__(self, ws_name, is_reset, is_cleanup=False):
        self.ws_name = ws_name
        self.is_reset = is_reset
        self.is_cleanup = is_cleanup
        self.org_ws = ""

    def __enter__(self):
        self.org_ws = workspace.CurrentWorkspace()
        if self.ws_name is not None:
            workspace.SwitchWorkspace(self.ws_name, True)
        if self.is_reset:
            workspace.ResetWorkspace()

        return workspace

    def __exit__(self, *args):
        if self.is_cleanup:
            workspace.ResetWorkspace()
        if self.ws_name is not None:
            workspace.SwitchWorkspace(self.org_ws)


def fetch_any_blob(name):
    bb = None
    try:
        bb = workspace.FetchBlob(name)
    except TypeError:
        bb = workspace.FetchInt8Blob(name)
    except Exception as e:
        logger.error("Get blob {} error: {}".format(name, e))

    return bb


# ==== torch/utils_caffe2/protobuf.py ==========================================


def get_pb_arg(pb, arg_name):
    for x in pb.arg:
        if x.name == arg_name:
            return x
    return None


def get_pb_arg_valf(pb, arg_name, default_val):
    arg = get_pb_arg(pb, arg_name)
    return arg.f if arg is not None else default_val


def get_pb_arg_floats(pb, arg_name, default_val):
    arg = get_pb_arg(pb, arg_name)
    return list(map(float, arg.floats)) if arg is not None else default_val


def get_pb_arg_ints(pb, arg_name, default_val):
    arg = get_pb_arg(pb, arg_name)
    return list(map(int, arg.ints)) if arg is not None else default_val


def get_pb_arg_vali(pb, arg_name, default_val):
    arg = get_pb_arg(pb, arg_name)
    return arg.i if arg is not None else default_val


def get_pb_arg_vals(pb, arg_name, default_val):
    arg = get_pb_arg(pb, arg_name)
    return arg.s if arg is not None else default_val


def get_pb_arg_valstrings(pb, arg_name, default_val):
    arg = get_pb_arg(pb, arg_name)
    return list(arg.strings) if arg is not None else default_val


def check_set_pb_arg(pb, arg_name, arg_attr, arg_value, allow_override=False):
    arg = get_pb_arg(pb, arg_name)
    if arg is None:
        arg = putils.MakeArgument(arg_name, arg_value)
        assert hasattr(arg, arg_attr)
        pb.arg.extend([arg])
    if allow_override and getattr(arg, arg_attr) != arg_value:
        logger.warning(
            "Override argument {}: {} -> {}".format(arg_name, getattr(arg, arg_attr), arg_value)
        )
        setattr(arg, arg_attr, arg_value)
    else:
        assert arg is not None
        assert getattr(arg, arg_attr) == arg_value, "Existing value {}, new value {}".format(
            getattr(arg, arg_attr), arg_value
        )


def _create_const_fill_op_from_numpy(name, tensor, device_option=None):
    assert type(tensor) == np.ndarray
    kTypeNameMapper = {
        np.dtype("float32"): "GivenTensorFill",
        np.dtype("int32"): "GivenTensorIntFill",
        np.dtype("int64"): "GivenTensorInt64Fill",
        np.dtype("uint8"): "GivenTensorStringFill",
    }

    args_dict = {}
    if tensor.dtype == np.dtype("uint8"):
        args_dict.update({"values": [str(tensor.data)], "shape": [1]})
    else:
        args_dict.update({"values": tensor, "shape": tensor.shape})

    if device_option is not None:
        args_dict["device_option"] = device_option

    return core.CreateOperator(kTypeNameMapper[tensor.dtype], [], [name], **args_dict)


def _create_const_fill_op_from_c2_int8_tensor(name, int8_tensor):
    assert type(int8_tensor) == workspace.Int8Tensor
    kTypeNameMapper = {
        np.dtype("int32"): "Int8GivenIntTensorFill",
        np.dtype("uint8"): "Int8GivenTensorFill",
    }

    tensor = int8_tensor.data
    assert tensor.dtype in [np.dtype("uint8"), np.dtype("int32")]
    values = tensor.tobytes() if tensor.dtype == np.dtype("uint8") else tensor

    return core.CreateOperator(
        kTypeNameMapper[tensor.dtype],
        [],
        [name],
        values=values,
        shape=tensor.shape,
        Y_scale=int8_tensor.scale,
        Y_zero_point=int8_tensor.zero_point,
    )


def create_const_fill_op(
    name: str,
    blob: Union[np.ndarray, workspace.Int8Tensor],
    device_option: Optional[caffe2_pb2.DeviceOption] = None,
) -> caffe2_pb2.OperatorDef:
    """
    Given a blob object, return the Caffe2 operator that creates this blob
    as constant. Currently support NumPy tensor and Caffe2 Int8Tensor.
    """

    tensor_type = type(blob)
    assert tensor_type in [
        np.ndarray,
        workspace.Int8Tensor,
    ], 'Error when creating const fill op for "{}", unsupported blob type: {}'.format(
        name, type(blob)
    )

    if tensor_type == np.ndarray:
        return _create_const_fill_op_from_numpy(name, blob, device_option)
    elif tensor_type == workspace.Int8Tensor:
        assert device_option is None
        return _create_const_fill_op_from_c2_int8_tensor(name, blob)


def construct_init_net_from_params(
    params: Dict[str, Any], device_options: Optional[Dict[str, caffe2_pb2.DeviceOption]] = None
) -> caffe2_pb2.NetDef:
    """
    Construct the init_net from params dictionary
    """
    init_net = caffe2_pb2.NetDef()
    device_options = device_options or {}
    for name, blob in params.items():
        if isinstance(blob, str):
            logger.warning(
                (
                    "Blob {} with type {} is not supported in generating init net,"
                    " skipped.".format(name, type(blob))
                )
            )
            continue
        init_net.op.extend(
            [create_const_fill_op(name, blob, device_option=device_options.get(name, None))]
        )
        init_net.external_output.append(name)
    return init_net


def get_producer_map(ssa):
    """
    Return dict from versioned blob to (i, j),
        where i is index of producer op, j is the index of output of that op.
    """
    producer_map = {}
    for i in range(len(ssa)):
        outputs = ssa[i][1]
        for j, outp in enumerate(outputs):
            producer_map[outp] = (i, j)
    return producer_map


def get_consumer_map(ssa):
    """
    Return dict from versioned blob to list of (i, j),
        where i is index of consumer op, j is the index of input of that op.
    """
    consumer_map = collections.defaultdict(list)
    for i in range(len(ssa)):
        inputs = ssa[i][0]
        for j, inp in enumerate(inputs):
            consumer_map[inp].append((i, j))
    return consumer_map


def get_params_from_init_net(
    init_net: caffe2_pb2.NetDef,
) -> [Dict[str, Any], Dict[str, caffe2_pb2.DeviceOption]]:
    """
    Take the output blobs from init_net by running it.
    Outputs:
        params: dict from blob name to numpy array
        device_options: dict from blob name to the device option of its creating op
    """
    # NOTE: this assumes that the params is determined by producer op with the
    # only exception be CopyGPUToCPU which is CUDA op but returns CPU tensor.
    def _get_device_option(producer_op):
        if producer_op.type == "CopyGPUToCPU":
            return caffe2_pb2.DeviceOption()
        else:
            return producer_op.device_option

    with ScopedWS("__get_params_from_init_net__", is_reset=True, is_cleanup=True) as ws:
        ws.RunNetOnce(init_net)
        params = {b: fetch_any_blob(b) for b in init_net.external_output}
    ssa, versions = core.get_ssa(init_net)
    producer_map = get_producer_map(ssa)
    device_options = {
        b: _get_device_option(init_net.op[producer_map[(b, versions[b])][0]])
        for b in init_net.external_output
    }
    return params, device_options


def _updater_raise(op, input_types, output_types):
    raise RuntimeError(
        "Failed to apply updater for op {} given input_types {} and"
        " output_types {}".format(op, input_types, output_types)
    )


def _generic_status_identifier(
    predict_net: caffe2_pb2.NetDef,
    status_updater: Callable,
    known_status: Dict[Tuple[str, int], Any],
) -> Dict[Tuple[str, int], Any]:
    """
    Statically infer the status of each blob, the status can be such as device type
        (CPU/GPU), layout (NCHW/NHWC), data type (float32/int8), etc. "Blob" here
        is versioned blob (Tuple[str, int]) in the format compatible with ssa.
    Inputs:
        predict_net: the caffe2 network
        status_updater: a callable, given an op and the status of its input/output,
            it returns the updated status of input/output. `None` is used for
            representing unknown status.
        known_status: a dict containing known status, used as initialization.
    Outputs:
        A dict mapping from versioned blob to its status
    """
    ssa, versions = core.get_ssa(predict_net)
    versioned_ext_input = [(b, 0) for b in predict_net.external_input]
    versioned_ext_output = [(b, versions[b]) for b in predict_net.external_output]
    all_versioned_blobs = set().union(*[set(x[0] + x[1]) for x in ssa])

    allowed_vbs = all_versioned_blobs.union(versioned_ext_input).union(versioned_ext_output)
    assert all(k in allowed_vbs for k in known_status)
    assert all(v is not None for v in known_status.values())
    _known_status = copy.deepcopy(known_status)

    def _check_and_update(key, value):
        assert value is not None
        if key in _known_status:
            if not _known_status[key] == value:
                raise RuntimeError(
                    "Confilict status for {}, existing status {}, new status {}".format(
                        key, _known_status[key], value
                    )
                )
        _known_status[key] = value

    def _update_i(op, ssa_i):
        versioned_inputs = ssa_i[0]
        versioned_outputs = ssa_i[1]

        inputs_status = [_known_status.get(b, None) for b in versioned_inputs]
        outputs_status = [_known_status.get(b, None) for b in versioned_outputs]

        new_inputs_status, new_outputs_status = status_updater(op, inputs_status, outputs_status)

        for versioned_blob, status in zip(
            versioned_inputs + versioned_outputs, new_inputs_status + new_outputs_status
        ):
            if status is not None:
                _check_and_update(versioned_blob, status)

    for op, ssa_i in zip(predict_net.op, ssa):
        _update_i(op, ssa_i)
    for op, ssa_i in zip(reversed(predict_net.op), reversed(ssa)):
        _update_i(op, ssa_i)

    # NOTE: This strictly checks all the blob from predict_net must be assgined
    # a known status. However sometimes it's impossible (eg. having deadend op),
    # we may relax this constraint if
    for k in all_versioned_blobs:
        if k not in _known_status:
            raise NotImplementedError(
                "Can not infer the status for {}. Currently only support the case where"
                " a single forward and backward pass can identify status for all blobs.".format(k)
            )

    return _known_status


def infer_device_type(
    predict_net: caffe2_pb2.NetDef,
    known_status: Dict[Tuple[str, int], Any],
    device_name_style: str = "caffe2",
) -> Dict[Tuple[str, int], str]:
    """Return the device type ("cpu" or "gpu"/"cuda") of each (versioned) blob"""

    assert device_name_style in ["caffe2", "pytorch"]
    _CPU_STR = "cpu"
    _GPU_STR = "gpu" if device_name_style == "caffe2" else "cuda"

    def _copy_cpu_to_gpu_updater(op, input_types, output_types):
        if input_types[0] == _GPU_STR or output_types[0] == _CPU_STR:
            _updater_raise(op, input_types, output_types)
        return ([_CPU_STR], [_GPU_STR])

    def _copy_gpu_to_cpu_updater(op, input_types, output_types):
        if input_types[0] == _CPU_STR or output_types[0] == _GPU_STR:
            _updater_raise(op, input_types, output_types)
        return ([_GPU_STR], [_CPU_STR])

    def _other_ops_updater(op, input_types, output_types):
        non_none_types = [x for x in input_types + output_types if x is not None]
        if len(non_none_types) > 0:
            the_type = non_none_types[0]
            if not all(x == the_type for x in non_none_types):
                _updater_raise(op, input_types, output_types)
        else:
            the_type = None
        return ([the_type for _ in op.input], [the_type for _ in op.output])

    def _device_updater(op, *args, **kwargs):
        return {
            "CopyCPUToGPU": _copy_cpu_to_gpu_updater,
            "CopyGPUToCPU": _copy_gpu_to_cpu_updater,
        }.get(op.type, _other_ops_updater)(op, *args, **kwargs)

    return _generic_status_identifier(predict_net, _device_updater, known_status)


# ==== torch/utils_caffe2/vis.py ===============================================


def _modify_blob_names(ops, blob_rename_f):
    ret = []

    def _replace_list(blob_list, replaced_list):
        del blob_list[:]
        blob_list.extend(replaced_list)

    for x in ops:
        cur = copy.deepcopy(x)
        _replace_list(cur.input, list(map(blob_rename_f, cur.input)))
        _replace_list(cur.output, list(map(blob_rename_f, cur.output)))
        ret.append(cur)

    return ret


def _rename_blob(name, blob_sizes, blob_ranges):
    def _list_to_str(bsize):
        ret = ", ".join([str(x) for x in bsize])
        ret = "[" + ret + "]"
        return ret

    ret = name
    if blob_sizes is not None and name in blob_sizes:
        ret += "\n" + _list_to_str(blob_sizes[name])
    if blob_ranges is not None and name in blob_ranges:
        ret += "\n" + _list_to_str(blob_ranges[name])

    return ret


# graph_name could not contain word 'graph'
def save_graph(net, file_name, graph_name="net", op_only=True, blob_sizes=None, blob_ranges=None):
    blob_rename_f = functools.partial(_rename_blob, blob_sizes=blob_sizes, blob_ranges=blob_ranges)
    return save_graph_base(net, file_name, graph_name, op_only, blob_rename_f)


def save_graph_base(net, file_name, graph_name="net", op_only=True, blob_rename_func=None):
    graph = None
    ops = net.op
    if blob_rename_func is not None:
        ops = _modify_blob_names(ops, blob_rename_func)
    if not op_only:
        graph = net_drawer.GetPydotGraph(ops, graph_name, rankdir="TB")
    else:
        graph = net_drawer.GetPydotGraphMinimal(
            ops, graph_name, rankdir="TB", minimal_dependency=True
        )

    try:
        par_dir = os.path.dirname(file_name)
        if not os.path.exists(par_dir):
            os.makedirs(par_dir)

        format = os.path.splitext(os.path.basename(file_name))[-1]
        if format == ".png":
            graph.write_png(file_name)
        elif format == ".pdf":
            graph.write_pdf(file_name)
        elif format == ".svg":
            graph.write_svg(file_name)
        else:
            print("Incorrect format {}".format(format))
    except Exception as e:
        print("Error when writing graph to image {}".format(e))

    return graph


# ==== torch/utils_toffee/aten_to_caffe2.py ====================================


def group_norm_replace_aten_with_caffe2(predict_net: caffe2_pb2.NetDef):
    """
    For ONNX exported model, GroupNorm will be represented as ATen op,
        this can be a drop in replacement from ATen to GroupNorm
    """
    count = 0
    for op in predict_net.op:
        if op.type == "ATen":
            op_name = get_pb_arg_vals(op, "operator", None)  # return byte in py3
            if op_name and op_name.decode() == "group_norm":
                op.arg.remove(get_pb_arg(op, "operator"))

                if get_pb_arg_vali(op, "cudnn_enabled", None):
                    op.arg.remove(get_pb_arg(op, "cudnn_enabled"))

                num_groups = get_pb_arg_vali(op, "num_groups", None)
                if num_groups is not None:
                    op.arg.remove(get_pb_arg(op, "num_groups"))
                    check_set_pb_arg(op, "group", "i", num_groups)

                op.type = "GroupNorm"
                count += 1
    if count > 1:
        logger.info("Replaced {} ATen operator to GroupNormOp".format(count))


# ==== torch/utils_toffee/alias.py =============================================


def alias(x, name, is_backward=False):
    if not torch.onnx.is_in_onnx_export():
        return x
    assert isinstance(x, torch.Tensor)
    return torch.ops._caffe2.AliasWithName(x, name, is_backward=is_backward)


def fuse_alias_placeholder(predict_net, init_net):
    """Remove AliasWithName placeholder and rename the input/output of it"""
    # First we finish all the re-naming
    for i, op in enumerate(predict_net.op):
        if op.type == "AliasWithName":
            assert len(op.input) == 1
            assert len(op.output) == 1
            name = get_pb_arg_vals(op, "name", None).decode()
            is_backward = bool(get_pb_arg_vali(op, "is_backward", 0))
            rename_op_input(predict_net, init_net, i, 0, name, from_producer=is_backward)
            rename_op_output(predict_net, i, 0, name)

    # Remove AliasWithName, should be very safe since it's a non-op
    new_ops = []
    for op in predict_net.op:
        if op.type != "AliasWithName":
            new_ops.append(op)
        else:
            # safety check
            assert op.input == op.output
            assert op.input[0] == op.arg[0].s.decode()
    del predict_net.op[:]
    predict_net.op.extend(new_ops)


# ==== torch/utils_caffe2/graph_transform.py ===================================


class IllegalGraphTransformError(ValueError):
    """When a graph transform function call can't be executed."""


def _rename_versioned_blob_in_proto(
    proto: caffe2_pb2.NetDef,
    old_name: str,
    new_name: str,
    version: int,
    ssa: List[Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]],
    start_versions: Dict[str, int],
    end_versions: Dict[str, int],
):
    """In given proto, rename all blobs with matched version"""
    # Operater list
    for op, i_th_ssa in zip(proto.op, ssa):
        versioned_inputs, versioned_outputs = i_th_ssa
        for i in range(len(op.input)):
            if versioned_inputs[i] == (old_name, version):
                op.input[i] = new_name
        for i in range(len(op.output)):
            if versioned_outputs[i] == (old_name, version):
                op.output[i] = new_name
    # external_input
    if start_versions.get(old_name, 0) == version:
        for i in range(len(proto.external_input)):
            if proto.external_input[i] == old_name:
                proto.external_input[i] = new_name
    # external_output
    if end_versions.get(old_name, 0) == version:
        for i in range(len(proto.external_output)):
            if proto.external_output[i] == old_name:
                proto.external_output[i] = new_name


def rename_op_input(
    predict_net: caffe2_pb2.NetDef,
    init_net: caffe2_pb2.NetDef,
    op_id: int,
    input_id: int,
    new_name: str,
    from_producer: bool = False,
):
    """
    Rename the op_id-th operator in predict_net, change it's input_id-th input's
        name to the new_name. It also does automatic re-route and change
        external_input and init_net if necessary.
    - It requires the input is only consumed by this op.
    - This function modifies predict_net and init_net in-place.
    - When from_producer is enable, this also updates other operators that consumes
        the same input. Be cautious because may trigger unintended behavior.
    """
    assert isinstance(predict_net, caffe2_pb2.NetDef)
    assert isinstance(init_net, caffe2_pb2.NetDef)

    init_net_ssa, init_net_versions = core.get_ssa(init_net)
    predict_net_ssa, predict_net_versions = core.get_ssa(
        predict_net, copy.deepcopy(init_net_versions)
    )

    versioned_inputs, versioned_outputs = predict_net_ssa[op_id]
    old_name, version = versioned_inputs[input_id]

    if from_producer:
        producer_map = get_producer_map(predict_net_ssa)
        if not (old_name, version) in producer_map:
            raise NotImplementedError(
                "Can't find producer, the input {} is probably from"
                " init_net, this is not supported yet.".format(old_name)
            )
        producer = producer_map[(old_name, version)]
        rename_op_output(predict_net, producer[0], producer[1], new_name)
        return

    def contain_targets(op_ssa):
        return (old_name, version) in op_ssa[0]

    is_consumer = [contain_targets(op_ssa) for op_ssa in predict_net_ssa]
    if sum(is_consumer) > 1:
        raise IllegalGraphTransformError(
            (
                "Input '{}' of operator(#{}) are consumed by other ops, please use"
                + " rename_op_output on the producer instead. Offending op: \n{}"
            ).format(old_name, op_id, predict_net.op[op_id])
        )

    # update init_net
    _rename_versioned_blob_in_proto(
        init_net, old_name, new_name, version, init_net_ssa, {}, init_net_versions
    )
    # update predict_net
    _rename_versioned_blob_in_proto(
        predict_net,
        old_name,
        new_name,
        version,
        predict_net_ssa,
        init_net_versions,
        predict_net_versions,
    )


def rename_op_output(predict_net: caffe2_pb2.NetDef, op_id: int, output_id: int, new_name: str):
    """
    Rename the op_id-th operator in predict_net, change it's output_id-th input's
        name to the new_name. It also does automatic re-route and change
        external_output and if necessary.
    - It allows multiple consumers of its output.
    - This function modifies predict_net in-place, doesn't need init_net.
    """
    assert isinstance(predict_net, caffe2_pb2.NetDef)

    ssa, blob_versions = core.get_ssa(predict_net)

    versioned_inputs, versioned_outputs = ssa[op_id]
    old_name, version = versioned_outputs[output_id]

    # update predict_net
    _rename_versioned_blob_in_proto(
        predict_net, old_name, new_name, version, ssa, {}, blob_versions
    )


def get_sub_graph_external_input_output(
    predict_net: caffe2_pb2.NetDef, sub_graph_op_indices: List[int]
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    """
    Return the list of external input/output of sub-graph,
    each element is tuple of the name and corresponding version in predict_net.

    external input/output is defined the same way as caffe2 NetDef.
    """
    ssa, versions = core.get_ssa(predict_net)

    all_inputs = []
    all_outputs = []
    for op_id in sub_graph_op_indices:
        all_inputs += [inp for inp in ssa[op_id][0] if inp not in all_inputs]
        all_outputs += list(ssa[op_id][1])  # ssa output won't repeat

    # for versioned blobs, external inputs are just those blob in all_inputs
    # but not in all_outputs
    ext_inputs = [inp for inp in all_inputs if inp not in all_outputs]

    # external outputs are essentially outputs of this subgraph that are used
    # outside of this sub-graph (including predict_net.external_output)
    all_other_inputs = sum(
        (ssa[i][0] for i in range(len(ssa)) if i not in sub_graph_op_indices),
        [(outp, versions[outp]) for outp in predict_net.external_output],
    )
    ext_outputs = [outp for outp in all_outputs if outp in set(all_other_inputs)]

    return ext_inputs, ext_outputs


class DiGraph:
    """A DAG representation of caffe2 graph, each vertice is a versioned blob."""

    def __init__(self):
        self.vertices = set()
        self.graph = collections.defaultdict(list)

    def add_edge(self, u, v):
        self.graph[u].append(v)
        self.vertices.add(u)
        self.vertices.add(v)

    # grab from https://www.geeksforgeeks.org/find-paths-given-source-destination/
    def get_all_paths(self, s, d):
        visited = {k: False for k in self.vertices}
        path = []
        all_paths = []

        def _get_all_paths_util(graph, u, d, visited, path):
            visited[u] = True
            path.append(u)
            if u == d:
                all_paths.append(copy.deepcopy(path))
            else:
                for i in graph[u]:
                    if not visited[i]:
                        _get_all_paths_util(graph, i, d, visited, path)
            path.pop()
            visited[u] = False

        _get_all_paths_util(self.graph, s, d, visited, path)
        return all_paths

    @staticmethod
    def from_ssa(ssa):
        graph = DiGraph()
        for op_id in range(len(ssa)):
            for inp in ssa[op_id][0]:
                for outp in ssa[op_id][1]:
                    graph.add_edge(inp, outp)
        return graph


def _get_dependency_chain(ssa, versioned_target, versioned_source):
    """
    Return the index list of relevant operator to produce target blob from source blob,
        if there's no dependency, return empty list.
    """

    # finding all paths between nodes can be O(N!), thus we can only search
    # in the subgraph using the op starting from the first consumer of source blob
    # to the producer of the target blob.
    consumer_map = get_consumer_map(ssa)
    producer_map = get_producer_map(ssa)
    start_op = min(x[0] for x in consumer_map[versioned_source]) - 15
    end_op = (
        producer_map[versioned_target][0] + 15 if versioned_target in producer_map else start_op
    )
    sub_graph_ssa = ssa[start_op : end_op + 1]
    if len(sub_graph_ssa) > 30:
        logger.warning(
            "Subgraph bebetween {} and {} is large (from op#{} to op#{}), it"
            " might take non-trival time to find all paths between them.".format(
                versioned_source, versioned_target, start_op, end_op
            )
        )

    dag = DiGraph.from_ssa(sub_graph_ssa)
    paths = dag.get_all_paths(versioned_source, versioned_target)  # include two ends
    ops_in_paths = [[producer_map[blob][0] for blob in path[1:]] for path in paths]
    return sorted(set().union(*[set(ops) for ops in ops_in_paths]))


def identify_reshape_sub_graph(predict_net: caffe2_pb2.NetDef) -> List[List[int]]:
    """
    Idenfity the reshape sub-graph in a protobuf.
    The reshape sub-graph is defined as matching the following pattern:

    (input_blob) -> Op_1 -> ... -> Op_N -> (new_shape) -─┐
        └-------------------------------------------> Reshape -> (output_blob)

    Return:
        List of sub-graphs, each sub-graph is represented as a list of indices
        of the relavent ops, [Op_1, Op_2, ..., Op_N, Reshape]
    """

    ssa, _ = core.get_ssa(predict_net)

    ret = []
    for i, op in enumerate(predict_net.op):
        if op.type == "Reshape":
            assert len(op.input) == 2
            input_ssa = ssa[i][0]
            data_source = input_ssa[0]
            shape_source = input_ssa[1]
            op_indices = _get_dependency_chain(ssa, shape_source, data_source)
            ret.append(op_indices + [i])
    return ret


def remove_reshape_for_fc(predict_net, params):
    """
    In PyTorch nn.Linear has to take 2D tensor, this often leads to reshape
        a 4D tensor to 2D by calling .view(). However this (dynamic) reshaping
        doesn't work well with ONNX and Int8 tools, and cause using extra
        ops (eg. ExpandDims) that might not be available on mobile.
    Luckily Caffe2 supports 4D tensor for FC, so we can remove those reshape
        after exporting ONNX model.
    """
    from caffe2.python import core

    # find all reshape sub-graph that can be removed, which is now all Reshape
    # sub-graph whose output is only consumed by FC.
    # TODO: to make it safer, we may need the actually value to better determine
    # if a Reshape before FC is removable.
    reshape_sub_graphs = identify_reshape_sub_graph(predict_net)
    sub_graphs_to_remove = []
    for reshape_sub_graph in reshape_sub_graphs:
        reshape_op_id = reshape_sub_graph[-1]
        assert predict_net.op[reshape_op_id].type == "Reshape"
        ssa, _ = core.get_ssa(predict_net)
        reshape_output = ssa[reshape_op_id][1][0]
        consumers = [i for i in range(len(ssa)) if reshape_output in ssa[i][0]]
        if all(predict_net.op[consumer].type == "FC" for consumer in consumers):
            # safety check if the sub-graph is isolated, for this reshape sub-graph,
            # it means it has one non-param external input and one external output.
            ext_inputs, ext_outputs = get_sub_graph_external_input_output(
                predict_net, reshape_sub_graph
            )
            non_params_ext_inputs = [inp for inp in ext_inputs if inp[1] != 0]
            if len(non_params_ext_inputs) == 1 and len(ext_outputs) == 1:
                sub_graphs_to_remove.append(reshape_sub_graph)

    # perform removing subgraph by:
    # 1: rename the Reshape's output to its input, then the graph can be
    #   seen as in-place itentify, meaning whose external input/output are the same.
    # 2: simply remove those ops.
    remove_op_ids = []
    params_to_remove = []
    for sub_graph in sub_graphs_to_remove:
        logger.info(
            "Remove Reshape sub-graph:\n{}".format(
                "".join(["(#{:>4})\n{}".format(i, predict_net.op[i]) for i in sub_graph])
            )
        )
        reshape_op_id = sub_graph[-1]
        new_reshap_output = predict_net.op[reshape_op_id].input[0]
        rename_op_output(predict_net, reshape_op_id, 0, new_reshap_output)
        ext_inputs, ext_outputs = get_sub_graph_external_input_output(predict_net, sub_graph)
        non_params_ext_inputs = [inp for inp in ext_inputs if inp[1] != 0]
        params_ext_inputs = [inp for inp in ext_inputs if inp[1] == 0]
        assert len(non_params_ext_inputs) == 1 and len(ext_outputs) == 1
        assert ext_outputs[0][0] == non_params_ext_inputs[0][0]
        assert ext_outputs[0][1] == non_params_ext_inputs[0][1] + 1
        remove_op_ids.extend(sub_graph)
        params_to_remove.extend(params_ext_inputs)

    predict_net = copy.deepcopy(predict_net)
    new_ops = [op for i, op in enumerate(predict_net.op) if i not in remove_op_ids]
    del predict_net.op[:]
    predict_net.op.extend(new_ops)
    for versioned_params in params_to_remove:
        name = versioned_params[0]
        logger.info("Remove params: {} from init_net and predict_net.external_input".format(name))
        del params[name]
        predict_net.external_input.remove(name)

    return predict_net, params


def fuse_copy_between_cpu_and_gpu(predict_net: caffe2_pb2.NetDef):
    """
    In-place fuse extra copy ops between cpu/gpu for the following case:
        a -CopyAToB-> b -CopyBToA> c1 -NextOp1-> d1
                        -CopyBToA> c2 -NextOp2-> d2
    The fused network will look like:
        a -NextOp1-> d1
          -NextOp2-> d2
    """

    _COPY_OPS = ["CopyCPUToGPU", "CopyGPUToCPU"]

    def _fuse_once(predict_net):
        ssa, blob_versions = core.get_ssa(predict_net)
        consumer_map = get_consumer_map(ssa)
        versioned_external_output = [
            (name, blob_versions[name]) for name in predict_net.external_output
        ]

        for op_id, op in enumerate(predict_net.op):
            if op.type in _COPY_OPS:
                fw_copy_versioned_output = ssa[op_id][1][0]
                consumer_ids = [x[0] for x in consumer_map[fw_copy_versioned_output]]
                reverse_op_type = _COPY_OPS[1 - _COPY_OPS.index(op.type)]

                is_fusable = (
                    len(consumer_ids) > 0
                    and fw_copy_versioned_output not in versioned_external_output
                    and all(
                        predict_net.op[_op_id].type == reverse_op_type
                        and ssa[_op_id][1][0] not in versioned_external_output
                        for _op_id in consumer_ids
                    )
                )

                if is_fusable:
                    for rv_copy_op_id in consumer_ids:
                        # making each NextOp uses "a" directly and removing Copy ops
                        rs_copy_versioned_output = ssa[rv_copy_op_id][1][0]
                        next_op_id, inp_id = consumer_map[rs_copy_versioned_output][0]
                        predict_net.op[next_op_id].input[inp_id] = op.input[0]
                    # remove CopyOps
                    new_ops = [
                        op
                        for i, op in enumerate(predict_net.op)
                        if i != op_id and i not in consumer_ids
                    ]
                    del predict_net.op[:]
                    predict_net.op.extend(new_ops)
                    return True

        return False

    # _fuse_once returns False is nothing can be fused
    while _fuse_once(predict_net):
        pass


def remove_dead_end_ops(net_def: caffe2_pb2.NetDef):
    """remove ops if its output is not used or not in external_output"""
    ssa, versions = core.get_ssa(net_def)
    versioned_external_output = [(name, versions[name]) for name in net_def.external_output]
    consumer_map = get_consumer_map(ssa)
    removed_op_ids = set()

    def _is_dead_end(versioned_blob):
        return not (
            versioned_blob in versioned_external_output
            or (
                len(consumer_map[versioned_blob]) > 0
                and all(x[0] not in removed_op_ids for x in consumer_map[versioned_blob])
            )
        )

    for i, ssa_i in reversed(list(enumerate(ssa))):
        versioned_outputs = ssa_i[1]
        if all(_is_dead_end(outp) for outp in versioned_outputs):
            removed_op_ids.add(i)

    # simply removing those deadend ops should have no effect to external_output
    new_ops = [op for i, op in enumerate(net_def.op) if i not in removed_op_ids]
    del net_def.op[:]
    net_def.op.extend(new_ops)
