# Copyright (c) Facebook, Inc. and its affiliates.

import copy
import io
import logging
import numpy as np
from typing import List
import onnx
import onnx.optimizer
import torch
from caffe2.proto import caffe2_pb2
from caffe2.python import core
from caffe2.python.onnx.backend import Caffe2Backend
from tabulate import tabulate
from termcolor import colored
from torch.onnx import OperatorExportTypes

from .shared import (
    ScopedWS,
    construct_init_net_from_params,
    fuse_alias_placeholder,
    fuse_copy_between_cpu_and_gpu,
    get_params_from_init_net,
    group_norm_replace_aten_with_caffe2,
    infer_device_type,
    remove_dead_end_ops,
    remove_reshape_for_fc,
    save_graph,
)

logger = logging.getLogger(__name__)


def export_onnx_model(model, inputs):
    """
    Trace and export a model to onnx format.

    Args:
        model (nn.Module):
        inputs (tuple[args]): the model will be called by `model(*inputs)`

    Returns:
        an onnx model
    """
    assert isinstance(model, torch.nn.Module)

    # make sure all modules are in eval mode, onnx may change the training state
    # of the module if the states are not consistent
    def _check_eval(module):
        assert not module.training

    model.apply(_check_eval)

    # Export the model to ONNX
    with torch.no_grad():
        with io.BytesIO() as f:
            torch.onnx.export(
                model,
                inputs,
                f,
                operator_export_type=OperatorExportTypes.ONNX_ATEN_FALLBACK,
                # verbose=True,  # NOTE: uncomment this for debugging
                # export_params=True,
            )
            onnx_model = onnx.load_from_string(f.getvalue())

    return onnx_model


def _op_stats(net_def):
    type_count = {}
    for t in [op.type for op in net_def.op]:
        type_count[t] = type_count.get(t, 0) + 1
    type_count_list = sorted(type_count.items(), key=lambda kv: kv[0])  # alphabet
    type_count_list = sorted(type_count_list, key=lambda kv: -kv[1])  # count
    return "\n".join("{:>4}x {}".format(count, name) for name, count in type_count_list)


def _assign_device_option(
    predict_net: caffe2_pb2.NetDef, init_net: caffe2_pb2.NetDef, tensor_inputs: List[torch.Tensor]
):
    """
    ONNX exported network doesn't have concept of device, assign necessary
    device option for each op in order to make it runable on GPU runtime.
    """

    def _get_device_type(torch_tensor):
        assert torch_tensor.device.type in ["cpu", "cuda"]
        assert torch_tensor.device.index == 0
        return torch_tensor.device.type

    def _assign_op_device_option(net_proto, net_ssa, blob_device_types):
        for op, ssa_i in zip(net_proto.op, net_ssa):
            if op.type in ["CopyCPUToGPU", "CopyGPUToCPU"]:
                op.device_option.CopyFrom(core.DeviceOption(caffe2_pb2.CUDA, 0))
            else:
                devices = [blob_device_types[b] for b in ssa_i[0] + ssa_i[1]]
                assert all(d == devices[0] for d in devices)
                if devices[0] == "cuda":
                    op.device_option.CopyFrom(core.DeviceOption(caffe2_pb2.CUDA, 0))

    # update ops in predict_net
    predict_net_input_device_types = {
        (name, 0): _get_device_type(tensor)
        for name, tensor in zip(predict_net.external_input, tensor_inputs)
    }
    predict_net_device_types = infer_device_type(
        predict_net, known_status=predict_net_input_device_types, device_name_style="pytorch"
    )
    predict_net_ssa, _ = core.get_ssa(predict_net)
    _assign_op_device_option(predict_net, predict_net_ssa, predict_net_device_types)

    # update ops in init_net
    init_net_ssa, versions = core.get_ssa(init_net)
    init_net_output_device_types = {
        (name, versions[name]): predict_net_device_types[(name, 0)]
        for name in init_net.external_output
    }
    init_net_device_types = infer_device_type(
        init_net, known_status=init_net_output_device_types, device_name_style="pytorch"
    )
    _assign_op_device_option(init_net, init_net_ssa, init_net_device_types)


def export_caffe2_detection_model(model: torch.nn.Module, tensor_inputs: List[torch.Tensor]):
    """
    Export a caffe2-compatible Detectron2 model to caffe2 format via ONNX.

    Arg:
        model: a caffe2-compatible version of detectron2 model, defined in caffe2_modeling.py
        tensor_inputs: a list of tensors that caffe2 model takes as input.
    """
    model = copy.deepcopy(model)
    assert isinstance(model, torch.nn.Module)
    assert hasattr(model, "encode_additional_info")

    # Export via ONNX
    logger.info(
        "Exporting a {} model via ONNX ...".format(type(model).__name__)
        + " Some warnings from ONNX are expected and are usually not to worry about."
    )
    onnx_model = export_onnx_model(model, (tensor_inputs,))
    # Convert ONNX model to Caffe2 protobuf
    init_net, predict_net = Caffe2Backend.onnx_graph_to_caffe2_net(onnx_model)
    ops_table = [[op.type, op.input, op.output] for op in predict_net.op]
    table = tabulate(ops_table, headers=["type", "input", "output"], tablefmt="pipe")
    logger.info(
        "ONNX export Done. Exported predict_net (before optimizations):\n" + colored(table, "cyan")
    )

    # Apply protobuf optimization
    fuse_alias_placeholder(predict_net, init_net)
    if any(t.device.type != "cpu" for t in tensor_inputs):
        fuse_copy_between_cpu_and_gpu(predict_net)
        remove_dead_end_ops(init_net)
        _assign_device_option(predict_net, init_net, tensor_inputs)
    params, device_options = get_params_from_init_net(init_net)
    predict_net, params = remove_reshape_for_fc(predict_net, params)
    init_net = construct_init_net_from_params(params, device_options)
    group_norm_replace_aten_with_caffe2(predict_net)

    # Record necessary information for running the pb model in Detectron2 system.
    model.encode_additional_info(predict_net, init_net)

    logger.info("Operators used in predict_net: \n{}".format(_op_stats(predict_net)))
    logger.info("Operators used in init_net: \n{}".format(_op_stats(init_net)))

    return predict_net, init_net


def run_and_save_graph(predict_net, init_net, tensor_inputs, graph_save_path):
    """
    Run the caffe2 model on given inputs, recording the shape and draw the graph.

    predict_net/init_net: caffe2 model.
    tensor_inputs: a list of tensors that caffe2 model takes as input.
    graph_save_path: path for saving graph of exported model.
    """

    logger.info("Saving graph of ONNX exported model to {} ...".format(graph_save_path))
    save_graph(predict_net, graph_save_path, op_only=False)

    # Run the exported Caffe2 net
    logger.info("Running ONNX exported model ...")
    with ScopedWS("__ws_tmp__", True) as ws:
        ws.RunNetOnce(init_net)
        initialized_blobs = set(ws.Blobs())
        uninitialized = [inp for inp in predict_net.external_input if inp not in initialized_blobs]
        for name, blob in zip(uninitialized, tensor_inputs):
            ws.FeedBlob(name, blob)

        try:
            ws.RunNetOnce(predict_net)
        except RuntimeError as e:
            logger.warning("Encountered RuntimeError: \n{}".format(str(e)))

        ws_blobs = {b: ws.FetchBlob(b) for b in ws.Blobs()}
        blob_sizes = {b: ws_blobs[b].shape for b in ws_blobs if isinstance(ws_blobs[b], np.ndarray)}

        logger.info("Saving graph with blob shapes to {} ...".format(graph_save_path))
        save_graph(predict_net, graph_save_path, op_only=False, blob_sizes=blob_sizes)

        return ws_blobs
