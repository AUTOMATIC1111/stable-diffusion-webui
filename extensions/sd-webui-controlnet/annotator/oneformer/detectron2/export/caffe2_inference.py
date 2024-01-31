# Copyright (c) Facebook, Inc. and its affiliates.

import logging
import numpy as np
from itertools import count
import torch
from caffe2.proto import caffe2_pb2
from caffe2.python import core

from .caffe2_modeling import META_ARCH_CAFFE2_EXPORT_TYPE_MAP, convert_batched_inputs_to_c2_format
from .shared import ScopedWS, get_pb_arg_vali, get_pb_arg_vals, infer_device_type

logger = logging.getLogger(__name__)


# ===== ref: mobile-vision predictor's 'Caffe2Wrapper' class ======
class ProtobufModel(torch.nn.Module):
    """
    Wrapper of a caffe2's protobuf model.
    It works just like nn.Module, but running caffe2 under the hood.
    Input/Output are tuple[tensor] that match the caffe2 net's external_input/output.
    """

    _ids = count(0)

    def __init__(self, predict_net, init_net):
        logger.info(f"Initializing ProtobufModel for: {predict_net.name} ...")
        super().__init__()
        assert isinstance(predict_net, caffe2_pb2.NetDef)
        assert isinstance(init_net, caffe2_pb2.NetDef)
        # create unique temporary workspace for each instance
        self.ws_name = "__tmp_ProtobufModel_{}__".format(next(self._ids))
        self.net = core.Net(predict_net)

        logger.info("Running init_net once to fill the parameters ...")
        with ScopedWS(self.ws_name, is_reset=True, is_cleanup=False) as ws:
            ws.RunNetOnce(init_net)
            uninitialized_external_input = []
            for blob in self.net.Proto().external_input:
                if blob not in ws.Blobs():
                    uninitialized_external_input.append(blob)
                    ws.CreateBlob(blob)
            ws.CreateNet(self.net)

        self._error_msgs = set()
        self._input_blobs = uninitialized_external_input

    def _infer_output_devices(self, inputs):
        """
        Returns:
            list[str]: list of device for each external output
        """

        def _get_device_type(torch_tensor):
            assert torch_tensor.device.type in ["cpu", "cuda"]
            assert torch_tensor.device.index == 0
            return torch_tensor.device.type

        predict_net = self.net.Proto()
        input_device_types = {
            (name, 0): _get_device_type(tensor) for name, tensor in zip(self._input_blobs, inputs)
        }
        device_type_map = infer_device_type(
            predict_net, known_status=input_device_types, device_name_style="pytorch"
        )
        ssa, versions = core.get_ssa(predict_net)
        versioned_outputs = [(name, versions[name]) for name in predict_net.external_output]
        output_devices = [device_type_map[outp] for outp in versioned_outputs]
        return output_devices

    def forward(self, inputs):
        """
        Args:
            inputs (tuple[torch.Tensor])

        Returns:
            tuple[torch.Tensor]
        """
        assert len(inputs) == len(self._input_blobs), (
            f"Length of inputs ({len(inputs)}) "
            f"doesn't match the required input blobs: {self._input_blobs}"
        )

        with ScopedWS(self.ws_name, is_reset=False, is_cleanup=False) as ws:
            for b, tensor in zip(self._input_blobs, inputs):
                ws.FeedBlob(b, tensor)

            try:
                ws.RunNet(self.net.Proto().name)
            except RuntimeError as e:
                if not str(e) in self._error_msgs:
                    self._error_msgs.add(str(e))
                    logger.warning("Encountered new RuntimeError: \n{}".format(str(e)))
                logger.warning("Catch the error and use partial results.")

            c2_outputs = [ws.FetchBlob(b) for b in self.net.Proto().external_output]
            # Remove outputs of current run, this is necessary in order to
            # prevent fetching the result from previous run if the model fails
            # in the middle.
            for b in self.net.Proto().external_output:
                # Needs to create uninitialized blob to make the net runable.
                # This is "equivalent" to: ws.RemoveBlob(b) then ws.CreateBlob(b),
                # but there'no such API.
                ws.FeedBlob(b, f"{b}, a C++ native class of type nullptr (uninitialized).")

        # Cast output to torch.Tensor on the desired device
        output_devices = (
            self._infer_output_devices(inputs)
            if any(t.device.type != "cpu" for t in inputs)
            else ["cpu" for _ in self.net.Proto().external_output]
        )

        outputs = []
        for name, c2_output, device in zip(
            self.net.Proto().external_output, c2_outputs, output_devices
        ):
            if not isinstance(c2_output, np.ndarray):
                raise RuntimeError(
                    "Invalid output for blob {}, received: {}".format(name, c2_output)
                )
            outputs.append(torch.tensor(c2_output).to(device=device))
        return tuple(outputs)


class ProtobufDetectionModel(torch.nn.Module):
    """
    A class works just like a pytorch meta arch in terms of inference, but running
    caffe2 model under the hood.
    """

    def __init__(self, predict_net, init_net, *, convert_outputs=None):
        """
        Args:
            predict_net, init_net (core.Net): caffe2 nets
            convert_outptus (callable): a function that converts caffe2
                outputs to the same format of the original pytorch model.
                By default, use the one defined in the caffe2 meta_arch.
        """
        super().__init__()
        self.protobuf_model = ProtobufModel(predict_net, init_net)
        self.size_divisibility = get_pb_arg_vali(predict_net, "size_divisibility", 0)
        self.device = get_pb_arg_vals(predict_net, "device", b"cpu").decode("ascii")

        if convert_outputs is None:
            meta_arch = get_pb_arg_vals(predict_net, "meta_architecture", b"GeneralizedRCNN")
            meta_arch = META_ARCH_CAFFE2_EXPORT_TYPE_MAP[meta_arch.decode("ascii")]
            self._convert_outputs = meta_arch.get_outputs_converter(predict_net, init_net)
        else:
            self._convert_outputs = convert_outputs

    def _convert_inputs(self, batched_inputs):
        # currently all models convert inputs in the same way
        return convert_batched_inputs_to_c2_format(
            batched_inputs, self.size_divisibility, self.device
        )

    def forward(self, batched_inputs):
        c2_inputs = self._convert_inputs(batched_inputs)
        c2_results = self.protobuf_model(c2_inputs)
        c2_results = dict(zip(self.protobuf_model.net.Proto().external_output, c2_results))
        return self._convert_outputs(batched_inputs, c2_inputs, c2_results)
