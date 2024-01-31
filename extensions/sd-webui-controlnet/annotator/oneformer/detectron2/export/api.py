# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import logging
import os
import torch
from caffe2.proto import caffe2_pb2
from torch import nn

from annotator.oneformer.detectron2.config import CfgNode
from annotator.oneformer.detectron2.utils.file_io import PathManager

from .caffe2_inference import ProtobufDetectionModel
from .caffe2_modeling import META_ARCH_CAFFE2_EXPORT_TYPE_MAP, convert_batched_inputs_to_c2_format
from .shared import get_pb_arg_vali, get_pb_arg_vals, save_graph

__all__ = [
    "Caffe2Model",
    "Caffe2Tracer",
]


class Caffe2Tracer:
    """
    Make a detectron2 model traceable with Caffe2 operators.
    This class creates a traceable version of a detectron2 model which:

    1. Rewrite parts of the model using ops in Caffe2. Note that some ops do
       not have GPU implementation in Caffe2.
    2. Remove post-processing and only produce raw layer outputs

    After making a traceable model, the class provide methods to export such a
    model to different deployment formats.
    Exported graph produced by this class take two input tensors:

    1. (1, C, H, W) float "data" which is an image (usually in [0, 255]).
       (H, W) often has to be padded to multiple of 32 (depend on the model
       architecture).
    2. 1x3 float "im_info", each row of which is (height, width, 1.0).
       Height and width are true image shapes before padding.

    The class currently only supports models using builtin meta architectures.
    Batch inference is not supported, and contributions are welcome.
    """

    def __init__(self, cfg: CfgNode, model: nn.Module, inputs):
        """
        Args:
            cfg (CfgNode): a detectron2 config used to construct caffe2-compatible model.
            model (nn.Module): An original pytorch model. Must be among a few official models
                in detectron2 that can be converted to become caffe2-compatible automatically.
                Weights have to be already loaded to this model.
            inputs: sample inputs that the given model takes for inference.
                Will be used to trace the model. For most models, random inputs with
                no detected objects will not work as they lead to wrong traces.
        """
        assert isinstance(cfg, CfgNode), cfg
        assert isinstance(model, torch.nn.Module), type(model)

        # TODO make it support custom models, by passing in c2 model directly
        C2MetaArch = META_ARCH_CAFFE2_EXPORT_TYPE_MAP[cfg.MODEL.META_ARCHITECTURE]
        self.traceable_model = C2MetaArch(cfg, copy.deepcopy(model))
        self.inputs = inputs
        self.traceable_inputs = self.traceable_model.get_caffe2_inputs(inputs)

    def export_caffe2(self):
        """
        Export the model to Caffe2's protobuf format.
        The returned object can be saved with its :meth:`.save_protobuf()` method.
        The result can be loaded and executed using Caffe2 runtime.

        Returns:
            :class:`Caffe2Model`
        """
        from .caffe2_export import export_caffe2_detection_model

        predict_net, init_net = export_caffe2_detection_model(
            self.traceable_model, self.traceable_inputs
        )
        return Caffe2Model(predict_net, init_net)

    def export_onnx(self):
        """
        Export the model to ONNX format.
        Note that the exported model contains custom ops only available in caffe2, therefore it
        cannot be directly executed by other runtime (such as onnxruntime or TensorRT).
        Post-processing or transformation passes may be applied on the model to accommodate
        different runtimes, but we currently do not provide support for them.

        Returns:
            onnx.ModelProto: an onnx model.
        """
        from .caffe2_export import export_onnx_model as export_onnx_model_impl

        return export_onnx_model_impl(self.traceable_model, (self.traceable_inputs,))

    def export_torchscript(self):
        """
        Export the model to a ``torch.jit.TracedModule`` by tracing.
        The returned object can be saved to a file by ``.save()``.

        Returns:
            torch.jit.TracedModule: a torch TracedModule
        """
        logger = logging.getLogger(__name__)
        logger.info("Tracing the model with torch.jit.trace ...")
        with torch.no_grad():
            return torch.jit.trace(self.traceable_model, (self.traceable_inputs,))


class Caffe2Model(nn.Module):
    """
    A wrapper around the traced model in Caffe2's protobuf format.
    The exported graph has different inputs/outputs from the original Pytorch
    model, as explained in :class:`Caffe2Tracer`. This class wraps around the
    exported graph to simulate the same interface as the original Pytorch model.
    It also provides functions to save/load models in Caffe2's format.'

    Examples:
    ::
        c2_model = Caffe2Tracer(cfg, torch_model, inputs).export_caffe2()
        inputs = [{"image": img_tensor_CHW}]
        outputs = c2_model(inputs)
        orig_outputs = torch_model(inputs)
    """

    def __init__(self, predict_net, init_net):
        super().__init__()
        self.eval()  # always in eval mode
        self._predict_net = predict_net
        self._init_net = init_net
        self._predictor = None

    __init__.__HIDE_SPHINX_DOC__ = True

    @property
    def predict_net(self):
        """
        caffe2.core.Net: the underlying caffe2 predict net
        """
        return self._predict_net

    @property
    def init_net(self):
        """
        caffe2.core.Net: the underlying caffe2 init net
        """
        return self._init_net

    def save_protobuf(self, output_dir):
        """
        Save the model as caffe2's protobuf format.
        It saves the following files:

            * "model.pb": definition of the graph. Can be visualized with
              tools like `netron <https://github.com/lutzroeder/netron>`_.
            * "model_init.pb": model parameters
            * "model.pbtxt": human-readable definition of the graph. Not
              needed for deployment.

        Args:
            output_dir (str): the output directory to save protobuf files.
        """
        logger = logging.getLogger(__name__)
        logger.info("Saving model to {} ...".format(output_dir))
        if not PathManager.exists(output_dir):
            PathManager.mkdirs(output_dir)

        with PathManager.open(os.path.join(output_dir, "model.pb"), "wb") as f:
            f.write(self._predict_net.SerializeToString())
        with PathManager.open(os.path.join(output_dir, "model.pbtxt"), "w") as f:
            f.write(str(self._predict_net))
        with PathManager.open(os.path.join(output_dir, "model_init.pb"), "wb") as f:
            f.write(self._init_net.SerializeToString())

    def save_graph(self, output_file, inputs=None):
        """
        Save the graph as SVG format.

        Args:
            output_file (str): a SVG file
            inputs: optional inputs given to the model.
                If given, the inputs will be used to run the graph to record
                shape of every tensor. The shape information will be
                saved together with the graph.
        """
        from .caffe2_export import run_and_save_graph

        if inputs is None:
            save_graph(self._predict_net, output_file, op_only=False)
        else:
            size_divisibility = get_pb_arg_vali(self._predict_net, "size_divisibility", 0)
            device = get_pb_arg_vals(self._predict_net, "device", b"cpu").decode("ascii")
            inputs = convert_batched_inputs_to_c2_format(inputs, size_divisibility, device)
            inputs = [x.cpu().numpy() for x in inputs]
            run_and_save_graph(self._predict_net, self._init_net, inputs, output_file)

    @staticmethod
    def load_protobuf(dir):
        """
        Args:
            dir (str): a directory used to save Caffe2Model with
                :meth:`save_protobuf`.
                The files "model.pb" and "model_init.pb" are needed.

        Returns:
            Caffe2Model: the caffe2 model loaded from this directory.
        """
        predict_net = caffe2_pb2.NetDef()
        with PathManager.open(os.path.join(dir, "model.pb"), "rb") as f:
            predict_net.ParseFromString(f.read())

        init_net = caffe2_pb2.NetDef()
        with PathManager.open(os.path.join(dir, "model_init.pb"), "rb") as f:
            init_net.ParseFromString(f.read())

        return Caffe2Model(predict_net, init_net)

    def __call__(self, inputs):
        """
        An interface that wraps around a Caffe2 model and mimics detectron2's models'
        input/output format. See details about the format at :doc:`/tutorials/models`.
        This is used to compare the outputs of caffe2 model with its original torch model.

        Due to the extra conversion between Pytorch/Caffe2, this method is not meant for
        benchmark. Because of the conversion, this method also has dependency
        on detectron2 in order to convert to detectron2's output format.
        """
        if self._predictor is None:
            self._predictor = ProtobufDetectionModel(self._predict_net, self._init_net)
        return self._predictor(inputs)
