# Copyright (c) Facebook, Inc. and its affiliates.

import functools
import io
import struct
import types
import torch

from annotator.oneformer.detectron2.modeling import meta_arch
from annotator.oneformer.detectron2.modeling.box_regression import Box2BoxTransform
from annotator.oneformer.detectron2.modeling.roi_heads import keypoint_head
from annotator.oneformer.detectron2.structures import Boxes, ImageList, Instances, RotatedBoxes

from .c10 import Caffe2Compatible
from .caffe2_patch import ROIHeadsPatcher, patch_generalized_rcnn
from .shared import (
    alias,
    check_set_pb_arg,
    get_pb_arg_floats,
    get_pb_arg_valf,
    get_pb_arg_vali,
    get_pb_arg_vals,
    mock_torch_nn_functional_interpolate,
)


def assemble_rcnn_outputs_by_name(image_sizes, tensor_outputs, force_mask_on=False):
    """
    A function to assemble caffe2 model's outputs (i.e. Dict[str, Tensor])
    to detectron2's format (i.e. list of Instances instance).
    This only works when the model follows the Caffe2 detectron's naming convention.

    Args:
        image_sizes (List[List[int, int]]): [H, W] of every image.
        tensor_outputs (Dict[str, Tensor]): external_output to its tensor.

        force_mask_on (Bool): if true, the it make sure there'll be pred_masks even
            if the mask is not found from tensor_outputs (usually due to model crash)
    """

    results = [Instances(image_size) for image_size in image_sizes]

    batch_splits = tensor_outputs.get("batch_splits", None)
    if batch_splits:
        raise NotImplementedError()
    assert len(image_sizes) == 1
    result = results[0]

    bbox_nms = tensor_outputs["bbox_nms"]
    score_nms = tensor_outputs["score_nms"]
    class_nms = tensor_outputs["class_nms"]
    # Detection will always success because Conv support 0-batch
    assert bbox_nms is not None
    assert score_nms is not None
    assert class_nms is not None
    if bbox_nms.shape[1] == 5:
        result.pred_boxes = RotatedBoxes(bbox_nms)
    else:
        result.pred_boxes = Boxes(bbox_nms)
    result.scores = score_nms
    result.pred_classes = class_nms.to(torch.int64)

    mask_fcn_probs = tensor_outputs.get("mask_fcn_probs", None)
    if mask_fcn_probs is not None:
        # finish the mask pred
        mask_probs_pred = mask_fcn_probs
        num_masks = mask_probs_pred.shape[0]
        class_pred = result.pred_classes
        indices = torch.arange(num_masks, device=class_pred.device)
        mask_probs_pred = mask_probs_pred[indices, class_pred][:, None]
        result.pred_masks = mask_probs_pred
    elif force_mask_on:
        # NOTE: there's no way to know the height/width of mask here, it won't be
        # used anyway when batch size is 0, so just set them to 0.
        result.pred_masks = torch.zeros([0, 1, 0, 0], dtype=torch.uint8)

    keypoints_out = tensor_outputs.get("keypoints_out", None)
    kps_score = tensor_outputs.get("kps_score", None)
    if keypoints_out is not None:
        # keypoints_out: [N, 4, #kypoints], where 4 is in order of (x, y, score, prob)
        keypoints_tensor = keypoints_out
        # NOTE: it's possible that prob is not calculated if "should_output_softmax"
        # is set to False in HeatmapMaxKeypoint, so just using raw score, seems
        # it doesn't affect mAP. TODO: check more carefully.
        keypoint_xyp = keypoints_tensor.transpose(1, 2)[:, :, [0, 1, 2]]
        result.pred_keypoints = keypoint_xyp
    elif kps_score is not None:
        # keypoint heatmap to sparse data structure
        pred_keypoint_logits = kps_score
        keypoint_head.keypoint_rcnn_inference(pred_keypoint_logits, [result])

    return results


def _cast_to_f32(f64):
    return struct.unpack("f", struct.pack("f", f64))[0]


def set_caffe2_compatible_tensor_mode(model, enable=True):
    def _fn(m):
        if isinstance(m, Caffe2Compatible):
            m.tensor_mode = enable

    model.apply(_fn)


def convert_batched_inputs_to_c2_format(batched_inputs, size_divisibility, device):
    """
    See get_caffe2_inputs() below.
    """
    assert all(isinstance(x, dict) for x in batched_inputs)
    assert all(x["image"].dim() == 3 for x in batched_inputs)

    images = [x["image"] for x in batched_inputs]
    images = ImageList.from_tensors(images, size_divisibility)

    im_info = []
    for input_per_image, image_size in zip(batched_inputs, images.image_sizes):
        target_height = input_per_image.get("height", image_size[0])
        target_width = input_per_image.get("width", image_size[1])  # noqa
        # NOTE: The scale inside im_info is kept as convention and for providing
        # post-processing information if further processing is needed. For
        # current Caffe2 model definitions that don't include post-processing inside
        # the model, this number is not used.
        # NOTE: There can be a slight difference between width and height
        # scales, using a single number can results in numerical difference
        # compared with D2's post-processing.
        scale = target_height / image_size[0]
        im_info.append([image_size[0], image_size[1], scale])
    im_info = torch.Tensor(im_info)

    return images.tensor.to(device), im_info.to(device)


class Caffe2MetaArch(Caffe2Compatible, torch.nn.Module):
    """
    Base class for caffe2-compatible implementation of a meta architecture.
    The forward is traceable and its traced graph can be converted to caffe2
    graph through ONNX.
    """

    def __init__(self, cfg, torch_model):
        """
        Args:
            cfg (CfgNode):
            torch_model (nn.Module): the detectron2 model (meta_arch) to be
                converted.
        """
        super().__init__()
        self._wrapped_model = torch_model
        self.eval()
        set_caffe2_compatible_tensor_mode(self, True)

    def get_caffe2_inputs(self, batched_inputs):
        """
        Convert pytorch-style structured inputs to caffe2-style inputs that
        are tuples of tensors.

        Args:
            batched_inputs (list[dict]): inputs to a detectron2 model
                in its standard format. Each dict has "image" (CHW tensor), and optionally
                "height" and "width".

        Returns:
            tuple[Tensor]:
                tuple of tensors that will be the inputs to the
                :meth:`forward` method. For existing models, the first
                is an NCHW tensor (padded and batched); the second is
                a im_info Nx3 tensor, where the rows are
                (height, width, unused legacy parameter)
        """
        return convert_batched_inputs_to_c2_format(
            batched_inputs,
            self._wrapped_model.backbone.size_divisibility,
            self._wrapped_model.device,
        )

    def encode_additional_info(self, predict_net, init_net):
        """
        Save extra metadata that will be used by inference in the output protobuf.
        """
        pass

    def forward(self, inputs):
        """
        Run the forward in caffe2-style. It has to use caffe2-compatible ops
        and the method will be used for tracing.

        Args:
            inputs (tuple[Tensor]): inputs defined by :meth:`get_caffe2_input`.
                They will be the inputs of the converted caffe2 graph.

        Returns:
            tuple[Tensor]: output tensors. They will be the outputs of the
                converted caffe2 graph.
        """
        raise NotImplementedError

    def _caffe2_preprocess_image(self, inputs):
        """
        Caffe2 implementation of preprocess_image, which is called inside each MetaArch's forward.
        It normalizes the input images, and the final caffe2 graph assumes the
        inputs have been batched already.
        """
        data, im_info = inputs
        data = alias(data, "data")
        im_info = alias(im_info, "im_info")
        mean, std = self._wrapped_model.pixel_mean, self._wrapped_model.pixel_std
        normalized_data = (data - mean) / std
        normalized_data = alias(normalized_data, "normalized_data")

        # Pack (data, im_info) into ImageList which is recognized by self.inference.
        images = ImageList(tensor=normalized_data, image_sizes=im_info)
        return images

    @staticmethod
    def get_outputs_converter(predict_net, init_net):
        """
        Creates a function that converts outputs of the caffe2 model to
        detectron2's standard format.
        The function uses information in `predict_net` and `init_net` that are
        available at inferene time. Therefore the function logic can be used in inference.

        The returned function has the following signature:

            def convert(batched_inputs, c2_inputs, c2_results) -> detectron2_outputs

        Where

            * batched_inputs (list[dict]): the original input format of the meta arch
            * c2_inputs (tuple[Tensor]): the caffe2 inputs.
            * c2_results (dict[str, Tensor]): the caffe2 output format,
                corresponding to the outputs of the :meth:`forward` function.
            * detectron2_outputs: the original output format of the meta arch.

        This function can be used to compare the outputs of the original meta arch and
        the converted caffe2 graph.

        Returns:
            callable: a callable of the above signature.
        """
        raise NotImplementedError


class Caffe2GeneralizedRCNN(Caffe2MetaArch):
    def __init__(self, cfg, torch_model):
        assert isinstance(torch_model, meta_arch.GeneralizedRCNN)
        torch_model = patch_generalized_rcnn(torch_model)
        super().__init__(cfg, torch_model)

        try:
            use_heatmap_max_keypoint = cfg.EXPORT_CAFFE2.USE_HEATMAP_MAX_KEYPOINT
        except AttributeError:
            use_heatmap_max_keypoint = False
        self.roi_heads_patcher = ROIHeadsPatcher(
            self._wrapped_model.roi_heads, use_heatmap_max_keypoint
        )

    def encode_additional_info(self, predict_net, init_net):
        size_divisibility = self._wrapped_model.backbone.size_divisibility
        check_set_pb_arg(predict_net, "size_divisibility", "i", size_divisibility)
        check_set_pb_arg(
            predict_net, "device", "s", str.encode(str(self._wrapped_model.device), "ascii")
        )
        check_set_pb_arg(predict_net, "meta_architecture", "s", b"GeneralizedRCNN")

    @mock_torch_nn_functional_interpolate()
    def forward(self, inputs):
        if not self.tensor_mode:
            return self._wrapped_model.inference(inputs)
        images = self._caffe2_preprocess_image(inputs)
        features = self._wrapped_model.backbone(images.tensor)
        proposals, _ = self._wrapped_model.proposal_generator(images, features)
        with self.roi_heads_patcher.mock_roi_heads():
            detector_results, _ = self._wrapped_model.roi_heads(images, features, proposals)
        return tuple(detector_results[0].flatten())

    @staticmethod
    def get_outputs_converter(predict_net, init_net):
        def f(batched_inputs, c2_inputs, c2_results):
            _, im_info = c2_inputs
            image_sizes = [[int(im[0]), int(im[1])] for im in im_info]
            results = assemble_rcnn_outputs_by_name(image_sizes, c2_results)
            return meta_arch.GeneralizedRCNN._postprocess(results, batched_inputs, image_sizes)

        return f


class Caffe2RetinaNet(Caffe2MetaArch):
    def __init__(self, cfg, torch_model):
        assert isinstance(torch_model, meta_arch.RetinaNet)
        super().__init__(cfg, torch_model)

    @mock_torch_nn_functional_interpolate()
    def forward(self, inputs):
        assert self.tensor_mode
        images = self._caffe2_preprocess_image(inputs)

        # explicitly return the images sizes to avoid removing "im_info" by ONNX
        # since it's not used in the forward path
        return_tensors = [images.image_sizes]

        features = self._wrapped_model.backbone(images.tensor)
        features = [features[f] for f in self._wrapped_model.head_in_features]
        for i, feature_i in enumerate(features):
            features[i] = alias(feature_i, "feature_{}".format(i), is_backward=True)
            return_tensors.append(features[i])

        pred_logits, pred_anchor_deltas = self._wrapped_model.head(features)
        for i, (box_cls_i, box_delta_i) in enumerate(zip(pred_logits, pred_anchor_deltas)):
            return_tensors.append(alias(box_cls_i, "box_cls_{}".format(i)))
            return_tensors.append(alias(box_delta_i, "box_delta_{}".format(i)))

        return tuple(return_tensors)

    def encode_additional_info(self, predict_net, init_net):
        size_divisibility = self._wrapped_model.backbone.size_divisibility
        check_set_pb_arg(predict_net, "size_divisibility", "i", size_divisibility)
        check_set_pb_arg(
            predict_net, "device", "s", str.encode(str(self._wrapped_model.device), "ascii")
        )
        check_set_pb_arg(predict_net, "meta_architecture", "s", b"RetinaNet")

        # Inference parameters:
        check_set_pb_arg(
            predict_net, "score_threshold", "f", _cast_to_f32(self._wrapped_model.test_score_thresh)
        )
        check_set_pb_arg(
            predict_net, "topk_candidates", "i", self._wrapped_model.test_topk_candidates
        )
        check_set_pb_arg(
            predict_net, "nms_threshold", "f", _cast_to_f32(self._wrapped_model.test_nms_thresh)
        )
        check_set_pb_arg(
            predict_net,
            "max_detections_per_image",
            "i",
            self._wrapped_model.max_detections_per_image,
        )

        check_set_pb_arg(
            predict_net,
            "bbox_reg_weights",
            "floats",
            [_cast_to_f32(w) for w in self._wrapped_model.box2box_transform.weights],
        )
        self._encode_anchor_generator_cfg(predict_net)

    def _encode_anchor_generator_cfg(self, predict_net):
        # serialize anchor_generator for future use
        serialized_anchor_generator = io.BytesIO()
        torch.save(self._wrapped_model.anchor_generator, serialized_anchor_generator)
        # Ideally we can put anchor generating inside the model, then we don't
        # need to store this information.
        bytes = serialized_anchor_generator.getvalue()
        check_set_pb_arg(predict_net, "serialized_anchor_generator", "s", bytes)

    @staticmethod
    def get_outputs_converter(predict_net, init_net):
        self = types.SimpleNamespace()
        serialized_anchor_generator = io.BytesIO(
            get_pb_arg_vals(predict_net, "serialized_anchor_generator", None)
        )
        self.anchor_generator = torch.load(serialized_anchor_generator)
        bbox_reg_weights = get_pb_arg_floats(predict_net, "bbox_reg_weights", None)
        self.box2box_transform = Box2BoxTransform(weights=tuple(bbox_reg_weights))
        self.test_score_thresh = get_pb_arg_valf(predict_net, "score_threshold", None)
        self.test_topk_candidates = get_pb_arg_vali(predict_net, "topk_candidates", None)
        self.test_nms_thresh = get_pb_arg_valf(predict_net, "nms_threshold", None)
        self.max_detections_per_image = get_pb_arg_vali(
            predict_net, "max_detections_per_image", None
        )

        # hack to reuse inference code from RetinaNet
        for meth in [
            "forward_inference",
            "inference_single_image",
            "_transpose_dense_predictions",
            "_decode_multi_level_predictions",
            "_decode_per_level_predictions",
        ]:
            setattr(self, meth, functools.partial(getattr(meta_arch.RetinaNet, meth), self))

        def f(batched_inputs, c2_inputs, c2_results):
            _, im_info = c2_inputs
            image_sizes = [[int(im[0]), int(im[1])] for im in im_info]
            dummy_images = ImageList(
                torch.randn(
                    (
                        len(im_info),
                        3,
                    )
                    + tuple(image_sizes[0])
                ),
                image_sizes,
            )

            num_features = len([x for x in c2_results.keys() if x.startswith("box_cls_")])
            pred_logits = [c2_results["box_cls_{}".format(i)] for i in range(num_features)]
            pred_anchor_deltas = [c2_results["box_delta_{}".format(i)] for i in range(num_features)]

            # For each feature level, feature should have the same batch size and
            # spatial dimension as the box_cls and box_delta.
            dummy_features = [x.clone()[:, 0:0, :, :] for x in pred_logits]
            # self.num_classess can be inferred
            self.num_classes = pred_logits[0].shape[1] // (pred_anchor_deltas[0].shape[1] // 4)

            results = self.forward_inference(
                dummy_images, dummy_features, [pred_logits, pred_anchor_deltas]
            )
            return meta_arch.GeneralizedRCNN._postprocess(results, batched_inputs, image_sizes)

        return f


META_ARCH_CAFFE2_EXPORT_TYPE_MAP = {
    "GeneralizedRCNN": Caffe2GeneralizedRCNN,
    "RetinaNet": Caffe2RetinaNet,
}
