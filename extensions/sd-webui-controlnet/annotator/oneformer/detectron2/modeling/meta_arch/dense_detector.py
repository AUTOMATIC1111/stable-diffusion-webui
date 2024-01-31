import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
from torch import Tensor, nn

from annotator.oneformer.detectron2.data.detection_utils import convert_image_to_rgb
from annotator.oneformer.detectron2.layers import move_device_like
from annotator.oneformer.detectron2.modeling import Backbone
from annotator.oneformer.detectron2.structures import Boxes, ImageList, Instances
from annotator.oneformer.detectron2.utils.events import get_event_storage

from ..postprocessing import detector_postprocess


def permute_to_N_HWA_K(tensor, K: int):
    """
    Transpose/reshape a tensor from (N, (Ai x K), H, W) to (N, (HxWxAi), K)
    """
    assert tensor.dim() == 4, tensor.shape
    N, _, H, W = tensor.shape
    tensor = tensor.view(N, -1, K, H, W)
    tensor = tensor.permute(0, 3, 4, 1, 2)
    tensor = tensor.reshape(N, -1, K)  # Size=(N,HWA,K)
    return tensor


class DenseDetector(nn.Module):
    """
    Base class for dense detector. We define a dense detector as a fully-convolutional model that
    makes per-pixel (i.e. dense) predictions.
    """

    def __init__(
        self,
        backbone: Backbone,
        head: nn.Module,
        head_in_features: Optional[List[str]] = None,
        *,
        pixel_mean,
        pixel_std,
    ):
        """
        Args:
            backbone: backbone module
            head: head module
            head_in_features: backbone features to use in head. Default to all backbone features.
            pixel_mean (Tuple[float]):
                Values to be used for image normalization (BGR order).
                To train on images of different number of channels, set different mean & std.
                Default values are the mean pixel value from ImageNet: [103.53, 116.28, 123.675]
            pixel_std (Tuple[float]):
                When using pre-trained models in Detectron1 or any MSRA models,
                std has been absorbed into its conv1 weights, so the std needs to be set 1.
                Otherwise, you can use [57.375, 57.120, 58.395] (ImageNet std)
        """
        super().__init__()

        self.backbone = backbone
        self.head = head
        if head_in_features is None:
            shapes = self.backbone.output_shape()
            self.head_in_features = sorted(shapes.keys(), key=lambda x: shapes[x].stride)
        else:
            self.head_in_features = head_in_features
        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)

    @property
    def device(self):
        return self.pixel_mean.device

    def _move_to_current_device(self, x):
        return move_device_like(x, self.pixel_mean)

    def forward(self, batched_inputs: List[Dict[str, Tensor]]):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            In training, dict[str, Tensor]: mapping from a named loss to a tensor storing the
            loss. Used during training only. In inference, the standard output format, described
            in :doc:`/tutorials/models`.
        """
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        features = [features[f] for f in self.head_in_features]
        predictions = self.head(features)

        if self.training:
            assert not torch.jit.is_scripting(), "Not supported"
            assert "instances" in batched_inputs[0], "Instance annotations are missing in training!"
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            return self.forward_training(images, features, predictions, gt_instances)
        else:
            results = self.forward_inference(images, features, predictions)
            if torch.jit.is_scripting():
                return results

            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results

    def forward_training(self, images, features, predictions, gt_instances):
        raise NotImplementedError()

    def preprocess_image(self, batched_inputs: List[Dict[str, Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        images = [self._move_to_current_device(x["image"]) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(
            images,
            self.backbone.size_divisibility,
            padding_constraints=self.backbone.padding_constraints,
        )
        return images

    def _transpose_dense_predictions(
        self, predictions: List[List[Tensor]], dims_per_anchor: List[int]
    ) -> List[List[Tensor]]:
        """
        Transpose the dense per-level predictions.

        Args:
            predictions: a list of outputs, each is a list of per-level
                predictions with shape (N, Ai x K, Hi, Wi), where N is the
                number of images, Ai is the number of anchors per location on
                level i, K is the dimension of predictions per anchor.
            dims_per_anchor: the value of K for each predictions. e.g. 4 for
                box prediction, #classes for classification prediction.

        Returns:
            List[List[Tensor]]: each prediction is transposed to (N, Hi x Wi x Ai, K).
        """
        assert len(predictions) == len(dims_per_anchor)
        res: List[List[Tensor]] = []
        for pred, dim_per_anchor in zip(predictions, dims_per_anchor):
            pred = [permute_to_N_HWA_K(x, dim_per_anchor) for x in pred]
            res.append(pred)
        return res

    def _ema_update(self, name: str, value: float, initial_value: float, momentum: float = 0.9):
        """
        Apply EMA update to `self.name` using `value`.

        This is mainly used for loss normalizer. In Detectron1, loss is normalized by number
        of foreground samples in the batch. When batch size is 1 per GPU, #foreground has a
        large variance and using it lead to lower performance. Therefore we maintain an EMA of
        #foreground to stabilize the normalizer.

        Args:
            name: name of the normalizer
            value: the new value to update
            initial_value: the initial value to start with
            momentum: momentum of EMA

        Returns:
            float: the updated EMA value
        """
        if hasattr(self, name):
            old = getattr(self, name)
        else:
            old = initial_value
        new = old * momentum + value * (1 - momentum)
        setattr(self, name, new)
        return new

    def _decode_per_level_predictions(
        self,
        anchors: Boxes,
        pred_scores: Tensor,
        pred_deltas: Tensor,
        score_thresh: float,
        topk_candidates: int,
        image_size: Tuple[int, int],
    ) -> Instances:
        """
        Decode boxes and classification predictions of one featuer level, by
        the following steps:
        1. filter the predictions based on score threshold and top K scores.
        2. transform the box regression outputs
        3. return the predicted scores, classes and boxes

        Args:
            anchors: Boxes, anchor for this feature level
            pred_scores: HxWxA,K
            pred_deltas: HxWxA,4

        Returns:
            Instances: with field "scores", "pred_boxes", "pred_classes".
        """
        # Apply two filtering to make NMS faster.
        # 1. Keep boxes with confidence score higher than threshold
        keep_idxs = pred_scores > score_thresh
        pred_scores = pred_scores[keep_idxs]
        topk_idxs = torch.nonzero(keep_idxs)  # Kx2

        # 2. Keep top k top scoring boxes only
        topk_idxs_size = topk_idxs.shape[0]
        if isinstance(topk_idxs_size, Tensor):
            # It's a tensor in tracing
            num_topk = torch.clamp(topk_idxs_size, max=topk_candidates)
        else:
            num_topk = min(topk_idxs_size, topk_candidates)
        pred_scores, idxs = pred_scores.topk(num_topk)
        topk_idxs = topk_idxs[idxs]

        anchor_idxs, classes_idxs = topk_idxs.unbind(dim=1)

        pred_boxes = self.box2box_transform.apply_deltas(
            pred_deltas[anchor_idxs], anchors.tensor[anchor_idxs]
        )
        return Instances(
            image_size, pred_boxes=Boxes(pred_boxes), scores=pred_scores, pred_classes=classes_idxs
        )

    def _decode_multi_level_predictions(
        self,
        anchors: List[Boxes],
        pred_scores: List[Tensor],
        pred_deltas: List[Tensor],
        score_thresh: float,
        topk_candidates: int,
        image_size: Tuple[int, int],
    ) -> Instances:
        """
        Run `_decode_per_level_predictions` for all feature levels and concat the results.
        """
        predictions = [
            self._decode_per_level_predictions(
                anchors_i,
                box_cls_i,
                box_reg_i,
                self.test_score_thresh,
                self.test_topk_candidates,
                image_size,
            )
            # Iterate over every feature level
            for box_cls_i, box_reg_i, anchors_i in zip(pred_scores, pred_deltas, anchors)
        ]
        return predictions[0].cat(predictions)  # 'Instances.cat' is not scriptale but this is

    def visualize_training(self, batched_inputs, results):
        """
        A function used to visualize ground truth images and final network predictions.
        It shows ground truth bounding boxes on the original image and up to 20
        predicted object bounding boxes on the original image.

        Args:
            batched_inputs (list): a list that contains input to the model.
            results (List[Instances]): a list of #images elements returned by forward_inference().
        """
        from annotator.oneformer.detectron2.utils.visualizer import Visualizer

        assert len(batched_inputs) == len(
            results
        ), "Cannot visualize inputs and results of different sizes"
        storage = get_event_storage()
        max_boxes = 20

        image_index = 0  # only visualize a single image
        img = batched_inputs[image_index]["image"]
        img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
        v_gt = Visualizer(img, None)
        v_gt = v_gt.overlay_instances(boxes=batched_inputs[image_index]["instances"].gt_boxes)
        anno_img = v_gt.get_image()
        processed_results = detector_postprocess(results[image_index], img.shape[0], img.shape[1])
        predicted_boxes = processed_results.pred_boxes.tensor.detach().cpu().numpy()

        v_pred = Visualizer(img, None)
        v_pred = v_pred.overlay_instances(boxes=predicted_boxes[0:max_boxes])
        prop_img = v_pred.get_image()
        vis_img = np.vstack((anno_img, prop_img))
        vis_img = vis_img.transpose(2, 0, 1)
        vis_name = f"Top: GT bounding boxes; Bottom: {max_boxes} Highest Scoring Results"
        storage.put_image(vis_name, vis_img)
