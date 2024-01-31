# Copyright (c) Facebook, Inc. and its affiliates.
import math
from typing import List, Tuple, Union
import torch
from fvcore.nn import giou_loss, smooth_l1_loss
from torch.nn import functional as F

from annotator.oneformer.detectron2.layers import cat, ciou_loss, diou_loss
from annotator.oneformer.detectron2.structures import Boxes

# Value for clamping large dw and dh predictions. The heuristic is that we clamp
# such that dw and dh are no larger than what would transform a 16px box into a
# 1000px box (based on a small anchor, 16px, and a typical image size, 1000px).
_DEFAULT_SCALE_CLAMP = math.log(1000.0 / 16)


__all__ = ["Box2BoxTransform", "Box2BoxTransformRotated", "Box2BoxTransformLinear"]


@torch.jit.script
class Box2BoxTransform(object):
    """
    The box-to-box transform defined in R-CNN. The transformation is parameterized
    by 4 deltas: (dx, dy, dw, dh). The transformation scales the box's width and height
    by exp(dw), exp(dh) and shifts a box's center by the offset (dx * width, dy * height).
    """

    def __init__(
        self, weights: Tuple[float, float, float, float], scale_clamp: float = _DEFAULT_SCALE_CLAMP
    ):
        """
        Args:
            weights (4-element tuple): Scaling factors that are applied to the
                (dx, dy, dw, dh) deltas. In Fast R-CNN, these were originally set
                such that the deltas have unit variance; now they are treated as
                hyperparameters of the system.
            scale_clamp (float): When predicting deltas, the predicted box scaling
                factors (dw and dh) are clamped such that they are <= scale_clamp.
        """
        self.weights = weights
        self.scale_clamp = scale_clamp

    def get_deltas(self, src_boxes, target_boxes):
        """
        Get box regression transformation deltas (dx, dy, dw, dh) that can be used
        to transform the `src_boxes` into the `target_boxes`. That is, the relation
        ``target_boxes == self.apply_deltas(deltas, src_boxes)`` is true (unless
        any delta is too large and is clamped).

        Args:
            src_boxes (Tensor): source boxes, e.g., object proposals
            target_boxes (Tensor): target of the transformation, e.g., ground-truth
                boxes.
        """
        assert isinstance(src_boxes, torch.Tensor), type(src_boxes)
        assert isinstance(target_boxes, torch.Tensor), type(target_boxes)

        src_widths = src_boxes[:, 2] - src_boxes[:, 0]
        src_heights = src_boxes[:, 3] - src_boxes[:, 1]
        src_ctr_x = src_boxes[:, 0] + 0.5 * src_widths
        src_ctr_y = src_boxes[:, 1] + 0.5 * src_heights

        target_widths = target_boxes[:, 2] - target_boxes[:, 0]
        target_heights = target_boxes[:, 3] - target_boxes[:, 1]
        target_ctr_x = target_boxes[:, 0] + 0.5 * target_widths
        target_ctr_y = target_boxes[:, 1] + 0.5 * target_heights

        wx, wy, ww, wh = self.weights
        dx = wx * (target_ctr_x - src_ctr_x) / src_widths
        dy = wy * (target_ctr_y - src_ctr_y) / src_heights
        dw = ww * torch.log(target_widths / src_widths)
        dh = wh * torch.log(target_heights / src_heights)

        deltas = torch.stack((dx, dy, dw, dh), dim=1)
        assert (src_widths > 0).all().item(), "Input boxes to Box2BoxTransform are not valid!"
        return deltas

    def apply_deltas(self, deltas, boxes):
        """
        Apply transformation `deltas` (dx, dy, dw, dh) to `boxes`.

        Args:
            deltas (Tensor): transformation deltas of shape (N, k*4), where k >= 1.
                deltas[i] represents k potentially different class-specific
                box transformations for the single box boxes[i].
            boxes (Tensor): boxes to transform, of shape (N, 4)
        """
        deltas = deltas.float()  # ensure fp32 for decoding precision
        boxes = boxes.to(deltas.dtype)

        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        wx, wy, ww, wh = self.weights
        dx = deltas[:, 0::4] / wx
        dy = deltas[:, 1::4] / wy
        dw = deltas[:, 2::4] / ww
        dh = deltas[:, 3::4] / wh

        # Prevent sending too large values into torch.exp()
        dw = torch.clamp(dw, max=self.scale_clamp)
        dh = torch.clamp(dh, max=self.scale_clamp)

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        x1 = pred_ctr_x - 0.5 * pred_w
        y1 = pred_ctr_y - 0.5 * pred_h
        x2 = pred_ctr_x + 0.5 * pred_w
        y2 = pred_ctr_y + 0.5 * pred_h
        pred_boxes = torch.stack((x1, y1, x2, y2), dim=-1)
        return pred_boxes.reshape(deltas.shape)


@torch.jit.script
class Box2BoxTransformRotated(object):
    """
    The box-to-box transform defined in Rotated R-CNN. The transformation is parameterized
    by 5 deltas: (dx, dy, dw, dh, da). The transformation scales the box's width and height
    by exp(dw), exp(dh), shifts a box's center by the offset (dx * width, dy * height),
    and rotate a box's angle by da (radians).
    Note: angles of deltas are in radians while angles of boxes are in degrees.
    """

    def __init__(
        self,
        weights: Tuple[float, float, float, float, float],
        scale_clamp: float = _DEFAULT_SCALE_CLAMP,
    ):
        """
        Args:
            weights (5-element tuple): Scaling factors that are applied to the
                (dx, dy, dw, dh, da) deltas. These are treated as
                hyperparameters of the system.
            scale_clamp (float): When predicting deltas, the predicted box scaling
                factors (dw and dh) are clamped such that they are <= scale_clamp.
        """
        self.weights = weights
        self.scale_clamp = scale_clamp

    def get_deltas(self, src_boxes, target_boxes):
        """
        Get box regression transformation deltas (dx, dy, dw, dh, da) that can be used
        to transform the `src_boxes` into the `target_boxes`. That is, the relation
        ``target_boxes == self.apply_deltas(deltas, src_boxes)`` is true (unless
        any delta is too large and is clamped).

        Args:
            src_boxes (Tensor): Nx5 source boxes, e.g., object proposals
            target_boxes (Tensor): Nx5 target of the transformation, e.g., ground-truth
                boxes.
        """
        assert isinstance(src_boxes, torch.Tensor), type(src_boxes)
        assert isinstance(target_boxes, torch.Tensor), type(target_boxes)

        src_ctr_x, src_ctr_y, src_widths, src_heights, src_angles = torch.unbind(src_boxes, dim=1)

        target_ctr_x, target_ctr_y, target_widths, target_heights, target_angles = torch.unbind(
            target_boxes, dim=1
        )

        wx, wy, ww, wh, wa = self.weights
        dx = wx * (target_ctr_x - src_ctr_x) / src_widths
        dy = wy * (target_ctr_y - src_ctr_y) / src_heights
        dw = ww * torch.log(target_widths / src_widths)
        dh = wh * torch.log(target_heights / src_heights)
        # Angles of deltas are in radians while angles of boxes are in degrees.
        # the conversion to radians serve as a way to normalize the values
        da = target_angles - src_angles
        da = (da + 180.0) % 360.0 - 180.0  # make it in [-180, 180)
        da *= wa * math.pi / 180.0

        deltas = torch.stack((dx, dy, dw, dh, da), dim=1)
        assert (
            (src_widths > 0).all().item()
        ), "Input boxes to Box2BoxTransformRotated are not valid!"
        return deltas

    def apply_deltas(self, deltas, boxes):
        """
        Apply transformation `deltas` (dx, dy, dw, dh, da) to `boxes`.

        Args:
            deltas (Tensor): transformation deltas of shape (N, k*5).
                deltas[i] represents box transformation for the single box boxes[i].
            boxes (Tensor): boxes to transform, of shape (N, 5)
        """
        assert deltas.shape[1] % 5 == 0 and boxes.shape[1] == 5

        boxes = boxes.to(deltas.dtype).unsqueeze(2)

        ctr_x = boxes[:, 0]
        ctr_y = boxes[:, 1]
        widths = boxes[:, 2]
        heights = boxes[:, 3]
        angles = boxes[:, 4]

        wx, wy, ww, wh, wa = self.weights

        dx = deltas[:, 0::5] / wx
        dy = deltas[:, 1::5] / wy
        dw = deltas[:, 2::5] / ww
        dh = deltas[:, 3::5] / wh
        da = deltas[:, 4::5] / wa

        # Prevent sending too large values into torch.exp()
        dw = torch.clamp(dw, max=self.scale_clamp)
        dh = torch.clamp(dh, max=self.scale_clamp)

        pred_boxes = torch.zeros_like(deltas)
        pred_boxes[:, 0::5] = dx * widths + ctr_x  # x_ctr
        pred_boxes[:, 1::5] = dy * heights + ctr_y  # y_ctr
        pred_boxes[:, 2::5] = torch.exp(dw) * widths  # width
        pred_boxes[:, 3::5] = torch.exp(dh) * heights  # height

        # Following original RRPN implementation,
        # angles of deltas are in radians while angles of boxes are in degrees.
        pred_angle = da * 180.0 / math.pi + angles
        pred_angle = (pred_angle + 180.0) % 360.0 - 180.0  # make it in [-180, 180)

        pred_boxes[:, 4::5] = pred_angle

        return pred_boxes


class Box2BoxTransformLinear(object):
    """
    The linear box-to-box transform defined in FCOS. The transformation is parameterized
    by the distance from the center of (square) src box to 4 edges of the target box.
    """

    def __init__(self, normalize_by_size=True):
        """
        Args:
            normalize_by_size: normalize deltas by the size of src (anchor) boxes.
        """
        self.normalize_by_size = normalize_by_size

    def get_deltas(self, src_boxes, target_boxes):
        """
        Get box regression transformation deltas (dx1, dy1, dx2, dy2) that can be used
        to transform the `src_boxes` into the `target_boxes`. That is, the relation
        ``target_boxes == self.apply_deltas(deltas, src_boxes)`` is true.
        The center of src must be inside target boxes.

        Args:
            src_boxes (Tensor): square source boxes, e.g., anchors
            target_boxes (Tensor): target of the transformation, e.g., ground-truth
                boxes.
        """
        assert isinstance(src_boxes, torch.Tensor), type(src_boxes)
        assert isinstance(target_boxes, torch.Tensor), type(target_boxes)

        src_ctr_x = 0.5 * (src_boxes[:, 0] + src_boxes[:, 2])
        src_ctr_y = 0.5 * (src_boxes[:, 1] + src_boxes[:, 3])

        target_l = src_ctr_x - target_boxes[:, 0]
        target_t = src_ctr_y - target_boxes[:, 1]
        target_r = target_boxes[:, 2] - src_ctr_x
        target_b = target_boxes[:, 3] - src_ctr_y

        deltas = torch.stack((target_l, target_t, target_r, target_b), dim=1)
        if self.normalize_by_size:
            stride_w = src_boxes[:, 2] - src_boxes[:, 0]
            stride_h = src_boxes[:, 3] - src_boxes[:, 1]
            strides = torch.stack([stride_w, stride_h, stride_w, stride_h], axis=1)
            deltas = deltas / strides

        return deltas

    def apply_deltas(self, deltas, boxes):
        """
        Apply transformation `deltas` (dx1, dy1, dx2, dy2) to `boxes`.

        Args:
            deltas (Tensor): transformation deltas of shape (N, k*4), where k >= 1.
                deltas[i] represents k potentially different class-specific
                box transformations for the single box boxes[i].
            boxes (Tensor): boxes to transform, of shape (N, 4)
        """
        # Ensure the output is a valid box. See Sec 2.1 of https://arxiv.org/abs/2006.09214
        deltas = F.relu(deltas)
        boxes = boxes.to(deltas.dtype)

        ctr_x = 0.5 * (boxes[:, 0] + boxes[:, 2])
        ctr_y = 0.5 * (boxes[:, 1] + boxes[:, 3])
        if self.normalize_by_size:
            stride_w = boxes[:, 2] - boxes[:, 0]
            stride_h = boxes[:, 3] - boxes[:, 1]
            strides = torch.stack([stride_w, stride_h, stride_w, stride_h], axis=1)
            deltas = deltas * strides

        l = deltas[:, 0::4]
        t = deltas[:, 1::4]
        r = deltas[:, 2::4]
        b = deltas[:, 3::4]

        pred_boxes = torch.zeros_like(deltas)
        pred_boxes[:, 0::4] = ctr_x[:, None] - l  # x1
        pred_boxes[:, 1::4] = ctr_y[:, None] - t  # y1
        pred_boxes[:, 2::4] = ctr_x[:, None] + r  # x2
        pred_boxes[:, 3::4] = ctr_y[:, None] + b  # y2
        return pred_boxes


def _dense_box_regression_loss(
    anchors: List[Union[Boxes, torch.Tensor]],
    box2box_transform: Box2BoxTransform,
    pred_anchor_deltas: List[torch.Tensor],
    gt_boxes: List[torch.Tensor],
    fg_mask: torch.Tensor,
    box_reg_loss_type="smooth_l1",
    smooth_l1_beta=0.0,
):
    """
    Compute loss for dense multi-level box regression.
    Loss is accumulated over ``fg_mask``.

    Args:
        anchors: #lvl anchor boxes, each is (HixWixA, 4)
        pred_anchor_deltas: #lvl predictions, each is (N, HixWixA, 4)
        gt_boxes: N ground truth boxes, each has shape (R, 4) (R = sum(Hi * Wi * A))
        fg_mask: the foreground boolean mask of shape (N, R) to compute loss on
        box_reg_loss_type (str): Loss type to use. Supported losses: "smooth_l1", "giou",
            "diou", "ciou".
        smooth_l1_beta (float): beta parameter for the smooth L1 regression loss. Default to
            use L1 loss. Only used when `box_reg_loss_type` is "smooth_l1"
    """
    if isinstance(anchors[0], Boxes):
        anchors = type(anchors[0]).cat(anchors).tensor  # (R, 4)
    else:
        anchors = cat(anchors)
    if box_reg_loss_type == "smooth_l1":
        gt_anchor_deltas = [box2box_transform.get_deltas(anchors, k) for k in gt_boxes]
        gt_anchor_deltas = torch.stack(gt_anchor_deltas)  # (N, R, 4)
        loss_box_reg = smooth_l1_loss(
            cat(pred_anchor_deltas, dim=1)[fg_mask],
            gt_anchor_deltas[fg_mask],
            beta=smooth_l1_beta,
            reduction="sum",
        )
    elif box_reg_loss_type == "giou":
        pred_boxes = [
            box2box_transform.apply_deltas(k, anchors) for k in cat(pred_anchor_deltas, dim=1)
        ]
        loss_box_reg = giou_loss(
            torch.stack(pred_boxes)[fg_mask], torch.stack(gt_boxes)[fg_mask], reduction="sum"
        )
    elif box_reg_loss_type == "diou":
        pred_boxes = [
            box2box_transform.apply_deltas(k, anchors) for k in cat(pred_anchor_deltas, dim=1)
        ]
        loss_box_reg = diou_loss(
            torch.stack(pred_boxes)[fg_mask], torch.stack(gt_boxes)[fg_mask], reduction="sum"
        )
    elif box_reg_loss_type == "ciou":
        pred_boxes = [
            box2box_transform.apply_deltas(k, anchors) for k in cat(pred_anchor_deltas, dim=1)
        ]
        loss_box_reg = ciou_loss(
            torch.stack(pred_boxes)[fg_mask], torch.stack(gt_boxes)[fg_mask], reduction="sum"
        )
    else:
        raise ValueError(f"Invalid dense box regression loss type '{box_reg_loss_type}'")
    return loss_box_reg
