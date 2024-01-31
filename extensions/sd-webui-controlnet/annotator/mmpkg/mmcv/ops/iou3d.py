# Copyright (c) OpenMMLab. All rights reserved.
import torch

from ..utils import ext_loader

ext_module = ext_loader.load_ext('_ext', [
    'iou3d_boxes_iou_bev_forward', 'iou3d_nms_forward',
    'iou3d_nms_normal_forward'
])


def boxes_iou_bev(boxes_a, boxes_b):
    """Calculate boxes IoU in the Bird's Eye View.

    Args:
        boxes_a (torch.Tensor): Input boxes a with shape (M, 5).
        boxes_b (torch.Tensor): Input boxes b with shape (N, 5).

    Returns:
        ans_iou (torch.Tensor): IoU result with shape (M, N).
    """
    ans_iou = boxes_a.new_zeros(
        torch.Size((boxes_a.shape[0], boxes_b.shape[0])))

    ext_module.iou3d_boxes_iou_bev_forward(boxes_a.contiguous(),
                                           boxes_b.contiguous(), ans_iou)

    return ans_iou


def nms_bev(boxes, scores, thresh, pre_max_size=None, post_max_size=None):
    """NMS function GPU implementation (for BEV boxes). The overlap of two
    boxes for IoU calculation is defined as the exact overlapping area of the
    two boxes. In this function, one can also set ``pre_max_size`` and
    ``post_max_size``.

    Args:
        boxes (torch.Tensor): Input boxes with the shape of [N, 5]
            ([x1, y1, x2, y2, ry]).
        scores (torch.Tensor): Scores of boxes with the shape of [N].
        thresh (float): Overlap threshold of NMS.
        pre_max_size (int, optional): Max size of boxes before NMS.
            Default: None.
        post_max_size (int, optional): Max size of boxes after NMS.
            Default: None.

    Returns:
        torch.Tensor: Indexes after NMS.
    """
    assert boxes.size(1) == 5, 'Input boxes shape should be [N, 5]'
    order = scores.sort(0, descending=True)[1]

    if pre_max_size is not None:
        order = order[:pre_max_size]
    boxes = boxes[order].contiguous()

    keep = torch.zeros(boxes.size(0), dtype=torch.long)
    num_out = ext_module.iou3d_nms_forward(boxes, keep, thresh)
    keep = order[keep[:num_out].cuda(boxes.device)].contiguous()
    if post_max_size is not None:
        keep = keep[:post_max_size]
    return keep


def nms_normal_bev(boxes, scores, thresh):
    """Normal NMS function GPU implementation (for BEV boxes). The overlap of
    two boxes for IoU calculation is defined as the exact overlapping area of
    the two boxes WITH their yaw angle set to 0.

    Args:
        boxes (torch.Tensor): Input boxes with shape (N, 5).
        scores (torch.Tensor): Scores of predicted boxes with shape (N).
        thresh (float): Overlap threshold of NMS.

    Returns:
        torch.Tensor: Remaining indices with scores in descending order.
    """
    assert boxes.shape[1] == 5, 'Input boxes shape should be [N, 5]'
    order = scores.sort(0, descending=True)[1]

    boxes = boxes[order].contiguous()

    keep = torch.zeros(boxes.size(0), dtype=torch.long)
    num_out = ext_module.iou3d_nms_normal_forward(boxes, keep, thresh)
    return order[keep[:num_out].cuda(boxes.device)].contiguous()
