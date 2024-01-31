# Copyright (c) OpenMMLab. All rights reserved.
from ..utils import ext_loader

ext_module = ext_loader.load_ext('_ext', ['bbox_overlaps'])


def bbox_overlaps(bboxes1, bboxes2, mode='iou', aligned=False, offset=0):
    """Calculate overlap between two set of bboxes.

    If ``aligned`` is ``False``, then calculate the ious between each bbox
    of bboxes1 and bboxes2, otherwise the ious between each aligned pair of
    bboxes1 and bboxes2.

    Args:
        bboxes1 (Tensor): shape (m, 4) in <x1, y1, x2, y2> format or empty.
        bboxes2 (Tensor): shape (n, 4) in <x1, y1, x2, y2> format or empty.
            If aligned is ``True``, then m and n must be equal.
        mode (str): "iou" (intersection over union) or iof (intersection over
            foreground).

    Returns:
        ious(Tensor): shape (m, n) if aligned == False else shape (m, 1)

    Example:
        >>> bboxes1 = torch.FloatTensor([
        >>>     [0, 0, 10, 10],
        >>>     [10, 10, 20, 20],
        >>>     [32, 32, 38, 42],
        >>> ])
        >>> bboxes2 = torch.FloatTensor([
        >>>     [0, 0, 10, 20],
        >>>     [0, 10, 10, 19],
        >>>     [10, 10, 20, 20],
        >>> ])
        >>> bbox_overlaps(bboxes1, bboxes2)
        tensor([[0.5000, 0.0000, 0.0000],
                [0.0000, 0.0000, 1.0000],
                [0.0000, 0.0000, 0.0000]])

    Example:
        >>> empty = torch.FloatTensor([])
        >>> nonempty = torch.FloatTensor([
        >>>     [0, 0, 10, 9],
        >>> ])
        >>> assert tuple(bbox_overlaps(empty, nonempty).shape) == (0, 1)
        >>> assert tuple(bbox_overlaps(nonempty, empty).shape) == (1, 0)
        >>> assert tuple(bbox_overlaps(empty, empty).shape) == (0, 0)
    """

    mode_dict = {'iou': 0, 'iof': 1}
    assert mode in mode_dict.keys()
    mode_flag = mode_dict[mode]
    # Either the boxes are empty or the length of boxes' last dimension is 4
    assert (bboxes1.size(-1) == 4 or bboxes1.size(0) == 0)
    assert (bboxes2.size(-1) == 4 or bboxes2.size(0) == 0)
    assert offset == 1 or offset == 0

    rows = bboxes1.size(0)
    cols = bboxes2.size(0)
    if aligned:
        assert rows == cols

    if rows * cols == 0:
        return bboxes1.new(rows, 1) if aligned else bboxes1.new(rows, cols)

    if aligned:
        ious = bboxes1.new_zeros(rows)
    else:
        ious = bboxes1.new_zeros((rows, cols))
    ext_module.bbox_overlaps(
        bboxes1, bboxes2, ious, mode=mode_flag, aligned=aligned, offset=offset)
    return ious
