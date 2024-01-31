import os.path as osp

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class PascalContextDataset(CustomDataset):
    """PascalContext dataset.

    In segmentation map annotation for PascalContext, 0 stands for background,
    which is included in 60 categories. ``reduce_zero_label`` is fixed to
    False. The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is
    fixed to '.png'.

    Args:
        split (str): Split txt file for PascalContext.
    """

    CLASSES = ('background', 'aeroplane', 'bag', 'bed', 'bedclothes', 'bench',
               'bicycle', 'bird', 'boat', 'book', 'bottle', 'building', 'bus',
               'cabinet', 'car', 'cat', 'ceiling', 'chair', 'cloth',
               'computer', 'cow', 'cup', 'curtain', 'dog', 'door', 'fence',
               'floor', 'flower', 'food', 'grass', 'ground', 'horse',
               'keyboard', 'light', 'motorbike', 'mountain', 'mouse', 'person',
               'plate', 'platform', 'pottedplant', 'road', 'rock', 'sheep',
               'shelves', 'sidewalk', 'sign', 'sky', 'snow', 'sofa', 'table',
               'track', 'train', 'tree', 'truck', 'tvmonitor', 'wall', 'water',
               'window', 'wood')

    PALETTE = [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
               [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
               [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
               [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
               [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
               [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
               [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
               [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
               [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
               [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
               [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
               [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
               [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
               [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
               [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255]]

    def __init__(self, split, **kwargs):
        super(PascalContextDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            split=split,
            reduce_zero_label=False,
            **kwargs)
        assert osp.exists(self.img_dir) and self.split is not None


@DATASETS.register_module()
class PascalContextDataset59(CustomDataset):
    """PascalContext dataset.

    In segmentation map annotation for PascalContext, 0 stands for background,
    which is included in 60 categories. ``reduce_zero_label`` is fixed to
    False. The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is
    fixed to '.png'.

    Args:
        split (str): Split txt file for PascalContext.
    """

    CLASSES = ('aeroplane', 'bag', 'bed', 'bedclothes', 'bench', 'bicycle',
               'bird', 'boat', 'book', 'bottle', 'building', 'bus', 'cabinet',
               'car', 'cat', 'ceiling', 'chair', 'cloth', 'computer', 'cow',
               'cup', 'curtain', 'dog', 'door', 'fence', 'floor', 'flower',
               'food', 'grass', 'ground', 'horse', 'keyboard', 'light',
               'motorbike', 'mountain', 'mouse', 'person', 'plate', 'platform',
               'pottedplant', 'road', 'rock', 'sheep', 'shelves', 'sidewalk',
               'sign', 'sky', 'snow', 'sofa', 'table', 'track', 'train',
               'tree', 'truck', 'tvmonitor', 'wall', 'water', 'window', 'wood')

    PALETTE = [[180, 120, 120], [6, 230, 230], [80, 50, 50], [4, 200, 3],
               [120, 120, 80], [140, 140, 140], [204, 5, 255], [230, 230, 230],
               [4, 250, 7], [224, 5, 255], [235, 255, 7], [150, 5, 61],
               [120, 120, 70], [8, 255, 51], [255, 6, 82], [143, 255, 140],
               [204, 255, 4], [255, 51, 7], [204, 70, 3], [0, 102, 200],
               [61, 230, 250], [255, 6, 51], [11, 102, 255], [255, 7, 71],
               [255, 9, 224], [9, 7, 230], [220, 220, 220], [255, 9, 92],
               [112, 9, 255], [8, 255, 214], [7, 255, 224], [255, 184, 6],
               [10, 255, 71], [255, 41, 10], [7, 255, 255], [224, 255, 8],
               [102, 8, 255], [255, 61, 6], [255, 194, 7], [255, 122, 8],
               [0, 255, 20], [255, 8, 41], [255, 5, 153], [6, 51, 255],
               [235, 12, 255], [160, 150, 20], [0, 163, 255], [140, 140, 140],
               [250, 10, 15], [20, 255, 0], [31, 255, 0], [255, 31, 0],
               [255, 224, 0], [153, 255, 0], [0, 0, 255], [255, 71, 0],
               [0, 235, 255], [0, 173, 255], [31, 0, 255]]

    def __init__(self, split, **kwargs):
        super(PascalContextDataset59, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            split=split,
            reduce_zero_label=True,
            **kwargs)
        assert osp.exists(self.img_dir) and self.split is not None
