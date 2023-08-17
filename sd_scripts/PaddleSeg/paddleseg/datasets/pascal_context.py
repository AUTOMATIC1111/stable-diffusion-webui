# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

from PIL import Image
from paddleseg.datasets import Dataset
from paddleseg.cvlibs import manager
from paddleseg.transforms import Compose


@manager.DATASETS.add_component
class PascalContext(Dataset):
    """
    PascalVOC2010 dataset `http://host.robots.ox.ac.uk/pascal/VOC/`.
    If you want to use pascal context dataset, please run the convert_voc2010.py in tools/data firstly.

    Args:
        transforms (list): Transforms for image.
        dataset_root (str): The dataset directory. Default: None
        mode (str): Which part of dataset to use. it is one of ('train', 'trainval', 'context', 'val').
            If you want to set mode to 'context', please make sure the dataset have been augmented. Default: 'train'.
        edge (bool, optional): Whether to compute edge while training. Default: False
    """
    NUM_CLASSES = 60
    IGNORE_INDEX = 255
    IMG_CHANNELS = 3

    def __init__(self,
                 transforms=None,
                 dataset_root=None,
                 mode='train',
                 edge=False):
        self.dataset_root = dataset_root
        self.transforms = Compose(transforms)
        mode = mode.lower()
        self.mode = mode
        self.file_list = list()
        self.num_classes = self.NUM_CLASSES
        self.ignore_index = self.IGNORE_INDEX
        self.edge = edge

        if mode not in ['train', 'trainval', 'val']:
            raise ValueError(
                "`mode` should be one of ('train', 'trainval', 'val') in PascalContext dataset, but got {}."
                .format(mode))

        if self.transforms is None:
            raise ValueError("`transforms` is necessary, but it is None.")
        if self.dataset_root is None:
            raise ValueError(
                "The dataset is not Found or the folder structure is nonconfoumance."
            )

        image_set_dir = os.path.join(self.dataset_root, 'ImageSets',
                                     'Segmentation')

        if mode == 'train':
            file_path = os.path.join(image_set_dir, 'train_context.txt')
        elif mode == 'val':
            file_path = os.path.join(image_set_dir, 'val_context.txt')
        elif mode == 'trainval':
            file_path = os.path.join(image_set_dir, 'trainval_context.txt')
        if not os.path.exists(file_path):
            raise RuntimeError(
                "PASCAL-Context annotations are not ready, "
                "Please make sure voc_context.py has been properly run.")

        img_dir = os.path.join(self.dataset_root, 'JPEGImages')
        label_dir = os.path.join(self.dataset_root, 'Context')

        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                image_path = os.path.join(img_dir, ''.join([line, '.jpg']))
                label_path = os.path.join(label_dir, ''.join([line, '.png']))
                self.file_list.append([image_path, label_path])
