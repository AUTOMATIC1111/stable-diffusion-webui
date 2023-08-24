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

from paddleseg.datasets import Dataset
from paddleseg.utils.download import download_file_and_uncompress
from paddleseg.utils import seg_env
from paddleseg.cvlibs import manager
from paddleseg.transforms import Compose

URL = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"


@manager.DATASETS.add_component
class PascalVOC(Dataset):
    """
    PascalVOC2012 dataset `http://host.robots.ox.ac.uk/pascal/VOC/`.
    If you want to augment the dataset, please run the voc_augment.py in tools/data.

    Args:
        transforms (list): Transforms for image.
        dataset_root (str): The dataset directory. Default: None
        mode (str, optional): Which part of dataset to use. it is one of ('train', 'trainval', 'trainaug', 'val').
            If you want to set mode to 'trainaug', please make sure the dataset have been augmented. Default: 'train'.
        edge (bool, optional): Whether to compute edge while training. Default: False
    """
    NUM_CLASSES = 21
    IGNORE_INDEX = 255
    IMG_CHANNELS = 3

    def __init__(self, transforms, dataset_root=None, mode='train', edge=False):
        self.dataset_root = dataset_root
        self.transforms = Compose(transforms)
        mode = mode.lower()
        self.mode = mode
        self.file_list = list()
        self.num_classes = self.NUM_CLASSES
        self.ignore_index = self.IGNORE_INDEX
        self.edge = edge

        if mode not in ['train', 'trainval', 'trainaug', 'val']:
            raise ValueError(
                "`mode` should be one of ('train', 'trainval', 'trainaug', 'val') in PascalVOC dataset, but got {}."
                .format(mode))

        if self.transforms is None:
            raise ValueError("`transforms` is necessary, but it is None.")

        if self.dataset_root is None:
            self.dataset_root = download_file_and_uncompress(
                url=URL,
                savepath=seg_env.DATA_HOME,
                extrapath=seg_env.DATA_HOME,
                extraname='VOCdevkit')
        elif not os.path.exists(self.dataset_root):
            self.dataset_root = os.path.normpath(self.dataset_root)
            savepath, extraname = self.dataset_root.rsplit(
                sep=os.path.sep, maxsplit=1)
            self.dataset_root = download_file_and_uncompress(
                url=URL,
                savepath=savepath,
                extrapath=savepath,
                extraname=extraname)

        image_set_dir = os.path.join(self.dataset_root, 'VOC2012', 'ImageSets',
                                     'Segmentation')
        if mode == 'train':
            file_path = os.path.join(image_set_dir, 'train.txt')
        elif mode == 'val':
            file_path = os.path.join(image_set_dir, 'val.txt')
        elif mode == 'trainval':
            file_path = os.path.join(image_set_dir, 'trainval.txt')
        elif mode == 'trainaug':
            file_path = os.path.join(image_set_dir, 'train.txt')
            file_path_aug = os.path.join(image_set_dir, 'aug.txt')

            if not os.path.exists(file_path_aug):
                raise RuntimeError(
                    "When `mode` is 'trainaug', Pascal Voc dataset should be augmented, "
                    "Please make sure voc_augment.py has been properly run when using this mode."
                )

        img_dir = os.path.join(self.dataset_root, 'VOC2012', 'JPEGImages')
        label_dir = os.path.join(self.dataset_root, 'VOC2012',
                                 'SegmentationClass')
        label_dir_aug = os.path.join(self.dataset_root, 'VOC2012',
                                     'SegmentationClassAug')

        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                image_path = os.path.join(img_dir, ''.join([line, '.jpg']))
                label_path = os.path.join(label_dir, ''.join([line, '.png']))
                self.file_list.append([image_path, label_path])
        if mode == 'trainaug':
            with open(file_path_aug, 'r') as f:
                for line in f:
                    line = line.strip()
                    image_path = os.path.join(img_dir, ''.join([line, '.jpg']))
                    label_path = os.path.join(label_dir_aug,
                                              ''.join([line, '.png']))
                    self.file_list.append([image_path, label_path])
