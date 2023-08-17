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

import paddle
import numpy as np
from PIL import Image

from paddleseg.cvlibs import manager
from paddleseg.transforms import Compose
import paddleseg.transforms.functional as F


@manager.DATASETS.add_component
class Dataset(paddle.io.Dataset):
    """
    Pass in a custom dataset that conforms to the format.

    Args:
        transforms (list): Transforms for image.
        dataset_root (str): The dataset directory.
        num_classes (int): Number of classes.
        mode (str, optional): which part of dataset to use. it is one of ('train', 'val', 'test'). Default: 'train'.
        train_path (str, optional): The train dataset file. When mode is 'train', train_path is necessary.
            The contents of train_path file are as follow:
            image1.jpg ground_truth1.png
            image2.jpg ground_truth2.png
        val_path (str. optional): The evaluation dataset file. When mode is 'val', val_path is necessary.
            The contents is the same as train_path
        test_path (str, optional): The test dataset file. When mode is 'test', test_path is necessary.
            The annotation file is not necessary in test_path file.
        separator (str, optional): The separator of dataset list. Default: ' '.
        edge (bool, optional): Whether to compute edge while training. Default: False

        Examples:

            import paddleseg.transforms as T
            from paddleseg.datasets import Dataset

            transforms = [T.RandomPaddingCrop(crop_size=(512,512)), T.Normalize()]
            dataset_root = 'dataset_root_path'
            train_path = 'train_path'
            num_classes = 2
            dataset = Dataset(transforms = transforms,
                              dataset_root = dataset_root,
                              num_classes = 2,
                              train_path = train_path,
                              mode = 'train')

    """

    def __init__(self,
                 mode,
                 dataset_root,
                 transforms,
                 num_classes,
                 img_channels=3,
                 train_path=None,
                 val_path=None,
                 test_path=None,
                 separator=' ',
                 ignore_index=255,
                 edge=False):
        self.dataset_root = dataset_root
        self.transforms = Compose(transforms, img_channels=img_channels)
        self.file_list = list()
        self.mode = mode.lower()
        self.num_classes = num_classes
        self.img_channels = img_channels
        self.ignore_index = ignore_index
        self.edge = edge

        if self.mode not in ['train', 'val', 'test']:
            raise ValueError(
                "mode should be 'train', 'val' or 'test', but got {}.".format(
                    self.mode))
        if not os.path.exists(self.dataset_root):
            raise FileNotFoundError('there is not `dataset_root`: {}.'.format(
                self.dataset_root))
        if self.transforms is None:
            raise ValueError("`transforms` is necessary, but it is None.")
        if num_classes < 1:
            raise ValueError(
                "`num_classes` should be greater than 1, but got {}".format(
                    num_classes))
        if img_channels not in [1, 3]:
            raise ValueError("`img_channels` should in [1, 3], but got {}".
                             format(img_channels))

        if self.mode == 'train':
            if train_path is None:
                raise ValueError(
                    'When `mode` is "train", `train_path` is necessary, but it is None.'
                )
            elif not os.path.exists(train_path):
                raise FileNotFoundError('`train_path` is not found: {}'.format(
                    train_path))
            else:
                file_path = train_path
        elif self.mode == 'val':
            if val_path is None:
                raise ValueError(
                    'When `mode` is "val", `val_path` is necessary, but it is None.'
                )
            elif not os.path.exists(val_path):
                raise FileNotFoundError('`val_path` is not found: {}'.format(
                    val_path))
            else:
                file_path = val_path
        else:
            if test_path is None:
                raise ValueError(
                    'When `mode` is "test", `test_path` is necessary, but it is None.'
                )
            elif not os.path.exists(test_path):
                raise FileNotFoundError('`test_path` is not found: {}'.format(
                    test_path))
            else:
                file_path = test_path

        with open(file_path, 'r') as f:
            for line in f:
                items = line.strip().split(separator)
                if len(items) != 2:
                    if self.mode == 'train' or self.mode == 'val':
                        raise ValueError(
                            "File list format incorrect! In training or evaluation task it should be"
                            " image_name{}label_name\\n".format(separator))
                    image_path = os.path.join(self.dataset_root, items[0])
                    label_path = None
                else:
                    image_path = os.path.join(self.dataset_root, items[0])
                    label_path = os.path.join(self.dataset_root, items[1])
                self.file_list.append([image_path, label_path])

    def __getitem__(self, idx):
        data = {}
        data['trans_info'] = []
        image_path, label_path = self.file_list[idx]
        data['img'] = image_path
        data['label'] = label_path
        # If key in gt_fields, the data[key] have transforms synchronous.
        data['gt_fields'] = []
        if self.mode == 'val':
            data = self.transforms(data)
            data['label'] = data['label'][np.newaxis, :, :]

        else:
            data['gt_fields'].append('label')
            data = self.transforms(data)
            if self.edge:
                edge_mask = F.mask_to_binary_edge(
                    data['label'], radius=2, num_classes=self.num_classes)
                data['edge'] = edge_mask
        return data

    def __len__(self):
        return len(self.file_list)
