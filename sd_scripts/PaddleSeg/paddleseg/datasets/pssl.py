# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import numpy as np

from paddleseg.datasets import Dataset
from paddleseg.cvlibs import manager
from paddleseg.transforms import Compose


@manager.DATASETS.add_component
class PSSLDataset(Dataset):
    """
    The PSSL dataset for segmentation. PSSL is short for Pseudo Semantic Segmentation Labels, where the pseudo label
    is computed by the Consensus explanation algorithm.

    The PSSL refers to "Distilling Ensemble of Explanations for Weakly-Supervised Pre-Training of Image Segmentation 
    Models" (https://arxiv.org/abs/2207.03335). 
    
    The Consensus explanation refers to "Cross-Model Consensus of Explanations and Beyond for Image Classification 
    Models: An Empirical Study" (https://arxiv.org/abs/2109.00707).

    To use this dataset, we need to additionally prepare the orignal ImageNet dataset, which has the folder structure
    as follows:

        imagenet_root
        |
        |--train
        |  |--n01440764
        |  |  |--n01440764_10026.JPEG
        |  |  |--...
        |  |--nxxxxxxxx
        |  |--...

    where only the "train" set is needed.

    The PSSL dataset has the folder structure as follows:

        pssl_root
        |
        |--train
        |  |--n01440764
        |  |  |--n01440764_10026.JPEG_eiseg.npz
        |  |  |--...
        |  |--nxxxxxxxx
        |  |--...
        |
        |--imagenet_lsvrc_2015_synsets.txt
        |--train.txt

    where "train.txt" and "imagenet_lsvrc_2015_synsets.txt" are included in the PSSL dataset.

    Args:
        transforms (list): Transforms for image.
        imagenet_root (str): The path to the original ImageNet dataset.
        pssl_root (str): The path to the PSSL dataset.
        mode (str, optional): Which part of dataset to use. it is one of ('train', 'val', 'test'). Default: 'train'.
        edge (bool, optional): Whether to compute edge while training. Default: False.
    """
    ignore_index = 1001  # 0~999 is target class, 1000 is bg
    NUM_CLASSES = 1001  # consider target class and bg
    IGNORE_INDEX = 1001
    IMG_CHANNELS = 3

    def __init__(self,
                 transforms,
                 imagenet_root,
                 pssl_root,
                 mode='train',
                 edge=False):
        mode = mode.lower()
        if mode not in ['train']:
            raise ValueError("mode should be 'train', but got {}.".format(mode))
        if transforms is None:
            raise ValueError("`transforms` is necessary, but it is None.")

        self.transforms = Compose(transforms)
        self.mode = mode
        self.edge = edge

        self.num_classes = self.NUM_CLASSES
        self.ignore_index = self.IGNORE_INDEX  # 1001
        self.file_list = []
        self.class_id_dict = {}

        if imagenet_root is None or not os.path.isdir(pssl_root):
            raise ValueError(
                "The dataset is not Found or the folder structure is nonconfoumance."
            )

        train_list_file = os.path.join(pssl_root, "train.txt")
        if not os.path.exists(train_list_file):
            raise ValueError("Train list file isn't exists.")
        for idx, line in enumerate(open(train_list_file)):
            # line: train/n04118776/n04118776_45912.JPEG_eiseg.npz
            label_path = line.strip()
            img_path = label_path.split('.JPEG')[0] + '.JPEG'
            label_path = os.path.join(pssl_root, label_path)
            img_path = os.path.join(imagenet_root, img_path)
            self.file_list.append([img_path, label_path])

        # mapping class name to class id.
        class_id_file = os.path.join(pssl_root,
                                     "imagenet_lsvrc_2015_synsets.txt")
        if not os.path.exists(class_id_file):
            raise ValueError("Class id file isn't exists.")
        for idx, line in enumerate(open(class_id_file)):
            class_name = line.strip()
            self.class_id_dict[class_name] = idx

    def __getitem__(self, idx):
        image_path, label_path = self.file_list[idx]
        data = {'img': image_path, 'trans_info': [], 'gt_fields': []}

        # get class_id by image path
        class_name = (image_path.split('/')[-1]).split('_')[0]
        class_id = self.class_id_dict[class_name]

        pssl_seg = np.load(label_path)['arr_0']
        gt_semantic_seg = np.zeros_like(pssl_seg, dtype=np.int64) + 1000
        # [0, 999] for imagenet classes, 1000 for background, others(-1) will be ignored during training.
        gt_semantic_seg[pssl_seg == 1] = class_id
        data['label'] = gt_semantic_seg

        if self.mode == 'train':
            data['gt_fields'].append('label')
        data = self.transforms(data)
        return data
