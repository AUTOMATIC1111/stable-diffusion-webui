# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from paddleseg.utils.download import download_file_and_uncompress
from paddleseg.utils import seg_env
from paddleseg.cvlibs import manager
from paddleseg.transforms import Compose
from paddleseg.datasets import Dataset

URL = 'https://bj.bcebos.com/paddleseg/dataset/hrf/hrf.zip'


@manager.DATASETS.add_component
class HRF(Dataset):
    """
    The HRF dataset is a dataset for retinal vessel segmentation which comprises 45 images and is organized as 15 subsets. Each subset contains one healthy fundus image, one image of patient with diabetic retinopathy and one glaucoma image. The image sizes are 3,304 x 2,336, with a training/testing image split of 21/24.
    (https://doi.org/10.1155/2013/154860)

    Args:
        transforms (list): Transforms for image.
        dataset_root (str): The dataset directory. Default: None
        edge (bool): whether extract edge infor in the output
        mode (str, optional): Which part of dataset to use. it is one of ('train', 'val', 'test'). Default: 'train'.
    """
    NUM_CLASSES = 2
    IGNORE_INDEX = 255
    IMG_CHANNELS = 3

    def __init__(self,
                 dataset_root=None,
                 transforms=None,
                 edge=False,
                 mode='train'):
        self.dataset_root = dataset_root
        self.transforms = Compose(transforms)
        mode = mode.lower()
        self.mode = mode
        self.edge = edge
        self.file_list = list()
        self.num_classes = self.NUM_CLASSES
        self.ignore_index = self.IGNORE_INDEX

        if mode not in ['train', 'val', 'test']:
            raise ValueError(
                "`mode` should be 'train', 'val' or 'test', but got {}.".format(
                    mode))

        if self.transforms is None:
            raise ValueError("`transforms` is necessary, but it is None.")

        if self.dataset_root is None:
            self.dataset_root = download_file_and_uncompress(
                url=URL,
                savepath=seg_env.DATA_HOME,
                extrapath=seg_env.DATA_HOME)
        elif not os.path.exists(self.dataset_root):
            self.dataset_root = os.path.normpath(self.dataset_root)
            savepath, extraname = self.dataset_root.rsplit(
                sep=os.path.sep, maxsplit=1)
            self.dataset_root = download_file_and_uncompress(
                url=URL,
                savepath=savepath,
                extrapath=savepath,
                extraname=extraname)

        if mode == 'train':
            file_path = os.path.join(self.dataset_root, 'train_list.txt')
        elif mode == 'val':
            file_path = os.path.join(self.dataset_root, 'val_list.txt')

        with open(file_path, 'r') as f:
            for line in f:
                items = line.strip().split()
                if len(items) != 2:
                    if mode == 'train' or mode == 'val':
                        raise Exception(
                            "File list format incorrect! It should be"
                            " image_name label_name\\n")
                    image_path = os.path.join(self.dataset_root, items[0])
                    grt_path = None
                else:
                    image_path = os.path.join(self.dataset_root, items[0])
                    grt_path = os.path.join(self.dataset_root, items[1])
                self.file_list.append([image_path, grt_path])
