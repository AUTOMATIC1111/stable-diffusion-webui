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

from .dataset import Dataset
from paddleseg.utils.download import download_file_and_uncompress
from paddleseg.utils import seg_env
from paddleseg.cvlibs import manager
from paddleseg.transforms import Compose

URL = "https://paddleseg.bj.bcebos.com/dataset/MiniDeepGlobeRoadExtraction.zip"


@manager.DATASETS.add_component
class MiniDeepGlobeRoadExtraction(Dataset):
    """
    MiniDeepGlobeRoadExtraction dataset is extraced from DeepGlobe CVPR2018 challenge (http://deepglobe.org/)

    There are 800 images in the training set and 200 images in the validation set.

    Args:
        dataset_root (str, optional): The dataset directory. Default: None.
        transforms (list, optional): Transforms for image. Default: None.
        mode (str, optional): Which part of dataset to use. It is one of ('train', 'val'). Default: 'train'.
        edge (bool, optional): Whether to compute edge while training. Default: False.
    """
    NUM_CLASSES = 2
    IGNORE_INDEX = 255
    IMG_CHANNELS = 3

    def __init__(self,
                 dataset_root=None,
                 transforms=None,
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

        if mode not in ['train', 'val']:
            raise ValueError(
                "`mode` should be 'train' or 'val', but got {}.".format(mode))

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
            file_path = os.path.join(self.dataset_root, 'train.txt')
        else:
            file_path = os.path.join(self.dataset_root, 'val.txt')

        with open(file_path, 'r') as f:
            for line in f:
                items = line.strip().split('|')
                if len(items) != 2:
                    if mode == 'train' or mode == 'val':
                        raise Exception(
                            "File list format incorrect! It should be"
                            " image_name|label_name\\n")
                    image_path = os.path.join(self.dataset_root, items[0])
                    grt_path = None
                else:
                    image_path = os.path.join(self.dataset_root, items[0])
                    grt_path = os.path.join(self.dataset_root, items[1])
                self.file_list.append([image_path, grt_path])
