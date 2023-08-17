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
import copy

import cv2
import numpy as np

from paddleseg.cvlibs import manager
from paddleseg.transforms import Compose
from paddleseg.datasets import Dataset
from paddleseg.utils.download import download_file_and_uncompress
from paddleseg.utils import seg_env
import paddleseg.transforms.functional as F

URL = "https://paddleseg.bj.bcebos.com/dataset/Supervisely_face.zip"


@manager.DATASETS.add_component
class SUPERVISELY(Dataset):
    """
    Supervise.ly dataset `https://supervise.ly/`.

    Args:
        common_transforms (list): A list of common image transformations for two inputs of portrait net.
        transforms1 (list): A list of image transformations for the first input of portrait net.
        transforms2 (list): A list of image transformations for the second input of portrait net.
        dataset_root (str, optional): The Supervise.ly dataset directory. Default: None.
        mode (str, optional): A subset of the entire dataset. It should be one of ('train', 'val'). Default: 'train'.
        edge (bool, optional): Whether to compute edge while training. Default: False
    """
    NUM_CLASSES = 2
    IGNORE_INDEX = 255
    IMG_CHANNELS = 3

    def __init__(self,
                 common_transforms,
                 transforms1,
                 transforms2,
                 dataset_root=None,
                 mode='train',
                 edge=False):
        self.dataset_root = dataset_root
        self.common_transforms = Compose(common_transforms)
        self.transforms = self.common_transforms
        if transforms1 is not None:
            self.transforms1 = Compose(transforms1, to_rgb=False)
        if transforms2 is not None:
            self.transforms2 = Compose(transforms2, to_rgb=False)
        mode = mode.lower()
        self.ignore_index = self.IGNORE_INDEX
        self.mode = mode
        self.num_classes = self.NUM_CLASSES
        self.input_width = 224
        self.input_height = 224

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
            path = os.path.join(dataset_root, 'supervisely_face_train_easy.txt')
        else:
            path = os.path.join(dataset_root, 'supervisely_face_test_easy.txt')
        with open(path, 'r') as f:
            files = f.readlines()
        files = ["/".join(file.split('/')[1:]) for file in files]
        img_files = [os.path.join(dataset_root, file).strip() for file in files]
        label_files = [
            os.path.join(dataset_root, file.replace('/img/', '/ann/')).strip()
            for file in files
        ]

        self.file_list = [
            [img_path, label_path]
            for img_path, label_path in zip(img_files, label_files)
        ]

    def __getitem__(self, item):
        image_path, label_path = self.file_list[item]
        im = cv2.imread(image_path)
        label = cv2.imread(label_path, 0)
        label[label > 0] = 1

        if self.mode == "val":
            common_data = self.common_transforms(dict(img=im, label=label))
            common_im, label = common_data['img'], common_data['label']
            im = np.float32(common_im[::-1, :, :])  # RGB => BGR
            im_aug = copy.deepcopy(im)
        else:
            common_data = self.common_transforms(dict(img=im, label=label))
            common_im, label = common_data['img'], common_data['label']
            common_im = np.transpose(common_im, [1, 2, 0])
            # add augmentation
            data = self.transforms1(dict(img=common_im))
            im = data['img']
            data = self.transforms2(dict(img=common_im))
            im_aug = data['img']

            im = np.float32(im[::-1, :, :])  # RGB => BGR
            im_aug = np.float32(im_aug[::-1, :, :])  # RGB => BGR

        label = cv2.resize(
            np.uint8(label), (self.input_width, self.input_height),
            interpolation=cv2.INTER_NEAREST)

        # add mask blur
        label = np.uint8(cv2.blur(label, (5, 5)))
        label[label >= 0.5] = 1
        label[label < 0.5] = 0

        edge_mask = F.mask_to_binary_edge(
            label, radius=4, num_classes=self.num_classes)
        edge_mask = np.transpose(edge_mask, [1, 2, 0]).squeeze(axis=-1)
        #im = np.concatenate([im_aug, im])
        if self.mode == "train":
            return dict(img=im, label=label, edge=edge_mask)
        else:
            return dict(img=im, label=label)
