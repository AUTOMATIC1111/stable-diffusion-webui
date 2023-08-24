# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import codecs
import os
import sys
import time

import yaml
import numpy as np
import cv2
import paddle
from paddle.inference import create_predictor, PrecisionType
from paddle.inference import Config as PredictConfig

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../../../')))

import paddleseg.transforms as T
from paddleseg.core.infer import reverse_transform
from paddleseg.cvlibs import manager
from paddleseg.utils import TimeAverager

from optic_flow_process import optic_flow_process


class DeployConfig:
    def __init__(self, path, vertical_screen):
        with codecs.open(path, 'r', 'utf-8') as file:
            self.dic = yaml.load(file, Loader=yaml.FullLoader)

            [width, height] = self.dic['Deploy']['transforms'][0]['target_size']
            if vertical_screen and width > height:
                self.dic['Deploy']['transforms'][0][
                    'target_size'] = [height, width]

        self._transforms = self._load_transforms(self.dic['Deploy'][
            'transforms'])
        self._dir = os.path.dirname(path)

    @property
    def transforms(self):
        return self._transforms

    @property
    def model(self):
        return os.path.join(self._dir, self.dic['Deploy']['model'])

    @property
    def params(self):
        return os.path.join(self._dir, self.dic['Deploy']['params'])

    def target_size(self):
        [width, height] = self.dic['Deploy']['transforms'][0]['target_size']
        return [width, height]

    def _load_transforms(self, t_list):
        com = manager.TRANSFORMS
        transforms = []
        for t in t_list:
            ctype = t.pop('type')
            transforms.append(com[ctype](**t))

        return transforms


class Predictor:
    def __init__(self, args):
        self.args = args
        self.cfg = DeployConfig(args.config, args.vertical_screen)
        self.compose = T.Compose(self.cfg.transforms)

        pred_cfg = PredictConfig(self.cfg.model, self.cfg.params)
        pred_cfg.disable_glog_info()
        if self.args.use_gpu:
            pred_cfg.enable_use_gpu(100, 0)

        self.predictor = create_predictor(pred_cfg)
        if self.args.test_speed:
            self.cost_averager = TimeAverager()

        if args.use_optic_flow:

            self.disflow = cv2.DISOpticalFlow_create(
                cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
            width, height = self.cfg.target_size()
            self.prev_gray = np.zeros((height, width), np.uint8)
            self.prev_cfd = np.zeros((height, width), np.float32)
            self.is_first_frame = True

    def run(self, img, bg):
        input_names = self.predictor.get_input_names()
        input_handle = self.predictor.get_input_handle(input_names[0])

        data = self.compose({'img': img})
        input_data = np.array([data['img']])

        input_handle.reshape(input_data.shape)
        input_handle.copy_from_cpu(input_data)
        if self.args.test_speed:
            start = time.time()

        self.predictor.run()

        if self.args.test_speed:
            self.cost_averager.record(time.time() - start)
        output_names = self.predictor.get_output_names()
        output_handle = self.predictor.get_output_handle(output_names[0])
        output = output_handle.copy_to_cpu()

        return self.postprocess(output, img, data, bg)

    def postprocess(self, pred_img, origin_img, data, bg):
        trans_info = data['trans_info']
        score_map = pred_img[0, 1, :, :]

        # post process
        if self.args.use_post_process:
            mask_original = score_map.copy()
            mask_original = (mask_original * 255).astype("uint8")
            _, mask_thr = cv2.threshold(mask_original, 240, 1,
                                        cv2.THRESH_BINARY)
            kernel_erode = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
            kernel_dilate = cv2.getStructuringElement(cv2.MORPH_CROSS, (25, 25))
            mask_erode = cv2.erode(mask_thr, kernel_erode)
            mask_dilate = cv2.dilate(mask_erode, kernel_dilate)
            score_map *= mask_dilate

        # optical flow
        if self.args.use_optic_flow:
            score_map = 255 * score_map
            cur_gray = cv2.cvtColor(origin_img, cv2.COLOR_BGR2GRAY)
            cur_gray = cv2.resize(cur_gray,
                                  (pred_img.shape[-1], pred_img.shape[-2]))
            optflow_map = optic_flow_process(cur_gray, score_map, self.prev_gray, self.prev_cfd, \
                    self.disflow, self.is_first_frame)
            self.prev_gray = cur_gray.copy()
            self.prev_cfd = optflow_map.copy()
            self.is_first_frame = False
            score_map = optflow_map / 255.

        score_map = score_map[np.newaxis, np.newaxis, ...]
        score_map = reverse_transform(
            paddle.to_tensor(score_map), trans_info, mode='bilinear')
        alpha = np.transpose(score_map.numpy().squeeze(1), [1, 2, 0])

        h, w, _ = origin_img.shape
        bg = cv2.resize(bg, (w, h))
        if bg.ndim == 2:
            bg = bg[..., np.newaxis]

        out = (alpha * origin_img + (1 - alpha) * bg).astype(np.uint8)
        return out
