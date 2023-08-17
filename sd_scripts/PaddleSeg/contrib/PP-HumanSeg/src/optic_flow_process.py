# coding: utf8
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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

import numpy as np


def human_seg_tracking(pre_gray, cur_gray, prev_cfd, dl_weights, disflow):
    """计算光流跟踪匹配点和光流图
    输入参数:
        pre_gray: 上一帧灰度图
        cur_gray: 当前帧灰度图
        prev_cfd: 上一帧光流图
        dl_weights: 融合权重图
        disflow: 光流数据结构
    返回值:
        is_track: 光流点跟踪二值图，即是否具有光流点匹配
        track_cfd: 光流跟踪图
    """
    check_thres = 8
    h, w = pre_gray.shape[:2]
    track_cfd = np.zeros_like(prev_cfd)
    is_track = np.zeros_like(pre_gray)
    flow_fw = disflow.calc(pre_gray, cur_gray, None)
    flow_bw = disflow.calc(cur_gray, pre_gray, None)
    flow_fw = np.round(flow_fw).astype(np.int)
    flow_bw = np.round(flow_bw).astype(np.int)
    y_list = np.array(range(h))
    x_list = np.array(range(w))
    yv, xv = np.meshgrid(y_list, x_list)
    yv, xv = yv.T, xv.T
    cur_x = xv + flow_fw[:, :, 0]
    cur_y = yv + flow_fw[:, :, 1]

    # 超出边界不跟踪
    not_track = (cur_x < 0) + (cur_x >= w) + (cur_y < 0) + (cur_y >= h)
    flow_bw[~not_track] = flow_bw[cur_y[~not_track], cur_x[~not_track]]
    not_track += (np.square(flow_fw[:, :, 0] + flow_bw[:, :, 0]) +
                  np.square(flow_fw[:, :, 1] + flow_bw[:, :, 1])) >= check_thres
    track_cfd[cur_y[~not_track], cur_x[~not_track]] = prev_cfd[~not_track]

    is_track[cur_y[~not_track], cur_x[~not_track]] = 1

    not_flow = np.all(np.abs(flow_fw) == 0,
                      axis=-1) * np.all(np.abs(flow_bw) == 0, axis=-1)
    dl_weights[cur_y[not_flow], cur_x[not_flow]] = 0.05
    return track_cfd, is_track, dl_weights


def human_seg_track_fuse(track_cfd, dl_cfd, dl_weights, is_track):
    """光流追踪图和人像分割结构融合
    输入参数:
        track_cfd: 光流追踪图
        dl_cfd: 当前帧分割结果
        dl_weights: 融合权重图
        is_track: 光流点匹配二值图
    返回
        cur_cfd: 光流跟踪图和人像分割结果融合图
    """
    fusion_cfd = dl_cfd.copy()
    is_track = is_track.astype(np.bool)
    fusion_cfd[is_track] = dl_weights[is_track] * dl_cfd[is_track] + (
        1 - dl_weights[is_track]) * track_cfd[is_track]
    # 确定区域
    index_certain = ((dl_cfd > 0.9) + (dl_cfd < 0.1)) * is_track
    index_less01 = (dl_weights < 0.1) * index_certain
    fusion_cfd[index_less01] = 0.3 * dl_cfd[index_less01] + 0.7 * track_cfd[
        index_less01]
    index_larger09 = (dl_weights >= 0.1) * index_certain
    fusion_cfd[index_larger09] = 0.4 * dl_cfd[index_larger09] + 0.6 * track_cfd[
        index_larger09]
    return fusion_cfd


def threshold_mask(img, thresh_bg, thresh_fg):
    dst = (img / 255.0 - thresh_bg) / (thresh_fg - thresh_bg)
    dst[np.where(dst > 1)] = 1
    dst[np.where(dst < 0)] = 0
    return dst.astype(np.float32)


def optic_flow_process(cur_gray, scoremap, prev_gray, pre_cfd, disflow,
                       is_init):
    """光流优化
    Args:
        cur_gray : 当前帧灰度图
        pre_gray : 前一帧灰度图
        pre_cfd  ：前一帧融合结果
        scoremap : 当前帧分割结果
        difflow  : 光流
        is_init : 是否第一帧
    Returns:
        fusion_cfd : 光流追踪图和预测结果融合图
    """
    h, w = scoremap.shape
    cur_cfd = scoremap.copy()

    if is_init:
        if h <= 64 or w <= 64:
            disflow.setFinestScale(1)
        elif h <= 160 or w <= 160:
            disflow.setFinestScale(2)
        else:
            disflow.setFinestScale(3)
        fusion_cfd = cur_cfd
    else:
        weights = np.ones((h, w), np.float32) * 0.3
        track_cfd, is_track, weights = human_seg_tracking(
            prev_gray, cur_gray, pre_cfd, weights, disflow)
        fusion_cfd = human_seg_track_fuse(track_cfd, cur_cfd, weights, is_track)

    return fusion_cfd
