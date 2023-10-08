# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import os

import numpy as np
import time
import paddle
import paddle.nn.functional as F

from paddleseg.utils import metrics, TimeAverager, calculate_eta, logger, progbar
from paddleseg.core import infer

np.set_printoptions(suppress=True)


def evaluate(model,
             eval_dataset,
             aug_eval=False,
             scales=1.0,
             flip_horizontal=False,
             flip_vertical=False,
             is_slide=False,
             stride=None,
             crop_size=None,
             precision='fp32',
             amp_level='O1',
             num_workers=0,
             print_detail=True,
             auc_roc=False):
    """
    Launch evalution.

    Args:
        model（nn.Layer): A semantic segmentation model.
        eval_dataset (paddle.io.Dataset): Used to read and process validation datasets.
        aug_eval (bool, optional): Whether to use mulit-scales and flip augment for evaluation. Default: False.
        scales (list|float, optional): Scales for augment. It is valid when `aug_eval` is True. Default: 1.0.
        flip_horizontal (bool, optional): Whether to use flip horizontally augment. It is valid when `aug_eval` is True. Default: True.
        flip_vertical (bool, optional): Whether to use flip vertically augment. It is valid when `aug_eval` is True. Default: False.
        is_slide (bool, optional): Whether to evaluate by sliding window. Default: False.
        stride (tuple|list, optional): The stride of sliding window, the first is width and the second is height.
            It should be provided when `is_slide` is True.
        crop_size (tuple|list, optional):  The crop size of sliding window, the first is width and the second is height.
            It should be provided when `is_slide` is True.
        precision (str, optional): Use AMP if precision='fp16'. If precision='fp32', the evaluation is normal.
        amp_level (str, optional): Auto mixed precision level. Accepted values are “O1” and “O2”: O1 represent mixed precision, the input data type of each operator will be casted by white_list and black_list; O2 represent Pure fp16, all operators parameters and input data will be casted to fp16, except operators in black_list, don’t support fp16 kernel and batchnorm. Default is O1(amp)
        num_workers (int, optional): Num workers for data loader. Default: 0.
        print_detail (bool, optional): Whether to print detailed information about the evaluation process. Default: True.
        auc_roc(bool, optional): whether add auc_roc metric

    Returns:
        float: The mIoU of validation datasets.
        float: The accuracy of validation datasets.
    """
    model.eval()
    nranks = paddle.distributed.ParallelEnv().nranks
    local_rank = paddle.distributed.ParallelEnv().local_rank
    if nranks > 1:
        # Initialize parallel environment if not done.
        if not paddle.distributed.parallel.parallel_helper._is_parallel_ctx_initialized(
        ):
            paddle.distributed.init_parallel_env()
    batch_sampler = paddle.io.DistributedBatchSampler(
        eval_dataset, batch_size=1, shuffle=False, drop_last=False)
    loader = paddle.io.DataLoader(
        eval_dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        return_list=True, )

    total_iters = len(loader)
    intersect_area_all = paddle.zeros([1], dtype='int64')
    pred_area_all = paddle.zeros([1], dtype='int64')
    label_area_all = paddle.zeros([1], dtype='int64')
    logits_all = None
    label_all = None

    if print_detail:
        logger.info("Start evaluating (total_samples: {}, total_iters: {})...".
                    format(len(eval_dataset), total_iters))
    #TODO(chenguowei): fix log print error with multi-gpus
    progbar_val = progbar.Progbar(
        target=total_iters, verbose=1 if nranks < 2 else 2)
    reader_cost_averager = TimeAverager()
    batch_cost_averager = TimeAverager()
    batch_start = time.time()
    with paddle.no_grad():
        for iter, data in enumerate(loader):
            reader_cost_averager.record(time.time() - batch_start)
            label = data['label'].astype('int64')

            if aug_eval:
                if precision == 'fp16':
                    with paddle.amp.auto_cast(
                            level=amp_level,
                            enable=True,
                            custom_white_list={
                                "elementwise_add", "batch_norm",
                                "sync_batch_norm"
                            },
                            custom_black_list={'bilinear_interp_v2'}):
                        pred, logits = infer.aug_inference(
                            model,
                            data['img'],
                            trans_info=data['trans_info'],
                            scales=scales,
                            flip_horizontal=flip_horizontal,
                            flip_vertical=flip_vertical,
                            is_slide=is_slide,
                            stride=stride,
                            crop_size=crop_size)
                else:
                    pred, logits = infer.aug_inference(
                        model,
                        data['img'],
                        trans_info=data['trans_info'],
                        scales=scales,
                        flip_horizontal=flip_horizontal,
                        flip_vertical=flip_vertical,
                        is_slide=is_slide,
                        stride=stride,
                        crop_size=crop_size)
            else:
                if precision == 'fp16':
                    with paddle.amp.auto_cast(
                            level=amp_level,
                            enable=True,
                            custom_white_list={
                                "elementwise_add", "batch_norm",
                                "sync_batch_norm"
                            },
                            custom_black_list={'bilinear_interp_v2'}):
                        pred, logits = infer.inference(
                            model,
                            data['img'],
                            trans_info=data['trans_info'],
                            is_slide=is_slide,
                            stride=stride,
                            crop_size=crop_size)
                else:
                    pred, logits = infer.inference(
                        model,
                        data['img'],
                        trans_info=data['trans_info'],
                        is_slide=is_slide,
                        stride=stride,
                        crop_size=crop_size)

            intersect_area, pred_area, label_area = metrics.calculate_area(
                pred,
                label,
                eval_dataset.num_classes,
                ignore_index=eval_dataset.ignore_index)

            # Gather from all ranks
            if nranks > 1:
                intersect_area_list = []
                pred_area_list = []
                label_area_list = []
                paddle.distributed.all_gather(intersect_area_list,
                                              intersect_area)
                paddle.distributed.all_gather(pred_area_list, pred_area)
                paddle.distributed.all_gather(label_area_list, label_area)

                # Some image has been evaluated and should be eliminated in last iter
                if (iter + 1) * nranks > len(eval_dataset):
                    valid = len(eval_dataset) - iter * nranks
                    intersect_area_list = intersect_area_list[:valid]
                    pred_area_list = pred_area_list[:valid]
                    label_area_list = label_area_list[:valid]

                for i in range(len(intersect_area_list)):
                    intersect_area_all = intersect_area_all + intersect_area_list[
                        i]
                    pred_area_all = pred_area_all + pred_area_list[i]
                    label_area_all = label_area_all + label_area_list[i]
            else:
                intersect_area_all = intersect_area_all + intersect_area
                pred_area_all = pred_area_all + pred_area
                label_area_all = label_area_all + label_area

                if auc_roc:
                    logits = F.softmax(logits, axis=1)
                    if logits_all is None:
                        logits_all = logits.numpy()
                        label_all = label.numpy()
                    else:
                        logits_all = np.concatenate(
                            [logits_all, logits.numpy()])  # (KN, C, H, W)
                        label_all = np.concatenate([label_all, label.numpy()])

            batch_cost_averager.record(
                time.time() - batch_start, num_samples=len(label))
            batch_cost = batch_cost_averager.get_average()
            reader_cost = reader_cost_averager.get_average()

            if local_rank == 0 and print_detail:
                progbar_val.update(iter + 1, [('batch_cost', batch_cost),
                                              ('reader cost', reader_cost)])
            reader_cost_averager.reset()
            batch_cost_averager.reset()
            batch_start = time.time()

    metrics_input = (intersect_area_all, pred_area_all, label_area_all)
    class_iou, miou = metrics.mean_iou(*metrics_input)
    acc, class_precision, class_recall = metrics.class_measurement(
        *metrics_input)
    kappa = metrics.kappa(*metrics_input)
    class_dice, mdice = metrics.dice(*metrics_input)

    if auc_roc:
        auc_roc = metrics.auc_roc(
            logits_all, label_all, num_classes=eval_dataset.num_classes)
        auc_infor = ' Auc_roc: {:.4f}'.format(auc_roc)

    if print_detail:
        infor = "[EVAL] #Images: {} mIoU: {:.4f} Acc: {:.4f} Kappa: {:.4f} Dice: {:.4f}".format(
            len(eval_dataset), miou, acc, kappa, mdice)
        infor = infor + auc_infor if auc_roc else infor
        logger.info(infor)
        logger.info("[EVAL] Class IoU: \n" + str(np.round(class_iou, 4)))
        logger.info("[EVAL] Class Precision: \n" + str(
            np.round(class_precision, 4)))
        logger.info("[EVAL] Class Recall: \n" + str(np.round(class_recall, 4)))
    return miou, acc, class_iou, class_precision, kappa
