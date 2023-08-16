#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/14 9:55 AM
# @Author  : wangdongming
# @Site    : 
# @File    : tagger.py
# @Software: Hifive
import os.path
import random
import typing
from PIL import Image, UnidentifiedImageError
from enum import IntEnum
from worker.dumper import dumper
from modules.postprocessing import run_extras
from loguru import logger
from tools.image import compress_image
from worker.handler import DumpTaskHandler
from handlers.utils import get_tmp_local_path, Tmp, upload_files
from worker.task import Task, TaskType, TaskProgress, TaskStatus
from handlers.Tagger.interrogator import Interrogator, WaifuDiffusionInterrogator
from typing import List, Dict


# interrogators: Dict[str, Interrogator] = {}

interrogators = {
    'wd14-convnextv2-v2': WaifuDiffusionInterrogator(
        'wd14-convnextv2-v2',
        repo_id='SmilingWolf/wd-v1-4-convnextv2-tagger-v2',
        revision='v2.0'
    ),
    'wd14-vit-v2': WaifuDiffusionInterrogator(
        'wd14-vit-v2',
        repo_id='SmilingWolf/wd-v1-4-vit-tagger-v2',
        revision='v2.0'
    ),
    'wd14-convnext-v2': WaifuDiffusionInterrogator(
        'wd14-convnext-v2',
        repo_id='SmilingWolf/wd-v1-4-convnext-tagger-v2',
        revision='v2.0'
    ),
    'wd14-swinv2-v2': WaifuDiffusionInterrogator(
        'wd14-swinv2-v2',
        repo_id='SmilingWolf/wd-v1-4-swinv2-tagger-v2',
        revision='v2.0'
    ),
    'wd14-convnextv2-v2-git': WaifuDiffusionInterrogator(
        'wd14-convnextv2-v2',
        repo_id='SmilingWolf/wd-v1-4-convnextv2-tagger-v2',
    ),
    'wd14-vit-v2-git': WaifuDiffusionInterrogator(
        'wd14-vit-v2-git',
        repo_id='SmilingWolf/wd-v1-4-vit-tagger-v2'
    ),
    'wd14-convnext-v2-git': WaifuDiffusionInterrogator(
        'wd14-convnext-v2-git',
        repo_id='SmilingWolf/wd-v1-4-convnext-tagger-v2'
    ),
    'wd14-swinv2-v2-git': WaifuDiffusionInterrogator(
        'wd14-swinv2-v2-git',
        repo_id='SmilingWolf/wd-v1-4-swinv2-tagger-v2'
    ),
    'wd14-vit': WaifuDiffusionInterrogator(
        'wd14-vit',
        repo_id='SmilingWolf/wd-v1-4-vit-tagger'),
    'wd14-convnext': WaifuDiffusionInterrogator(
        'wd14-convnext',
        repo_id='SmilingWolf/wd-v1-4-convnext-tagger'
    ),
}


def get_tagger(
        images: List[str],  # 图片路径
        interrogator_name: str,  # 反推模型
        threshold: float,  # 反推阈值
        additional_tags: str,  # 附加标签
        exclude_tags: str,  # 排除标签
        sort_by_alphabetical_order: bool,  # 按照首字母排序
        add_confident_as_weight: bool = False,  # 将置信度作为权重写入生成的Tags (不推荐勾选)
        replace_underscore: bool = True,  # 使用空格替代下划线 (推荐勾选)
        replace_underscore_excludes: str = None,  # 排除项(逗号分隔)
        escape_tag: bool = True,  # 转义括号(防止误识别为权重信息，推荐勾选)
        unload_model_after_running: bool = True):  # 完成后从显存中卸载模型 (推荐勾选)

    tags_res = {}
    # 判断反推模型存在
    # interrogator = 'wd14-vit-v2-git'
    if interrogator_name not in interrogators:
        raise OSError(f'cannot found interrogator:{interrogator_name}')

    # 加载反推模型
    interrogator: Interrogator = interrogators[interrogator_name]

    def split_str(s: str, separator=',') -> List[str]:
        return [x.strip() for x in s.split(separator) if x]

    # 阈值参数
    postprocess_opts = (
        threshold,
        split_str(additional_tags),
        split_str(exclude_tags),
        sort_by_alphabetical_order,
        add_confident_as_weight,
        replace_underscore,
        split_str(replace_underscore_excludes or ''),
        escape_tag
    )

    for path in images:
        image = None
        try:
            local_image = get_tmp_local_path(path)
            if not (local_image and os.path.isfile(local_image)):
                raise OSError(f'cannot found image:{path}')
            image = Image.open(local_image)
        except UnidentifiedImageError:
            logger.exception(f'${image} is not supported image type, image path:{path}')
            continue
        ratings, tags = interrogator.interrogate(image)
        processed_tags = Interrogator.postprocess_tags(
            tags,
            *postprocess_opts
        )

        tags = ', '.join(processed_tags)
        if path not in tags_res.keys():
            basename = os.path.basename(local_image)
            tags_res[basename] = tags

    if unload_model_after_running:
        interrogator.unload()  # 卸载模型

    return tags_res


class TaggerTaskMinorTaskType(IntEnum):
    Default = 0
    SingleTagger = 1


class TaggerTask:

    # TODO 改成对应的参数
    def __init__(
            self,
            images: List[str],  # 图片路径
            interrogator: str,  # 反推模型
            threshold: float = 0.35,  # 反推阈值
            additional_tags: str = "",  # 附加标签
            exclude_tags: str = "",  # 排除标签
            sort_by_alphabetical_order: bool = False,  # 按照首字母排序
            add_confident_as_weight: bool = False,  # 将置信度作为权重写入生成的Tags (不推荐勾选)
            replace_underscore: bool = True,  # 使用空格替代下划线 (推荐勾选)
            replace_underscore_excludes: str = "",  # 排除项(逗号分隔)
            escape_tag: bool = True,  # 转义括号(防止误识别为权重信息，推荐勾选)
            unload_model_after_running: bool = True):
        self.images = list(images)
        self.interrogator = interrogator
        self.threshold = threshold
        self.additional_tags = additional_tags
        self.exclude_tags = exclude_tags
        self.sort_by_alphabetical_order = sort_by_alphabetical_order
        self.add_confident_as_weight = add_confident_as_weight
        self.replace_underscore = replace_underscore
        self.replace_underscore_excludes = replace_underscore_excludes
        self.escape_tag = escape_tag
        self.unload_model_after_running = unload_model_after_running

    # TODO 功能入口
    @classmethod
    def exec_task(cls, task: Task):
        t = TaggerTask(
            task['images'],  # task['images']
            task.get('interrogator'),
            task.get('threshold', 0.35),
            task.get('additional_tags', ""),
            task.get('exclude_tags', ""),
            task.get('sort_by_alphabetical_order', False),
            task.get('add_confident_as_weight', False),
            task.get('replace_underscore', True),
            task.get('replace_underscore_excludes', ""),
            task.get('escape_tag', True),
            task.get('unload_model_after_running', True),
        )
        # 改成Tageer反推
        return get_tagger(t.images, t.interrogator, t.threshold, t.additional_tags, t.exclude_tags,
                          t.sort_by_alphabetical_order,
                          t.add_confident_as_weight, t.replace_underscore, t.replace_underscore_excludes,
                          t.escape_tag, t.unload_model_after_running)


class TaggerTaskHandler(DumpTaskHandler):

    def __init__(self):
        super(TaggerTaskHandler, self).__init__(TaskType.Tagger)

    def _exec(self, task: Task) -> typing.Iterable[TaskProgress]:
        minor_type = TaggerTaskMinorTaskType(task.minor_type)
        if minor_type <= TaggerTaskMinorTaskType.SingleTagger:
            yield from self.__exec_Tagger_task(task)

    def __exec_Tagger_task(self, task: Task):
        p = TaskProgress.new_running(task, "tagger image ...")
        yield p
        result = TaggerTask.exec_task(task)

        p.task_progress = random.randint(30, 70)
        yield p
        if result:
            # 返回一个字典。{image_path:tag}
            p = TaskProgress.new_finish(task, result)
            p.task_desc = f'tagger task:{task.id} finished.'
        else:
            p.status = TaskStatus.Failed
            p.task_desc = f'tagger task:{task.id} failed.'
        yield p

    def _save_images(self, task: Task, r: typing.List):
        high, low = [], []
        if r:
            images = r[0]
            for image in images:
                filename = task.id + '.png'
                full_path = os.path.join(Tmp, filename)
                image.save(full_path)
                high.append(full_path)
                low_file = os.path.join(Tmp, 'low-' + filename)
                if not os.path.isfile(low_file):
                    compress_image(full_path, low_file)
                low.append(low_file)

        return high, low
