#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/5/8 12:37 PM
# @Author  : wangdongming
# @Site    : 
# @File    : preprocess.py
# @Software: Hifive
import copy
import os.path
from worker.dumper import dumper
from worker.task import Task
from .typex import PreprocessTask
from worker.task import TaskProgress
from tools.image import thumbnail
from .utils import get_tmp_local_path, Tmp, upload_files
from tools.file import zip_compress, zip_uncompress, find_files_from_dir
from modules.textual_inversion.preprocess import preprocess_sub_dir


def exec_preprocess_task(job: Task):
    task = PreprocessTask(job)
    tmp_zip = get_tmp_local_path(task.zip)

    if os.path.isfile(tmp_zip):
        target_dir = os.path.join(Tmp, job.id)
        os.makedirs(target_dir, exist_ok=True)
        zip_uncompress(tmp_zip, target_dir)
        p = TaskProgress.new_ready(job, 'ready preprocess')
        yield p
        processed_dir = os.path.join(Tmp, job.id + '-preprocessed')
        params = copy.copy(task.params)

        def progress_callback(progress):
            if progress > 95:
                return
            p = TaskProgress.new_running(job, 'run preprocess', progress)
            dumper.dump_task_progress(p)

        params.update(
            {
                'progress_cb': progress_callback
            }
        )

        if not task.ignore:
            preprocess_sub_dir(
                target_dir,
                processed_dir,
                **params
            )
        else:
            processed_dir = target_dir

        images = build_thumbnail_tag(processed_dir)
        new_zip = build_zip(processed_dir)
        keys = upload_files(True, new_zip)
        task_result = {
            'images': images,
            'processed_key': keys[0] if keys else None
        }
        p = TaskProgress.new_finish(job, task_result)
        yield p

    else:
        p = TaskProgress.new_failed(job, f'failed preprocess: cannot found zip:{task.zip}')
        yield p


def build_zip(target_dir):
    dirname = os.path.dirname(Tmp)
    filename = os.path.basename(target_dir) + '.zip'
    dst = os.path.join(dirname, filename)

    zip_compress(target_dir, dst)
    return dst


def build_thumbnail_tag(target_dir):
    images = {}
    tag_files = []
    for file in find_files_from_dir(target_dir, "txt", "png"):
        basename, ex = os.path.splitext(os.path.basename(file))
        ex = str(ex).lstrip('.').lower()
        if ex == 'png':
            thumbnail = create_thumbnail(file)
            thumbnail_key = upload_files(True, thumbnail)
            images[basename] = {
                'filename': os.path.basename(file),
                'dirname': os.path.dirname(file).replace(target_dir, '').lstrip('/'),
                'thumbnail': thumbnail_key
            }
        else:
            tag_files.append(file)

    for file in tag_files:
        basename, ex = os.path.splitext(os.path.basename(file))
        with open(file, "w") as f:
            lines = f.readlines()
            images[basename]['tag'] = ' '.join(lines)

    return images


def create_thumbnail(image_path):
    dirname = os.path.dirname(image_path)
    basename = os.path.dirname(image_path)
    dst = os.path.join(dirname, 'thumb-' + basename)
    thumbnail(image_path, dst)
    return dst
