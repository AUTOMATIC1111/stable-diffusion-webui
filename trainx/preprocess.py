#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/5/8 12:37 PM
# @Author  : wangdongming
# @Site    : 
# @File    : preprocess.py
# @Software: Hifive
import copy
import os.path
import shutil

from loguru import logger
from worker.dumper import dumper
from worker.task import Task
from .typex import PreprocessTask
from worker.task import TaskProgress
from tools.image import thumbnail
from .utils import get_tmp_local_path, Tmp, upload_files
from tools.file import zip_compress, zip_uncompress, find_files_from_dir
from modules.textual_inversion.preprocess import preprocess_sub_dir

ImagesEx = ["png", 'jpeg', 'jpg']


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
            p = TaskProgress.new_running(job, 'run preprocess', int(progress * 0.7))
            dumper.dump_task_progress(p)

        params.update(
            {
                'progress_cb': progress_callback
            }
        )

        copy_captions(target_dir, processed_dir)
        if not task.ignore:
            preprocess_sub_dir(
                target_dir,
                processed_dir,
                **params
            )
        else:
            processed_dir = target_dir
        p = TaskProgress.new_running(job, 'upload thumbnail', 80)
        yield p
        images = build_thumbnail_tag(processed_dir)
        p = TaskProgress.new_running(job, 'upload thumbnail', 90)
        yield p
        new_zip = build_zip(processed_dir)
        keys = upload_files(True, new_zip)
        task_result = {
            'images': list(images),
            'processed_key': keys[0] if keys else None
        }
        p = TaskProgress.new_finish(job, task_result)
        yield p

    else:
        p = TaskProgress.new_failed(job, f'failed preprocess: cannot found zip:{task.zip}')
        yield p


def build_zip(target_dir):
    # dirname = os.path.dirname(Tmp)
    filename = os.path.basename(target_dir) + '.zip'
    dst = os.path.join(Tmp, filename)

    zip_compress(target_dir, dst)
    return dst


def build_thumbnail_tag(target_dir):
    images = {}
    tag_files = []
    for file in find_files_from_dir(target_dir, "txt", *ImagesEx):
        basename, ex = os.path.splitext(os.path.basename(file))
        dirname = os.path.dirname(file)
        ex = str(ex).lstrip('.').lower()
        if 'MACOSX' in dirname:
            continue
        if ex in ImagesEx:
            thumb = create_thumbnail(file)
            if thumb and os.path.isfile(thumb):
                thumbnail_key = upload_files(True, thumb)
                images[basename] = {
                    'filename': os.path.basename(file),
                    'dirname': os.path.dirname(file).replace(target_dir, '').lstrip('/'),
                    'thumbnail': thumbnail_key[0] if thumbnail_key else ''
                }

                os.remove(thumb)
        else:
            tag_files.append(file)

    for file in tag_files:
        basename, ex = os.path.splitext(os.path.basename(file))
        try:
            with open(file) as f:
                lines = f.readlines()
                if basename in images:
                    images[basename]['tag'] = ' '.join(lines)
                else:
                    # 提前预置了TXT，预处理后png名称会重命名。
                    def get_rename_image():
                        for k in images.keys():
                            if str(k).endswith(basename):
                                return k
                    rename = get_rename_image()
                    if not rename:
                        raise KeyError(f'cannot found image key:{basename}')

                    images[rename]['tag'] = ' '.join(lines)
        except Exception as ex:
            print(f'cannot read caption file:{file}, err:{ex}')

    return images.values()


def create_thumbnail(image_path):
    dirname = os.path.dirname(image_path)
    basename = os.path.basename(image_path)
    dst = os.path.join(dirname, 'thumb-' + basename)
    try:
        thumbnail(image_path, dst)
    except Exception as ex:
        logger.exception('cannot gen thumbnail')
    else:
        return dst


def copy_captions(src, dst):
    for file in find_files_from_dir(src, "txt"):
        basename = file.replace(src, '').lstrip('/')
        dirname = os.path.dirname(basename)
        os.makedirs(os.path.join(dst, dirname), exist_ok=True)
        shutil.copy(file, os.path.join(dst, basename))

