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

from PIL import Image
from loguru import logger
from worker.dumper import dumper
from worker.task import Task
from trainx.typex import PreprocessTask
from worker.task import TaskProgress
from tools.image import thumbnail
from trainx.utils import get_tmp_local_path, Tmp, upload_files
from tools.file import zip_compress, zip_uncompress, find_files_from_dir
from trainx.text_inversion import preprocess_sub_dir

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
        resolution, min_reso, max_reso = get_batch_image_size(processed_dir)
        images = build_thumbnail_tag(processed_dir)
        p = TaskProgress.new_running(job, 'upload thumbnail', 90)
        yield p
        new_zip = build_zip(processed_dir)
        keys = upload_files(True, new_zip)
        task_result = {
            'images': list(images),
            'processed_key': keys[0] if keys else None,
            'resolution': resolution,
            'min_reso': min_reso,
            'max_reso': max_reso,
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
                    # 替换掉_为空格
                    images[basename]['tag'] = ' '.join(lines).replace('_', ' ')
                else:
                    # 提前预置了TXT，预处理后png名称会重命名。
                    def get_rename_image():
                        for k in images.keys():
                            if str(k).endswith(basename):
                                return k
                    rename = get_rename_image()
                    if not rename:
                        raise KeyError(f'cannot found image key:{basename}')
                    if not images[rename]['tag']:
                        # tag 已经存在可能是反推出来的说明要保留
                        images[rename]['tag'] = ' '.join(lines).replace('_', ' ')
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


def get_batch_image_size(target_dir):
    current = []
    min_reso, max_reso = 100000, -1
    for image_path in find_files_from_dir(target_dir, *ImagesEx):
        with Image.open(image_path) as image:
            if not current and current is not None:
                current = image.size
            elif current != image.size:
                current = None

            max_reso = max(max(image.size), max_reso)
            min_reso = min(min(image.size), min_reso)
    return [x for x in current or []], min_reso, max_reso


def debug_task():
    t = {
        'task_id': 'test_pre',
        'base_model_path': 'xingzheaidraw/sd-web/models/system/Lora/2023/05/22/478d15657df6464ce9cc07a40bde3c1d06cb2a74e51c21e2f458012aff3406eb.safetensors',
        'model_hash': '478d15657df6464ce9cc07a40bde3c1d06cb2a74e51c21e2f458012aff3406eb',
        'alwayson_scripts': {},
        'user_id': 'test_user',
        'task_type': 4,
        'minor_type': 1,
        "interrogate_model": "wd14-vit-v2",
        "process_width": 512,
        "process_height": 512,
        "preprocess_txt_action": "copy",
        "process_flip": False,
        "process_split": False,
        "split_threshold": 0.5,
        "overlap_ratio": 0.2,
        "process_focal_crop": False,
        "process_focal_crop_face_weight": 0.9,
        "process_focal_crop_entropy_weight": 0.15,
        "process_focal_crop_edges_weight": 0.5,
        "process_focal_crop_debug": False,
        "process_multicrop": False,
        "process_multicrop_mindim": 384,
        "process_multicrop_maxdim": 768,
        "process_multicrop_minarea": 4096,
        "process_multicrop_maxarea": 409600,
        "process_multicrop_objective": "Maximize area",
        "process_multicrop_threshold": 0.1,
        "process_keep_original_size": False,
        "zip_key": "xingzheaidraw/sd-tmp/2023/06/28/file-qjry3qvzjvzqkl.zip",
        "ignore": False,
        "regex_tokens": [
            {
                "sub_folder": "asd",
                "need": False
            }
        ],
        "resolution": None,
        "min_reso": 0,
        "max_reso": 0,
    }

    return Task(**t)