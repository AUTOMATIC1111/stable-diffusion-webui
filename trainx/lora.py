#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/5/8 9:06 PM
# @Author  : wangdongming
# @Site    : 
# @File    : train_lora.py
# @Software: Hifive
import os.path
import shutil
import typing
import torch
import psutil

from Crypto.Hash import SHA256
from loguru import logger
from worker.task import Task, TaskType, TaskProgress, TrainEpoch
from sd_scripts.train_network_all import train_with_params
from .typex import TrainLoraTask
from .utils import upload_files
from worker.task_send import RedisSender
from multiprocessing import Process
from modules.devices import torch_gc


def get_train_models(train_lora_task: TrainLoraTask, model_name: str):
    models = []
    for file in os.listdir(train_lora_task.output_dir):
        full = os.path.join(train_lora_task.output_dir, file)
        if os.path.isfile(full) and file.startswith(model_name):
            _, ex = os.path.splitext(file)
            if ex.lower() == '.safetensors':
                models.append(full)

    def sort_model(module_path):
        basename, _ = os.path.splitext(os.path.basename(module_path))

        if '-' not in basename:
            return 10000001
        seg = basename.split('-')[-1]
        if seg == 'last':
            return 10000000
        try:
            x = int(seg)
        except:
            x = 0
        return x

    return sorted(models, key=sort_model)


def exec_train_lora_task(task: Task, dump_func: typing.Callable = None):
    train_lora_task = TrainLoraTask(task)
    kwargs = train_lora_task.build_command_args()
    p = TaskProgress.new_ready(task, 'ready')
    yield p
    logger.info("=============>>>> start train lora <<<<=============")
    logger.info(">>> command args:")
    for k, v in kwargs.items():
        logger.info(f"> args: {k}: {v}")
    logger.info("====================================================")
    p = TaskProgress.new_running(task, 'running', 0)

    def progress_callback(epoch, loss, num_train_epochs):
        print(f">>> update progress, epoch:{epoch},loss:{loss},len:{len(p.train.epoch)}")
        progress = epoch / num_train_epochs * 100 * 0.9
        p.train.add_epoch_log(TrainEpoch(epoch, loss))
        p.task_progress = progress
        if callable(dump_func):
            dump_func(p)

    ok = train_with_params(callback=progress_callback, **kwargs)
    # start_train_process(task, kwargs, dump_func)
    # local_models = get_train_models(train_lora_task, kwargs['output_name'])
    # ok = local_models is not None and len(local_models) > 0
    torch_gc()
    local_models = get_train_models(train_lora_task, kwargs['output_name'])
    if ok:
        logger.info("=============>>>> end of train <<<<=============")
        material = train_lora_task.compress_train_material(p.train.format_epoch_log())
        result = {
            'material': None,
            'models': []
        }

        cover = train_lora_task.get_model_cover_key()
        for m in local_models:
            # rename
            dirname = os.path.dirname(m)
            basename = os.path.basename(m)
            without, ex = os.path.splitext(basename)
            sha256 = SHA256.new(basename.encode()).hexdigest()
            array = without.split('-')
            epoch = array[-1] if len(array) > 1 else ''
            hash_file_path = os.path.join(dirname, sha256+ex)

            shutil.move(m, hash_file_path)
            key = upload_files(False, hash_file_path)
            result['models'].append({
                'key': key[0] if key else '',
                'thumbnail_path': cover,
                'hash': sha256,
                'epoch': epoch
            })

        if os.path.isfile(material):
            material_keys = upload_files(False, material)
            result['material'] = material_keys[0] if material_keys else ''
        # notify web server
        sender = RedisSender()
        sender.notify_train_task(task)

        fp = TaskProgress.new_finish(task, {
            'train': result
        }, True)
        fp.train = p.train

        yield fp
    else:
        p = TaskProgress.new_failed(task, 'train failed(unknown errors)')
        yield p


def do_train_with_process(task: Task, kwargs: typing.Mapping, dump_progress_cb: typing.Callable):
    p = TaskProgress.new_running(task, 'running', 0)

    def progress_callback(epoch, loss, num_train_epochs):
        print(f">>> update progress, epoch:{epoch},loss:{loss},len:{len(p.train.epoch)}")
        progress = epoch / num_train_epochs * 100 * 0.9
        p.train.add_epoch_log(TrainEpoch(epoch, loss))
        p.task_progress = progress
        if callable(dump_progress_cb):
            dump_progress_cb(p)

    ok = train_with_params(callback=progress_callback, **kwargs)
    if ok:
        train_lora_task = TrainLoraTask(task)
        logger.info("=============>>>> end of train <<<<=============")
        material = train_lora_task.compress_train_material(p.train.format_epoch_log())
        result = {
            'material': None,
            'models': []
        }

        local_models = get_train_models(train_lora_task, kwargs['output_name'])
        for m in local_models:
            # rename
            dirname = os.path.dirname(m)
            basename = os.path.basename(m)
            without, ex = os.path.splitext(basename)
            sha256 = SHA256.new(basename.encode()).hexdigest()
            array = without.split('-')
            epoch = array[-1] if len(array) > 1 else ''
            hash_file_path = os.path.join(dirname, sha256 + ex)

            shutil.move(m, hash_file_path)
            key = upload_files(False, hash_file_path)
            result['models'].append({
                'key': key[0] if key else '',
                'hash': sha256,
                'epoch': epoch
            })

        if os.path.isfile(material):
            material_keys = upload_files(False, material)
            result['material'] = material_keys[0] if material_keys else ''
        # notify web server
        sender = RedisSender()
        sender.notify_train_task(task)

        fp = TaskProgress.new_finish(task, {
            'train': result
        }, True)
        fp.train = p.train
        if callable(dump_progress_cb):
            dump_progress_cb(fp)
    else:
        p = TaskProgress.new_failed(task, 'train failed(unknown errors)')
        if callable(dump_progress_cb):
            dump_progress_cb(p)


def start_train_process(task: Task, kwargs: typing.Mapping, dump_progress_cb: typing.Callable):
    torch.multiprocessing.set_start_method('spawn')

    for p in psutil.process_iter():
        if 'train_worker' == p.name():
            p.kill()

    proc = Process(target=do_train_with_process,
                   name='train_worker',
                   args=(task, kwargs, dump_progress_cb))

    print(f'start sub process:{proc.pid}')
    proc.start()
    proc.join()

