#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/5/8 3:19 PM
# @Author  : wangdongming
# @Site    : 
# @File    : typex.py
# @Software: Hifive
import json
import os.path
import time
import typing
from enum import IntEnum
from collections import UserDict
from worker.task import Task, TaskType
from .utils import get_tmp_local_path, Tmp
from tools.file import zip_uncompress, getdirsize, zip_compress
from modules.textual_inversion.preprocess import PreprocessTxtAction


class PreprocessTask(UserDict):

    @property
    def zip(self):
        return self.get('zip_key')

    @property
    def ignore(self):
        return self.get('ignore')

    @property
    def params(self):
        interrogate_model = self['interrogate_model']
        process_caption_deepbooru = interrogate_model == 'deepbooru'
        process_caption = interrogate_model == 'clip'
        return {
            'process_width': self['process_width'],
            'process_height': self['process_height'],
            'preprocess_txt_action': self.get('preprocess_txt_action', PreprocessTxtAction.Ignore),
            'process_flip': self['process_flip'],
            'process_split': self['process_split'],
            'process_caption': process_caption,
            'process_caption_deepbooru': process_caption_deepbooru,
            'split_threshold': self.get('split_threshold', 0.5),
            'overlap_ratio': self.get('overlap_ratio', 0.2),
            'process_focal_crop': self.get('process_focal_crop', False),
            'process_focal_crop_face_weight': self.get('process_focal_crop_face_weight', 0.9),
            'process_focal_crop_entropy_weight': self.get('process_focal_crop_entropy_weight', 0.15),
            'process_focal_crop_edges_weight': self.get('process_focal_crop_edges_weight', 0.5),
            'process_focal_crop_debug': self.get('process_focal_crop_debug', False),
            'process_multicrop': self.get('process_multicrop', False),
            'process_multicrop_mindim': self.get('process_multicrop_mindim', 384),
            'process_multicrop_maxdim': self.get('process_multicrop_maxdim', 768),
            'process_multicrop_minarea': self.get('process_multicrop_minarea', 4096),
            'process_multicrop_maxarea': self.get('process_multicrop_maxarea', 409600),
            'process_multicrop_objective': self.get('process_multicrop_objective', "Maximize area"),
            'process_multicrop_threshold': self.get('process_multicrop_threshold', 0.1),
        }

    def __init__(self, task: Task):
        super(PreprocessTask, self).__init__(task)

    @classmethod
    def debug_task(self):
        t = {
            'task_id': 'test_preprocess',
            'user_id': 'test_user',
            'task_type': TaskType.Train,
            'model_hash':  'train',
            'create_at': int(time.time()),
            'interrogate_model': 'deepbooru',
            'process_width': 512,
            'process_height': 512,
            'preprocess_txt_action': 'ignore',
            'process_split': False,
            'process_flip': False,
            'split_threshold': 0.5,
            'overlap_ratio': 0.2,
            'process_focal_crop': False,
            'process_focal_crop_face_weight': 0.9,
            'process_focal_crop_entropy_weight': 0.15,
            'process_focal_crop_edges_weight': 0.5,
            'process_focal_crop_debug': False,
            'process_multicrop': False,
            'process_multicrop_mindim': 384,
            'process_multicrop_maxdim': 768,
            'process_multicrop_minarea': 4096,
            'process_multicrop_maxarea': 409600,
            'process_multicrop_objective': "Maximize area",
            'process_multicrop_threshold': 0.1,
            'zip_key': '/data/wdm/lora_train_data.zip',
            'ignore': False,
        }
        return Task(**t)


class TrainMinorTaskType(IntEnum):
    Preprocess = 1
    Train = 2


# ===================================================================
# ============================== lora ===============================
# ===================================================================

class SerializationObj:

    def to_dict(self):
        pr = {}
        for name in dir(self):
            value = getattr(self, name)
            try:
                if not name.startswith('_') and not callable(value):
                    if hasattr(value, 'to_dict'):
                        to_dict_func = getattr(value, 'to_dict')
                        if callable(to_dict_func):
                            value = value.to_dict()
                    pr[name] = value
            except:
                pass
        return pr


class TrainLoraBaseConfig(SerializationObj):

    def __init__(self, task: Task):
        self.model_name = task.value('model_name', requires=True)
        self.model_space = task.value('model_space', requires=True)
        self.model_desc = task.value('model_desc', '')
        self.base_model = task.value('base_model', requires=True)
        self.base_lora = task.value('base_lora', '')


class TrainLoraTrainConfig(SerializationObj):

    def __init__(self, task: Task):
        self.resolution = task.value('resolution', requires=True)
        self.num_repeats = task.value('num_repeats', default=1)
        self.batch_size = task.value('batch_size', default=4)
        self.epoch = task.value('epoch', default=20)
        self.save_every_n_epochs = task.value('save_every_n_epochs', default=5)
        self.clip_skip = task.value('clip_skip', default=1)
        self.seed = task.value('seed', default=None)


class TrainLoraNetConfig(SerializationObj):

    def __init__(self, task: Task):
        self.network_dim = task.value('network_dim', default=32)
        self.network_alpha = task.value('network_alpha', default=1)
        self.optimizer_type = task.value('optimizer_type', default='AdamW8bit')
        self.unet_lr = task.value('unet_lr', default=0.0001)
        self.text_encoder_lr = task.value('text_encoder_lr', default=0.0001)
        self.learning_rate = task.value('learning_rate', default=0.0001)
        self.lr_scheduler = task.value('lr_scheduler', default='constant')
        self.lr_scheduler_num_cycles = task.value('lr_scheduler_num_cycles', default=1)
        train_module = task.value('train_module', default='')
        train_module = train_module.lower()
        self.network_train_text_encoder_only = train_module == 'encoder'
        self.network_train_unet_only = train_module == 'unet'


class TrainLoraParams(SerializationObj):

    def __init__(self, task: Task):
        self.base = TrainLoraBaseConfig(task)
        self.train = TrainLoraTrainConfig(task)
        self.net = TrainLoraNetConfig(task)


class TrainLoraTask(UserDict):

    def __init__(self, task: Task):
        super(TrainLoraTask, self).__init__(task)
        self.id = task.id
        self.orig_task = task

    @property
    def images(self):
        for item in self['images']:
            yield {
                'filename': item['filename'],
                'tag': item['tag'],
                'dirname': item['dirname']
            }

    @property
    def resolution(self):
        '''
        512 或者 [512,512]格式
        '''
        res = self['resolution']
        if isinstance(res, (list, tuple)):
            if len(res) > 1:
                return f'[{res[0]},{res[1]}]'
            else:
                return f'{res[0]}'
        elif isinstance(res, (str, bytes)):
            if isinstance(res, bytes):
                res = res.decode('utf8')
            if ',' in res:
                array = res.split(',')
                return f'[{array[0]},{array[1]}]'
        return str(res)

    @property
    def output_dir(self):
        return os.path.join(Tmp, self.id)

    @property
    def toml(self):
        return os.path.join(Tmp, self.id + '.toml')

    def image_dir(self):
        zip_file = get_tmp_local_path(self['processed_key'])
        image_dir = os.path.join(Tmp, self.id)
        if not os.path.isdir(image_dir) and getdirsize(image_dir) == 0:
            zip_uncompress(zip_file, image_dir)
        return image_dir

    def rewrite_caption(self, image_dir):
        for item in self.images:
            image_path = os.path.join(image_dir, item['dirname'], item['filename'])
            if os.path.isfile(image_path):
                name, _ = os.path.splitext(item['filename'])
                caption_path = os.path.join(image_dir, item['dirname'], name + '.txt')
                caption = item['tag']
                caption = str(caption) if not isinstance(caption, bytes) else caption.decode('utf8')
                caption = caption.strip().replace('\n', ' ')
                if caption:
                    with open(caption_path, 'w+') as f:
                        f.write(caption)

    def create_toml(self, image_dir):
        params = self.train_param()

        with open(self.toml, 'w+') as f:
            space_4 = ' ' * 4
            f.write('[general]\n')
            f.write('enable_bucket = false                       # 是否使用Aspect Ratio Bucketing\n')
            f.write('\n')
            f.write('[[datasets]]\n')
            f.write(f'resolution = {self.resolution}             # 学习分辨率\n')
            f.write(f'batch_size = {self["batch_size"]}          # 批量大小\n')
            f.write('\n')
            f.write(space_4 + '[[datasets.subsets]]\n')
            f.write(space_4 + f'image_dir = "{image_dir}"                 # 指定包含训练图像的文件夹\n')
            f.write(space_4 + f'class_tokens =             # 指定类别\n')
            f.write(space_4 + f'num_repeats = {params.train.num_repeats}  # 正则化图像的迭代次数，基本上1就可以了\n')
        return self.toml

    def train_param(self):
        if not hasattr(self, 'train_params'):
            self.train_params = TrainLoraParams(self.orig_task)
        return self.train_params

    def dump_train_config(self, image_dir):
        params = self.train_param()
        with open(os.path.join(image_dir, 'train.json'), "w+") as f:
            f.write(json.dumps(params.to_dict()))

    def build_command_args(self):
        image_dir = self.image_dir()
        self.rewrite_caption(image_dir)
        toml = self.create_toml(image_dir)
        params = self.train_param()
        base_model = get_tmp_local_path(params.base.base_model)
        base_lora = get_tmp_local_path(params.base.base_lora)

        args = [
            f'--pretrained_model_name_or_path="{base_model}"',
            f'--dataset_config="{toml}"',
            f'--output_dir="{self.output_dir}"',
            f'--output_name={params.base.model_name}',
            '--save_model_as=safetensors',
            '--prior_loss_weight=1.0',
            '--network_module=networks.lora',
            '--mixed_precision="fp16"',
            '--xformers',
            f'--max_train_epochs={params.train.epoch}',
            f'--save_every_n_epochs={params.train.save_every_n_epochs}',
            f'--train_batch_size={params.train.batch_size}',
            f'--learning_rate={params.net.learning_rate}',
            f'--text_encoder_lr={params.net.text_encoder_lr}',
            f'--lr_scheduler={params.net.lr_scheduler}',
            f'--network_dim={params.net.network_dim}',
            f'--network_alpha={params.net.network_alpha}',
            f'--optimizer_type="{params.net.optimizer_type}"',
            f'--network_alpha={params.net.network_alpha}',
            f'--lr_scheduler_num_cycles={params.net.lr_scheduler_num_cycles}',
            f'--seed={params.train.seed}',
            f'--clip_skip={params.train.clip_skip}',
            f'--network_weights={base_lora}'
        ]

        if params.net.network_train_text_encoder_only:
            args.append('--network_train_text_encoder_only')
        if params.net.network_train_unet_only:
            args.append('--network_train_unet_only')
        return args

    def compress_train_material(self):
        image_dir = self.image_dir()
        self.dump_train_config(image_dir)
        dst = os.path.join(Tmp, f'train-material-{self.id}.zip')
        zip_compress(image_dir, dst)
        return dst
