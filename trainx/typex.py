#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/5/8 3:19 PM
# @Author  : wangdongming
# @Site    : 
# @File    : typex.py
# @Software: Hifive
import json
import os.path
import shutil
import time
import typing
from enum import IntEnum
from collections import UserDict
from Crypto.Hash import SHA256
from worker.task import Task, TaskType
from .utils import get_tmp_local_path, Tmp, upload_files, ModelType, get_model_local_path
from tools.file import zip_uncompress, getdirsize, zip_compress, find_files_from_dir
from modules.textual_inversion.preprocess import PreprocessTxtAction


class PreprocessTask(UserDict):

    @property
    def zip(self):
        return self.get('zip_key')

    @property
    def id(self):
        return self["task_id"]

    @property
    def image_keys(self):
        return self.get('image_keys') or []

    @property
    def ignore(self):
        return self.get('ignore')

    @property
    def params(self):
        interrogate_model = str(self['interrogate_model'] or "")
        # process_caption_deepbooru = 'deepbooru' in interrogate_model
        # process_caption = 'clip' in interrogate_model

        process_caption_deepbooru = False
        process_caption = False
        caption_wd_interrogator_name = None

        for name in interrogate_model.split(','):
            name = name.strip().lower()
            if name == 'deepbooru':
                process_caption_deepbooru = True
            elif 'clip' == name:
                process_caption = True
            elif name.startswith('wd14-') or name.startswith('wd-v1-'):
                caption_wd_interrogator_name = name

        return {
            'process_width': self['process_width'],
            'process_height': self['process_height'],
            'preprocess_txt_action': self.get('preprocess_txt_action', PreprocessTxtAction.Ignore),
            'process_flip': self['process_flip'],
            'process_split': self['process_split'],
            'process_caption': process_caption,
            'process_caption_deepbooru': process_caption_deepbooru,
            'caption_wd_interrogator_name': caption_wd_interrogator_name,
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
            'process_keep_original_size': self.get('process_keep_original_size', False)
        }

    def __init__(self, task: Task):
        super(PreprocessTask, self).__init__(task)

    @classmethod
    def debug_task(self):
        t = {
            'task_id': 'test_preprocess-1',
            'user_id': 'test_user',
            'task_type': TaskType.Train,
            'minor_type': TrainMinorTaskType.Preprocess,
            'model_hash': 'train',
            'create_at': int(time.time()),
            'interrogate_model': 'deepbooru',
            'process_width': 512,
            'process_height': 512,
            'preprocess_txt_action': 'ignore',
            'process_split': False,
            'process_flip': False,
            'split_threshold': 0.5,
            'overlap_ratio': 0.2,
            'process_focal_crop': True,
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
            'zip_key': 'xingzheaidraw/sd-tmp/2023/05/16/file-qq6prnwx92kj51.zip',
            'ignore': False,
            'process_keep_original_size': False,
        }
        return Task(**t)

    def download_images_with_keys(self):
        local_files = []
        for i, image in enumerate(self.image_keys):
            if len(local_files) > 60:
                continue
            if isinstance(image, dict):
                image_key = image['key']
                target_dir = os.path.join(Tmp, self.id, image['relative_path'])
            else:
                image_key = image
                target_dir = os.path.join(Tmp, self.id)
            file = get_tmp_local_path(image_key, dir=target_dir)
            if os.path.isfile(file):
                local_files.append(file)
        return local_files


class TrainMinorTaskType(IntEnum):
    Preprocess = 1
    Lora = 2
    DigitalDoppelganger = 3


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
        self.group_id = task.value('group_id', requires=True)
        self.model_desc = task.value('model_desc', '')
        self.base_model = task.value('base_model', requires=True)
        self.base_lora = task.value('base_lora', '')
        self.class_tokens = task.value('class_tokens', requires=True) or []


class TrainLoraTrainConfig(SerializationObj):

    def __init__(self, task: Task):
        self.resolution = task.value('resolution', requires=True)
        num_repeats = task.value('num_repeats', requires=True)
        self.batch_size = task.value('batch_size', default=4)
        self.epoch = task.value('epoch', default=20)
        self.save_every_n_epochs = task.value('save_every_n_epochs', default=5)
        self.save_last_n_epochs = task.value('save_last_n_epochs', default=2)
        self.clip_skip = task.value('clip_skip', default=1)
        self.seed = task.value('seed', default=None)
        if isinstance(num_repeats, list):
            self.num_repeats = {}
            for item in num_repeats:
                self.num_repeats[item['sub_folder']] = item['num']
        elif isinstance(num_repeats, int):
            self.num_repeats = num_repeats
        else:
            raise TypeError('num_repeats type error')


class TrainLoraNetConfig(SerializationObj):

    def __init__(self, task: Task):
        self.network_dim = task.value('network_dim', default=32)
        self.network_alpha = task.value('network_alpha', default=1)
        self.optimizer_type = task.value('optimizer_type', default='Lion')
        self.unet_lr = task.value('unet_lr', default=0.0001)
        self.text_encoder_lr = task.value('text_encoder_lr', default=0.0001)
        self.learning_rate = task.value('learning_rate', default=0.0001)
        self.lr_scheduler = task.value('lr_scheduler', default='constant')
        self.lr_scheduler_num_cycles = task.value('lr_scheduler_num_cycles', default=1)
        self.lr_scheduler_power = task.get('lr_scheduler_power', 1)  # Polynomial power for polynomial scheduler
        train_module = task.value('train_module', default='')
        train_module = train_module.lower()
        self.network_train_text_encoder_only = train_module == 'encoder'
        self.network_train_unet_only = train_module == 'unet'
        self.unet_lr = self.get_lr(self.unet_lr)
        self.text_encoder_lr = self.get_lr(self.text_encoder_lr)
        self.learning_rate = self.get_lr(self.learning_rate)

    def get_lr(self, v):
        if v is None or v < 0:
            return 0
        return v


class AdvancedConfig(SerializationObj):

    def set_property_value(self, task: Task, name: str, default: typing.Any = None, requires: bool = False):
        v = task.value(name, default, requires)
        setattr(self, name, v)

    def __init__(self, task: Task):
        self.v2 = task.get('v2', False)
        self.v_parameterization = task.get('v_parameterization', False)
        self.save_precision = task.get('save_precision', None) or 'fp16'
        self.max_token_length = task.get('max_token_length', 75)  # (default for 75, 150 or 225)
        self.reg_tokens = task.get('reg_tokens', None)
        self.list_reg_data_dir = task.get('list_reg_data_dir', None)
        self.list_reg_repeats = task.get('list_reg_repeats', None)
        self.cache_latents = task.get('cache_latents', False)
        self.cache_latents_to_disk = task.get('cache_latents_to_disk', False)
        self.enable_bucket = task.get('enable_bucket', True)
        self.min_bucket_reso = task.get('min_bucket_reso', 256)
        self.max_bucket_reso = task.get('max_bucket_reso', 1024)
        self.bucket_reso_steps = task.get('bucket_reso_steps', 64)
        self.token_warmup_min = task.get('token_warmup_min', 1)
        self.token_warmup_step = task.get('token_warmup_step', 0)
        self.caption_dropout_rate = task.get('caption_dropout_rate', 0)
        self.caption_dropout_every_n_epochs = task.get('caption_dropout_every_n_epochs', 0)
        self.caption_tag_dropout_rate = task.get('caption_tag_dropout_rate', 0.0)  # 0~1
        self.shuffle_caption = task.get('shuffle_caption', False)  # shuffle comma-separated caption
        self.weighted_captions = task.get('weighted_captions', False)  # 使用带权重的 token，不推荐与 shuffle_caption 一同开启
        self.keep_tokens = task.get('keep_tokens', 0)
        self.color_aug = task.get('color_aug', False)
        self.flip_aug = task.get('flip_aug', False)
        self.face_crop_aug_range = task.get('face_crop_aug_range', None)
        self.random_crop = task.get('random_crop', False)
        self.lowram = task.get('lowram', True)
        self.mem_eff_attn = task.get('mem_eff_attn', False)
        self.xformers = task.get('xformers', os.getenv('XFORMERS', "0") == "1")
        self.vae = task.get('vae', None)
        self.set_property_value(task, 'max_data_loader_n_workers', 8)
        self.set_property_value(task, 'persistent_data_loader_workers', True)
        self.set_property_value(task, 'max_train_steps', 1600)
        self.set_property_value(task, 'gradient_checkpointing', True)
        self.set_property_value(task, 'gradient_accumulation_steps', 1)

        self.set_property_value(task, 'mixed_precision',  os.getenv("MIXED_PRECISION", "no"))
        self.set_property_value(task, 'full_fp16', True)
        self.set_property_value(task, 'enable_preview', False)
        self.set_property_value(task, 'sample_prompts', None)
        self.set_property_value(task, 'sample_sampler',
                                       'ddim')  # ["ddim","pndm","lms","euler","euler_a","heun","dpm_2","dpm_2_a","dpmsolver","dpmsolver++","dpmsingle","k_lms","k_euler","k_euler_a","k_dpm_2","k_dpm_2_a",]
        self.set_property_value(task, 'sample_every_n_epochs', None)
        self.set_property_value(task, 'network_module', "networks.lora")
        network_module = getattr(self, 'network_module')
        if network_module and not str(network_module).startswith('networks'):
            if network_module != "lycoris":
                setattr(self, 'network_module', 'networks.' + network_module)
            else:
                setattr(self, 'network_module', 'lycoris.kohya')

        self.set_property_value(task, 'conv_dim', None)
        self.set_property_value(task, 'conv_alpha', None)
        self.set_property_value(task, 'unit', 8)
        self.set_property_value(task, 'dropout', 0)
        self.set_property_value(task, 'algo', 'lora')  # ['lora','loha','lokr','ia3']
        self.set_property_value(task, 'enable_block_weights', False)
        self.set_property_value(task, 'block_dims', None)
        self.set_property_value(task, 'block_alphas', None)
        self.set_property_value(task, 'conv_block_dims', None)
        self.set_property_value(task, 'conv_block_alphas')
        self.set_property_value(task, 'down_lr_weight')
        self.set_property_value(task, 'mid_lr_weight')
        self.set_property_value(task, 'up_lr_weight')
        self.set_property_value(task, 'block_lr_zero_threshold', 0.0)
        self.set_property_value(task, 'weight_decay')
        self.set_property_value(task, 'betas')
        self.set_property_value(task, 'max_grad_norm', 1.0)
        self.set_property_value(task, 'prior_loss_weight', 1.0)
        self.set_property_value(task, 'min_snr_gamma')
        self.set_property_value(task, 'noise_offset')
        self.set_property_value(task, 'adaptive_noise_scale')
        self.set_property_value(task, 'multires_noise_iterations')
        self.set_property_value(task, 'multires_noise_discount', 0.3)


class TrainLoraParams(SerializationObj):

    def __init__(self, task: Task):
        self.base = TrainLoraBaseConfig(task)
        self.train = TrainLoraTrainConfig(task)
        self.net = TrainLoraNetConfig(task)
        self.advanced = AdvancedConfig(task)


class TrainLoraTask(UserDict):

    def __init__(self, task: Task):
        super(TrainLoraTask, self).__init__(task)
        self.orig_task = task

    @property
    def id(self):
        return self.orig_task.id

    @property
    def hash_id(self):
        return SHA256.new(self.id.encode()).hexdigest()

    @property
    def images(self):
        images = self.get('images') or []
        for item in images:
            yield {
                'filename': item.get('filename') or item['name'],
                'tag': item.get('tags') or item['tag'],
                'dirname': item.get('dirname') or item['folder']
            }

    @property
    def resolution(self):
        '''
        512 或者 512,512格式
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
                return f'{array[0].strip()},{array[1].strip()}'
        return str(res)

    def auto_batch_size(self):
        batch_size = self["batch_size"]
        # arr = self.resolution.split(",")
        # w, h = arr[0], arr[-1]
        #
        # r = int(w) * int(h)
        # if r < 512 * 768:
        #     return min(batch_size, 6)
        # elif r > 1024 * 1024:
        #     return min(batch_size, 4)

        return batch_size

    @property
    def output_dir(self):
        return os.path.join(Tmp, self.id)

    @property
    def toml(self):
        return os.path.join(Tmp, self.id + '.toml')

    def image_dir(self):
        # 兼容数字人训练，processed_key已经是本地路径
        if os.path.isdir(self['processed_key']):
            return self['processed_key']

        zip_file = get_tmp_local_path(self['processed_key'])
        image_dir = os.path.join(Tmp, self.id)
        if not os.path.isdir(image_dir) and getdirsize(image_dir) == 0:
            zip_uncompress(zip_file, image_dir)

        return image_dir

    def rewrite_caption(self, image_dir):
        for item in self.images:
            dirname = item.get('folder') or item['dirname']
            filename = item.get('name') or item['filename']
            image_path = os.path.join(image_dir, dirname, filename)
            if os.path.isfile(image_path):
                name, _ = os.path.splitext(filename)
                caption_path = os.path.join(image_dir, dirname, name + '.txt')
                caption = item['tag']
                caption = str(caption) if not isinstance(caption, bytes) else caption.decode('utf8')
                caption = list((x for x in caption.strip().replace('\n', ' ').split(',') if x))
                if caption:
                    with open(caption_path, 'w+') as f:
                        f.write(','.join(caption))
                else:
                    raise ValueError(f'cannot found caption:{filename}')

    def create_toml(self, image_dir):
        params = self.train_param()

        with open(self.toml, 'w+') as f:
            space_4 = ' ' * 4
            f.write('[general]\n')
            f.write('enable_bucket = false                       # 是否使用Aspect Ratio Bucketing\n')
            f.write('\n')
            f.write('[[datasets]]\n')
            f.write(f'resolution = {self.resolution}             # 学习分辨率\n')
            f.write(f'batch_size = {self.auto_batch_size()}          # 批量大小\n')
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

    def dump_train_config(self, image_dir, kwargs):
        data = dict((k, v) for k, v in kwargs.items())
        with open(os.path.join(image_dir, 'train.json'), "w+") as f:

            for key in ['pretrained_model_name_or_path', 'network_weights']:
                if key in data:
                    data[key] = '******.safetensors'
            f.write(json.dumps(data))

    def build_command_args(self):
        image_dir = self.image_dir()
        self.rewrite_caption(image_dir)
        # toml = self.create_toml(image_dir)
        params = self.train_param()
        base_model = get_model_local_path(params.base.base_model, ModelType.CheckPoint)
        base_lora = get_model_local_path(params.base.base_lora, ModelType.Lora) if params.base.base_lora else ''
        save_last_n_epochs = params.train.save_last_n_epochs

        class_tokens, list_train_data_dir, num_repeats = [], [], []
        for i, item in enumerate(self.train_params.base.class_tokens):
            sub_folder = item['sub_folder'].replace(' ', "")
            tokens = item['trigger']
            list_train_data_dir.append(os.path.join(image_dir, sub_folder))
            class_tokens.append(tokens)
            if isinstance(params.train.num_repeats, dict):
                num_repeats.append(params.train.num_repeats[sub_folder])
            else:
                num_repeats.append(params.train.num_repeats)

        reg_tokens, list_reg_data_dir, list_reg_repeats = [], [], []
        for i, item in enumerate(params.advanced.reg_tokens or []):
            sub_folder = item['sub_folder'].replace(' ', "")
            tokens = item['trigger']
            list_reg_repeats.append(os.path.join(image_dir, sub_folder))
            reg_tokens.append(tokens)
            if isinstance(params.advanced.list_reg_repeats, dict):
                num_repeats.append(params.advanced.list_reg_repeats[sub_folder])
            else:
                num_repeats.append(params.advanced.list_reg_repeats)
        key = self.orig_task.get('model_name') or self.hash_id[:16]

        kwargs = {
            'output_dir': self.output_dir,
            'pretrained_model_name_or_path': base_model,
            'network_weights': base_lora,
            # 'train_data_dir': image_dir,
            'list_train_data_dir': list_train_data_dir,
            'trigger_words': class_tokens,
            'output_name': key,
            'save_model_as': 'safetensors',
            'save_every_n_epochs': params.train.save_every_n_epochs,
            'batch_size': self.auto_batch_size(),
            'epoch': params.train.epoch,
            'resolution': self.resolution,
            'num_repeats': num_repeats,
            'clip_skip': params.train.clip_skip,
            'seed': params.train.seed if params.train.seed >= 1 else 1,
            'network_dim': params.net.network_dim,
            'network_alpha': params.net.network_alpha,
            'learning_rate': params.net.learning_rate,
            'unet_lr': params.net.unet_lr,
            'text_encoder_lr': params.net.text_encoder_lr,
            'optimizer_type': params.net.optimizer_type,
            'network_train_unet_only': params.net.network_train_unet_only,
            'network_train_text_encoder_only': params.net.network_train_text_encoder_only,
            # 'reg_data_dir': '',
            'save_last_n_epochs': save_last_n_epochs,
            'caption_extension': ".txt",
            'lr_scheduler': params.net.lr_scheduler,
            'lr_scheduler_num_cycles': params.net.lr_scheduler_num_cycles,

        }
        advanced = params.advanced.to_dict()
        for k, v in advanced.items():
            if v == -1 or v == '':
                advanced[k] = None

        kwargs.update({
            # reg_tokens, list_reg_data_dir, list_reg_repeats
            'list_reg_repeats': list_reg_repeats,
            'list_reg_data_dir': list_reg_data_dir,
            'reg_tokens': reg_tokens
        })
        kwargs.update(advanced)

        return kwargs

    def compress_train_material(self, train_log: str, kwargs: typing.Mapping):
        image_dir = self.image_dir()
        self.dump_train_config(image_dir, kwargs)
        with open('train_log', 'w+') as f:
            f.write(train_log)

        def filter_safetensors(x):
            _, ex = os.path.splitext(x)
            return str(ex).lower().lstrip('.') == 'safetensors'

        dst = os.path.join(Tmp, f'mater-{self.id}.zip')
        zip_compress(image_dir, dst, None)
        return dst

    def get_model_cover_key(self):
        image_dir = self.image_dir()
        if not hasattr(self, 'model_cover'):
            for file in find_files_from_dir(image_dir, 'png', 'jpg', 'jpeg'):
                key = upload_files(True, file)
                setattr(self, 'model_cover', key)
                if key:
                    return key[0]
        return getattr(self, 'model_cover')

    @classmethod
    def debug_task(cls):
        t = {
            'resolution': '512,512',
            'processed_key': 'xingzheaidraw/sd-web/resources/TrainLoraSamples.zip',
            'model_name': 'test_train(lora)',
            'group_id': 'group-x87qrm7mzm4wwp',
            'model_desc': 'test only',
            'base_model': 'xingzheaidraw/models/system/Stable-diffusion/2023/05/06/0389907e714c9239261269f21eb511a9585e4884c75d17ecafabc74b7c9baad8.ckpt',
            'num_repeats': [{'sub_folder': 'jpg2', "num": 1}],
            'batch_size': 4,
            'epoch': 20,
            'save_last_n_epochs': 10,
            'save_every_n_epochs': 2,
            'clip_skip': 1,
            'seed': 100001,
            'network_dim': 32,
            'network_alpha': 1,
            'optimizer_type': 'Lion',
            'train_module': 'all',
            'task_id': 'test_train_lora',
            'user_id': 'test_user',
            'task_type': TaskType.Train,
            'minor_type': TrainMinorTaskType.Lora,
            'model_hash': 'train',
            'create_at': int(time.time()),
            'class_tokens': [
                {
                    'sub_folder': 'jpg2',
                    'trigger': 'liuxiang'
                }
            ]
        }

        return Task(**t)


class DigitalDoppelgangerTask(PreprocessTask):

    def __init__(self, task: Task):
        super(DigitalDoppelgangerTask, self).__init__(task)
        self.train_type = task.value('train_type', 0)  # 自动训练类型
        self.output_dir = os.path.join(Tmp, self.id+"-out")
        os.makedirs(self.output_dir, exist_ok=True)
        self.general_model_path = "models"
        if not os.path.isdir(os.path.join("models", "tag_models")):
            raise OSError(f'cannot found tag model fodler:{ os.path.join("models", "tag_models")}')

        base_model = task.value('base_model', requires=True)
        self.base_model = get_model_local_path(base_model, ModelType.CheckPoint)
        self.input_dir = os.path.join(Tmp, self.id+"-in")
        os.makedirs(self.input_dir, exist_ok=True)

    def download_move_input_images(self):
        images = self.download_images_with_keys()
        if images:
            for img in images:
                shutil.move(img, os.path.join(self.input_dir, os.path.basename(img)))

            return self.input_dir

    def get_model_cover_key(self):
        image_dir = self.input_dir
        if not hasattr(self, 'model_cover'):
            for file in find_files_from_dir(image_dir, 'png', 'jpg', 'jpeg'):
                key = upload_files(True, file)
                setattr(self, 'model_cover', key)
                if key:
                    return key[0]
        return getattr(self, 'model_cover')

    @classmethod
    def debug_task(cls):
        t = {
            'model_name': 'test_train(Digital)',
            'group_id': 'group-x87qrm7mzm4wwp',
            'model_desc': 'test digital',
            'base_model': 'xingzheaidraw/models/system/Stable-diffusion/2023/05/06/0389907e714c9239261269f21eb511a9585e4884c75d17ecafabc74b7c9baad8.ckpt',
            'task_id': 'test_train_digital',
            'user_id': 'test_user',
            'task_type': TaskType.Train,
            'minor_type': TrainMinorTaskType.DigitalDoppelganger,
            'model_hash': 'train',
            'create_at': int(time.time()),
            'image_keys': [
                'xingzheaidraw/sd-web/output/admin/txt2img/samples/2023/08/07/low-t2i-e5ovy8elmre4wp_5-0.png'
            ]
        }

        return Task(**t)

