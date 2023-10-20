import os
import torch
from safetensors.torch import load_file
import typing
from handlers.txt2img import Txt2ImgTask, Txt2ImgTaskHandler
from worker.task import TaskType, TaskProgress, Task, TaskStatus
from copy import deepcopy
import numpy as np
from typing import List, Union, Dict, Set, Tuple
import cv2
import numpy as np
from modules.sd_models import CheckpointInfo


def load_pt_file(file):
    state_dict = torch.load(file, map_location="cpu")
    return state_dict

def load_ckpt_file(file):
    checkpoint = torch.load(file, map_location="cpu")
    state_dict = checkpoint['state_dict']
    return state_dict

def load_safetensors_file(file):
    sd = load_file(file)
    return sd

def base_model_version(keys, progress, lora_modules_num, all_modules_num):
    first_stage_count = 0
    cond_stage_count = 0
    diffusion_model_count = 0
    conditioner_stage_count = 0
    encoder_stage_count = 0
    decoder_stage_count = 0

    for key in keys:
        if "first_stage_model" in key:
            first_stage_count += 1
        elif "cond_stage_model" in key:
            cond_stage_count += 1
        elif "model.diffusion_model" in key:
            diffusion_model_count += 1
        elif "conditioner" in key:
            conditioner_stage_count += 1
        elif "encoder" in key:
            encoder_stage_count += 1
        elif "decoder" in key:
            decoder_stage_count += 1
            
    print("first_stage_count:", first_stage_count)
    print("cond_stage_count:", cond_stage_count)
    print("diffusion_model_count:", diffusion_model_count)
    print("conditioner_stage_count:", conditioner_stage_count)
    if cond_stage_count != 0 :
        progress.version = 1
        progress.cate = "base"
        print("该模型为sd1.5的底膜")
    elif conditioner_stage_count != 0 :
        progress.version = 2
        progress.cate = "base"
        print("该模型为sdxl的底膜")
    elif encoder_stage_count == 106 and decoder_stage_count == 138:
        progress.version = 2
        progress.cate = "vae"
        print("该模型为sdxl的vae")
    elif first_stage_count == 0:
        if lora_modules_num != 0:
            if lora_modules_num <= 700:
                progress.version = 1
                progress.cate = "lora"
                print("该模型为基于sd1.5 lora模型")
            elif lora_modules_num > 700:
                progress.version = 2
                progress.cate = "lora"
                print("该模型为基于XL lora模型")
        else:
            if all_modules_num == 6:
                print("该模型为基于sd1.5的embedding")
            elif all_modules_num == 2:
                print("该模型为基于sdxl的embedding")
    else:
        progress.version = 1
        print("该模型为未知模型")


def model_check_info(filename, sha256=None):
    checkpoint = CheckpointInfo(filename, sha256)
    return checkpoint


class ModelCheckTask(Txt2ImgTask):
    def __init__(self,
                base_model_path: str,
                 ):
        self.base_model_path = base_model_path


    @classmethod
    def exec_task(cls, task: Task):

        return task


class ModelCheckTaskType(Txt2ImgTask):
    ModelCheckAction = 1  


class ModelCheckTaskHandler(Txt2ImgTaskHandler):
    def __init__(self):
        super(ModelCheckTaskHandler, self).__init__()
        self.task_type = TaskType.ModelCheck

    def _exec(self, task: Task) -> typing.Iterable[TaskProgress]:
        # 根据任务的不同类型：执行不同的任务
            yield from self._exec_modelcheck(task)



    def _exec_modelcheck(self, task: Task) -> typing.Iterable[TaskProgress]:
        try:
            file = task['base_model_path']
            base_model_path = self._get_local_checkpoint(task)
            model_info = model_check_info(base_model_path, None) 
            print("model_info:", vars(model_info))

            print("打印task:",vars(task))
            progress = TaskProgress.new_ready(task, f'model loaded, run model_check...')
            yield progress
            base_model = model_info.filename
            print(f"loading: {base_model}")
            if os.path.splitext(base_model)[1] == ".safetensors":
                sd = load_safetensors_file(base_model)
            elif os.path.splitext(base_model)[1] == ".ckpt":
                sd = load_ckpt_file(base_model)
            elif os.path.splitext(base_model)[1] == ".pt":
                sd = load_pt_file(base_model)
            else:
                raise Exception("unknown file type")
        except Exception as e:
            progress.status = TaskStatus.Failed
            progress.task_desc = f'"无法识别的文件名后缀或模型无法正常加载" {e}'
            yield progress
            return

        try:
            values = []
            keys = list(sd.keys())

            for key in keys:
                if "lora_up" in key or "lora_down" in key:
                    values.append((key, sd[key]))
            lora_modules_num = len(values)
            print(f"number of LoRA modules: {lora_modules_num}")


            for key in [k for k in keys if k not in values]:
                values.append((key, sd[key]))
            all_modules_num = len(values)
            print(f"number of all modules: {all_modules_num}")
            

            base_model_version(keys, progress, lora_modules_num, all_modules_num)
            yield progress
        except Exception as e:
            progress.status = TaskStatus.Failed
            progress.task_desc = f'"模型正常加载但是检测失败了" {e}'
            yield progress