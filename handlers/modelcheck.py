import os
import traceback

import torch
import typing
from safetensors.torch import load_file
from handlers.txt2img import Txt2ImgTask, Txt2ImgTaskHandler
from worker.task import TaskType, TaskProgress, Task, TaskStatus, SerializationObj
from loguru import logger
from enum import IntEnum
from modules.sd_models import CheckpointInfo
from handlers.typex import ModelType


class SdModelVer(IntEnum):
    Unknown = -1
    SD15 = 1
    SDXL = 2


class CheckResult(SerializationObj):

    def __init__(self, version: SdModelVer = SdModelVer.SD15, model_type: ModelType = ModelType.CheckPoint):
        self.version = version
        self.category = model_type.name
        self.model_type = model_type


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


def base_model_version(keys, lora_modules_num, all_modules_num) -> typing.Optional[CheckResult]:
    first_stage_count = 0
    cond_stage_count = 0
    diffusion_model_count = 0
    conditioner_stage_count = 0
    encoder_stage_count = 0
    decoder_stage_count = 0
    res = None

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

    logger.debug("first_stage_count:", first_stage_count)
    logger.debug("cond_stage_count:", cond_stage_count)
    logger.debug("diffusion_model_count:", diffusion_model_count)
    logger.debug("conditioner_stage_count:", conditioner_stage_count)

    if cond_stage_count != 0:
        logger.debug("该模型为sd1.5的底膜")
        res = CheckResult()
    elif conditioner_stage_count != 0:
        res = CheckResult(SdModelVer.SDXL)
        logger.debug("该模型为sdxl的底膜")
    # elif encoder_stage_count == 106 and decoder_stage_count == 138:
    #     res = CheckResult(SdModelVer.SDXL, ModelType.VAE)
    #     logger.debug("该模型为sdxl的vae")
    # # todo: 是否缺失1.5版本VAE
    elif first_stage_count == 0:
        if lora_modules_num != 0:
            if lora_modules_num <= 1050:
                res = CheckResult(SdModelVer.SD15, ModelType.Lora)
                logger.debug("该模型为基于sd1.5 lora模型")
            elif lora_modules_num > 1050:
                res = CheckResult(SdModelVer.SDXL, ModelType.Lora)
                logger.debug("该模型为基于XL lora模型")
        else:
            if all_modules_num == 6:
                res = CheckResult(SdModelVer.SD15, ModelType.Embedding)
                logger.debug("该模型为基于sd1.5的embedding")
            elif all_modules_num == 2:
                res = CheckResult(SdModelVer.SDXL, ModelType.Embedding)
                logger.debug("该模型为基于sdxl的embedding")
    else:
        logger.debug("该模型为未知模型")
        res = CheckResult(SdModelVer.Unknown, ModelType.Unknown)
    return res


def model_check_info(filename, sha256=None):
    checkpoint = CheckpointInfo(filename, sha256)
    return checkpoint


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
            base_model_path = self._get_local_checkpoint(task)
            progress = TaskProgress.new_ready(task, f'model loaded, run model_check...')
            yield progress

            model_info = model_check_info(base_model_path, None)
            print("model_info:", vars(model_info))
            print("打印task:", vars(task))

            progress = TaskProgress.new_running(task, f'task running, run model_check...')
            yield progress

            base_model = model_info.filename
            logger.info(f"loading: {base_model}")
            ex = os.path.splitext(base_model)[-1].lower()

            if ex == ".safetensors":
                sd = load_safetensors_file(base_model)
            elif ex == ".ckpt":
                sd = load_ckpt_file(base_model)
            elif ex == ".pt":
                sd = load_pt_file(base_model)
            else:
                raise Exception("unknown file type")

        except Exception as e:
            task_desc = f'"无法识别的文件名后缀或模型无法正常加载" {e}'
            progress = TaskProgress.new_failed(task, task_desc, traceback.format_exc())
            yield progress
            return

        try:
            values = []
            keys = list(sd.keys())

            for key in keys:
                if str(key).startswith('lora_'):
                    values.append((key, sd[key]))
            lora_modules_num = len(values)
            logger.info(f"number of LoRA modules: {lora_modules_num}")

            for key in [k for k in keys if k not in values]:
                values.append((key, sd[key]))
            all_modules_num = len(values)
            logger.info(f"number of all modules: {all_modules_num}")

            r = base_model_version(keys, lora_modules_num, all_modules_num)
            if r and r.model_type == ModelType.Unknown:
                progress = TaskProgress.new_failed(task, "未能识别")
                yield progress
            else:
                result = r.to_dict() if r else {}
                progress.set_finish_result(result)
                yield progress

        except Exception as e:
            task_desc = f'"模型正常加载但是检测失败了" {e}'
            progress = TaskProgress.new_failed(task, task_desc, traceback.format_exc())
            yield progress
