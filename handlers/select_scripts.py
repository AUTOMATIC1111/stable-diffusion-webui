#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/12/15 10:12 AM
# @Author  : wangdongming
# @Site    : 
# @File    : select_scripts.py
# @Software: xingzhe.ai
import sys
if sys.version.startswith('3.10'):
    from strenum import StrEnum
else:
    from enum import StrEnum


class SelectScriptNames(StrEnum):

    XYZ = "X/Y/Z plot"


class SupportedXYZAxis(StrEnum):
    PromptSR = "Prompt S/R"
    Sampler = "Sampler"
    Checkpoint = "Checkpoint name"
    Steps = 'Steps'
    CFG = 'CFG Scale'
    DenoisingStrength = 'Denoising'
    VAE = 'VAE'
    # 以下为自定义
    Lora = 'Lora'  # 如何确定LORA的权重
    LoraWeight = 'LoraWeight'  # 如何判断对应LORA

