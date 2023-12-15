#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/12/15 10:12 AM
# @Author  : wangdongming
# @Site    : 
# @File    : select_scripts.py
# @Software: xingzhe.ai
import sys
import typing
from scripts.xyz_grid import axis_options

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
    Nothing = 'Nothing'
    # 以下为自定义
    Lora = 'Lora'  # 如何确定LORA的权重
    LoraWeight = 'LoraWeight'  # 如何判断对应LORA


DropdownAxisTypes = [
    SupportedXYZAxis.Sampler,
    SupportedXYZAxis.Checkpoint,
    SupportedXYZAxis.VAE
]


class AxisData:

    def __init__(self, axis_type, values):
        self.axis_type = SupportedXYZAxis(axis_type)
        self.values = values
        self.axis_type_idx = 0
        for i, opt in axis_options:
            if opt.label == axis_type:
                self.axis_type_idx = i
                break

    def tolist(self):
        values = [self.axis_type_idx, '', '']
        if self.axis_type in DropdownAxisTypes:
            index = -1
        else:
            index = 1
        values[index] = values
        return values


class XYZScriptArgs:

    def __init__(self,
                 axis_type_x: str,
                 axis_type_y: str,
                 axis_type_z: str,
                 axis_values_x: str,
                 axis_values_y: str,
                 axis_values_z: str,
                 draw_legend: bool = True,  # 显示轴类型和值
                 no_fixed_seeds: bool = False,  # 保持随机种子为-1
                 include_lone_images: bool = False,  # 预览次级图
                 include_sub_grids: bool = False,  # 预览次级宫格图
                 margin_size: int = 0,  # 宫格图边框像素
                 ):
        self.axis_x = AxisData(axis_type_x, axis_values_x)
        self.axis_y = AxisData(axis_type_y, axis_values_y)
        self.axis_z = AxisData(axis_type_z, axis_values_z)
        self.draw_legend = draw_legend
        self.margin_size = margin_size
        self.no_fixed_seeds = no_fixed_seeds
        self.include_sub_grids = include_sub_grids
        self.include_lone_images = include_lone_images

    def format_script_args(self):
        args = self.axis_x.tolist() + self.axis_y.tolist() + self.axis_z.tolist()
        args += [
            self.draw_legend,
            self.no_fixed_seeds,
            self.include_lone_images,
            self.include_sub_grids,
            self.margin_size,
            False,  # Use text inputs instead of dropdowns值固定设置为false
        ]
        return args

    @classmethod
    def from_dict(cls, args: typing.Dict):
        axis_type_x = args['axis_type_x']
        axis_values_x = args['axis_values_x']
        axis_type_y = args['axis_type_y']
        axis_values_y = args['axis_values_y']
        axis_type_z = args['axis_type_z']
        axis_values_z = args['axis_values_z']

        keys = [
            'axis_type_x',
            'axis_type_y',
            'axis_type_z',
            'axis_values_x',
            'axis_values_y',
            'axis_values_z',
        ]
        kwargs = {}
        for k in args:
            if k not in keys:
                kwargs[k] = args[k]

        return cls(
            axis_type_x,
            axis_type_y,
            axis_type_z,
            axis_values_x,
            axis_values_y,
            axis_values_z,
            **kwargs
        )



