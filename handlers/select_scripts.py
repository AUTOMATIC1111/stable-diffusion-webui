#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/12/15 10:12 AM
# @Author  : wangdongming
# @Site    : 
# @File    : select_scripts.py
# @Software: xingzhe.ai
import sys
import typing
from scripts.xyz_grid import axis_options, AxisOption, format_value
from modules.shared import cmd_opts

if sys.version.startswith('3.10'):
    from strenum import StrEnum
else:
    from enum import StrEnum

PromptSRFlag = 'SR-'


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
    SingleLora = 'SingleLora'  # 一维LORA，values中含有每个Lora权重
    PlaneLora = 'PlaneLora'    # 二维LORA，需要与LoraWeight配合使用
    LoraWeight = 'LoraWeight'  # 需要与PlaneLora配合


DropdownAxisTypes = [
    SupportedXYZAxis.Sampler,
    SupportedXYZAxis.Checkpoint,
    SupportedXYZAxis.VAE
]


class AxisValue:

    def __init__(self, ):
        pass


class AxisValues:

    def __init__(self, axis_type: SupportedXYZAxis, axis_values: typing.Union[typing.Sequence, str]):
        self.values = axis_values
        if axis_type == SupportedXYZAxis.SingleLora:
            if not isinstance(axis_values, list):
                raise ValueError(f'single lora values type error, expect list but get {type(axis_values)}')
            values = [PromptSRFlag + SupportedXYZAxis.SingleLora]
            for item in axis_values:
                if isinstance(item, dict):  # 传入 [{"name": "xxx", "value": 0.8}, {"name": "yyy", "value": 0.8}]格式
                    values.append(f"<lora:{item['name']}:{item['value']}>")
                elif isinstance(item, tuple):  # 传入[("xxx", w), ("yyy", w2)] 格式
                    values.append(f"<lora:{item[0]}:{item[1]}>")
                elif isinstance(item, str):  # 传入["<lora:xxx:w>","<lora:yyy:w2>"]格式
                    values.append(item)
            self.values = ','.join(values)
        elif axis_type == SupportedXYZAxis.PlaneLora:
            if not isinstance(axis_values, list):
                raise ValueError(f'plane lora values type error, expect list but get {type(axis_values)}')
            values = [PromptSRFlag + SupportedXYZAxis.PlaneLora]
            for item in axis_values:
                if isinstance(item, dict):  # 传入 [{"name": "lora1"}, {"name": "lora2"}]格式
                    values.append(item['name'])
                elif isinstance(item, str):  # 传入["lora1", "lora2"...]格式
                    values.append(item)
            self.values = ','.join(values)
        elif axis_type == SupportedXYZAxis.LoraWeight:
            if not isinstance(axis_values, list):
                raise ValueError(f'plane lora values type error, expect list but get {type(axis_values)}')
            values = [PromptSRFlag + SupportedXYZAxis.LoraWeight]
            for item in axis_values:
                values.append(str(item))
            self.values = ','.join(values)


class AxisData:

    def __init__(self, axis_type, values):
        self.axis_type = SupportedXYZAxis(axis_type)
        self.data = AxisValues(self.axis_type, values)
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
        values[index] = self.data.values
        return values


class XYZScriptArgs:

    def __init__(self,
                 axis_type_x: typing.Union[SupportedXYZAxis, str],
                 axis_type_y: typing.Union[SupportedXYZAxis, str],
                 axis_type_z: typing.Union[SupportedXYZAxis, str],
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
        axis_type_x = args.get('axis_type_x', 'Nothing')
        axis_values_x = args.get('axis_values_x', '')
        axis_type_y = args.get('axis_type_y', 'Nothing')  # Nothing
        axis_values_y = args.get('axis_values_y', '')  # ""
        axis_type_z = args.get('axis_type_z', 'Nothing')
        axis_values_z = args.get('axis_values_z', '')

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


def apply_prompt(p, x, xs):
    p.prompt = p.prompt.replace(xs[0], x)
    p.negative_prompt = p.negative_prompt.replace(xs[0], x)


def apply_single_lora(p, x, xs):
    flag = xs[0]
    if flag not in p.prompt:
        p.prompt += flag
    if x == flag:
        x = ''  # none
    apply_prompt(p, x, xs)


def apply_plane_lora(p, x, xs):
    flag = xs[0]
    if flag not in p.prompt:
        p.prompt += f"<lora:{flag}:{PromptSRFlag + SupportedXYZAxis.LoraWeight}>"
    if x == flag:
        x = 'none'  # none
    apply_prompt(p, x, xs)


def apply_lora_weights(p, x, xs):
    flag = xs[0]
    if flag not in p.prompt:
        p.prompt += f"<lora:{flag}:{PromptSRFlag + SupportedXYZAxis.LoraWeight}>"
    if x == flag:
        x = "0"  # none
    apply_prompt(p, x, xs)


if cmd_opts.worker:
    axis_options.extend([
        AxisOption(SupportedXYZAxis.SingleLora, str, apply_single_lora, format_value=format_value),
        AxisOption(SupportedXYZAxis.PlaneLora, str, apply_plane_lora, format_value=format_value),
        AxisOption(SupportedXYZAxis.LoraWeight, str, apply_lora_weights, format_value=format_value),

    ])

