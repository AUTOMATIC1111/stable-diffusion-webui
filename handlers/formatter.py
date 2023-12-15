#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/6 6:11 PM
# @Author  : wangdongming
# @Site    : 
# @File    : formatter.py
# @Software: Hifive
import typing
from modules.scripts import Script
from tools.reflection import find_classes
from handlers.select_scripts import SelectScriptNames, XYZScriptArgs


class AlwaysonScriptArgsFormatter:

    def name(self):
        return 'base'

    def format(self, is_img2img: bool, alwayson_scripts: typing.Union[typing.Sequence[typing.Any], typing.Mapping]) \
            -> typing.Sequence[typing.Any]:
        return alwayson_scripts


class SelectScriptArgsFormatter:

    def name(self):
        return 'select-script'

    def format(self, is_img2img: bool, selectable_script: Script, script_args: typing.Union[typing.Dict, typing.List]) \
            -> typing.Sequence[typing.Any]:
        if isinstance(script_args, dict):
            if selectable_script.name == SelectScriptNames.XYZ:
                xyz_script_args = XYZScriptArgs.from_dict(script_args)
                return xyz_script_args.format_script_args()

        return script_args


formatters = {

}


def init_formatters():
    for cls in find_classes('handlers.extension'):
        if issubclass(cls, AlwaysonScriptArgsFormatter):
            ins = cls()
            name = ins.name()
            if not ins.name() or name in formatters:
                continue

            formatters[ins.name()] = ins.format


def format_alwayson_script_args(name, is_img2img, args):
    if not formatters:
        init_formatters()
    name = name.replace('-', '_') if name not in formatters else name
    format = formatters.get(name)
    if not format:
        return args
    return format(is_img2img, args)


def format_select_script_args(is_img2img, select_script, script_args):
    formatter = SelectScriptArgsFormatter()
    return formatter.format(is_img2img, select_script, script_args)

