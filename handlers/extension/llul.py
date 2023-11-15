#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/13 1:59 PM
# @Author  : wangdongming
# @Site    :
# @File    : llul.py
# @Software: Hifive
import collections
import os.path
import typing
import tempfile
from handlers.utils import get_tmp_local_path
from handlers.formatter import AlwaysonScriptArgsFormatter

LLuL = 'LLuL'


class LLuLFormatter(AlwaysonScriptArgsFormatter):

    def name(self):
        return LLuL

    def format(self, is_img2img: bool, args: typing.Union[typing.Sequence[typing.Any], typing.Mapping]) \
            -> typing.Sequence[typing.Any]:
        def obj_to_array(obj: typing.Mapping) -> typing.Sequence:
            #   enabled,
            #             multiply,
            #             weight,
            #             understand,
            #             layers,
            #             apply_to,
            #             start_steps,
            #             max_steps,
            #             up,
            #             up_aa,
            #             down,
            #             down_aa,
            #             intp,
            #             x,
            #             y,
            #             force_float,
            #             use_mask,
            #             mask,
            #             add_area_image,

            if isinstance(obj, dict):
                mask = obj.get('mask')
                if mask:
                    tmp_file = get_tmp_local_path(mask)
                    TmpFileObj = collections.namedtuple('TmpFileObj', 'name')
                    mask = TmpFileObj(tmp_file)

                return [obj.get('enabled', False),
                        obj.get('multiply', 1),
                        obj.get('weight', 0.15),
                        obj.get('understand', False),
                        obj.get('layers', 'OUT'),
                        obj.get('apply_to') or ['OUT'],
                        obj.get('start_steps', 5),
                        obj.get('max_steps', 0),
                        obj.get('up', 'Bilinear',),
                        obj.get('up_aa', False),
                        obj.get('down', 'Pooling Max'),
                        obj.get('down_aa', False),
                        obj.get('intp', 'Lerp'),
                        str(int(obj.get('x', 0))),
                        str(int(obj.get('y', 0))),
                        obj.get('force_float', False),
                        obj.get('use_mask', False),
                        mask,
                        obj.get('add_area_image', True),
                        ]
            return obj

        llul_script_args = args
        if isinstance(args, dict):
            llul_script_args = obj_to_array(args)
        else:
            llul_script_args = []
            for x in args:
                llul_script_args.extend(obj_to_array(x))

        return llul_script_args


