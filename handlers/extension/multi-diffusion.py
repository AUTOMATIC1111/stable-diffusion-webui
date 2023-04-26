#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/13 3:24 PM
# @Author  : wangdongming
# @Site    : 
# @File    : multi-diffusion.py
# @Software: Hifive
import typing
from handlers.formatter import AlwaysonScriptArgsFormatter

Multidiffusion = "Tiled-Diffusion"
TiledVAE = 'Tiled-VAE'


class MultiDiffusionFormatter(AlwaysonScriptArgsFormatter):

    def name(self):
        return Multidiffusion

    def format(self, is_img2img: bool, args: typing.Union[typing.Sequence[typing.Any], typing.Mapping]) \
            -> typing.Sequence[typing.Any]:
        def obj_to_array(obj: typing.Mapping) -> typing.Sequence:
            # 如果是[OBJ1, OBJ2]形式的，需要转换为ARRAY
            if isinstance(obj, dict):
                return [args['enabled'],
                        args['override_image_size'],
                        args['image_width'],
                        args['image_height'],
                        args['keep_input_size'],
                        args['tile_width'],
                        args['tile_height'],
                        args['overlap'],
                        args['batch_size'],
                        args['upscaler_index'],
                        args['scale_factor'],
                        args['control_tensor_cpu']
                ]

            return obj

        if is_img2img:
            if isinstance(args, dict):
                posex_script_args = obj_to_array(args)
            else:
                posex_script_args = [obj_to_array(x) for x in args]
        else:
            posex_script_args = obj_to_array(args)
        return posex_script_args


class MultiVAEFormatter(AlwaysonScriptArgsFormatter):

    def name(self):
        return TiledVAE

    def format(self, is_img2img: bool, args: typing.Union[typing.Sequence[typing.Any], typing.Mapping]) \
            -> typing.Sequence[typing.Any]:
        def obj_to_array(obj: typing.Mapping) -> typing.Sequence:
            # 如果是[OBJ1, OBJ2]形式的，需要转换为ARRAY
            if isinstance(obj, dict):
                return [args['enabled'],
                        args['vae_to_gpu'],
                        args['fast_decoder'],
                        args['fast_encoder'],
                        args['color_fix'],
                        args['encoder_tile_size'],
                        args['decoder_tile_size'],
                ]

            return obj

        if is_img2img:
            if isinstance(args, dict):
                posex_script_args = obj_to_array(args)
            else:
                posex_script_args = [obj_to_array(x) for x in args]
        else:
            posex_script_args = obj_to_array(args)
        return posex_script_args

