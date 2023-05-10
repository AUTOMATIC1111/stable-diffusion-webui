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
                array = [
                    obj['enabled'],
                    obj['method'],
                    obj['noise_inverse'],
                    obj['noise_inverse_steps'],
                    obj['noise_inverse_retouch'],
                    obj['noise_inverse_renoise_strength'],
                    obj['noise_inverse_renoise_kernel'],
                    obj['overwrite_image_size'],
                    obj['keep_input_size'],
                    obj['image_width'],
                    obj['image_height'],
                    obj['tile_width'],
                    obj['tile_height'],
                    obj['overlap'],
                    obj['batch_size'],
                    obj['upscaler'],
                    obj['scale_factor'],
                    obj['control_tensor_cpu'],
                    obj['enable_bbox_control'],
                    obj['draw_background'],
                    obj['causal_layers'],
                ]
                controls = obj['controls']
                if len(controls) > 8:
                    raise ValueError('region length err')
                for ctl in controls:
                    array.extend([
                        ctl['enable'],
                        ctl['x'],
                        ctl['y'],
                        ctl['w'],
                        ctl['h'],
                        ctl['prompt'],
                        ctl['neg_prompt'],
                        ctl['blend_mode'],
                        ctl['feather_ratio'],
                        ctl['seed'],
                    ])

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

