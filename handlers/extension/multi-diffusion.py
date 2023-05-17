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
                    obj.get('overwrite_size', False),
                    obj['keep_input_size'],
                    obj['image_width'],
                    obj['image_height'],
                    obj['tile_width'],
                    obj['tile_height'],
                    obj['overlap'],
                    obj['batch_size'],
                    obj.get('upscaler') or obj.get('upscaler_name'),
                    obj['scale_factor'],
                    obj['noise_inverse'],
                    obj['noise_inverse_steps'],
                    obj['noise_inverse_retouch'],
                    obj['noise_inverse_renoise_strength'],
                    obj['noise_inverse_renoise_kernel'],
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
                return array

            return obj

        if is_img2img:
            if isinstance(args, dict):
                md_script_args = obj_to_array(args)
            else:
                md_script_args = []
                for x in args:
                    md_script_args.extend(obj_to_array(x))
        else:
            md_script_args = obj_to_array(args)
        return md_script_args


class MultiVAEFormatter(AlwaysonScriptArgsFormatter):

    def name(self):
        return TiledVAE

    def format(self, is_img2img: bool, args: typing.Union[typing.Sequence[typing.Any], typing.Mapping]) \
            -> typing.Sequence[typing.Any]:
        def obj_to_array(obj: typing.Mapping) -> typing.Sequence:
            #          enabled,
            #             encoder_tile_size, decoder_tile_size,
            #             vae_to_gpu, fast_decoder, fast_encoder, color_fix,
            if isinstance(obj, dict):
                return [obj['enabled'],
                        obj['encoder_tile_size'],
                        obj['decoder_tile_size'],
                        obj['vae_to_gpu'],
                        obj['fast_decoder'],
                        obj['fast_encoder'],
                        obj['color_fix'],
                ]

            return obj

        if is_img2img:
            if isinstance(args, dict):
                mv_script_args = obj_to_array(args)
            else:
                mv_script_args = []
                for x in args:
                    mv_script_args.extend(obj_to_array(x))
        else:
            mv_script_args = obj_to_array(args)
        return mv_script_args

