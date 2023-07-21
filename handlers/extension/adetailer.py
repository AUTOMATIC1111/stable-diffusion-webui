#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/7/19 8:51 PM
# @Author  : wangdongming
# @Site    : 
# @File    : adetailer.py
# @Software: Hifive

# ADetailer

import collections
import os.path
import typing
import tempfile
from handlers.utils import get_tmp_local_path
from handlers.formatter import AlwaysonScriptArgsFormatter

ADetailer = 'ADetailer'


class ADetailerFormatter(AlwaysonScriptArgsFormatter):

    def name(self):
        return ADetailer

    def format(self, is_img2img: bool, args: typing.Union[typing.Sequence[typing.Any], typing.Mapping]) \
            -> typing.Sequence[typing.Any]:
        def obj_to_array(obj: typing.Mapping) -> typing.Sequence:
            if isinstance(obj, dict):
                #  ad_model: str = "None"
                #     ad_prompt: str = ""
                #     ad_negative_prompt: str = ""
                #     ad_confidence: confloat(ge=0.0, le=1.0) = 0.3
                #     ad_mask_min_ratio: confloat(ge=0.0, le=1.0) = 0.0
                #     ad_mask_max_ratio: confloat(ge=0.0, le=1.0) = 1.0
                #     ad_dilate_erode: int = 4
                #     ad_x_offset: int = 0
                #     ad_y_offset: int = 0
                #     ad_mask_merge_invert: Literal["None", "Merge", "Merge and Invert"] = "None"
                #     ad_mask_blur: NonNegativeInt = 4
                #     ad_denoising_strength: confloat(ge=0.0, le=1.0) = 0.4
                #     ad_inpaint_only_masked: bool = True
                #     ad_inpaint_only_masked_padding: NonNegativeInt = 32
                #     ad_use_inpaint_width_height: bool = False
                #     ad_inpaint_width: PositiveInt = 512
                #     ad_inpaint_height: PositiveInt = 512
                #     ad_use_steps: bool = False
                #     ad_steps: PositiveInt = 28
                #     ad_use_cfg_scale: bool = False
                #     ad_cfg_scale: NonNegativeFloat = 7.0
                #     ad_use_sampler: bool = False
                #     ad_sampler: str = "DPM++ 2M Karras"
                #     ad_use_noise_multiplier: bool = False
                #     ad_noise_multiplier: confloat(ge=0.5, le=1.5) = 1.0
                #     ad_restore_face: bool = False
                #     ad_controlnet_model: constr(regex=cn_model_regex) = "None"
                #     ad_controlnet_module: Optional[constr(regex=r".*inpaint.*|^None$")] = None
                #     ad_controlnet_weight: confloat(ge=0.0, le=1.0) = 1.0
                #     ad_controlnet_guidance_start: confloat(ge=0.0, le=1.0) = 0.0
                #     ad_controlnet_guidance_end: confloat(ge=0.0, le=1.0) = 1.0
                #     is_api: bool = True
                d = {
                    'is_api': obj.get('is_api', True),
                    'ad_controlnet_guidance_end': obj.get('ad_controlnet_guidance_end', 1),
                    'ad_controlnet_guidance_start': obj.get('ad_controlnet_guidance_start', 0),
                    'ad_controlnet_weight': obj.get('ad_controlnet_weight', 1),
                    'ad_controlnet_module': obj.get('ad_controlnet_module', 'None'),
                    'ad_controlnet_model': obj.get('ad_controlnet_model', 'None'),
                    'ad_restore_face': obj.get('ad_restore_face', False),
                    'ad_noise_multiplier': obj.get('ad_noise_multiplier', 1),
                    'ad_use_noise_multiplier': obj.get('ad_use_noise_multiplier', False),
                    'ad_sampler': obj.get('ad_sampler', "DPM++ 2M Karras"),
                    'ad_use_sampler': obj.get('ad_use_sampler', False),
                    'ad_cfg_scale': obj.get('ad_cfg_scale', 7),
                    'ad_use_cfg_scale': obj.get('ad_use_cfg_scale', False),
                    'ad_steps': obj.get('ad_steps', 28),
                    'ad_use_steps': obj.get('ad_use_steps', False),
                    'ad_inpaint_height': obj.get('ad_inpaint_height', 512),
                    'ad_inpaint_width': obj.get('ad_inpaint_width', 512),
                    'ad_use_inpaint_width_height': obj.get('ad_use_inpaint_width_height', False),
                    'ad_inpaint_only_masked_padding': obj.get('ad_inpaint_only_masked_padding', 32),
                    'ad_inpaint_only_masked': obj.get('ad_inpaint_only_masked', True),
                    'ad_denoising_strength': obj.get('ad_denoising_strength', 0.4),
                    'ad_mask_blur': obj.get('ad_mask_blur', 4),
                    'ad_mask_merge_invert': obj.get('ad_mask_merge_invert') or 'None',
                    'ad_y_offset': obj.get('ad_y_offset', 0),
                    'ad_x_offset': obj.get('ad_x_offset', 0),
                    'ad_dilate_erode': obj.get('ad_dilate_erode', 4),
                    'ad_mask_max_ratio': obj.get('ad_mask_max_ratio', 1),
                    'ad_mask_min_ratio': obj.get('ad_mask_min_ratio', 0),
                    'ad_confidence': obj.get('ad_confidence', 0.3),
                    'ad_negative_prompt': obj.get('ad_negative_prompt') or "",
                    'ad_prompt': obj.get('ad_prompt') or "",
                    'ad_model': obj.get('ad_model') or 'None',  # 'mediapipe_face_full'
                }

                return [d]
            return [obj]

        ad_script_args = []
        if args:
            ad_script_args.append(True)
            if isinstance(args, dict):
                ad_script_args = obj_to_array(args)
            else:
                for x in args:
                    ad_script_args.extend(obj_to_array(x))

        return ad_script_args

