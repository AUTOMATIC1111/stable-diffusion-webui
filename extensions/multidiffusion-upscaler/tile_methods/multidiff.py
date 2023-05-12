#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/13 6:17 PM
# @Author  : wangdongming
# @Site    : 
# @File    : multidiff.py
# @Software: Hifive

import torch

from modules import devices, extra_networks
from modules.shared import state
from tile_methods.absdiff import TiledDiffusion
from tile_utils.utils import *
from tile_utils.typex import *


class MultiDiffusion(TiledDiffusion):
    """
        Multi-Diffusion Implementation
        https://arxiv.org/abs/2302.08113
    """

    def __init__(self, p: Processing, *args, **kwargs):
        super().__init__(p, *args, **kwargs)
        assert p.sampler_name != 'UniPC', 'MultiDiffusion is not compatible with UniPC!'

        # For ddim sampler we need to cache the pred_x0
        self.x_pred_buffer = None

    def hook(self):
        if self.is_kdiff:
            # For K-Diffusion sampler with uniform prompt, we hijack into the inner model for simplicity
            # Otherwise, the masked-redraw will break due to the init_latent
            self.sampler: CFGDenoiser
            self.sampler_forward = self.sampler.inner_model.forward
            self.sampler.inner_model.forward = self.kdiff_forward
        else:
            self.sampler: VanillaStableDiffusionSampler
            self.sampler_forward = self.sampler.orig_p_sample_ddim
            self.sampler.orig_p_sample_ddim = self.ddim_forward

    @staticmethod
    def unhook():
        # no need to unhook MultiDiffusion as it only hook the sampler,
        # which will be destroyed after the painting is done
        pass

    def reset_buffer(self, x_in: Tensor):
        super().reset_buffer(x_in)

        # ddim needs to cache pred0
        if self.is_ddim:
            if self.x_pred_buffer is None:
                self.x_pred_buffer = torch.zeros_like(x_in, device=x_in.device)
            else:
                self.x_pred_buffer.zero_()

    @custom_bbox
    def init_custom_bbox(self, *args):
        super().init_custom_bbox(*args)

        for bbox in self.custom_bboxes:
            if bbox.blend_mode == BlendMode.BACKGROUND:
                self.weights[bbox.slicer] += 1.0

    ''' ↓↓↓ kernel hijacks ↓↓↓ '''

    def repeat_cond_dict(self, cond_input: CondDict, bboxes: List[CustomBBox]) -> CondDict:
        cond = cond_input['c_crossattn'][0]
        # repeat the condition on its first dim
        cond_shape = cond.shape
        cond = cond.repeat((len(bboxes),) + (1,) * (len(cond_shape) - 1))
        image_cond = cond_input['c_concat'][0]
        if image_cond.shape[2] == self.h and image_cond.shape[3] == self.w:
            image_cond_list = []
            for bbox in bboxes:
                image_cond_list.append(image_cond[bbox.slicer])
            image_cond_tile = torch.cat(image_cond_list, dim=0)
        else:
            image_cond_shape = image_cond.shape
            image_cond_tile = image_cond.repeat((len(bboxes),) + (1,) * (len(image_cond_shape) - 1))
        return {"c_crossattn": [cond], "c_concat": [image_cond_tile]}

    @torch.no_grad()
    @keep_signature
    def kdiff_forward(self, x_in: Tensor, sigma_in: Tensor, cond: CondDict) -> Tensor:
        '''
        This function hijacks `k_diffusion.external.CompVisDenoiser.forward()`
        So its signature should be the same as the original function, especially the "cond" should be with exactly the same name
        '''

        assert CompVisDenoiser.forward

        def org_func(x: Tensor):
            return self.sampler_forward(x, sigma_in, cond=cond)

        def repeat_func(x_tile: Tensor, bboxes: List[CustomBBox]):
            # For kdiff sampler, the dim 0 of input x_in is:
            #   = batch_size * (num_AND + 1)   if not an edit model
            #   = batch_size * (num_AND + 2)   otherwise
            sigma_in_tile = sigma_in.repeat(len(bboxes))
            new_cond = self.repeat_cond_dict(cond, bboxes)
            x_tile_out = self.sampler_forward(x_tile, sigma_in_tile, cond=new_cond)
            return x_tile_out

        def custom_func(x: Tensor, bbox_id: int, bbox: CustomBBox):
            return self.kdiff_custom_forward(x, sigma_in, cond, bbox_id, bbox, self.sampler_forward)

        return self.sample_one_step(x_in, org_func, repeat_func, custom_func)

    @torch.no_grad()
    @keep_signature
    def ddim_forward(self, x_in: Tensor, cond_in: Union[CondDict, Tensor], ts: Tensor,
                     unconditional_conditioning: Tensor, *args, **kwargs) -> Tuple[Tensor, Tensor]:
        '''
        This function will replace the original p_sample_ddim function in ldm/diffusionmodels/ddim.py
        So its signature should be the same as the original function,
        Particularly, the unconditional_conditioning should be with exactly the same name
        '''

        assert VanillaStableDiffusionSampler.p_sample_ddim_hook

        def org_func(x: Tensor):
            return self.sampler_forward(x, cond_in, ts, unconditional_conditioning=unconditional_conditioning, *args,
                                        **kwargs)

        def repeat_func(x_tile: Tensor, bboxes: List[CustomBBox]):
            if isinstance(cond_in, dict):
                ts_tile = ts.repeat(len(bboxes))
                cond_tile = self.repeat_cond_dict(cond_in, bboxes)
                ucond_tile = self.repeat_cond_dict(unconditional_conditioning, bboxes)
            else:
                ts_tile = ts.repeat(len(bboxes))
                cond_shape = cond_in.shape
                cond_tile = cond_in.repeat((len(bboxes),) + (1,) * (len(cond_shape) - 1))
                ucond_shape = unconditional_conditioning.shape
                ucond_tile = unconditional_conditioning.repeat((len(bboxes),) + (1,) * (len(ucond_shape) - 1))
            x_tile_out, x_pred = self.sampler_forward(
                x_tile, cond_tile, ts_tile,
                unconditional_conditioning=ucond_tile,
                *args, **kwargs)
            return x_tile_out, x_pred

        def custom_func(x: Tensor, bbox_id: int, bbox: CustomBBox):
            # before the final forward, we can set the control tensor
            def forward_func(x, *args, **kwargs):
                self.set_controlnet_tensors(bbox_id, 2 * x.shape[0])
                return self.sampler_forward(x, *args, **kwargs)

            return self.ddim_custom_forward(x, cond_in, bbox, ts, forward_func, *args, **kwargs)

        return self.sample_one_step(x_in, org_func, repeat_func, custom_func)

    def sample_one_step(self, x_in: Tensor, org_func: Callable, repeat_func: Callable, custom_func: Callable) -> Union[
        Tensor, Tuple[Tensor, Tensor]]:
        '''
        this method splits the whole latent and process in tiles
            - x_in: current whole U-Net latent
            - org_func: original forward function, when use highres
            - denoise_func: one step denoiser for grid tile
            - denoise_custom_func: one step denoiser for custom tile
        '''

        N, C, H, W = x_in.shape
        if H != self.h or W != self.w:
            self.reset_controlnet_tensors()
            return org_func(x_in)

        # clear buffer canvas
        self.reset_buffer(x_in)

        # Background sampling (grid bbox)
        if self.draw_background:
            for batch_id, bboxes in enumerate(self.batched_bboxes):
                if state.interrupted: return x_in

                # batching
                x_tile_list = []
                for bbox in bboxes:
                    x_tile_list.append(x_in[bbox.slicer])
                x_tile = torch.cat(x_tile_list, dim=0)

                # controlnet tiling
                # FIXME: is_denoise is default to False, however it is set to True in case of MixtureOfDiffusers
                self.switch_controlnet_tensors(batch_id, N, len(bboxes))

                # compute tiles
                if self.is_kdiff:
                    x_tile_out = repeat_func(x_tile, bboxes)
                    for i, bbox in enumerate(bboxes):
                        self.x_buffer[bbox.slicer] += x_tile_out[i * N:(i + 1) * N, :, :, :]
                else:
                    x_tile_out, x_tile_pred = repeat_func(x_tile, bboxes)
                    for i, bbox in enumerate(bboxes):
                        self.x_buffer[bbox.slicer] += x_tile_out[i * N:(i + 1) * N, :, :, :]
                        self.x_pred_buffer[bbox.slicer] += x_tile_pred[i * N:(i + 1) * N, :, :, :]

                # update progress bar
                self.update_pbar()

        # Custom region sampling (custom bbox)
        x_feather_buffer = None
        x_feather_mask = None
        x_feather_count = None
        x_feather_pred_buffer = None
        if len(self.custom_bboxes) > 0:
            for bbox_id, bbox in enumerate(self.custom_bboxes):
                if state.interrupted: return x_in

                if not self.p.disable_extra_networks:
                    with devices.autocast():
                        extra_networks.activate(self.p, bbox.extra_network_data)

                x_tile = x_in[bbox.slicer]

                if self.is_kdiff:
                    # retrieve original x_in from construncted input
                    x_tile_out = custom_func(x_tile, bbox_id, bbox)

                    if bbox.blend_mode == BlendMode.BACKGROUND:
                        self.x_buffer[bbox.slicer] += x_tile_out
                    elif bbox.blend_mode == BlendMode.FOREGROUND:
                        if x_feather_buffer is None:
                            x_feather_buffer = torch.zeros_like(self.x_buffer)
                            x_feather_mask = torch.zeros((1, 1, H, W), device=x_in.device)
                            x_feather_count = torch.zeros((1, 1, H, W), device=x_in.device)
                        x_feather_buffer[bbox.slicer] += x_tile_out
                        x_feather_mask[bbox.slicer] += bbox.feather_mask
                        x_feather_count[bbox.slicer] += 1
                else:
                    x_tile_out, x_tile_pred = custom_func(x_tile, bbox_id, bbox)

                    if bbox.blend_mode == BlendMode.BACKGROUND:
                        self.x_buffer[bbox.slicer] += x_tile_out
                        self.x_pred_buffer[bbox.slicer] += x_tile_pred
                    elif bbox.blend_mode == BlendMode.FOREGROUND:
                        if x_feather_buffer is None:
                            x_feather_buffer = torch.zeros_like(self.x_buffer)
                            x_feather_pred_buffer = torch.zeros_like(self.x_pred_buffer)
                            x_feather_mask = torch.zeros((1, 1, H, W), device=x_in.device)
                            x_feather_count = torch.zeros((1, 1, H, W), device=x_in.device)
                        x_feather_buffer[bbox.slicer] += x_tile_out
                        x_feather_pred_buffer[bbox.slicer] += x_tile_pred
                        x_feather_mask[bbox.slicer] += bbox.feather_mask
                        x_feather_count[bbox.slicer] += 1

                if not self.p.disable_extra_networks:
                    with devices.autocast():
                        extra_networks.deactivate(self.p, bbox.extra_network_data)

                # update progress bar
                self.update_pbar()

        # Averaging background buffer
        x_out = torch.where(self.weights > 1, self.x_buffer / self.weights, self.x_buffer)
        if self.is_ddim:
            x_pred_out = torch.where(self.weights > 1, self.x_pred_buffer / self.weights, self.x_pred_buffer)

        # Foreground Feather blending
        if x_feather_buffer is not None:
            # Average overlapping feathered regions
            x_feather_buffer = torch.where(x_feather_count > 1, x_feather_buffer / x_feather_count, x_feather_buffer)
            x_feather_mask = torch.where(x_feather_count > 1, x_feather_mask / x_feather_count, x_feather_mask)
            # Weighted average with original x_buffer
            x_out = torch.where(x_feather_count > 0, x_out * (1 - x_feather_mask) + x_feather_buffer * x_feather_mask,
                                x_out)
            if self.is_ddim:
                x_feather_pred_buffer = torch.where(x_feather_count > 1, x_feather_pred_buffer / x_feather_count,
                                                    x_feather_pred_buffer)
                x_pred_out = torch.where(x_feather_count > 0,
                                         x_pred_out * (1 - x_feather_mask) + x_feather_pred_buffer * x_feather_mask,
                                         x_pred_out)

        return x_out if self.is_kdiff else (x_out, x_pred_out)

    def get_noise(self, x_in: Tensor, sigma_in: Tensor, cond_in: Dict[str, Tensor], step: int) -> Tensor:
        # NOTE: The following code is analytically wrong but aesthetically beautiful
        local_cond_in = cond_in.copy()

        def org_func(x: Tensor):
            return shared.sd_model.apply_model(x, sigma_in, cond=local_cond_in)

        def repeat_func(x_tile: Tensor, bboxes: List[CustomBBox]):
            sigma_in_tile = sigma_in.repeat(len(bboxes))
            new_cond = self.repeat_cond_dict(local_cond_in, bboxes)
            x_tile_out = shared.sd_model.apply_model(x_tile, sigma_in_tile, cond=new_cond)
            return x_tile_out

        def custom_func(x: Tensor, bbox_id: int, bbox: CustomBBox):
            # The negative prompt in custom bbox should not be used for noise inversion
            # otherwise the result will be astonishingly bad.
            cond = Condition.reconstruct_cond(bbox.cond, step)
            image_cond = local_cond_in['c_concat'][0]
            if image_cond.shape[2:] == (self.h, self.w):
                image_cond = image_cond[bbox.slicer]
            image_conditioning = image_cond
            cond_in = {"c_concat": [image_conditioning], "c_crossattn": [cond]}
            return shared.sd_model.apply_model(x, sigma_in, cond=cond_in)

        return self.sample_one_step(x_in, org_func, repeat_func, custom_func)
