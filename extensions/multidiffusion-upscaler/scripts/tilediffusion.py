#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
# ------------------------------------------------------------------------
#
#   Tiled Diffusion for Automatic1111 WebUI
#
#   Introducing revolutionary large image drawing methods:
#       MultiDiffusion and Mixture of Diffusers!
#
#   Techniques is not originally proposed by me, please refer to
#
#   MultiDiffusion: https://multidiffusion.github.io
#   Mixture of Diffusers: https://github.com/albarji/mixture-of-diffusers
#
#   The script contains a few optimizations including:
#       - symmetric tiling bboxes
#       - cached tiling weights
#       - batched denoising
#       - advanced prompt control for each tile
#
# ------------------------------------------------------------------------
#
#   This script hooks into the original sampler and decomposes the latent
#   image, sampled separately and run weighted average to merge them back.
#
#   Advantages:
#   - Allows for super large resolutions (2k~8k) for both txt2img and img2img.
#   - The merged output is completely seamless without any post-processing.
#   - Training free. No need to train a new model, and you can control the
#       text prompt for each tile.
#
#   Drawbacks:
#   - Depending on your parameter settings, the process can be very slow,
#       especially when overlap is relatively large.
#   - The gradient calculation is not compatible with this hack. It
#       will break any backward() or torch.autograd.grad() that passes UNet.
#
#   How it works (insanely simple!)
#   1) The latent image x_t is split into tiles
#   2) The tiles are denoised by original sampler to get x_t-1
#   3) The tiles are added together, but divided by how many times each pixel
#       is added.
#
#   Enjoy!
#
#   @author: LI YI @ Nanyang Technological University - Singapore
#   @date: 2023-03-03
#   @license: MIT License
#
#   Please give me a star if you like this project!
#
# ------------------------------------------------------------------------
'''
from pathlib import Path
import json
import torch
import numpy as np
import gradio as gr

from modules import sd_samplers, images, shared, scripts, devices, processing
from modules.shared import opts
from modules.processing import opt_f, get_fixed_seed
from modules.ui import gr_show
from tile_methods.absdiff import TiledDiffusion
from tile_methods.multidiff import MultiDiffusion
from tile_methods.mixtureofdiff import MixtureOfDiffusers
from tile_utils.utils import *
from tile_utils.typex import *

SD_WEBUI_PATH = Path.cwd()
ME_PATH = SD_WEBUI_PATH / 'extensions' / 'multidiffusion-upscaler-for-automatic1111'
CFG_PATH = ME_PATH / 'region_configs'
BBOX_MAX_NUM = min(getattr(shared.cmd_opts, "md_max_regions", 8), 16)


class Script(scripts.Script):

    def __init__(self):
        self.controlnet_script: ModuleType = None
        self.delegate: TiledDiffusion = None
        self.noise_inverse_cache: NoiseInverseCache = None

    def title(self):
        return "Tiled Diffusion"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        tab = 't2i' if not is_img2img else 'i2i'
        is_t2i = 'true' if not is_img2img else 'false'

        with gr.Accordion('Tiled Diffusion', open=False):
            with gr.Row(variant='compact') as tab_enable:
                enabled = gr.Checkbox(label='Enable Tiled Diffusion', value=False, elem_id=self.elem_id("enable"))
                overwrite_size = gr.Checkbox(label='Overwrite image size', value=False, visible=not is_img2img,
                                             elem_id=self.elem_id("overwrite_image_size"))
                keep_input_size = gr.Checkbox(label='Keep input image size', value=True, visible=is_img2img,
                                              elem_id=self.elem_id("keep_input_size"))

            with gr.Row(variant='compact', visible=False) as tab_size:
                image_width = gr.Slider(minimum=256, maximum=16384, step=16, label='Image width', value=1024,
                                        elem_id=f'MD-overwrite-width-{tab}')
                image_height = gr.Slider(minimum=256, maximum=16384, step=16, label='Image height', value=1024,
                                         elem_id=f'MD-overwrite-height-{tab}')
                overwrite_size.change(fn=gr_show, inputs=overwrite_size, outputs=tab_size)

            with gr.Row(variant='compact') as tab_param:
                method = gr.Dropdown(label='Method', choices=[e.value for e in Method], value=Method.MULTI_DIFF.value,
                                     elem_id=self.elem_id("method"))
                control_tensor_cpu = gr.Checkbox(label='Move ControlNet tensor to CPU (if applicable)', value=False,
                                                 elem_id=self.elem_id("control_tensor_cpu"))
                reset_status = gr.Button(value='Free GPU', variant='tool', elem_id=self.elem_id("reset_status"))
                reset_status.click(fn=self.reset_and_gc, show_progress=False)

            with gr.Group() as tab_tile:
                with gr.Row(variant='compact'):
                    tile_width = gr.Slider(minimum=16, maximum=256, step=16, label='Latent tile width', value=96,
                                           elem_id=self.elem_id("latent_tile_width"))
                    tile_height = gr.Slider(minimum=16, maximum=256, step=16, label='Latent tile height', value=96,
                                            elem_id=self.elem_id("latent_tile_height"))

                with gr.Row(variant='compact'):
                    overlap = gr.Slider(minimum=0, maximum=256, step=4, label='Latent tile overlap', value=48,
                                        elem_id=self.elem_id("latent_overlap"))
                    batch_size = gr.Slider(minimum=1, maximum=8, step=1, label='Latent tile batch size', value=4,
                                           elem_id=self.elem_id("latent_batch_size"))

            with gr.Row(variant='compact', visible=is_img2img) as tab_upscale:
                upscaler_name = gr.Dropdown(label='Upscaler', choices=[x.name for x in shared.sd_upscalers],
                                            value="None", elem_id='MD-upscaler-index')
                scale_factor = gr.Slider(minimum=1.0, maximum=8.0, step=0.05, label='Scale Factor', value=2.0,
                                         elem_id='MD-upscaler-factor')

            with gr.Accordion('Noise Inversion', open=True, visible=is_img2img) as tab_noise_inv:
                with gr.Row(variant='compact'):
                    noise_inverse = gr.Checkbox(label='Enable Noise Inversion', value=False,
                                                elem_id=self.elem_id("noise_inverse"))
                    noise_inverse_steps = gr.Slider(minimum=1, maximum=200, step=1, label='Inversion steps', value=10,
                                                    elem_id=self.elem_id("noise_inverse_steps"))
                    gr.HTML(
                        '<p>Please test on small images before actual upscale. Default params require denoise <= 0.6</p>')
                with gr.Row(variant='compact'):
                    noise_inverse_retouch = gr.Slider(minimum=1, maximum=100, step=0.1, label='Retouch', value=1,
                                                      elem_id=self.elem_id("noise_inverse_retouch"))
                    noise_inverse_renoise_strength = gr.Slider(minimum=0, maximum=2, step=0.01,
                                                               label='Renoise strength', value=1,
                                                               elem_id=self.elem_id("noise_inverse_renoise_strength"))
                    noise_inverse_renoise_kernel = gr.Slider(minimum=2, maximum=512, step=1,
                                                             label='Renoise kernel size', value=64,
                                                             elem_id=self.elem_id("noise_inverse_renoise_kernel"))

            # The control includes txt2img and img2img, we use t2i and i2i to distinguish them
            with gr.Group(elem_id=f'MD-bbox-control-{tab}') as tab_bbox:
                with gr.Accordion('Region Prompt Control', open=False):
                    with gr.Row(variant='compact'):
                        enable_bbox_control = gr.Checkbox(label='Enable Control', value=False,
                                                          elem_id=self.elem_id("enable_bbox_control"))
                        draw_background = gr.Checkbox(label='Draw full canvas background', value=False,
                                                      elem_id=self.elem_id("draw_background"))
                        causal_layers = gr.Checkbox(label='Causalize layers', value=False, visible=False,
                                                    elem_id=self.elem_id("causal_layers"))  # NOTE: currently not used

                    with gr.Row(variant='compact'):
                        create_button = gr.Button(value="Create txt2img canvas" if not is_img2img else "From img2img",
                                                  elem_id=self.elem_id("create_button"))

                    bbox_controls: List[Component] = []  # control set for each bbox
                    with gr.Row(variant='compact'):
                        ref_image = gr.Image(label='Ref image (for conviently locate regions)', image_mode=None,
                                             elem_id=f'MD-bbox-ref-{tab}', interactive=True)
                        if not is_img2img:
                            # gradio has a serious bug: it cannot accept multiple inputs when you use both js and fn.
                            # to workaround this, we concat the inputs into a single string and parse it in js
                            def create_t2i_ref(string):
                                w, h = [int(x) for x in string.split('x')]
                                w = max(w, opt_f)
                                h = max(h, opt_f)
                                return np.zeros(shape=(h, w, 3), dtype=np.uint8) + 255

                            create_button.click(
                                fn=create_t2i_ref,
                                inputs=overwrite_size,
                                outputs=ref_image,
                                _js='onCreateT2IRefClick')
                        else:
                            create_button.click(fn=None, outputs=ref_image, _js='onCreateI2IRefClick')

                    with gr.Row(variant='compact'):
                        cfg_name = gr.Textbox(label='Custom Config File', value='config.json',
                                              elem_id=self.elem_id("cfg_name"))
                        cfg_dump = gr.Button(value='💾 Save', variant='tool', elem_id=self.elem_id("cfg_dump"))
                        cfg_load = gr.Button(value='⚙️ Load', variant='tool', elem_id=self.elem_id("cfg_load"))

                    with gr.Row(variant='compact'):
                        cfg_tip = gr.HTML(value='', visible=False, elem_id=self.elem_id("cfg_tip"))

                    for i in range(BBOX_MAX_NUM):
                        # Only when displaying & png generate info we use index i+1, in other cases we use i
                        with gr.Accordion(f'Region {i + 1}', open=False, elem_id=f'MD-accordion-{tab}-{i}'):
                            with gr.Row(variant='compact'):
                                e = gr.Checkbox(label=f'Enable Region {i + 1}', value=False)
                                e.change(fn=None, inputs=e, outputs=e, _js=f'e => onBoxEnableClick({is_t2i}, {i}, e)',
                                         show_progress=False)

                                blend_mode = gr.Dropdown(label='Type', choices=[e.value for e in BlendMode],
                                                         value=BlendMode.BACKGROUND.value,
                                                         elem_id=f'MD-{tab}-{i}-blend-mode')
                                feather_ratio = gr.Slider(label='Feather', value=0.2, minimum=0, maximum=1, step=0.05,
                                                          visible=False, elem_id=f'MD-{tab}-{i}-feather')

                                blend_mode.change(fn=lambda x: gr_show(x == BlendMode.FOREGROUND.value),
                                                  inputs=blend_mode, outputs=feather_ratio, show_progress=False)

                            with gr.Row(variant='compact'):
                                x = gr.Slider(label='x', value=0.4, minimum=0.0, maximum=1.0, step=0.0001,
                                              elem_id=f'MD-{tab}-{i}-x')
                                y = gr.Slider(label='y', value=0.4, minimum=0.0, maximum=1.0, step=0.0001,
                                              elem_id=f'MD-{tab}-{i}-y')

                            with gr.Row(variant='compact'):
                                w = gr.Slider(label='w', value=0.2, minimum=0.0, maximum=1.0, step=0.0001,
                                              elem_id=f'MD-{tab}-{i}-w')
                                h = gr.Slider(label='h', value=0.2, minimum=0.0, maximum=1.0, step=0.0001,
                                              elem_id=f'MD-{tab}-{i}-h')

                                x.change(fn=None, inputs=x, outputs=x, _js=f'v => onBoxChange({is_t2i}, {i}, "x", v)',
                                         show_progress=False)
                                y.change(fn=None, inputs=y, outputs=y, _js=f'v => onBoxChange({is_t2i}, {i}, "y", v)',
                                         show_progress=False)
                                w.change(fn=None, inputs=w, outputs=w, _js=f'v => onBoxChange({is_t2i}, {i}, "w", v)',
                                         show_progress=False)
                                h.change(fn=None, inputs=h, outputs=h, _js=f'v => onBoxChange({is_t2i}, {i}, "h", v)',
                                         show_progress=False)

                            prompt = gr.Text(show_label=False, placeholder=f'Prompt, will append to your {tab} prompt',
                                             max_lines=2, elem_id=f'MD-{tab}-{i}-prompt')
                            neg_prompt = gr.Text(show_label=False, placeholder='Negative Prompt, will also be appended',
                                                 max_lines=1, elem_id=f'MD-{tab}-{i}-neg-prompt')
                            with gr.Row(variant='compact'):
                                seed = gr.Number(label='Seed', value=-1, visible=True, elem_id=f'MD-{tab}-{i}-seed')
                                random_seed = gr.Button(value='🎲', variant='tool', elem_id=f'MD-{tab}-{i}-random_seed')
                                reuse_seed = gr.Button(value='♻️', variant='tool', elem_id=f'MD-{tab}-{i}-reuse_seed')
                                random_seed.click(fn=lambda: -1, outputs=seed, show_progress=False)
                                reuse_seed.click(fn=None, inputs=seed, outputs=seed,
                                                 _js=f'e => getSeedInfo({is_t2i}, {i + 1}, e)', show_progress=False)

                        control = [e, x, y, w, h, prompt, neg_prompt, blend_mode, feather_ratio, seed]
                        assert len(control) == NUM_BBOX_PARAMS
                        bbox_controls.extend(control)

                    # NOTE: dynamically hard coded!!
                    load_regions_js = '''
                        function onBoxChangeAll(ref_image, cfg_name, ...args) {
                            const is_t2i = %s;
                            const n_bbox = %d;
                            const n_ctrl = %d;
                            for (let i=0; i<n_bbox; i++) {
                                onBoxEnableClick(is_t2i, i, args[i * n_ctrl + 0])
                                onBoxChange(is_t2i, i, "x", args[i * n_ctrl + 1]);
                                onBoxChange(is_t2i, i, "y", args[i * n_ctrl + 2]);
                                onBoxChange(is_t2i, i, "w", args[i * n_ctrl + 3]);
                                onBoxChange(is_t2i, i, "h", args[i * n_ctrl + 4]);
                            }
                            updateBoxes(true);
                            updateBoxes(false);
                            return args_to_array(arguments);
                        }
                    ''' % (is_t2i, BBOX_MAX_NUM, NUM_BBOX_PARAMS)
                    cfg_dump.click(fn=self.dump_regions, inputs=[cfg_name, *bbox_controls], outputs=cfg_tip,
                                   show_progress=False)
                    cfg_load.click(fn=self.load_regions, _js=load_regions_js,
                                   inputs=[ref_image, cfg_name, *bbox_controls], outputs=[*bbox_controls, cfg_tip],
                                   show_progress=False)

        return [
            enabled, method,
            overwrite_size, keep_input_size, image_width, image_height,
            tile_width, tile_height, overlap, batch_size,
            upscaler_name, scale_factor,
            noise_inverse, noise_inverse_steps, noise_inverse_retouch, noise_inverse_renoise_strength,
            noise_inverse_renoise_kernel,
            control_tensor_cpu,
            enable_bbox_control, draw_background, causal_layers,
            *bbox_controls,
        ]

    def process(self, p: Processing,
                enabled: bool, method: str,
                overwrite_size: bool, keep_input_size: bool, image_width: int, image_height: int,
                tile_width: int, tile_height: int, overlap: int, tile_batch_size: int,
                upscaler_name: str, scale_factor: float,
                noise_inverse: bool, noise_inverse_steps: int, noise_inverse_retouch: float,
                noise_inverse_renoise_strength: float, noise_inverse_renoise_kernel: int,
                control_tensor_cpu: bool,
                enable_bbox_control: bool, draw_background: bool, causal_layers: bool,
                *bbox_control_states: List[Any],
                ):

        # unhijack & unhook, in case it broke at last time
        self.reset()

        if not enabled: return

        ''' upscale '''
        # store canvas size settings
        if hasattr(p, "init_images"):
            p.init_images_original_md = [img.copy() for img in p.init_images]
        p.width_original_md = p.width
        p.height_original_md = p.height

        is_img2img = hasattr(p, "init_images") and len(p.init_images) > 0
        if is_img2img:  # img2img, TODO: replace with `images.resize_image()`
            idx = [x.name for x in shared.sd_upscalers].index(upscaler_name)
            upscaler = shared.sd_upscalers[idx]
            init_img = p.init_images[0]
            init_img = images.flatten(init_img, opts.img2img_background_color)
            if upscaler.name != "None":
                print(f"[Tiled Diffusion] upscaling image with {upscaler.name}...")
                image = upscaler.scaler.upscale(init_img, scale_factor, upscaler.data_path)
                p.extra_generation_params["Tiled Diffusion upscaler"] = upscaler.name
                p.extra_generation_params["Tiled Diffusion scale factor"] = scale_factor
                # For webui folder based batch processing, the length of init_images is not 1
                # We need to replace all images with the upsampled one
                for i in range(len(p.init_images)):
                    p.init_images[i] = image
            else:
                image = init_img

            # decide final canvas size
            if keep_input_size:
                p.width = image.width
                p.height = image.height
            elif upscaler.name != "None":
                p.width = scale_factor * p.width_original_md
                p.height = scale_factor * p.height_original_md
        elif overwrite_size:  # txt2img
            p.width = image_width
            p.height = image_height

        ''' sanitiy check '''
        chks = [
            splitable(p.width, p.height, tile_width, tile_height, overlap),
            enable_bbox_control,
            is_img2img and noise_inverse,
        ]
        if not any(chks):
            print("[Tiled Diffusion] ignore tiling when there's only 1 tile or nothing to do :)")
            return

        bbox_settings = build_bbox_settings(bbox_control_states) if enable_bbox_control else []

        if 'png info':
            info = {}
            p.extra_generation_params["Tiled Diffusion"] = info

            info['Method'] = method
            info['Tile tile width'] = tile_width
            info['Tile tile height'] = tile_height
            info['Tile Overlap'] = overlap
            info['Tile batch size'] = tile_batch_size

            if is_img2img:
                if upscaler.name != "None":
                    info['Upscaler'] = upscaler.name
                    info['Upscale factor'] = scale_factor
                if keep_input_size:
                    info['Keep input size'] = keep_input_size
                if noise_inverse:
                    info['NoiseInv'] = noise_inverse
                    info['NoiseInv Steps'] = noise_inverse_steps
                    info['NoiseInv Retouch'] = noise_inverse_retouch
                    info['NoiseInv Renoise strength'] = noise_inverse_renoise_strength
                    info['NoiseInv Kernel size'] = noise_inverse_renoise_kernel

            if enable_bbox_control:
                info["Region control"] = {f'Region {i + 1}': v._asdict() for i, v in bbox_settings.items()}

        ''' ControlNet hackin '''
        try:
            from scripts.cldm import ControlNet
            # fix controlnet multi-batch issue

            def align(self, hint, h, w):
                if len(hint.shape) == 3:
                    hint = hint.unsqueeze(0)
                _, _, h1, w1 = hint.shape
                if (h, w) != (h1, w1):
                    hint = torch.nn.functional.interpolate(hint, size=(h, w), mode="nearest")
                return hint

            ControlNet.align = align

            for script in p.scripts.scripts + p.scripts.alwayson_scripts:
                if hasattr(script, "latest_network") and script.title().lower() == "controlnet":
                    self.controlnet_script = script
                    print("[Tiled Diffusion] ControlNet found, support is enabled.")
                    break
        except ImportError:
            pass

        ''' hijack inner APIs '''
        sd_samplers.create_sampler_original_md = sd_samplers.create_sampler
        sd_samplers.create_sampler = lambda name, model: self.create_sampler_hijack(
            name, model, p, Method(method),
            tile_width, tile_height, overlap, tile_batch_size,
            noise_inverse, noise_inverse_steps, noise_inverse_retouch,
            noise_inverse_renoise_strength, noise_inverse_renoise_kernel,
            control_tensor_cpu,
            enable_bbox_control, draw_background, causal_layers,
            bbox_settings,
        )

        if enable_bbox_control:
            processing.create_random_tensors_original_md = processing.create_random_tensors
            processing.create_random_tensors = lambda *args, **kwargs: self.create_random_tensors_hijack(
                bbox_settings,
                *args, **kwargs,
            )

    def postprocess_batch(self, p: Processing, enabled, *args, **kwargs):
        if not enabled: return

        if self.delegate is not None: self.delegate.reset_controlnet_tensors()

    def postprocess(self, p: Processing, processed, enabled, *args):
        if not enabled: return

        # unhijack & unhook
        self.reset()

        # restore canvas size settings
        if hasattr(p, 'init_images') and hasattr(p, 'init_images_original_md'):
            p.init_images.clear()  # NOTE: do NOT change the list object, compatible with shallow copy of XYZ-plot
            p.init_images.extend(p.init_images_original_md)
            del p.init_images_original_md
        p.width = p.width_original_md;
        del p.width_original_md
        p.height = p.height_original_md;
        del p.height_original_md

        # clean up noise inverse latent for folder-based processing
        if hasattr(p, 'noise_inverse_latent'):
            del p.noise_inverse_latent

    ''' ↓↓↓ inner API hijack ↓↓↓ '''

    def create_sampler_hijack(
            self, name: str, model: LatentDiffusion, p: Processing, method: Method,
            tile_width: int, tile_height: int, overlap: int, tile_batch_size: int,
            noise_inverse: bool, noise_inverse_steps: int, noise_inverse_retouch: float,
            noise_inverse_renoise_strength: float, noise_inverse_renoise_kernel: int,
            control_tensor_cpu: bool,
            enable_bbox_control: bool, draw_background: bool, causal_layers: bool,
            bbox_settings: Dict[int, BBoxSettings]
    ):

        if self.delegate is not None:
            # samplers are stateless, we reuse it if possible
            if self.delegate.sampler_name == name:
                # before we reuse the sampler, we refresh the control tensor
                # so that we are compatible with ControlNet batch processing
                if self.controlnet_script:
                    self.delegate.prepare_controlnet_tensors(refresh=True)
                return self.delegate.sampler_raw
            else:
                self.reset()

        flag_noise_inverse = hasattr(p, "init_images") and len(p.init_images) > 0 and noise_inverse
        if flag_noise_inverse:
            print('warn: noise inversion only supports the Euler sampler, switch to it sliently...')
            name = 'Euler'
            p.sampler_name = name

        # create a sampler with the original function
        sampler = sd_samplers.create_sampler_original_md(name, model)
        if method == Method.MULTI_DIFF:
            delegate_cls = MultiDiffusion
        elif method == Method.MIX_DIFF:
            delegate_cls = MixtureOfDiffusers
        else:
            raise NotImplementedError(f"Method {method} not implemented.")

        # delegate hacks into the `sampler` with context of `p`
        delegate = delegate_cls(p, sampler)

        # setup **optional** supports through `init_*`, make everything relatively pluggable!!
        if flag_noise_inverse:
            get_cache_callback = self.noise_inverse_get_cache
            set_cache_callback = lambda x0, xt, prompts: self.noise_inverse_set_cache(p, x0, xt, prompts,
                                                                                      noise_inverse_steps,
                                                                                      noise_inverse_retouch)
            delegate.init_noise_inverse(noise_inverse_steps, noise_inverse_retouch, get_cache_callback,
                                        set_cache_callback, noise_inverse_renoise_strength,
                                        noise_inverse_renoise_kernel)
        if not enable_bbox_control or draw_background:
            delegate.init_grid_bbox(tile_width, tile_height, overlap, tile_batch_size)
        if enable_bbox_control:
            delegate.init_custom_bbox(bbox_settings, draw_background, causal_layers)
        if self.controlnet_script:
            delegate.init_controlnet(self.controlnet_script, control_tensor_cpu)

        # init everything done, perform sanity check & pre-computations
        delegate.init_done()
        # hijack the behaviours
        delegate.hook()

        self.delegate = delegate

        info = (
                f"{method.value} hooked into {name!r} sampler, " +
                f"Tile size: {tile_width}x{tile_height}, " +
                f"Tile batches: {len(self.delegate.batched_bboxes)}, " +
                f"Batch size: {tile_batch_size}."
        )
        exts = [
            f"NoiseInv" if flag_noise_inverse else None,
            f"RegionCtrl" if enable_bbox_control else None,
            f"ContrlNet" if self.controlnet_script else None,
        ]
        ext_info = ', '.join([e for e in exts if e])
        if ext_info: ext_info = f' (ext: {ext_info})'
        print(info + ext_info)

        return delegate.sampler_raw

    def create_random_tensors_hijack(
            self, bbox_settings,
            shape, seeds, subseeds=None, subseed_strength=0.0, seed_resize_from_h=0, seed_resize_from_w=0, p=None,
    ):
        org_random_tensors = processing.create_random_tensors_original_md(shape, seeds, subseeds, subseed_strength,
                                                                          seed_resize_from_h, seed_resize_from_w, p)
        height, width = shape[1], shape[2]
        background_noise = torch.zeros_like(org_random_tensors)
        background_noise_count = torch.zeros((1, 1, height, width), device=org_random_tensors.device)
        foreground_noise = torch.zeros_like(org_random_tensors)
        foreground_noise_count = torch.zeros((1, 1, height, width), device=org_random_tensors.device)

        for _, v in bbox_settings.items():
            seed = get_fixed_seed(v.seed)  # NOTE: fix seed here, cannot fillback to png info
            x, y, w, h = v.x, v.y, v.w, v.h
            # convert to pixel
            x = int(x * width)
            y = int(y * height)
            w = math.ceil(w * width)
            h = math.ceil(h * height)
            # clamp
            x = max(0, x)
            y = max(0, y)
            w = min(width - x, w)
            h = min(height - y, h)
            # create random tensor
            torch.manual_seed(seed)
            rand_tensor = torch.randn((1, org_random_tensors.shape[1], h, w), device=devices.cpu)
            if BlendMode(v.blend_mode) == BlendMode.BACKGROUND:
                background_noise[:, :, y:y + h, x:x + w] += rand_tensor.to(background_noise.device)
                background_noise_count[:, :, y:y + h, x:x + w] += 1
            elif BlendMode(v.blend_mode) == BlendMode.FOREGROUND:
                foreground_noise[:, :, y:y + h, x:x + w] += rand_tensor.to(foreground_noise.device)
                foreground_noise_count[:, :, y:y + h, x:x + w] += 1
            else:
                raise NotImplementedError

        # average
        background_noise = torch.where(background_noise_count > 1, background_noise / background_noise_count,
                                       background_noise)
        foreground_noise = torch.where(foreground_noise_count > 1, foreground_noise / foreground_noise_count,
                                       foreground_noise)
        # paste two layers to original random tensor
        org_random_tensors = torch.where(background_noise_count > 0, background_noise, org_random_tensors)
        org_random_tensors = torch.where(foreground_noise_count > 0, foreground_noise, org_random_tensors)
        return org_random_tensors

    ''' ↓↓↓ helper methods ↓↓↓ '''

    def dump_regions(self, cfg_name, *bbox_controls):
        if not cfg_name: return gr_value(f'<span style="color:red">Config file name cannot be empty.</span>',
                                         visible=True)

        bbox_settings = build_bbox_settings(bbox_controls)
        data = {'bbox_controls': [v._asdict() for v in bbox_settings.values()]}

        CFG_PATH.mkdir(exist_ok=True, parents=True)
        fp = CFG_PATH / cfg_name
        with open(fp, 'w', encoding='utf-8') as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)

        return gr_value(f'Config saved to {fp}.', visible=True)

    def load_regions(self, ref_image, cfg_name, *bbox_controls):
        if ref_image is None:
            return [gr_value(v) for v in bbox_controls] + [
                gr_value(f'<span style="color:red">Please create or upload a ref image first.</span>', visible=True)]
        fp = CFG_PATH / cfg_name
        if not fp.exists():
            return [gr_value(v) for v in bbox_controls] + [
                gr_value(f'<span style="color:red">Config {fp} not found.</span>', visible=True)]

        try:
            with open(fp, 'r', encoding='utf-8') as fh:
                data = json.load(fh)
        except Exception as e:
            return [gr_value(v) for v in bbox_controls] + [
                gr_value(f'<span style="color:red">Failed to load config {fp}: {e}</span>', visible=True)]

        num_boxes = len(data['bbox_controls'])
        data_list = []
        for i in range(BBOX_MAX_NUM):
            if i < num_boxes:
                for k in BBoxSettings._fields:
                    if k in data['bbox_controls'][i]:
                        data_list.append(data['bbox_controls'][i][k])
                    else:
                        data_list.append(None)
            else:
                data_list.extend(DEFAULT_BBOX_SETTINGS)

        return [gr_value(v) for v in data_list] + [gr_value(f'Config loaded from {fp}.', visible=True)]

    def noise_inverse_set_cache(self, p: ProcessingImg2Img, x0: Tensor, xt: Tensor, prompts: List[str], steps: int,
                                retouch: float):
        self.noise_inverse_cache = NoiseInverseCache(p.sd_model.sd_model_hash, x0, xt, steps, retouch, prompts)

    def noise_inverse_get_cache(self):
        return self.noise_inverse_cache

    def reset(self):
        ''' unhijack inner APIs '''
        if hasattr(sd_samplers, "create_sampler_original_md"):
            sd_samplers.create_sampler = sd_samplers.create_sampler_original_md
            del sd_samplers.create_sampler_original_md
        if hasattr(processing, "create_random_tensors_original_md"):
            processing.create_random_tensors = processing.create_random_tensors_original_md
            del processing.create_random_tensors_original_md
        MultiDiffusion.unhook()
        MixtureOfDiffusers.unhook()
        self.delegate = None

    def reset_and_gc(self):
        self.reset()
        self.noise_inverse_cache = None

        import gc;
        gc.collect()
        devices.torch_gc()

        try:
            import os
            import psutil
            mem = psutil.Process(os.getpid()).memory_info()
            print(f'[Mem] rss: {mem.rss / 2 ** 30:.3f} GB, vms: {mem.vms / 2 ** 30:.3f} GB')
            from modules.shared import mem_mon as vram_mon
            free, total = vram_mon.cuda_mem_get_info()
            print(f'[VRAM] free: {free / 2 ** 30:.3f} GB, total: {total / 2 ** 30:.3f} GB')
        except:
            pass