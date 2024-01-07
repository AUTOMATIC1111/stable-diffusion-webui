import os
import time
import math
from typing import List, Union
import cv2
import numpy as np
import diffusers
from PIL import Image
from modules.control import util
from modules.control import unit
from modules.control import processors
from modules.control.units import controlnet # lllyasviel ControlNet
from modules.control.units import xs # VisLearn ControlNet-XS
from modules.control.units import lite # Kohya ControlLLLite
from modules.control.units import t2iadapter # TencentARC T2I-Adapter
from modules.control.units import reference # ControlNet-Reference
from scripts import ipadapter # pylint: disable=no-name-in-module
from modules import devices, shared, errors, processing, images, sd_models, scripts # pylint: disable=ungrouped-imports


debug = shared.log.trace if os.environ.get('SD_CONTROL_DEBUG', None) is not None else lambda *args, **kwargs: None
debug('Trace: CONTROL')
pipe = None
original_pipeline = None


class ControlProcessing(processing.StableDiffusionProcessingImg2Img):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.strength = None
        self.adapter_conditioning_scale = None
        self.adapter_conditioning_factor = None
        self.guess_mode = None
        self.controlnet_conditioning_scale = None
        self.control_guidance_start = None
        self.control_guidance_end = None
        self.reference_attn = None
        self.reference_adain = None
        self.attention_auto_machine_weight = None
        self.gn_auto_machine_weight = None
        self.style_fidelity = None
        self.ref_image = None
        self.image = None
        self.query_weight = None
        self.adain_weight = None
        self.adapter_conditioning_factor = 1.0
        self.attention = 'Attention'
        self.fidelity = 0.5
        self.override = None

    def sample(self, conditioning, unconditional_conditioning, seeds, subseeds, subseed_strength, prompts): # abstract
        pass

    def init_hr(self):
        if self.resize_name == 'None' or self.scale_by == 1.0:
            return
        self.is_hr_pass = True
        self.hr_force = True
        self.hr_upscaler = self.resize_name
        self.hr_upscale_to_x, self.hr_upscale_to_y = int(self.width * self.scale_by), int(self.height * self.scale_by)
        self.hr_upscale_to_x, self.hr_upscale_to_y = 8 * math.ceil(self.hr_upscale_to_x / 8), 8 * math.ceil(self.hr_upscale_to_y / 8)
        # hypertile_set(self, hr=True)
        shared.state.job_count = 2 * self.n_iter
        shared.log.debug(f'Control hires: upscaler="{self.hr_upscaler}" upscale={self.scale_by} size={self.hr_upscale_to_x}x{self.hr_upscale_to_y}')


def restore_pipeline():
    global pipe # pylint: disable=global-statement
    pipe = None
    if original_pipeline is not None:
        shared.sd_model = original_pipeline
        debug(f'Control restored pipeline: class={shared.sd_model.__class__.__name__}')
    devices.torch_gc()


def control_run(units: List[unit.Unit], inputs, inits, mask, unit_type: str, is_generator: bool, input_type: int,
                prompt, negative, styles, steps, sampler_index,
                seed, subseed, subseed_strength, seed_resize_from_h, seed_resize_from_w,
                cfg_scale, clip_skip, image_cfg_scale, diffusers_guidance_rescale, sag_scale, full_quality, restore_faces, tiling,
                hdr_clamp, hdr_boundary, hdr_threshold, hdr_center, hdr_channel_shift, hdr_full_shift, hdr_maximize, hdr_max_center, hdr_max_boundry,
                resize_mode_before, resize_name_before, width_before, height_before, scale_by_before, selected_scale_tab_before,
                resize_mode_after, resize_name_after, width_after, height_after, scale_by_after, selected_scale_tab_after,
                denoising_strength, batch_count, batch_size, mask_blur, mask_overlap,
                video_skip_frames, video_type, video_duration, video_loop, video_pad, video_interpolate,
                ip_adapter, ip_scale, ip_image,
                *input_script_args
        ):
    global pipe, original_pipeline # pylint: disable=global-statement
    debug(f'Control {unit_type}: input={inputs} init={inits} type={input_type}')
    if inputs is None or (type(inputs) is list and len(inputs) == 0):
        inputs = [None]
    output_images: List[Image.Image] = [] # output images
    active_process: List[processors.Processor] = [] # all active preprocessors
    active_model: List[Union[controlnet.ControlNet, xs.ControlNetXS, t2iadapter.Adapter]] = [] # all active models
    active_strength: List[float] = [] # strength factors for all active models
    active_start: List[float] = [] # start step for all active models
    active_end: List[float] = [] # end step for all active models
    processed_image: Image.Image = None # last processed image
    if mask is not None and input_type == 0:
        input_type = 1 # inpaint always requires control_image

    p = ControlProcessing(
        prompt = prompt,
        negative_prompt = negative,
        styles = styles,
        steps = steps,
        sampler_name = processing.get_sampler_name(sampler_index),
        hr_sampler_name = processing.get_sampler_name(sampler_index),
        seed = seed,
        subseed = subseed,
        subseed_strength = subseed_strength,
        seed_resize_from_h = seed_resize_from_h,
        seed_resize_from_w = seed_resize_from_w,
        cfg_scale = cfg_scale,
        clip_skip = clip_skip,
        image_cfg_scale = image_cfg_scale,
        diffusers_guidance_rescale = diffusers_guidance_rescale,
        sag_scale = sag_scale,
        full_quality = full_quality,
        restore_faces = restore_faces,
        tiling = tiling,
        hdr_clamp = hdr_clamp,
        hdr_boundary = hdr_boundary,
        hdr_threshold = hdr_threshold,
        hdr_center = hdr_center,
        hdr_channel_shift = hdr_channel_shift,
        hdr_full_shift = hdr_full_shift,
        hdr_maximize = hdr_maximize,
        hdr_max_center = hdr_max_center,
        hdr_max_boundry = hdr_max_boundry,
        resize_mode = resize_mode_before if resize_name_before != 'None' else 0,
        resize_name = resize_name_before,
        scale_by = scale_by_before,
        selected_scale_tab = selected_scale_tab_before,
        denoising_strength = denoising_strength,
        n_iter = batch_count,
        batch_size = batch_size,
        mask_blur=mask_blur,
        outpath_samples=shared.opts.outdir_samples or shared.opts.outdir_control_samples,
        outpath_grids=shared.opts.outdir_grids or shared.opts.outdir_control_grids,
    )
    processing.process_init(p)

    if resize_mode_before != 0 or inputs is None or inputs == [None]:
        p.width, p.height = width_before, height_before # pylint: disable=attribute-defined-outside-init
    else:
        del p.width
        del p.height

    t0 = time.time()
    for u in units:
        if not u.enabled or u.type != unit_type:
            continue
        if unit_type == 'adapter' and u.adapter.model is not None:
            active_process.append(u.process)
            active_model.append(u.adapter)
            active_strength.append(float(u.strength))
            p.adapter_conditioning_factor = u.factor
            shared.log.debug(f'Control T2I-Adapter unit: process={u.process.processor_id} model={u.adapter.model_id} strength={u.strength} factor={u.factor}')
        elif unit_type == 'controlnet' and u.controlnet.model is not None:
            active_process.append(u.process)
            active_model.append(u.controlnet)
            active_strength.append(float(u.strength))
            active_start.append(float(u.start))
            active_end.append(float(u.end))
            p.guess_mode = u.guess
            shared.log.debug(f'Control ControlNet unit: process={u.process.processor_id} model={u.controlnet.model_id} strength={u.strength} guess={u.guess} start={u.start} end={u.end}')
        elif unit_type == 'xs' and u.controlnet.model is not None:
            active_process.append(u.process)
            active_model.append(u.controlnet)
            active_strength.append(float(u.strength))
            active_start.append(float(u.start))
            active_end.append(float(u.end))
            shared.log.debug(f'Control ControlNet-XS unit: process={u.process.processor_id} model={u.controlnet.model_id} strength={u.strength} guess={u.guess} start={u.start} end={u.end}')
        elif unit_type == 'lite' and u.controlnet.model is not None:
            active_process.append(u.process)
            active_model.append(u.controlnet)
            active_strength.append(float(u.strength))
            shared.log.debug(f'Control ControlNet-XS unit: process={u.process.processor_id} model={u.controlnet.model_id} strength={u.strength} guess={u.guess} start={u.start} end={u.end}')
        elif unit_type == 'reference':
            p.override = u.override
            p.attention = u.attention
            p.query_weight = float(u.query_weight)
            p.adain_weight = float(u.adain_weight)
            p.fidelity = u.fidelity
            shared.log.debug('Control Reference unit')
        else:
            active_process.append(u.process)
            # active_model.append(model)
            active_strength.append(float(u.strength))
    p.ops.append('control')

    has_models = False
    selected_models: List[Union[controlnet.ControlNetModel, xs.ControlNetXSModel, t2iadapter.AdapterModel]] = None
    if unit_type == 'adapter' or unit_type == 'controlnet' or unit_type == 'xs' or unit_type == 'lite':
        if len(active_model) == 0:
            selected_models = None
        elif len(active_model) == 1:
            selected_models = active_model[0].model if active_model[0].model is not None else None
            p.extra_generation_params["Control model"] = (active_model[0].model_id or '') if active_model[0].model is not None else None
            has_models = selected_models is not None
        else:
            selected_models = [m.model for m in active_model if m.model is not None]
            p.extra_generation_params["Control model"] = ', '.join([(m.model_id or '') for m in active_model if m.model is not None])
            has_models = len(selected_models) > 0
        use_conditioning = active_strength[0] if len(active_strength) == 1 else list(active_strength) # strength or list[strength]
    else:
        pass

    debug(f'Control: run type={unit_type} models={has_models}')
    if unit_type == 'adapter' and has_models:
        p.extra_generation_params["Control mode"] = 'T2I-Adapter'
        p.extra_generation_params["Control conditioning"] = use_conditioning
        p.task_args['adapter_conditioning_scale'] = use_conditioning
        instance = t2iadapter.AdapterPipeline(selected_models, shared.sd_model)
        pipe = instance.pipeline
        if inits is not None:
            shared.log.warning('Control: T2I-Adapter does not support separate init image')
    elif unit_type == 'controlnet' and has_models:
        p.extra_generation_params["Control mode"] = 'ControlNet'
        p.extra_generation_params["Control conditioning"] = use_conditioning
        p.task_args['controlnet_conditioning_scale'] = use_conditioning
        p.task_args['control_guidance_start'] = active_start[0] if len(active_start) == 1 else list(active_start)
        p.task_args['control_guidance_end'] = active_end[0] if len(active_end) == 1 else list(active_end)
        p.task_args['guess_mode'] = p.guess_mode
        instance = controlnet.ControlNetPipeline(selected_models, shared.sd_model)
        pipe = instance.pipeline
    elif unit_type == 'xs' and has_models:
        p.extra_generation_params["Control mode"] = 'ControlNet-XS'
        p.extra_generation_params["Control conditioning"] = use_conditioning
        p.controlnet_conditioning_scale = use_conditioning
        p.control_guidance_start = active_start[0] if len(active_start) == 1 else list(active_start)
        p.control_guidance_end = active_end[0] if len(active_end) == 1 else list(active_end)
        instance = xs.ControlNetXSPipeline(selected_models, shared.sd_model)
        pipe = instance.pipeline
        if inits is not None:
            shared.log.warning('Control: ControlNet-XS does not support separate init image')
    elif unit_type == 'lite' and has_models:
        p.extra_generation_params["Control mode"] = 'ControlLLLite'
        p.extra_generation_params["Control conditioning"] = use_conditioning
        p.controlnet_conditioning_scale = use_conditioning
        instance = lite.ControlLLitePipeline(shared.sd_model)
        pipe = instance.pipeline
        if inits is not None:
            shared.log.warning('Control: ControlLLLite does not support separate init image')
    elif unit_type == 'reference':
        p.extra_generation_params["Control mode"] = 'Reference'
        p.extra_generation_params["Control attention"] = p.attention
        p.task_args['reference_attn'] = 'Attention' in p.attention
        p.task_args['reference_adain'] = 'Adain' in p.attention
        p.task_args['attention_auto_machine_weight'] = p.query_weight
        p.task_args['gn_auto_machine_weight'] = p.adain_weight
        p.task_args['style_fidelity'] = p.fidelity
        instance = reference.ReferencePipeline(shared.sd_model)
        pipe = instance.pipeline
        if inits is not None:
            shared.log.warning('Control: ControlNet-XS does not support separate init image')
    else: # run in txt2img/img2img mode
        if len(active_strength) > 0:
            p.strength = active_strength[0]
        pipe = diffusers.AutoPipelineForText2Image.from_pipe(shared.sd_model) # use set_diffuser_pipe
        instance = None

    debug(f'Control pipeline: class={pipe.__class__} args={vars(p)}')
    t1, t2, t3 = time.time(), 0, 0
    status = True
    frame = None
    video = None
    output_filename = None
    index = 0
    frames = 0

    original_pipeline = shared.sd_model
    if pipe is not None:
        shared.sd_model = pipe
        if not ((shared.opts.diffusers_model_cpu_offload or shared.cmd_opts.medvram) or (shared.opts.diffusers_seq_cpu_offload or shared.cmd_opts.lowvram)):
            shared.sd_model.to(shared.device)
        shared.sd_model.to(device=devices.device, dtype=devices.dtype)
        debug(f'Control device={devices.device} dtype={devices.dtype}')
        sd_models.copy_diffuser_options(shared.sd_model, original_pipeline) # copy options from original pipeline
        sd_models.set_diffuser_options(shared.sd_model)

    try:
        with devices.inference_context():
            if isinstance(inputs, str): # only video, the rest is a list
                if input_type == 2: # separate init image
                    if isinstance(inits, str) and inits != inputs:
                        shared.log.warning('Control: separate init video not support for video input')
                        input_type = 1
                try:
                    video = cv2.VideoCapture(inputs)
                    if not video.isOpened():
                        msg = f'Control: video open failed: path={inputs}'
                        shared.log.error(msg)
                        restore_pipeline()
                        return msg
                    frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = int(video.get(cv2.CAP_PROP_FPS))
                    w, h = int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    codec = util.decode_fourcc(video.get(cv2.CAP_PROP_FOURCC))
                    status, frame = video.read()
                    if status:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    shared.log.debug(f'Control: input video: path={inputs} frames={frames} fps={fps} size={w}x{h} codec={codec}')
                except Exception as e:
                    msg = f'Control: video open failed: path={inputs} {e}'
                    shared.log.error(msg)
                    restore_pipeline()
                    return msg

            while status:
                processed_image = None
                if frame is not None:
                    inputs = [Image.fromarray(frame)] # cv2 to pil
                for i, input_image in enumerate(inputs):
                    debug(f'Control Control image: {i + 1} of {len(inputs)}')
                    if shared.state.skipped:
                        shared.state.skipped = False
                        continue
                    if shared.state.interrupted:
                        shared.state.interrupted = False
                        restore_pipeline()
                        return 'Control interrupted'
                    # get input
                    if isinstance(input_image, str):
                        try:
                            input_image = Image.open(inputs[i])
                        except Exception as e:
                            shared.log.error(f'Control: image open failed: path={inputs[i]} type=control error={e}')
                            continue
                    # match init input
                    if input_type == 1:
                        debug('Control Init image: same as control')
                        init_image = input_image
                    elif inits is None:
                        debug('Control Init image: none')
                        init_image = None
                    elif isinstance(inits[i], str):
                        debug(f'Control: init image: {inits[i]}')
                        try:
                            init_image = Image.open(inits[i])
                        except Exception as e:
                            shared.log.error(f'Control: image open failed: path={inits[i]} type=init error={e}')
                            continue
                    else:
                        debug(f'Control Init image: {i % len(inits) + 1} of {len(inits)}')
                        init_image = inits[i % len(inits)]
                    index += 1
                    if video is not None and index % (video_skip_frames + 1) != 0:
                        continue

                    # resize before
                    if resize_mode_before != 0 and resize_name_before != 'None':
                        if selected_scale_tab_before == 1 and input_image is not None:
                            width_before, height_before = int(input_image.width * scale_by_before), int(input_image.height * scale_by_before)
                        if input_image is not None:
                            p.extra_generation_params["Control resize"] = f'{resize_name_before}'
                            debug(f'Control resize: op=before image={input_image} width={width_before} height={height_before} mode={resize_mode_before} name={resize_name_before}')
                            input_image = images.resize_image(resize_mode_before, input_image, width_before, height_before, resize_name_before)
                    if input_image is not None:
                        p.width = input_image.width
                        p.height = input_image.height
                        debug(f'Control: input image={input_image}')

                    # process
                    if input_image is None:
                        p.image = None
                        processed_image = None
                        debug(f'Control: process=None image={p.image} mask={mask}')
                    elif mask is not None and not has_models:
                        processed_image = mask
                        debug(f'Control: process=None image={p.image} mask={mask}')
                    elif len(active_process) == 0 and unit_type == 'reference':
                        p.ref_image = p.override or input_image
                        p.task_args['ref_image'] = p.ref_image
                        debug(f'Control: process=None image={p.ref_image}')
                        if p.ref_image is None:
                            msg = 'Control: attempting reference mode but image is none'
                            shared.log.error(msg)
                            restore_pipeline()
                            return msg
                        processed_image = p.ref_image
                    elif len(active_process) == 1:
                        image_mode = 'L' if unit_type == 'adapter' and len(active_model) > 0 and ('Canny' in active_model[0].model_id or 'Sketch' in active_model[0].model_id) else 'RGB'
                        p.image = active_process[0](input_image, image_mode)
                        p.task_args['image'] = p.image
                        p.extra_generation_params["Control process"] = active_process[0].processor_id
                        debug(f'Control: process={active_process[0].processor_id} image={p.image}')
                        if p.image is None:
                            msg = 'Control: attempting process but output is none'
                            shared.log.error(msg)
                            restore_pipeline()
                            return msg
                        processed_image = p.image
                    else:
                        if len(active_process) > 0:
                            p.image = []
                            for i, process in enumerate(active_process): # list[image]
                                image_mode = 'L' if unit_type == 'adapter' and len(active_model) > i and ('Canny' in active_model[i].model_id or 'Sketch' in active_model[i].model_id) else 'RGB'
                                p.image.append(process(input_image, image_mode))
                        else:
                            p.image = [input_image]
                        p.task_args['image'] = p.image
                        p.extra_generation_params["Control process"] = [p.processor_id for p in active_process]
                        debug(f'Control: process={[p.processor_id for p in active_process]} image={p.image}')
                        if any(img is None for img in p.image):
                            msg = 'Control: attempting process but output is none'
                            shared.log.error(msg)
                            restore_pipeline()
                            return msg
                        processed_image = [np.array(i) for i in p.image]
                        processed_image = util.blend(processed_image) # blend all processed images into one
                        processed_image = Image.fromarray(processed_image)

                    if unit_type == 'controlnet' and input_type == 1: # Init image same as control
                        p.task_args['image'] = input_image
                        p.task_args['control_image'] = p.image
                        p.task_args['strength'] = p.denoising_strength
                    elif unit_type == 'controlnet' and input_type == 2: # Separate init image
                        p.task_args['control_image'] = p.image
                        p.task_args['strength'] = p.denoising_strength
                        if init_image is None:
                            shared.log.warning('Control: separate init image not provided')
                            p.task_args['image'] = input_image
                        else:
                            p.task_args['image'] = init_image

                    if is_generator:
                        image_txt = f'{processed_image.width}x{processed_image.height}' if processed_image is not None else 'None'
                        msg = f'process | {index} of {frames if video is not None else len(inputs)} | {"Image" if video is None else "Frame"} {image_txt}'
                        debug(f'Control yield: {msg}')
                        yield (None, processed_image, f'Control {msg}')
                    t2 += time.time() - t2

                    # prepare pipeline
                    if hasattr(p, 'init_images'):
                        del p.init_images # control never uses init_image as-is
                    if pipe is not None:
                        if not has_models and (unit_type == 'controlnet' or unit_type == 'adapter' or unit_type == 'xs' or unit_type == 'lite'): # run in txt2img/img2img/inpaint mode
                            if mask is not None:
                                p.task_args['strength'] = denoising_strength
                                p.image_mask = mask
                                p.inpaint_full_res = False
                                p.init_images = [input_image]
                                # if mask_overlap > 0:
                                #    p.task_args['padding_mask_crop'] = mask_overlap # TODO enable once fixed in diffusers
                                shared.sd_model = sd_models.set_diffuser_pipe(shared.sd_model, sd_models.DiffusersTaskType.INPAINTING)
                            elif processed_image is not None:
                                p.init_images = [processed_image]
                                shared.sd_model = sd_models.set_diffuser_pipe(shared.sd_model, sd_models.DiffusersTaskType.IMAGE_2_IMAGE)
                            else:
                                p.init_hr()
                                shared.sd_model = sd_models.set_diffuser_pipe(shared.sd_model, sd_models.DiffusersTaskType.TEXT_2_IMAGE)
                        elif unit_type == 'reference':
                            p.is_control = True
                            shared.sd_model = sd_models.set_diffuser_pipe(shared.sd_model, sd_models.DiffusersTaskType.TEXT_2_IMAGE)
                        else: # actual control
                            p.is_control = True
                            if mask is not None:
                                p.task_args['strength'] = denoising_strength
                                p.image_mask = mask
                                p.inpaint_full_res = False
                                # if mask_overlap > 0:
                                #    p.task_args['padding_mask_crop'] = mask_overlap # TODO enable once fixed in diffusers
                                shared.sd_model = sd_models.set_diffuser_pipe(shared.sd_model, sd_models.DiffusersTaskType.INPAINTING) # only controlnet supports inpaint
                            elif 'control_image' in p.task_args:
                                shared.sd_model = sd_models.set_diffuser_pipe(shared.sd_model, sd_models.DiffusersTaskType.IMAGE_2_IMAGE) # only controlnet supports img2img
                            else:
                                shared.sd_model = sd_models.set_diffuser_pipe(shared.sd_model, sd_models.DiffusersTaskType.TEXT_2_IMAGE)
                            if unit_type == 'lite':
                                instance.apply(selected_models, p.image, use_conditioning)

                    # ip adapter
                    if ipadapter.apply(shared.sd_model, p, ip_adapter, ip_scale, ip_image or input_image):
                        original_pipeline.feature_extractor = shared.sd_model.feature_extractor
                        original_pipeline.image_encoder = shared.sd_model.image_encoder

                    # pipeline
                    output = None
                    if pipe is not None: # run new pipeline
                        debug(f'Control exec pipeline: task={sd_models.get_diffusers_task(pipe)} class={pipe.__class__}')
                        debug(f'Control exec pipeline: p={vars(p)}')
                        debug(f'Control exec pipeline: args={p.task_args} image={p.task_args.get("image", None)} control={p.task_args.get("control_image", None)} mask={p.task_args.get("mask_image", None)} ref={p.task_args.get("ref_image", None)}')
                        p.scripts = scripts.scripts_control
                        p.script_args = input_script_args
                        processed = p.scripts.run(p, *input_script_args)
                        if processed is None:
                            processed: processing.Processed = processing.process_images(p) # run actual pipeline
                        output = processed.images if processed is not None else None
                        # output = pipe(**vars(p)).images # alternative direct pipe exec call
                    else: # blend all processed images and return
                        output = [processed_image]
                    t3 += time.time() - t3

                    # outputs
                    if output is not None and len(output) > 0:
                        output_image = output[0]
                        if output_image is not None:

                            # resize after
                            if selected_scale_tab_after == 1:
                                width_after = int(output_image.width * scale_by_after)
                                height_after = int(output_image.height * scale_by_after)
                            if resize_mode_after != 0 and resize_name_after != 'None':
                                debug(f'Control resize: op=after image={output_image} width={width_after} height={height_after} mode={resize_mode_after} name={resize_name_after}')
                                output_image = images.resize_image(resize_mode_after, output_image, width_after, height_after, resize_name_after)

                            output_images.append(output_image)
                            if is_generator:
                                image_txt = f'{output_image.width}x{output_image.height}' if output_image is not None else 'None'
                                if video is not None:
                                    msg = f'Control output | {index} of {frames} skip {video_skip_frames} | Frame {image_txt}'
                                else:
                                    msg = f'Control output | {index} of {len(inputs)} | Image {image_txt}'
                                yield (output_image, processed_image, msg) # result is control_output, proces_output

                if video is not None and frame is not None:
                    status, frame = video.read()
                    if status:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    debug(f'Control: video frame={index} frames={frames} status={status} skip={index % (video_skip_frames + 1)} progress={index/frames:.2f}')
                else:
                    status = False

            if video is not None:
                video.release()

            shared.log.info(f'Control: pipeline units={len(active_model)} process={len(active_process)} time={t3-t0:.2f} init={t1-t0:.2f} proc={t2-t1:.2f} ctrl={t3-t2:.2f} outputs={len(output_images)}')
    except Exception as e:
        shared.log.error(f'Control pipeline failed: type={unit_type} units={len(active_model)} error={e}')
        errors.display(e, 'Control')

    shared.sd_model = original_pipeline
    pipe = None
    devices.torch_gc()

    if len(output_images) == 0:
        output_images = None
        image_txt = 'images=None'
    elif len(output_images) == 1:
        output_images = output_images[0]
        image_txt = f'| Images 1 | Size {output_images.width}x{output_images.height}' if output_image is not None else 'None'
    else:
        image_txt = f'| Images {len(output_images)} | Size {output_images[0].width}x{output_images[0].height}' if output_image is not None else 'None'

    if video_type != 'None' and isinstance(output_images, list):
        p.do_not_save_grid = True # pylint: disable=attribute-defined-outside-init
        output_filename = images.save_video(p, filename=None, images=output_images, video_type=video_type, duration=video_duration, loop=video_loop, pad=video_pad, interpolate=video_interpolate, sync=True)
        image_txt = f'| Frames {len(output_images)} | Size {output_images[0].width}x{output_images[0].height}'

    image_txt += f' | {util.dict2str(p.extra_generation_params)}'
    if hasattr(instance, 'restore'):
        instance.restore()
    restore_pipeline()
    debug(f'Control ready: {image_txt}')
    if is_generator:
        yield (output_images, processed_image, f'Control ready {image_txt}', output_filename)
    else:
        return (output_images, processed_image, f'Control ready {image_txt}', output_filename)
