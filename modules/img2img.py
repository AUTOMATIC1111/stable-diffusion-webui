import os
import itertools # SBM Batch frames
import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageEnhance, ImageChops, UnidentifiedImageError
import modules.scripts
from modules import shared, processing, images
from modules.generation_parameters_copypaste import create_override_settings_dict
from modules.ui import plaintext_to_html
from modules.memstats import memory_stats


debug = shared.log.trace if os.environ.get('SD_PROCESS_DEBUG', None) is not None else lambda *args, **kwargs: None
debug('Trace: PROCESS')


def process_batch(p, input_files, input_dir, output_dir, inpaint_mask_dir, args):
    shared.log.debug(f'batch: {input_files}|{input_dir}|{output_dir}|{inpaint_mask_dir}')
    processing.fix_seed(p)
    if input_files is not None and len(input_files) > 0:
        image_files = [f.name for f in input_files]
    else:
        if not os.path.isdir(input_dir):
            shared.log.error(f"Process batch: directory not found: {input_dir}")
            return
        image_files = os.listdir(input_dir)
    is_inpaint_batch = False
    if inpaint_mask_dir:
        inpaint_masks = os.listdir(inpaint_mask_dir)
        is_inpaint_batch = len(inpaint_masks) > 0
    if is_inpaint_batch:
        shared.log.info(f"Process batch: inpaint batch masks={len(inpaint_masks)}")
    save_normally = output_dir == ''
    p.do_not_save_grid = True
    p.do_not_save_samples = not save_normally
    shared.state.job_count = len(image_files) * p.n_iter
    if shared.opts.batch_frame_mode: # SBM Frame mode is on, process each image in batch with same seed
        window_size = p.batch_size
        btcrept = 1
        p.seed = [p.seed] * window_size # SBM MONKEYPATCH: Need to change processing to support a fixed seed value.
        p.subseed = [p.subseed] * window_size # SBM MONKEYPATCH
        shared.log.info(f"Process batch: inputs={len(image_files)} parallel={window_size} outputs={p.n_iter} per input ")
    else: # SBM Frame mode is off, standard operation of repeating same images with sequential seed.
        window_size = 1
        btcrept = p.batch_size
        shared.log.info(f"Process batch: inputs={len(image_files)} outputs={p.n_iter * p.batch_size} per input")
    for i in range(0, len(image_files), window_size):
        if shared.state.skipped:
            shared.state.skipped = False
        if shared.state.interrupted:
            break
        batch_image_files = image_files[i:i+window_size]
        batch_images = []
        for image_file in batch_image_files:
            try:
                img = Image.open(image_file)
                if p.scale_by != 1:
                    p.width = int(img.width * p.scale_by)
                    p.height = int(img.height * p.scale_by)
            except UnidentifiedImageError as e:
                shared.log.error(f"Image error: {e}")
                continue
            img = ImageOps.exif_transpose(img)
            batch_images.append(img)
        batch_images = batch_images * btcrept # Standard mode sends the same image per batchsize.
        p.init_images = batch_images

        if is_inpaint_batch:
            # try to find corresponding mask for an image using simple filename matching
            batch_mask_images = []
            for image_file in batch_image_files:
                mask_image_path = os.path.join(inpaint_mask_dir, os.path.basename(image_file))
                # if not found use first one ("same mask for all images" use-case)
                if mask_image_path not in inpaint_masks:
                    mask_image_path = inpaint_masks[0]
                mask_image = Image.open(mask_image_path)
                batch_mask_images.append(mask_image)
            batch_mask_images = batch_mask_images * btcrept
            p.image_mask = batch_mask_images

        batch_image_files = batch_image_files * btcrept # List used for naming later.

        proc = modules.scripts.scripts_img2img.run(p, *args)
        if proc is None:
            proc = processing.process_images(p)
        for n, (image, image_file) in enumerate(itertools.zip_longest(proc.images,batch_image_files)):
            basename = ''
            if shared.opts.use_original_name_batch:
                forced_filename, ext = os.path.splitext(os.path.basename(image_file))
            else:
                forced_filename = None
                ext = shared.opts.samples_format
            if len(proc.images) > 1:
                basename = f'{n + i}' if shared.opts.batch_frame_mode else f'{n}'
            else:
                basename = ''
            if output_dir == '':
                output_dir = shared.opts.outdir_img2img_samples
            if not save_normally:
                os.makedirs(output_dir, exist_ok=True)
            geninfo, items = images.read_info_from_image(image)
            for k, v in items.items():
                image.info[k] = v
            images.save_image(image, path=output_dir, basename=basename, seed=None, prompt=None, extension=ext, info=geninfo, short_filename=True, no_prompt=True, grid=False, pnginfo_section_name="extras", existing_info=image.info, forced_filename=forced_filename)
        shared.log.debug(f'Processed: images={len(batch_image_files)} memory={memory_stats()} batch')


def img2img(id_task: str, mode: int,
            prompt, negative_prompt, prompt_styles,
            init_img,
            sketch,
            init_img_with_mask,
            inpaint_color_sketch,
            inpaint_color_sketch_orig,
            init_img_inpaint,
            init_mask_inpaint,
            steps,
            sampler_index,
            mask_blur, mask_alpha,
            inpainting_fill,
            full_quality, restore_faces, tiling,
            n_iter, batch_size,
            cfg_scale, image_cfg_scale,
            diffusers_guidance_rescale,
            sag_scale,
            refiner_start,
            clip_skip,
            denoising_strength,
            seed, subseed, subseed_strength, seed_resize_from_h, seed_resize_from_w,
            selected_scale_tab,
            height, width,
            scale_by,
            resize_mode, resize_name,
            inpaint_full_res, inpaint_full_res_padding, inpainting_mask_invert,
            img2img_batch_files, img2img_batch_input_dir, img2img_batch_output_dir, img2img_batch_inpaint_mask_dir,
            hdr_clamp, hdr_boundary, hdr_threshold, hdr_center, hdr_channel_shift, hdr_full_shift, hdr_maximize, hdr_max_center, hdr_max_boundry,
            override_settings_texts,
            *args): # pylint: disable=unused-argument

    if shared.sd_model is None:
        shared.log.warning('Model not loaded')
        return [], '', '', 'Error: model not loaded'

    debug(f'img2img: id_task={id_task}|mode={mode}|prompt={prompt}|negative_prompt={negative_prompt}|prompt_styles={prompt_styles}|init_img={init_img}|sketch={sketch}|init_img_with_mask={init_img_with_mask}|inpaint_color_sketch={inpaint_color_sketch}|inpaint_color_sketch_orig={inpaint_color_sketch_orig}|init_img_inpaint={init_img_inpaint}|init_mask_inpaint={init_mask_inpaint}|steps={steps}|sampler_index={sampler_index}||mask_blur={mask_blur}|mask_alpha={mask_alpha}|inpainting_fill={inpainting_fill}|full_quality={full_quality}|restore_faces={restore_faces}|tiling={tiling}|n_iter={n_iter}|batch_size={batch_size}|cfg_scale={cfg_scale}|image_cfg_scale={image_cfg_scale}|clip_skip={clip_skip}|denoising_strength={denoising_strength}|seed={seed}|subseed{subseed}|subseed_strength={subseed_strength}|seed_resize_from_h={seed_resize_from_h}|seed_resize_from_w={seed_resize_from_w}|selected_scale_tab={selected_scale_tab}|height={height}|width={width}|scale_by={scale_by}|resize_mode={resize_mode}|resize_name={resize_name}|inpaint_full_res={inpaint_full_res}|inpaint_full_res_padding={inpaint_full_res_padding}|inpainting_mask_invert={inpainting_mask_invert}|img2img_batch_files={img2img_batch_files}|img2img_batch_input_dir={img2img_batch_input_dir}|img2img_batch_output_dir={img2img_batch_output_dir}|img2img_batch_inpaint_mask_dir={img2img_batch_inpaint_mask_dir}|override_settings_texts={override_settings_texts}')

    if mode == 5:
        if img2img_batch_files is None or len(img2img_batch_files) == 0:
            shared.log.debug('Init bactch images not set')
        elif init_img:
            shared.log.debug('Init image not set')

    if sampler_index is None:
        sampler_index = 0

    override_settings = create_override_settings_dict(override_settings_texts)

    if mode == 0:  # img2img
        if init_img is None:
            return [], '', '', 'Error: init image not provided'
        image = init_img.convert("RGB")
        mask = None
    elif mode == 1:  # img2img sketch
        if sketch is None:
            return [], '', '', 'Error: sketch image not provided'
        image = sketch.convert("RGB")
        mask = None
    elif mode == 2:  # inpaint
        if init_img_with_mask is None:
            return [], '', '', 'Error: init image with mask not provided'
        image = init_img_with_mask["image"]
        mask = init_img_with_mask["mask"]
        alpha_mask = ImageOps.invert(image.split()[-1]).convert('L').point(lambda x: 255 if x > 0 else 0, mode='1')
        mask = ImageChops.lighter(alpha_mask, mask.convert('L')).convert('L')
        image = image.convert("RGB")
    elif mode == 3:  # inpaint sketch
        if inpaint_color_sketch is None:
            return [], '', '', 'Error: color sketch image not provided'
        image = inpaint_color_sketch
        orig = inpaint_color_sketch_orig or inpaint_color_sketch
        pred = np.any(np.array(image) != np.array(orig), axis=-1)
        mask = Image.fromarray(pred.astype(np.uint8) * 255, "L")
        mask = ImageEnhance.Brightness(mask).enhance(1 - mask_alpha / 100)
        blur = ImageFilter.GaussianBlur(mask_blur)
        image = Image.composite(image.filter(blur), orig, mask.filter(blur))
        image = image.convert("RGB")
    elif mode == 4:  # inpaint upload mask
        if init_img_inpaint is None:
            return [], '', '', 'Error: inpaint image not provided'
        image = init_img_inpaint
        mask = init_mask_inpaint
    else:
        shared.log.error(f'Image processing unknown mode: {mode}')
        image = None
        mask = None
    if image is not None:
        image = ImageOps.exif_transpose(image)
        if selected_scale_tab == 1 and resize_mode != 0:
            width = int(image.width * scale_by)
            height = int(image.height * scale_by)

    p = processing.StableDiffusionProcessingImg2Img(
        sd_model=shared.sd_model,
        outpath_samples=shared.opts.outdir_samples or shared.opts.outdir_img2img_samples,
        outpath_grids=shared.opts.outdir_grids or shared.opts.outdir_img2img_grids,
        prompt=prompt,
        negative_prompt=negative_prompt,
        styles=prompt_styles,
        seed=seed,
        subseed=subseed,
        subseed_strength=subseed_strength,
        seed_resize_from_h=seed_resize_from_h,
        seed_resize_from_w=seed_resize_from_w,
        seed_enable_extras=True,
        sampler_name = processing.get_sampler_name(sampler_index, img=True),
        batch_size=batch_size,
        n_iter=n_iter,
        steps=steps,
        cfg_scale=cfg_scale,
        clip_skip=clip_skip,
        width=width,
        height=height,
        full_quality=full_quality,
        restore_faces=restore_faces,
        tiling=tiling,
        init_images=[image],
        mask=mask,
        mask_blur=mask_blur,
        inpainting_fill=inpainting_fill,
        resize_mode=resize_mode,
        resize_name=resize_name,
        denoising_strength=denoising_strength,
        image_cfg_scale=image_cfg_scale,
        diffusers_guidance_rescale=diffusers_guidance_rescale,
        sag_scale=sag_scale,
        refiner_start=refiner_start,
        inpaint_full_res=inpaint_full_res != 0,
        inpaint_full_res_padding=inpaint_full_res_padding,
        inpainting_mask_invert=inpainting_mask_invert,
        hdr_clamp=hdr_clamp, hdr_boundary=hdr_boundary, hdr_threshold=hdr_threshold,
        hdr_center=hdr_center, hdr_channel_shift=hdr_channel_shift, hdr_full_shift=hdr_full_shift,
        hdr_maximize=hdr_maximize, hdr_max_center=hdr_max_center, hdr_max_boundry=hdr_max_boundry,
        override_settings=override_settings,
    )
    if selected_scale_tab == 1 and resize_mode != 0:
        p.scale_by = scale_by
    p.scripts = modules.scripts.scripts_img2img
    p.script_args = args
    if mask:
        p.extra_generation_params["Mask blur"] = mask_blur
        p.extra_generation_params["Mask alpha"] = mask_alpha
        p.extra_generation_params["Mask invert"] = inpainting_mask_invert
        p.extra_generation_params["Mask content"] = inpainting_fill
        p.extra_generation_params["Mask area"] = inpaint_full_res
        p.extra_generation_params["Mask padding"] = inpaint_full_res_padding
    p.is_batch = mode == 5
    if p.is_batch:
        process_batch(p, img2img_batch_files, img2img_batch_input_dir, img2img_batch_output_dir, img2img_batch_inpaint_mask_dir, args)
        processed = processing.Processed(p, [], p.seed, "")
    else:
        processed = modules.scripts.scripts_img2img.run(p, *args)
        if processed is None:
            processed = processing.process_images(p)
    p.close()
    generation_info_js = processed.js() if processed is not None else ''
    return processed.images, generation_info_js, processed.info, plaintext_to_html(processed.comments)
