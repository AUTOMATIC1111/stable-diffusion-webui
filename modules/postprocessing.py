import os
import tempfile
from typing import List

from PIL import Image

from modules import shared, images, devices, scripts, scripts_postprocessing, generation_parameters_copypaste
from modules.shared import opts


def run_postprocessing(extras_mode, image, image_folder: List[tempfile.NamedTemporaryFile], input_dir, output_dir, show_extras_results, *args, save_output: bool = True):
    devices.torch_gc()
    shared.state.begin('extras')
    image_data = []
    image_names = []
    image_fullnames = []
    image_ext = []
    outputs = []
    params = {}
    infotext = ''
    if extras_mode == 1:
        for img in image_folder:
            if isinstance(img, Image.Image):
                image = img
                fn = ''
                ext = None
            else:
                try:
                    image = Image.open(os.path.abspath(img.name))
                except Exception as e:
                    shared.log.error(f'Failed to open image: file="{img.name}" {e}')
                    continue
                fn, ext = os.path.splitext(img.orig_name)
                image_fullnames.append(img.name)
            image_data.append(image)
            image_names.append(fn)
            image_ext.append(ext)
        shared.log.debug(f'Process: mode=batch inputs={len(image_folder)} images={len(image_data)}')
    elif extras_mode == 2:
        assert not shared.cmd_opts.hide_ui_dir_config, '--hide-ui-dir-config option must be disabled'
        assert input_dir, 'input directory not selected'
        image_list = os.listdir(input_dir)
        for filename in image_list:
            fn = os.path.join(input_dir, filename)
            try:
                image = Image.open(fn)
            except Exception as e:
                shared.log.error(f'Failed to open image: file="{fn}" {e}')
                continue
            image_fullnames.append(fn)
            image_data.append(image)
            image_names.append(fn)
            image_ext.append(None)
        shared.log.debug(f'Process: mode=folder inputs={input_dir} files={len(image_list)} images={len(image_data)}')
    else:
        image_data.append(image)
        image_names.append(None)
        image_ext.append(None)
    if extras_mode == 2 and output_dir != '':
        outpath = output_dir
    else:
        outpath = opts.outdir_samples or opts.outdir_extras_samples
    processed_images = []
    for image, name, ext in zip(image_data, image_names, image_ext): # pylint: disable=redefined-argument-from-local
        shared.log.debug(f'Process: image={image} {args}')
        infotext = ''
        if shared.state.interrupted:
            shared.log.debug('Postprocess interrupted')
            break
        if image is None:
            continue
        shared.state.textinfo = name
        pp = scripts_postprocessing.PostprocessedImage(image.convert("RGB"))
        scripts.scripts_postproc.run(pp, args)
        geninfo, items = images.read_info_from_image(image)
        params = generation_parameters_copypaste.parse_generation_parameters(geninfo)
        for k, v in items.items():
            pp.image.info[k] = v
        if 'parameters' in items:
            infotext = items['parameters'] + ', '
        infotext = infotext + ", ".join([k if k == v else f'{k}: {generation_parameters_copypaste.quote(v)}' for k, v in pp.info.items() if v is not None])
        pp.image.info["postprocessing"] = infotext
        processed_images.append(pp.image)
        if save_output:
            if opts.use_original_name_batch and name is not None:
                forced_filename = os.path.splitext(os.path.basename(name))[0]
                images.save_image(pp.image, path=outpath, extension=ext or opts.samples_format, info=infotext, short_filename=True, no_prompt=True, grid=False, pnginfo_section_name="extras", existing_info=pp.image.info, forced_filename=forced_filename)
            else:
                images.save_image(pp.image, path=outpath, extension=ext or opts.samples_format, info=infotext, short_filename=True, no_prompt=True, grid=False, pnginfo_section_name="extras", existing_info=pp.image.info)
        if extras_mode != 2 or show_extras_results:
            outputs.append(pp.image)
        image.close()
    scripts.scripts_postproc.postprocess(processed_images, args)

    devices.torch_gc()
    return outputs, infotext, params


def run_extras(extras_mode, resize_mode, image, image_folder, input_dir, output_dir, show_extras_results, gfpgan_visibility, codeformer_visibility, codeformer_weight, upscaling_resize, upscaling_resize_w, upscaling_resize_h, upscaling_crop, extras_upscaler_1, extras_upscaler_2, extras_upscaler_2_visibility, upscale_first: bool, save_output: bool = True): #pylint: disable=unused-argument
    """old handler for API"""

    args = scripts.scripts_postproc.create_args_for_run({
        "Upscale": {
            "upscale_mode": resize_mode,
            "upscale_by": upscaling_resize,
            "upscale_to_width": upscaling_resize_w,
            "upscale_to_height": upscaling_resize_h,
            "upscale_crop": upscaling_crop,
            "upscaler_1_name": extras_upscaler_1,
            "upscaler_2_name": extras_upscaler_2,
            "upscaler_2_visibility": extras_upscaler_2_visibility,
        },
        "GFPGAN": {
            "gfpgan_visibility": gfpgan_visibility,
        },
        "CodeFormer": {
            "codeformer_visibility": codeformer_visibility,
            "codeformer_weight": codeformer_weight,
        },
    })

    return run_postprocessing(extras_mode, image, image_folder, input_dir, output_dir, show_extras_results, *args, save_output=save_output)
