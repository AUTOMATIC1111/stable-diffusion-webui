import os
import tempfile
from typing import List

from PIL import Image

from modules import shared, images, devices, scripts, scripts_postprocessing, ui_common, generation_parameters_copypaste
from modules.shared import opts


def run_postprocessing(extras_mode, image, image_folder: List[tempfile.NamedTemporaryFile], input_dir, output_dir, show_extras_results, *args, save_output: bool = True):
    devices.torch_gc()
    shared.state.begin()
    shared.state.job = 'extras'
    image_data = []
    image_names = []
    image_ext = []
    outputs = []
    if extras_mode == 1:
        for img in image_folder:
            if isinstance(img, Image.Image):
                image = img
                fn = ''
                ext = None
            else:
                image = Image.open(os.path.abspath(img.name))
                fn, ext = os.path.splitext(img.orig_name)
            image_data.append(image)
            image_names.append(fn)
            image_ext.append(ext)
    elif extras_mode == 2:
        assert not shared.cmd_opts.hide_ui_dir_config, '--hide-ui-dir-config option must be disabled'
        assert input_dir, 'input directory not selected'
        image_list = shared.listfiles(input_dir)
        for filename in image_list:
            try:
                image = Image.open(filename)
            except Exception:
                continue
            image_data.append(image)
            image_names.append(filename)
            image_ext.append(None)
    else:
        image_data.append(image)
        image_names.append(None)
        image_ext.append(None)
    if extras_mode == 2 and output_dir != '':
        outpath = output_dir
    else:
        outpath = opts.outdir_samples or opts.outdir_extras_samples
    for image, name, ext in zip(image_data, image_names, image_ext):
        infotext = ''
        if shared.state.interrupted:
            shared.log.debug('Postprocess interrupted')
            break
        if image is None:
            continue
        shared.state.textinfo = name
        pp = scripts_postprocessing.PostprocessedImage(image.convert("RGB"))
        scripts.scripts_postproc.run(pp, args)
        if opts.use_original_name_batch and name is not None:
            basename = os.path.splitext(os.path.basename(name))[0]
        else:
            basename = ''
        _geninfo, items = images.read_info_from_image(image)
        for k, v in items.items():
            pp.image.info[k] = v
        if 'parameters' in items:
            infotext = items['parameters'] + ', '
        infotext = infotext + ", ".join([k if k == v else f'{k}: {generation_parameters_copypaste.quote(v)}' for k, v in pp.info.items() if v is not None])
        pp.image.info["postprocessing"] = infotext
        if save_output:
            images.save_image(pp.image, path=outpath, basename=basename, seed=None, prompt=None, extension=ext or opts.samples_format, info=infotext, short_filename=True, no_prompt=True, grid=False, pnginfo_section_name="extras", existing_info=pp.image.info, forced_filename=None)
        if extras_mode != 2 or show_extras_results:
            outputs.append(pp.image)

    devices.torch_gc()
    return outputs, ui_common.infotext_to_html(infotext), pp.image.info


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
