import os

from PIL import Image

from modules import shared, images, devices, scripts, scripts_postprocessing, ui_common, generation_parameters_copypaste
from modules.shared import opts


def run_postprocessing(extras_mode, image, image_folder, input_dir, output_dir, show_extras_results, *args, save_output: bool = True):
    devices.torch_gc()

    shared.state.begin(job="extras")

    outputs = []

    def get_images(extras_mode, image, image_folder, input_dir):
        if extras_mode == 1:
            for img in image_folder:
                if isinstance(img, Image.Image):
                    image = img
                    fn = ''
                else:
                    image = Image.open(os.path.abspath(img.name))
                    fn = os.path.splitext(img.orig_name)[0]
                yield image, fn
        elif extras_mode == 2:
            assert not shared.cmd_opts.hide_ui_dir_config, '--hide-ui-dir-config option must be disabled'
            assert input_dir, 'input directory not selected'

            image_list = shared.listfiles(input_dir)
            for filename in image_list:
                yield filename, filename
        else:
            assert image, 'image not selected'
            yield image, None

    if extras_mode == 2 and output_dir != '':
        outpath = output_dir
    else:
        outpath = opts.outdir_samples or opts.outdir_extras_samples

    infotext = ''

    data_to_process = list(get_images(extras_mode, image, image_folder, input_dir))
    shared.state.job_count = len(data_to_process)

    for image_placeholder, name in data_to_process:
        image_data: Image.Image

        shared.state.nextjob()
        shared.state.textinfo = name
        shared.state.skipped = False

        if shared.state.interrupted:
            break

        if isinstance(image_placeholder, str):
            try:
                image_data = Image.open(image_placeholder)
            except Exception:
                continue
        else:
            image_data = image_placeholder

        shared.state.assign_current_image(image_data)

        parameters, existing_pnginfo = images.read_info_from_image(image_data)
        if parameters:
            existing_pnginfo["parameters"] = parameters

        initial_pp = scripts_postprocessing.PostprocessedImage(image_data.convert("RGB"))

        scripts.scripts_postproc.run(initial_pp, args)

        if shared.state.skipped:
            continue

        used_suffixes = {}
        for pp in [initial_pp, *initial_pp.extra_images]:
            suffix = pp.get_suffix(used_suffixes)

            if opts.use_original_name_batch and name is not None:
                basename = os.path.splitext(os.path.basename(name))[0]
                forced_filename = basename + suffix
            else:
                basename = ''
                forced_filename = None

            infotext = ", ".join([k if k == v else f'{k}: {generation_parameters_copypaste.quote(v)}' for k, v in pp.info.items() if v is not None])

            if opts.enable_pnginfo:
                pp.image.info = existing_pnginfo
                pp.image.info["postprocessing"] = infotext

            if save_output:
                fullfn, _ = images.save_image(pp.image, path=outpath, basename=basename, extension=opts.samples_format, info=infotext, short_filename=True, no_prompt=True, grid=False, pnginfo_section_name="extras", existing_info=existing_pnginfo, forced_filename=forced_filename, suffix=suffix)

                if pp.caption:
                    caption_filename = os.path.splitext(fullfn)[0] + ".txt"
                    if os.path.isfile(caption_filename):
                        with open(caption_filename, encoding="utf8") as file:
                            existing_caption = file.read().strip()
                    else:
                        existing_caption = ""

                    action = shared.opts.postprocessing_existing_caption_action
                    if action == 'Prepend' and existing_caption:
                        caption = f"{existing_caption} {pp.caption}"
                    elif action == 'Append' and existing_caption:
                        caption = f"{pp.caption} {existing_caption}"
                    elif action == 'Keep' and existing_caption:
                        caption = existing_caption
                    else:
                        caption = pp.caption

                    caption = caption.strip()
                    if caption:
                        with open(caption_filename, "w", encoding="utf8") as file:
                            file.write(caption)

            if extras_mode != 2 or show_extras_results:
                outputs.append(pp.image)

        image_data.close()

    devices.torch_gc()
    shared.state.end()
    return outputs, ui_common.plaintext_to_html(infotext), ''


def run_postprocessing_webui(id_task, *args, **kwargs):
    return run_postprocessing(*args, **kwargs)


def run_extras(extras_mode, resize_mode, image, image_folder, input_dir, output_dir, show_extras_results, gfpgan_visibility, codeformer_visibility, codeformer_weight, upscaling_resize, upscaling_resize_w, upscaling_resize_h, upscaling_crop, extras_upscaler_1, extras_upscaler_2, extras_upscaler_2_visibility, upscale_first: bool, save_output: bool = True):
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
            "enable": True,
            "gfpgan_visibility": gfpgan_visibility,
        },
        "CodeFormer": {
            "enable": True,
            "codeformer_visibility": codeformer_visibility,
            "codeformer_weight": codeformer_weight,
        },
    })

    return run_postprocessing(extras_mode, image, image_folder, input_dir, output_dir, show_extras_results, *args, save_output=save_output)
