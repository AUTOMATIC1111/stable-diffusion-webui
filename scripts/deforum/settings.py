import os
import json

def load_args(args_dict,anim_args_dict, custom_settings_file, root):
    print(f"reading custom settings from {custom_settings_file}")
    if not os.path.isfile(custom_settings_file):
        print('The custom settings file does not exist. The in-notebook settings will be used instead')
    else:
        with open(custom_settings_file, "r") as f:
            jdata = json.loads(f.read())
            root.animation_prompts = jdata["prompts"]
            for i, k in enumerate(args_dict):
                if k in jdata:
                    args_dict[k] = jdata[k]
                else:
                    print(f"key {k} doesn't exist in the custom settings data! using the default value of {args_dict[k]}")
            for i, k in enumerate(anim_args_dict):
                if k in jdata:
                    anim_args_dict[k] = jdata[k]
                else:
                    print(f"key {k} doesn't exist in the custom settings data! using the default value of {anim_args_dict[k]}")
            print(args_dict)
            print(anim_args_dict)

import gradio as gr

# In gradio gui settings save/load
def save_settings(settings_path, override_settings_with_file, custom_settings_file, animation_mode, max_frames, border, angle, zoom, translation_x, translation_y, translation_z, rotation_3d_x, rotation_3d_y, rotation_3d_z, flip_2d_perspective, perspective_flip_theta, perspective_flip_phi, perspective_flip_gamma, perspective_flip_fv, noise_schedule, strength_schedule, contrast_schedule, cfg_scale_schedule, fov_schedule, near_schedule, far_schedule, seed_schedule, color_coherence, diffusion_cadence, use_depth_warping, midas_weight, near_plane, far_plane, fov, padding_mode, sampling_mode, save_depth_maps, video_init_path, extract_nth_frame, overwrite_extracted_frames, use_mask_video, video_mask_path, interpolate_key_frames, interpolate_x_frames, resume_from_timestring, resume_timestring, prompts, animation_prompts, W, H, restore_faces, tiling, enable_hr, firstphase_width, firstphase_height, seed, sampler, seed_enable_extras, subseed, subseed_strength, seed_resize_from_w, seed_resize_from_h, steps, ddim_eta, n_batch, make_grid, grid_rows, save_settings, save_samples, display_samples, save_sample_per_step, show_sample_per_step, override_these_with_webui, batch_name, filename_format, seed_behavior, use_init, from_img2img_instead_of_link, strength_0_no_init, strength, init_image, use_mask, use_alpha_as_mask, invert_mask, overlay_mask, mask_file, mask_brightness_adjust, mask_overlay_blur):
    from scripts.deforum.args import pack_args, pack_anim_args
    args_dict = pack_args(W, H, restore_faces, tiling, enable_hr, firstphase_width, firstphase_height, seed, sampler, seed_enable_extras, subseed, subseed_strength, seed_resize_from_w, seed_resize_from_h, steps, ddim_eta, n_batch, make_grid, grid_rows, save_settings, save_samples, display_samples, save_sample_per_step, show_sample_per_step, override_these_with_webui, batch_name, filename_format, seed_behavior, use_init, from_img2img_instead_of_link, strength_0_no_init, strength, init_image, use_mask, use_alpha_as_mask, invert_mask, overlay_mask, mask_file, mask_brightness_adjust, mask_overlay_blur)
    anim_args_dict = pack_anim_args(animation_mode, max_frames, border, angle, zoom, translation_x, translation_y, translation_z, rotation_3d_x, rotation_3d_y, rotation_3d_z, flip_2d_perspective, perspective_flip_theta, perspective_flip_phi, perspective_flip_gamma, perspective_flip_fv, noise_schedule, strength_schedule, contrast_schedule, cfg_scale_schedule, fov_schedule, near_schedule, far_schedule, seed_schedule, color_coherence, diffusion_cadence, use_depth_warping, midas_weight, near_plane, far_plane, fov, padding_mode, sampling_mode, save_depth_maps, video_init_path, extract_nth_frame, overwrite_extracted_frames, use_mask_video, video_mask_path, interpolate_key_frames, interpolate_x_frames, resume_from_timestring, resume_timestring)
    #print(f"{animation_prompts}")
    args_dict["prompts"] = json.loads(animation_prompts)
    #print(f"{prompts}")
    print(f"saving custom settings to {settings_path}")
    with open(settings_path, "w") as f:
        f.write(json.dumps({**args_dict, **anim_args_dict}, ensure_ascii=False, indent=4))
    return [""]

def save_video_settings(video_settings_path, skip_video_for_run_all, fps, output_format, ffmpeg_location, add_soundtrack, soundtrack_path, use_manual_settings, render_steps, max_video_frames, path_name_modifier, image_path, mp4_path):
    from scripts.deforum.args import pack_video_args
    video_args_dict = pack_video_args(skip_video_for_run_all, fps, output_format, ffmpeg_location, add_soundtrack, soundtrack_path, use_manual_settings, render_steps, max_video_frames, path_name_modifier, image_path, mp4_path)
    print(f"saving video settings to {video_settings_path}")
    with open(video_settings_path, "w") as f:
        f.write(json.dumps(video_args_dict, ensure_ascii=False, indent=4))
    return [""]

def load_settings(settings_path, override_settings_with_file, custom_settings_file, animation_mode, max_frames, border, angle, zoom, translation_x, translation_y, translation_z, rotation_3d_x, rotation_3d_y, rotation_3d_z, flip_2d_perspective, perspective_flip_theta, perspective_flip_phi, perspective_flip_gamma, perspective_flip_fv, noise_schedule, strength_schedule, contrast_schedule, cfg_scale_schedule, fov_schedule, near_schedule, far_schedule, seed_schedule, color_coherence, diffusion_cadence, use_depth_warping, midas_weight, near_plane, far_plane, fov, padding_mode, sampling_mode, save_depth_maps, video_init_path, extract_nth_frame, overwrite_extracted_frames, use_mask_video, video_mask_path, interpolate_key_frames, interpolate_x_frames, resume_from_timestring, resume_timestring, prompts, animation_prompts, W, H, restore_faces, tiling, enable_hr, firstphase_width, firstphase_height, seed, sampler, seed_enable_extras, subseed, subseed_strength, seed_resize_from_w, seed_resize_from_h, steps, ddim_eta, n_batch, make_grid, grid_rows, save_settings, save_samples, display_samples, save_sample_per_step, show_sample_per_step, override_these_with_webui, batch_name, filename_format, seed_behavior, use_init, from_img2img_instead_of_link, strength_0_no_init, strength, init_image, use_mask, use_alpha_as_mask, invert_mask, overlay_mask, mask_file, mask_brightness_adjust, mask_overlay_blur):
    print(f"reading custom settings from {settings_path}")
    data = locals()
    data.pop("settings_path")
    jdata = {}
    if not os.path.isfile(settings_path):
        print('The custom settings file does not exist. The values will be unchanged.')
        return [override_settings_with_file, custom_settings_file, animation_mode, max_frames, border, angle, zoom, translation_x, translation_y, translation_z, rotation_3d_x, rotation_3d_y, rotation_3d_z, flip_2d_perspective, perspective_flip_theta, perspective_flip_phi, perspective_flip_gamma, perspective_flip_fv, noise_schedule, strength_schedule, contrast_schedule, cfg_scale_schedule, fov_schedule, near_schedule, far_schedule, seed_schedule, color_coherence, diffusion_cadence, use_depth_warping, midas_weight, near_plane, far_plane, fov, padding_mode, sampling_mode, save_depth_maps, video_init_path, extract_nth_frame, overwrite_extracted_frames, use_mask_video, video_mask_path, interpolate_key_frames, interpolate_x_frames, resume_from_timestring, resume_timestring, prompts, animation_prompts, W, H, restore_faces, tiling, enable_hr, firstphase_width, firstphase_height, seed, sampler, seed_enable_extras, subseed, subseed_strength, seed_resize_from_w, seed_resize_from_h, steps, ddim_eta, n_batch, make_grid, grid_rows, save_settings, save_samples, display_samples, save_sample_per_step, show_sample_per_step, override_these_with_webui, batch_name, filename_format, seed_behavior, use_init, from_img2img_instead_of_link, strength_0_no_init, strength, init_image, use_mask, use_alpha_as_mask, invert_mask, overlay_mask, mask_file, mask_brightness_adjust, mask_overlay_blur, ""]
    else:
        with open(settings_path, "r") as f:
            jdata = json.loads(f.read())
    ret = []

    if 'animation_prompts' in jdata:
        jdata['prompts'] = jdata['animation_prompts']#compatibility with old versions

    for key in data:
        if key == 'sampler':
            sampler_val = jdata[key]
            if type(sampler_val) == int:
                from modules.sd_samplers import samplers_for_img2img
                ret.append(samplers_for_img2img[sampler_val].name)
            else:
                ret.append(sampler_val)

        elif key in jdata:
            ret.append(jdata[key])
        else:
            if key == 'animation_prompts':
                ret.append(json.dumps(jdata['prompts'], ensure_ascii=False, indent=4))
            else:
                ret.append(data[key])

    #stuff
    ret.append("")

    return ret

def load_video_settings(video_settings_path, skip_video_for_run_all, fps, output_format, ffmpeg_location, add_soundtrack, soundtrack_path, use_manual_settings, render_steps, max_video_frames, path_name_modifier, image_path, mp4_path):  
    print(f"reading custom video settings from {video_settings_path}")
    data = locals()
    data.pop("video_settings_path")
    jdata = {}
    if not os.path.isfile(video_settings_path):
        print('The custom video settings file does not exist. The values will be unchanged.')
        return [skip_video_for_run_all, fps, output_format, ffmpeg_location, add_soundtrack, soundtrack_path, use_manual_settings, render_steps, max_video_frames, path_name_modifier, image_path, mp4_path, ""]
    else:
        with open(video_settings_path, "r") as f:
            jdata = json.loads(f.read())
    ret = []

    for key in data:
        if key in jdata:
            ret.append(jdata[key])
        else:
            ret.append(data[key])
    
    #stuff
    ret.append("")
    
    return ret