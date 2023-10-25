import gc
import os
import copy
import glob
import cv2
import numpy as np
from PIL import Image
import torch
import gradio as gr
from collections import OrderedDict
from scipy.ndimage import binary_dilation
from modules import scripts, shared
from modules.ui import gr_show
from modules.safe import unsafe_torch_load, load
from modules.devices import device, torch_gc, cpu
from modules.paths import models_path
from handlers.clothes_repair.clothes_seg.predictor import SamPredictorHQ
from handlers.clothes_repair.clothes_seg.build_sam_hq import sam_model_registry
from handlers.clothes_repair.clothes_seg.clothes_dino import dino_model_list, dino_predict_internal, show_boxes, clear_dino_cache


sam_model_cache = OrderedDict()
# scripts_sam_model_dir = os.path.join(scripts.basedir(), "models/sam")
sd_sam_model_dir = os.path.join(models_path, "sam")
# sam_model_dir = sd_sam_model_dir if os.path.exists(sd_sam_model_dir) else scripts_sam_model_dir
sam_model_dir = sd_sam_model_dir
sam_model_list = [
    f for f in os.listdir(sam_model_dir) if os.path.isfile(os.path.join(sam_model_dir, f)) and f.split('.')[-1] != 'txt'
] if os.path.isdir(sd_sam_model_dir) else []
sam_device = device


def show_masks(image_np, masks: np.ndarray, alpha=0.5):
    image = copy.deepcopy(image_np)
    np.random.seed(0)
    for mask in masks:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        image[mask] = image[mask] * (1 - alpha) + 255 * color.reshape(1, 1, -1) * alpha
    return image.astype(np.uint8)


def load_sam_model(sam_checkpoint):
    model_type = sam_checkpoint.split('.')[0]
    if 'hq' not in model_type:
        model_type = '_'.join(model_type.split('_')[:-1])
    sam_checkpoint_path = os.path.join(sam_model_dir, sam_checkpoint)
    torch.load = unsafe_torch_load
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint_path)
    sam.to(device=sam_device)
    sam.eval()
    torch.load = load
    return sam


def clear_sam_cache():
    sam_model_cache.clear()
    gc.collect()
    torch_gc()


def clear_cache():
    clear_sam_cache()
    clear_dino_cache()
    # clear_sem_sam_cache()


def garbage_collect(sam):
    if shared.cmd_opts.lowvram:
        sam.to(cpu)
    gc.collect()
    torch_gc()


def init_sam_model(sam_model_name):
    print(f"Initializing SAM to {sam_device}")
    if sam_model_name in sam_model_cache:
        sam = sam_model_cache[sam_model_name]
        if shared.cmd_opts.lowvram or (str(sam_device) not in str(sam.device)):
            sam.to(device=sam_device)
        return sam
    elif sam_model_name in sam_model_list:
        clear_sam_cache()
        sam_model_cache[sam_model_name] = load_sam_model(sam_model_name)
        return sam_model_cache[sam_model_name]
    else:
        raise Exception(
            f"{sam_model_name} not found, please download model to models/sam.")


def create_mask_output(image_np, masks, boxes_filt):
    print("Creating output image")
    mask_images, masks_gallery, matted_images = [], [], []
    boxes_filt = boxes_filt.numpy().astype(int) if boxes_filt is not None else None
    for mask in masks:
        masks_gallery.append(Image.fromarray(np.any(mask, axis=0)))
        blended_image = show_masks(show_boxes(image_np, boxes_filt), mask)
        mask_images.append(Image.fromarray(blended_image))
        image_np_copy = copy.deepcopy(image_np)
        image_np_copy[~np.any(mask, axis=0)] = np.array([0, 0, 0, 0])
        matted_images.append(Image.fromarray(image_np_copy))
    return mask_images + masks_gallery + matted_images


def dilate_mask(mask, dilation_amt):
    x, y = np.meshgrid(np.arange(dilation_amt), np.arange(dilation_amt))
    center = dilation_amt // 2
    dilation_kernel = ((x - center)**2 + (y - center)**2 <= center**2).astype(np.uint8)
    dilated_binary_img = binary_dilation(mask, dilation_kernel)
    dilated_mask = Image.fromarray(dilated_binary_img.astype(np.uint8) * 255)
    return dilated_mask, dilated_binary_img


def create_mask_batch_output(
    input_image_file, dino_batch_dest_dir,
    image_np, masks, boxes_filt, batch_dilation_amt,
    dino_batch_save_image, dino_batch_save_mask, dino_batch_save_background, dino_batch_save_image_with_mask):
    print("Creating batch output image")
    filename, ext = os.path.splitext(os.path.basename(input_image_file))
    ext = ".png" # JPEG not compatible with RGBA
    for idx, mask in enumerate(masks):
        blended_image = show_masks(show_boxes(image_np, boxes_filt), mask)
        merged_mask = np.any(mask, axis=0)
        if dino_batch_save_background:
            merged_mask = ~merged_mask
        if batch_dilation_amt:
            _, merged_mask = dilate_mask(merged_mask, batch_dilation_amt)
        image_np_copy = copy.deepcopy(image_np)
        image_np_copy[~merged_mask] = np.array([0, 0, 0, 0])
        if dino_batch_save_image:
            output_image = Image.fromarray(image_np_copy)
            output_image.save(os.path.join(dino_batch_dest_dir, f"{filename}_{idx}_output{ext}"))
        if dino_batch_save_mask:
            output_mask = Image.fromarray(merged_mask)
            output_mask.save(os.path.join(dino_batch_dest_dir, f"{filename}_{idx}_mask{ext}"))
        if dino_batch_save_image_with_mask:
            output_blend = Image.fromarray(blended_image)
            output_blend.save(os.path.join(dino_batch_dest_dir, f"{filename}_{idx}_blend{ext}"))


def create_mask_batch_output_simplified(
    input_image_file, dino_batch_dest_dir,
    image_np, masks, boxes_filt, batch_dilation_amt,
    dino_batch_save_image, dino_batch_save_mask, dino_batch_save_background, dino_batch_save_image_with_mask):
    print("Creating batch output image")
    # filename, ext = os.path.splitext(os.path.basename(input_image_file))
    # ext = ".png" # JPEG not compatible with RGBA
    output_image_batch = []
    output_mask_batch = []
    for idx, mask in enumerate(masks):
        blended_image = show_masks(show_boxes(image_np, boxes_filt), mask)
        merged_mask = np.any(mask, axis=0)
        if dino_batch_save_background:
            merged_mask = ~merged_mask
        if batch_dilation_amt:
            _, merged_mask = dilate_mask(merged_mask, batch_dilation_amt)
        image_np_copy = copy.deepcopy(image_np)
        image_np_copy[~merged_mask] = np.array([0, 0, 0, 0])
        if dino_batch_save_image:
            output_image = Image.fromarray(image_np_copy)
            # output_image.save(os.path.join(dino_batch_dest_dir, f"{filename}_{idx}_output{ext}"))
            output_image_batch.append(output_image)
        if dino_batch_save_mask:
            output_mask = Image.fromarray(merged_mask)
            # output_mask.save(os.path.join(dino_batch_dest_dir, f"{filename}_{idx}_mask{ext}"))
            output_mask_batch.append(output_mask)
        if dino_batch_save_image_with_mask:
            output_blend = Image.fromarray(blended_image)
            # output_blend.save(os.path.join(dino_batch_dest_dir, f"{filename}_{idx}_blend{ext}"))
    # return output_image, output_mask
    return output_image_batch, output_mask_batch


def mask_predict(input_image):
    if input_image.mode != "RGBA":
        input_image = input_image.convert("RGBA")
    print("Start SAM Processing")
    sam_model_name = sam_model_list[0]
    positive_points = []
    negative_points = []
    if sam_model_name is None:
        return [], "SAM model not found. Please download SAM model from extension README."
    if input_image is None:
        return [], "SAM requires an input image. Please upload an image first."
    image_np = np.array(input_image)
    image_np_rgb = image_np[..., :3]
    text_prompt = "Clothing, Apparel, Garments, Costume"
    dino_enabled = text_prompt is not None
    box_threshold = 0.3
    dino_model_name = dino_model_list[0]
    boxes_filt, install_success = dino_predict_internal(input_image, dino_model_name, text_prompt, box_threshold)

    sam = init_sam_model(sam_model_name)
    print(f"Running SAM Inference {image_np_rgb.shape}")
    predictor = SamPredictorHQ(sam, 'hq' in sam_model_name)
    predictor.set_image(image_np_rgb)
    if dino_enabled and boxes_filt.shape[0] > 1:
        sam_predict_status = f"SAM inference with {boxes_filt.shape[0]} boxes, point prompts discarded"
        print(sam_predict_status)
        transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image_np.shape[:2])
        masks, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes.to(sam_device),
            multimask_output=False)
        masks = masks.permute(1, 0, 2, 3).cpu().numpy()
    else:
        num_box = 0 if boxes_filt is None else boxes_filt.shape[0]
        num_points = len(positive_points) + len(negative_points)
        if num_box == 0 and num_points == 0:
            garbage_collect(sam)
            return [], "You neither added point prompts nor enabled GroundingDINO. Segmentation cannot be generated."
        sam_predict_status = f"SAM inference with {num_box} box, {len(positive_points)} positive prompts, {len(negative_points)} negative prompts"
        print(sam_predict_status)
        point_coords = np.array(positive_points + negative_points)
        point_labels = np.array([1] * len(positive_points) + [0] * len(negative_points))
        box = copy.deepcopy(boxes_filt[0].numpy()) if boxes_filt is not None and boxes_filt.shape[0] > 0 else None
        masks, _, _ = predictor.predict(
            point_coords=point_coords if len(point_coords) > 0 else None,
            point_labels=point_labels if len(point_coords) > 0 else None,
            box=box,
            multimask_output=False)
        masks = masks[:, None, ...]
        masks = np.transpose(masks, (1, 0, 2, 3)).squeeze() * 255
    garbage_collect(sam)
    masks_gallery = []
    for mask in masks:
        masks_gallery.append(Image.fromarray(np.any(mask, axis=0)))
    return masks_gallery


def dino_batch_simplified(batch_text_prompt, batch_box_threshold, dino_batch_source_dir,  dino_batch_save_mask, batch_dilation_amt=0,
         dino_batch_dest_dir=None, dino_batch_output_per_image=3, dino_batch_save_image=True, dino_batch_save_background=False,
        dino_batch_save_image_with_mask=False):
    if batch_text_prompt is None or batch_text_prompt == "":
        return "Please add text prompts to generate masks"
    print("Start batch processing")
    batch_sam_model_name = sam_model_list[0]
    batch_dino_model_name = dino_model_list[0]
    sam = init_sam_model(batch_sam_model_name)
    predictor = SamPredictorHQ(sam, 'hq' in batch_sam_model_name)

    process_info = ""
    install_success = True
    all_files = glob.glob(os.path.join(dino_batch_source_dir, "*"))
    # all_files = dino_batch_source_dir

    image_sum = []
    mask_sum = []
    for image_index, input_image_file in enumerate(all_files):
        print(f"Processing {image_index}/{len(all_files)} {input_image_file}")
        try:
            input_image = Image.open(input_image_file).convert("RGBA")
        except:
            print(f"File {input_image_file} not image, skipped.")
            continue
        image_np = np.array(input_image)
        image_np_rgb = image_np[..., :3]

        boxes_filt, install_success = dino_predict_internal(input_image, batch_dino_model_name, batch_text_prompt,
                                                            batch_box_threshold)

        if boxes_filt is None or boxes_filt.shape[0] == 0:
            msg = f"GroundingDINO generated 0 box for image {input_image_file}, please lower the box threshold if you want any segmentation for this image. "
            print(msg)
            process_info += (msg + "\n")
            continue

        predictor.set_image(image_np_rgb)
        transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image_np.shape[:2])
        masks, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes.to(sam_device),
            multimask_output=True)

        masks = masks.permute(1, 0, 2, 3).cpu().numpy()
        boxes_filt = boxes_filt.cpu().numpy().astype(int)

        output_image, output_mask = create_mask_batch_output_simplified(
            input_image_file, dino_batch_dest_dir,
            image_np, masks, boxes_filt, batch_dilation_amt,
            dino_batch_save_image, dino_batch_save_mask, dino_batch_save_background, dino_batch_save_image_with_mask)
        image_sum.append((output_image))
        mask_sum.append((output_mask))
    garbage_collect(sam)
    out_list = []
    for i in range(len(all_files)):
        out_list.append(image_sum[i][0])
        out_list.append(mask_sum[i][0])
        out_list.append(image_sum[i][1])
        out_list.append(mask_sum[i][1])
        out_list.append(image_sum[i][2])
        out_list.append(mask_sum[i][2])
    return out_list


def dino_batch_process(
        batch_sam_model_name, batch_dino_model_name, batch_text_prompt, batch_box_threshold, batch_dilation_amt,
        dino_batch_source_dir, dino_batch_dest_dir,
        dino_batch_output_per_image, dino_batch_save_image, dino_batch_save_mask, dino_batch_save_background,
        dino_batch_save_image_with_mask):
    if batch_text_prompt is None or batch_text_prompt == "":
        return "Please add text prompts to generate masks"
    print("Start batch processing")
    sam = init_sam_model(batch_sam_model_name)
    predictor = SamPredictorHQ(sam, 'hq' in batch_sam_model_name)

    process_info = ""
    install_success = True
    all_files = glob.glob(os.path.join(dino_batch_source_dir, "*"))
    for image_index, input_image_file in enumerate(all_files):
        print(f"Processing {image_index}/{len(all_files)} {input_image_file}")
        try:
            input_image = Image.open(input_image_file).convert("RGBA")
        except:
            print(f"File {input_image_file} not image, skipped.")
            continue
        image_np = np.array(input_image)
        image_np_rgb = image_np[..., :3]

        boxes_filt, install_success = dino_predict_internal(input_image, batch_dino_model_name, batch_text_prompt,
                                                            batch_box_threshold)
        if boxes_filt is None or boxes_filt.shape[0] == 0:
            msg = f"GroundingDINO generated 0 box for image {input_image_file}, please lower the box threshold if you want any segmentation for this image. "
            print(msg)
            process_info += (msg + "\n")
            continue

        predictor.set_image(image_np_rgb)
        transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image_np.shape[:2])
        masks, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes.to(sam_device),
            multimask_output=(dino_batch_output_per_image == 1))

        masks = masks.permute(1, 0, 2, 3).cpu().numpy()
        boxes_filt = boxes_filt.cpu().numpy().astype(int)

        create_mask_batch_output(
            input_image_file, dino_batch_dest_dir,
            image_np, masks, boxes_filt, batch_dilation_amt,
            dino_batch_save_image, dino_batch_save_mask, dino_batch_save_background, dino_batch_save_image_with_mask)

    garbage_collect(sam)
    return process_info + "Done" + (
        "" if install_success else f". However, GroundingDINO installment has failed. Your process automatically fall back to local groundingdino. See your terminal for more detail")







