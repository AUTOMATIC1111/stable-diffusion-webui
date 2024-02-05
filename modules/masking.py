from types import SimpleNamespace
from typing import List
import os
import sys
import time
import gradio as gr
import numpy as np
import cv2
from PIL import Image, ImageFilter, ImageOps
from transformers import SamModel, SamImageProcessor, MaskGenerationPipeline
from modules import shared, errors, devices, ui_components, ui_symbols, paths
from modules.memstats import memory_stats


def get_crop_region(mask, pad=0):
    """finds a rectangular region that contains all masked ares in an image. Returns (x1, y1, x2, y2) coordinates of the rectangle.
    For example, if a user has painted the top-right part of a 512x512 image", the result may be (256, 0, 512, 256)"""
    h, w = mask.shape
    crop_left = 0
    for i in range(w):
        if not (mask[:, i] == 0).all():
            break
        crop_left += 1
    crop_right = 0
    for i in reversed(range(w)):
        if not (mask[:, i] == 0).all():
            break
        crop_right += 1
    crop_top = 0
    for i in range(h):
        if not (mask[i] == 0).all():
            break
        crop_top += 1
    crop_bottom = 0
    for i in reversed(range(h)):
        if not (mask[i] == 0).all():
            break
        crop_bottom += 1
    return (
        int(max(crop_left-pad, 0)),
        int(max(crop_top-pad, 0)),
        int(min(w - crop_right + pad, w)),
        int(min(h - crop_bottom + pad, h))
    )


def expand_crop_region(crop_region, processing_width, processing_height, image_width, image_height):
    """expands crop region get_crop_region() to match the ratio of the image the region will processed in; returns expanded region
    for example, if user drew mask in a 128x32 region, and the dimensions for processing are 512x512, the region will be expanded to 128x128."""
    x1, y1, x2, y2 = crop_region
    ratio_crop_region = (x2 - x1) / (y2 - y1)
    ratio_processing = processing_width / processing_height

    if ratio_crop_region > ratio_processing:
        desired_height = (x2 - x1) / ratio_processing
        desired_height_diff = int(desired_height - (y2-y1))
        y1 -= desired_height_diff//2
        y2 += desired_height_diff - desired_height_diff//2
        if y2 >= image_height:
            diff = y2 - image_height
            y2 -= diff
            y1 -= diff
        if y1 < 0:
            y2 -= y1
            y1 -= y1
        if y2 >= image_height:
            y2 = image_height
    else:
        desired_width = (y2 - y1) * ratio_processing
        desired_width_diff = int(desired_width - (x2-x1))
        x1 -= desired_width_diff//2
        x2 += desired_width_diff - desired_width_diff//2
        if x2 >= image_width:
            diff = x2 - image_width
            x2 -= diff
            x1 -= diff
        if x1 < 0:
            x2 -= x1
            x1 -= x1
        if x2 >= image_width:
            x2 = image_width

    return x1, y1, x2, y2


def fill(image, mask):
    """fills masked regions with colors from image using blur. Not extremely effective."""
    image_mod = Image.new('RGBA', (image.width, image.height))
    image_masked = Image.new('RGBa', (image.width, image.height))
    image_masked.paste(image.convert("RGBA").convert("RGBa"), mask=ImageOps.invert(mask.convert('L')))
    image_masked = image_masked.convert('RGBa')
    for radius, repeats in [(256, 1), (64, 1), (16, 2), (4, 4), (2, 2), (0, 1)]:
        blurred = image_masked.filter(ImageFilter.GaussianBlur(radius)).convert('RGBA')
        for _ in range(repeats):
            image_mod.alpha_composite(blurred)
    return image_mod.convert("RGB")


"""
[docs](https://huggingface.co/docs/transformers/v4.36.1/en/model_doc/sam#overview)
TODO:
- PerSAM
- REMBG
- https://huggingface.co/docs/transformers/tasks/semantic_segmentation
- transformers.pipeline.MaskGenerationPipeline: https://huggingface.co/models?pipeline_tag=mask-generation
- transformers.pipeline.ImageSegmentationPipeline: https://huggingface.co/models?pipeline_tag=image-segmentation
"""

MODELS = {
    'None': None,
    'Facebook SAM ViT Base': 'facebook/sam-vit-base',
    'Facebook SAM ViT Large': 'facebook/sam-vit-large',
    'Facebook SAM ViT Huge': 'facebook/sam-vit-huge',
    'SlimSAM Uniform': 'Zigeng/SlimSAM-uniform-50',
    'SlimSAM Uniform Tiny': 'Zigeng/SlimSAM-uniform-77',
    'Rembg Silueta': 'silueta',
    'Rembg U2Net': 'u2net',
    'Rembg ISNet': 'isnet',
    # "u2net_human_seg",
    # "isnet-general-use",
    # "isnet-anime",
}
COLORMAP = ['autumn', 'bone', 'jet', 'winter', 'rainbow', 'ocean', 'summer', 'spring', 'cool', 'hsv', 'pink', 'hot', 'parula', 'magma', 'inferno', 'plasma', 'viridis', 'cividis', 'twilight', 'shifted', 'turbo', 'deepgreen']
cache_dir = 'models/control/segment'
generator: MaskGenerationPipeline = None
debug = shared.log.trace if os.environ.get('SD_MASK_DEBUG', None) is not None else lambda *args, **kwargs: None
debug('Trace: MASK')
busy = False
btn_mask = None
btn_lama = None
lama_model = None
controls = []
opts = SimpleNamespace(**{
    'model': None,
    'auto_mask': 'None',
    'mask_only': False,
    'mask_blur': 0.01,
    'mask_padding': 0,
    'mask_erode': 0.01,
    'mask_dilate': 0.01,
    'seg_iou_thresh': 0.5,
    'seg_score_thresh': 0.5,
    'seg_nms_thresh': 0.5,
    'seg_overlap_ratio': 0.3,
    'seg_points_per_batch': 64,
    'seg_topK': 50,
    'seg_colormap': 'pink',
    'preview_type': 'Composite',
    'seg_live': True,
    'weight_original': 0.5,
    'weight_mask': 0.5,
    'kernel_iterations': 1,
    'invert': False
})


def init_model(selected_model: str):
    global busy, generator # pylint: disable=global-statement
    model_path = MODELS[selected_model]
    if model_path is None: # none
        if generator is not None:
            shared.log.debug('Segment unloading model')
        opts.model = None
        generator = None
        devices.torch_gc()
        return selected_model
    if 'Rembg' in selected_model: # rembg
        opts.model = model_path
        generator = None
        devices.torch_gc()
        return selected_model
    if opts.model != selected_model or generator is None: # sam pipeline
        busy = True
        t0 = time.time()
        shared.log.debug(f'Segment loading: model={selected_model} path={model_path}')
        model = SamModel.from_pretrained(model_path, cache_dir=cache_dir).to(device=devices.device)
        processor = SamImageProcessor.from_pretrained(model_path, cache_dir=cache_dir)
        generator = MaskGenerationPipeline(
            model=model,
            image_processor=processor,
            device=devices.device,
            # output_bboxes_mask=False,
            # output_rle_masks=False,
        )
        devices.torch_gc()
        shared.log.debug(f'Segment loaded: model={selected_model} path={model_path} time={time.time()-t0:.2f}s')
        opts.model = selected_model
        busy = False
    return selected_model


def run_segment(input_image: gr.Image, input_mask: np.ndarray):
    outputs = None
    with devices.inference_context():
        try:
            outputs = generator(
                input_image,
                points_per_batch=opts.seg_points_per_batch,
                pred_iou_thresh=opts.seg_iou_thresh,
                stability_score_thresh=opts.seg_score_thresh,
                crops_nms_thresh=opts.seg_nms_thresh,
                crop_overlap_ratio=opts.seg_overlap_ratio,
                crops_n_layers=0,
                crop_n_points_downscale_factor=1,
            )
        except Exception as e:
            shared.log.error(f'Segment error: {e}')
            errors.display(e, 'Segment')
            return outputs
    devices.torch_gc()
    i = 1
    combined_mask = np.zeros(input_mask.shape, dtype='uint8')
    input_mask_size = np.count_nonzero(input_mask)
    debug(f'Segment SAM: {vars(opts)}')
    for mask in outputs['masks']:
        mask = mask.astype('uint8')
        mask_size = np.count_nonzero(mask)
        if mask_size == 0:
            continue
        overlap = 0
        if input_mask_size > 0:
            if mask.shape != input_mask.shape:
                mask = cv2.resize(mask, (input_mask.shape[1], input_mask.shape[0]), interpolation=cv2.INTER_CUBIC)
            overlap = cv2.bitwise_and(mask, input_mask)
            overlap = np.count_nonzero(overlap)
            if overlap == 0:
                continue
        mask = (opts.seg_topK + 1 - i) * mask * (255 // opts.seg_topK) # set grayscale intensity so we can recolor
        combined_mask = combined_mask + mask
        debug(f'Segment mask: i={i} size={input_image.width}x{input_image.height} masked={mask_size}px overlap={overlap} score={outputs["scores"][i-1]:.2f}')
        i += 1
        if i > opts.seg_topK:
            break
    return combined_mask


def run_rembg(input_image: Image, input_mask: np.ndarray):
    try:
        import rembg
    except Exception as e:
        shared.log.error(f'Segment Rembg load failed: {e}')
        return input_mask
    if "U2NET_HOME" not in os.environ:
        os.environ["U2NET_HOME"] = os.path.join(paths.models_path, "Rembg")
    args = {
        'data': input_image,
        'only_mask': True,
        'post_process_mask': False,
        'bgcolor': None,
        'alpha_matting': False,
        'alpha_matting_foreground_threshold': 240,
        'alpha_matting_background_threshold': 10,
        'alpha_matting_erode_size': int(opts.mask_erode * 40),
        'session': rembg.new_session(opts.model),
    }
    mask = rembg.remove(**args)
    mask = np.array(mask)
    if len(input_mask.shape) > 2:
        mask = cv2.cvtColor(input_mask, cv2.COLOR_RGB2GRAY)
    binary_input = cv2.threshold(input_mask, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    binary_output = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    if binary_input.shape != binary_output.shape:
        binary_output = cv2.resize(binary_output, binary_input.shape[:2], interpolation=cv2.INTER_LINEAR)
    binary_overlap = cv2.bitwise_and(binary_input, binary_output)
    input_size = np.count_nonzero(binary_input)
    overlap_size = np.count_nonzero(binary_overlap)
    debug(f'Segment Rembg: {args} overlap={overlap_size}')
    if input_size > 0 and overlap_size == 0:
        mask = np.invert(mask)
    return mask


def get_mask(input_image: gr.Image, input_mask: gr.Image):
    t0 = time.time()
    if input_mask is not None:
        output_mask = np.array(input_mask)
        if len(output_mask.shape) > 2:
            output_mask = cv2.cvtColor(output_mask, cv2.COLOR_RGB2GRAY)
        binary_mask = cv2.threshold(output_mask, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        mask_size = np.count_nonzero(binary_mask)
    else:
        output_mask = None
        mask_size = 0
    if mask_size == 0 and opts.auto_mask != 'None': # mask_size == 0
        output_mask = np.array(input_image)
        if opts.auto_mask == 'Threshold':
            output_mask = cv2.cvtColor(output_mask, cv2.COLOR_RGB2GRAY)
            output_mask = cv2.threshold(output_mask, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        elif opts.auto_mask == 'Edge':
            output_mask = cv2.cvtColor(output_mask, cv2.COLOR_RGB2GRAY)
            output_mask = cv2.threshold(output_mask, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            # output_mask = cv2.Canny(output_mask, 50, 150) # run either canny or threshold before contouring
            contours, _hierarchy = cv2.findContours(output_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True) # sort contours by area with largest first
            contours = contours[:opts.seg_topK] # limit to top K contours
            output_mask = np.zeros(output_mask.shape, dtype='uint8')
            largest_size = cv2.contourArea(contours[0]) if len(contours) > 0 else 0
            for i, contour in enumerate(contours):
                area_size = cv2.contourArea(contour)
                luminance = int(255.0 * area_size / largest_size)
                if luminance < 1:
                    break
                cv2.drawContours(output_mask, contours, i, (luminance), -1)
        elif opts.auto_mask == 'Grayscale':
            lab_image = cv2.cvtColor(output_mask, cv2.COLOR_RGB2LAB)
            l_channel, a, b = cv2.split(lab_image)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) # applying CLAHE to L-channel
            cl = clahe.apply(l_channel)
            lab_image = cv2.merge((cl, a, b)) # merge the CLAHE enhanced L-channel with the a and b channel
            lab_image = cv2.cvtColor(lab_image, cv2.COLOR_LAB2RGB)
            output_mask = cv2.cvtColor(lab_image, cv2.COLOR_RGB2GRAY)
        t1 = time.time()
        debug(f'Segment auto-mask: mode={opts.auto_mask} time={t1-t0:.2f}')
        return output_mask
    else: # no mask or empty mask and no auto-mask
        return output_mask


def run_mask(input_image: gr.Image, input_mask: gr.Image = None, return_type: str = None, mask_blur: int = None, mask_padding: int = None, segment_enable=True, invert=None):
    debug(f'Run mask: function={sys._getframe(1).f_code.co_name}') # pylint: disable=protected-access

    if input_image is None:
        return input_mask
    if isinstance(input_image, list):
        input_image = input_image[0]
    if isinstance(input_image, dict):
        input_mask = input_image.get('mask', None)
        input_image = input_image.get('image', None)
    if input_image is None:
        return input_mask

    t0 = time.time()
    input_mask = get_mask(input_image, input_mask) # perform optional auto-masking
    if input_mask is None:
        return None

    if invert is not None:
        opts.invert = invert
    if mask_blur is not None: # compatibility with old img2img values which have different range
        opts.mask_blur = mask_blur / min(input_image.width, input_image.height)
    if mask_padding is not None:
        opts.mask_dilate = mask_padding / min(input_image.width, input_image.height)
        opts.mask_padding = mask_padding
    else:
        opts.mask_padding = int(opts.mask_dilate * input_image.height / 4) + 1

    if opts.model is None or not segment_enable:
        mask = input_mask
    elif generator is None:
        mask = run_rembg(input_image, input_mask)
    else:
        mask = run_segment(input_image, input_mask)
    mask = cv2.resize(mask, (input_image.width, input_image.height), interpolation=cv2.INTER_LINEAR)

    debug(f'Mask opts: {opts}')
    debug(f'Segment mask: mask={mask.shape}')
    if opts.mask_erode > 0:
        try:
            kernel = np.ones((int(opts.mask_erode * input_image.height / 4) + 1, int(opts.mask_erode * input_image.width / 4) + 1), np.uint8)
            cv2_mask = cv2.erode(mask, kernel, iterations=opts.kernel_iterations) # remove noise
            mask = cv2_mask
            debug(f'Segment erode={opts.mask_erode} kernel={kernel.shape} mask={mask.shape}')
        except Exception as e:
            shared.log.error(f'Segment erode: {e}')
    if opts.mask_dilate > 0:
        try:
            kernel = np.ones((int(opts.mask_dilate * input_image.height / 4) + 1, int(opts.mask_dilate * input_image.width / 4) + 1), np.uint8)
            cv2_mask = cv2.dilate(mask, kernel, iterations=opts.kernel_iterations) # expand area
            mask = cv2_mask
            debug(f'Segment dilate={opts.mask_dilate} kernel={kernel.shape} mask={mask.shape}')
        except Exception as e:
            shared.log.error(f'Segment dilate: {e}')
    if opts.mask_blur > 0:
        try:
            sigmax, sigmay = 1 + int(opts.mask_blur * input_image.width / 4), 1 + int(opts.mask_blur * input_image.height / 4)
            cv2_mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=sigmax, sigmaY=sigmay) # blur mask
            mask = cv2_mask
            debug(f'Segment blur={opts.mask_blur} x={sigmax} y={sigmay} mask={mask.shape}')
        except Exception as e:
            shared.log.error(f'Segment blur: {e}')
    if opts.invert:
        mask = np.invert(mask)

    mask_size = np.count_nonzero(mask)
    total_size = np.prod(mask.shape)
    area_size = np.count_nonzero(mask)
    t1 = time.time()

    return_type = return_type or opts.preview_type

    shared.log.debug(f'Segment mask: size={input_image.width}x{input_image.height} masked={mask_size}px area={area_size/total_size:.2f} auto={opts.auto_mask} type={return_type} time={t1-t0:.2f}')
    if return_type == 'None':
        return input_mask
    elif return_type == 'Binary':
        binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1] # otsu uses mean instead of threshold
        return Image.fromarray(binary_mask)
    elif return_type == 'Masked':
        orig = np.array(input_image)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        masked_image = cv2.bitwise_and(orig, mask)
        return Image.fromarray(masked_image)
    elif return_type == 'Grayscale':
        return Image.fromarray(mask)
    elif return_type == 'Color':
        colored_mask = cv2.applyColorMap(mask, COLORMAP.index(opts.seg_colormap)) # recolor mask
        return Image.fromarray(colored_mask)
    elif return_type == 'Composite':
        colored_mask = cv2.applyColorMap(mask, COLORMAP.index(opts.seg_colormap)) # recolor mask
        orig = np.array(input_image)
        combined_image = cv2.addWeighted(orig, opts.weight_original, colored_mask, opts.weight_mask, 0)
        return Image.fromarray(combined_image)
    else:
        shared.log.error(f'Segment unknown return type: {return_type}')
    return input_mask


def run_lama(input_image: gr.Image, input_mask: gr.Image = None):
    global lama_model # pylint: disable=global-statement
    if isinstance(input_image, dict):
        input_mask = input_image.get('mask', None)
        input_image = input_image.get('image', None)
    if input_image is None:
        return None
    input_mask = run_mask(input_image, input_mask, return_type='Grayscale')
    if lama_model is None:
        import modules.lama
        shared.log.debug(f'LaMa loading: model={modules.lama.LAMA_MODEL_URL}')
        lama_model = modules.lama.SimpleLama()
        shared.log.debug(f'LaMa loaded: {memory_stats()}')
    lama_model.model.to(devices.device)
    result = lama_model(input_image, input_mask)
    if shared.opts.control_move_processor:
        lama_model.model.to('cpu')
    return result


def run_mask_live(input_image: gr.Image):
    global busy # pylint: disable=global-statement
    if opts.seg_live:
        if not busy:
            busy = True
            res = run_mask(input_image)
            busy = False
            return res
    else:
        return None


def create_segment_ui():
    def update_opts(*args):
        opts.seg_live = args[0]
        opts.mask_only = args[1]
        opts.invert = args[2]
        opts.mask_blur = args[3]
        opts.mask_erode = args[4]
        opts.mask_dilate = args[5]
        opts.auto_mask = args[6]
        opts.seg_score_thresh = args[7]
        opts.seg_iou_thresh = args[8]
        opts.seg_nms_thresh = args[9]
        opts.preview_type = args[10]
        opts.seg_colormap = args[11]

    global btn_mask, btn_lama # pylint: disable=global-statement
    with gr.Accordion(open=False, label="Mask", elem_id="control_mask", elem_classes=["small-accordion"]):
        controls.clear()
        with gr.Row():
            controls.append(gr.Checkbox(label="Live update", value=True))
            btn_mask = ui_components.ToolButton(value=ui_symbols.refresh, visible=True)
            btn_lama = ui_components.ToolButton(value=ui_symbols.image, visible=True)
        with gr.Row():
            controls.append(gr.Checkbox(label="Inpaint masked only", value=False))
            controls.append(gr.Checkbox(label="Invert mask", value=False))
        with gr.Row():
            controls.append(gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label='Blur', value=0.01, elem_id="control_mask_blur"))
            controls.append(gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label='Erode', value=0.01, elem_id="control_mask_erode"))
            controls.append(gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label='Dilate', value=0.01, elem_id="control_mask_dilate"))
        with gr.Row():
            controls.append(gr.Dropdown(label="Auto-mask", choices=['None', 'Threshold', 'Edge', 'Grayscale'], value='None'))
            selected_model = gr.Dropdown(label="Auto-segment", choices=MODELS.keys(), value='None')
        with gr.Row():
            controls.append(gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label='Score', value=0.5, visible=False))
            controls.append(gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label='IOU', value=0.5, visible=False))
            controls.append(gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label='NMS', value=0.5, visible=False))
        with gr.Row():
            controls.append(gr.Dropdown(label="Preview", choices=['None', 'Masked', 'Binary', 'Grayscale', 'Color', 'Composite'], value='Composite'))
            controls.append(gr.Dropdown(label="Colormap", choices=COLORMAP, value='pink'))

        selected_model.change(fn=init_model, inputs=[selected_model], outputs=[selected_model])
        for control in controls:
            control.change(fn=update_opts, inputs=controls, outputs=[])
        return controls


def bind_controls(image_controls: List[gr.Image], preview_image: gr.Image, output_image: gr.Image):
    for image_control in image_controls:
        btn_mask.click(run_mask, inputs=[image_control], outputs=[preview_image])
        btn_lama.click(run_lama, inputs=[image_control], outputs=[output_image])
        image_control.edit(fn=run_mask_live, inputs=[image_control], outputs=[preview_image])
        for control in controls:
            control.change(fn=run_mask_live, inputs=[image_control], outputs=[preview_image])
