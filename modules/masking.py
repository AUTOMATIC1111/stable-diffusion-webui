from types import SimpleNamespace
import os
import time
import gradio as gr
import numpy as np
import cv2
from PIL import Image, ImageFilter, ImageOps
from transformers import SamModel, SamImageProcessor, MaskGenerationPipeline
from modules import shared, errors, devices, ui_components, ui_symbols


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
    # 'Tiny Random': 'fxmarty/sam-vit-tiny-random',
}
COLORMAP = ['autumn', 'bone', 'jet', 'winter', 'rainbow', 'ocean', 'summer', 'spring', 'cool', 'hsv', 'pink', 'hot', 'parula', 'magma', 'inferno', 'plasma', 'viridis', 'cividis', 'twilight', 'shifted', 'turbo', 'deepgreen']
cache_dir = 'models/control/segment'
loaded_model = None
model: SamModel = None
processor: SamImageProcessor = None
generator: MaskGenerationPipeline = None
debug = shared.log.trace if os.environ.get('SD_MASK_DEBUG', None) is not None else lambda *args, **kwargs: None
debug('Trace: MASK')
busy = False
btn_segment = None
controls = []
opts = SimpleNamespace(**{
    'mask_blur': 0.01,
    'mask_erode': 0.01,
    'mask_dilate': 0.01,
    'seg_iou_thresh': 0.5,
    'seg_score_thresh': 0.5,
    'seg_nms_thresh': 0.5,
    'seg_overlap_ratio': 0.3,
    'seg_points_per_batch': 64,
    'seg_topK': 50,
    'seg_colormap': 'pink',
    'preview_type': 'composite',
    'seg_live': True,
    'weight_original': 0.5,
    'weight_mask': 0.5,
    'kernel_iterations': 1,
})


def init_model(selected_model: str):
    global busy, loaded_model, model, processor, generator # pylint: disable=global-statement
    if selected_model == "None":
        if model is not None:
            shared.log.debug('Segment unloading model')
        model = None
        loaded_model = None
        processor = None
        generator = None
        devices.torch_gc()
        return selected_model
    model_path = MODELS[selected_model]
    if model_path is not None and (loaded_model != selected_model or model is None or processor is None):
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
    debug(f'Segment: {vars(opts)}')
    for mask in outputs['masks']:
        mask = mask.astype('uint8')
        mask_size = np.count_nonzero(mask)
        if mask_size == 0:
            continue
        overlap = 0
        if input_mask_size > 0:
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


def run_mask(input_image: gr.Image, input_mask: gr.Image = None, return_type: str = None, mask_blur: int = None, mask_padding: int = None, segment_enable=True):
    if input_image is None:
        return input_mask
    if isinstance(input_image, list):
        input_image = input_image[0]
    if isinstance(input_image, dict):
        input_mask = input_image.get('mask', None)
        input_image = input_image.get('image', None)
    if input_mask is None:
        input_mask = input_image.convert('L')
        input_mask = input_mask.point(lambda x: 255 if x > 127 else 0)
    else:
        input_mask = input_mask.convert('L')
    shared.log.debug(f'Segment mask: input={input_image} mask={input_mask} type={return_type}')

    input_mask = np.array(input_mask) // 255
    t0 = time.time()

    if mask_blur is not None:
        opts.mask_blur = mask_blur / min(input_image.width, input_image.height)
    if mask_padding is not None:
        opts.mask_dilate = mask_padding / min(input_image.width, input_image.height)

    if generator is None or not segment_enable:
        mask = input_mask * 255
    else:
        mask = run_segment(input_image, input_mask)

    if mask is None:
        shared.log.error('Segment error: no mask')
        return input_mask

    debug(f'Segment mask: mask={mask.shape}')
    if opts.mask_erode > 0:
        try:
            kernel = np.ones((int(opts.mask_erode * input_image.height / 4) + 1, int(opts.mask_erode * input_image.width / 4) + 1), np.uint8)
            cv2_mask = cv2.erode(mask, kernel, iterations=opts.kernel_iterations) # remove noise
            mask = cv2_mask
            debug(f'Segment erode={opts.mask_erode} kernel={kernel} mask={mask.shape}')
        except Exception as e:
            shared.log.error(f'Segment erode: {e}')
    if opts.mask_dilate > 0:
        try:
            kernel = np.ones((int(opts.mask_dilate * input_image.height / 4) + 1, int(opts.mask_dilate * input_image.width / 4) + 1), np.uint8)
            cv2_mask = cv2.dilate(mask, kernel, iterations=opts.kernel_iterations) # expand area
            mask = cv2_mask
            debug(f'Segment dilate={opts.mask_dilate} kernel={kernel} mask={mask.shape}')
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

    mask_size = np.count_nonzero(mask)
    total_size = np.prod(mask.shape)
    area_size = np.count_nonzero(mask)
    colored_mask = cv2.applyColorMap(mask, COLORMAP.index(opts.seg_colormap)) # recolor mask
    combined_image = cv2.addWeighted(np.array(input_image), opts.weight_original, colored_mask, opts.weight_mask, 0)
    binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1] # otsu uses mean instead of threshold
    t1 = time.time()

    return_type = return_type or opts.preview_type
    shared.log.debug(f'Segment mask opts: size={input_image.width}x{input_image.height} masked={mask_size}px area={area_size/total_size:.2f} time={t1-t0:.2f}')
    if return_type == 'none':
        return input_mask
    elif return_type == 'binary':
        return Image.fromarray(binary_mask)
    elif return_type == 'grayscale':
        return Image.fromarray(mask)
    elif return_type == 'color':
        return Image.fromarray(colored_mask)
    elif return_type == 'composite':
        return Image.fromarray(combined_image)
    return input_mask


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
        opts.mask_blur = args[1]
        opts.mask_erode = args[2]
        opts.mask_dilate = args[3]
        opts.seg_score_thresh = args[4]
        opts.seg_iou_thresh = args[5]
        opts.seg_nms_thresh = args[6]
        opts.preview_type = args[7]
        opts.seg_colormap = args[8]

    def display_controls(selected_model):
        return 4 * [gr.update(visible=True)] + (len(controls) - 4) * [gr.update(visible=selected_model != 'None')]

    global btn_segment # pylint: disable=global-statement
    with gr.Accordion(open=False, label="Mask", elem_id="control_mask", elem_classes=["small-accordion"]):
        controls.clear()
        with gr.Row():
            controls.append(gr.Checkbox(label="Live update", value=True))
        with gr.Row():
            controls.append(gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label='Blur', value=0.01, elem_id="control_mask_blur"))
            controls.append(gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label='Erode', value=0.01, elem_id="control_mask_erode"))
            controls.append(gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label='Dilate', value=0.01, elem_id="control_mask_dilate"))
        with gr.Row():
            selected_model = gr.Dropdown(label="Auto-segment", choices=MODELS.keys(), value='None')
            btn_segment = ui_components.ToolButton(value=ui_symbols.refresh, visible=False)
        with gr.Row():
            controls.append(gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label='Score', value=0.5, visible=False))
            controls.append(gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label='IOU', value=0.5, visible=False))
            controls.append(gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label='NMS', value=0.5, visible=False))
        with gr.Row():
            controls.append(gr.Dropdown(label="Preview", choices=['none', 'binary', 'grayscale', 'color', 'composite'], value='composite'))
            controls.append(gr.Dropdown(label="Colormap", choices=COLORMAP, value='pink'))

        selected_model.change(fn=init_model, inputs=[selected_model], outputs=[selected_model])
        selected_model.change(fn=display_controls, inputs=[selected_model], outputs=controls)
        for control in controls:
            control.change(fn=update_opts, inputs=controls, outputs=[])


def bind_controls(input_image: gr.Image, preview_image: gr.Image):
    btn_segment.click(run_mask, inputs=[input_image], outputs=[preview_image])
    input_image.edit(fn=run_mask_live, inputs=[input_image], outputs=[preview_image])
    for control in controls:
        control.change(fn=run_mask_live, inputs=[input_image], outputs=[preview_image])
