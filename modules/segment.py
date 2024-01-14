"""
[docs](https://huggingface.co/docs/transformers/v4.36.1/en/model_doc/sam#overview)
TODO:
- PerSAM
- transformers.pipeline.MaskGenerationPipeline: https://huggingface.co/models?pipeline_tag=mask-generation
- transformers.pipeline.ImageSegmentationPipeline: https://huggingface.co/models?pipeline_tag=image-segmentation
"""

from transformers import SamModel, SamImageProcessor, MaskGenerationPipeline
from PIL import Image
import gradio as gr
import numpy as np
import cv2
from modules import shared, devices


MODELS = {
    'None': None,
    'Facebook SAM ViT Base': 'facebook/sam-vit-base',
    'Facebook SAM ViT Large': 'facebook/sam-vit-large',
    'Facebook SAM ViT Huge': 'facebook/sam-vit-huge',
    'SlimSAM Uniform': 'Zigeng/SlimSAM-uniform-50',
}
COLORMAP = ['autumn', 'bone', 'jet', 'winter', 'rainbow', 'ocean', 'summer', 'spring', 'cool', 'hsv', 'pink', 'hot', 'parula', 'magma', 'inferno', 'plasma', 'viridis', 'cividis', 'twilight', 'shifted', 'turbo', 'deepgreen']
cache_dir = 'models/control/segment'
loaded_model = None
model: SamModel = None
processor: SamImageProcessor = None


def init(selected_model: str, input_image: gr.Image):
    global loaded_model, model, processor # pylint: disable=global-statement
    if input_image is None or input_image.get('image', None) is None:
        return False
    if selected_model == "None":
        return False
    if selected_model == "None":
        model = None
        loaded_model = None
        processor = None
    model_path = MODELS[selected_model]
    if model_path is not None and (loaded_model != selected_model or model is None or processor is None):
        shared.log.debug(f'Segment loading: model={selected_model} path={model_path}')
        model = SamModel.from_pretrained(model_path, cache_dir=cache_dir).to(device=devices.device)
        processor = SamImageProcessor.from_pretrained(model_path, cache_dir=cache_dir)
        shared.log.debug(f'Segment loaded: model={selected_model} path={model_path}')
    if model is None or processor is None:
        return False
    return True


# run as auto-mask with all possible masks
def run_segment(selected_model: str, input_image: gr.Image, points_per_batch=64, pred_iou_thresh=0.75, stability_score_thresh=0.85, crops_nms_thresh=0.5, crop_overlap_ratio=0.3, topK=25, colormap='jet', erode=0, dilate=0):
    if not init(selected_model, input_image):
        return gr.update(), None
    input_mask = input_image.get('mask', None) or Image.new('L', input_image.get('image', None).size, 255)
    input_mask = input_mask.convert('L')
    input_image = input_image.get('image', None)
    generator: MaskGenerationPipeline = MaskGenerationPipeline(model=model, image_processor=processor, device=devices.device)
    with devices.inference_context():
        outputs = generator(
            input_image,
            points_per_batch=points_per_batch,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            crops_nms_thresh=crops_nms_thresh,
            crop_overlap_ratio=crop_overlap_ratio,
        )
    combined_mask = np.zeros(input_mask.size, dtype='uint8')
    input_mask = np.array(input_mask) // 255
    input_mask_size = np.count_nonzero(input_mask)
    print('HERE', input_mask.shape, input_mask_size)
    i = 1
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
        # TODO erode,dilate
        if erode > 0:
            mask = cv2.erode(mask, np.ones((erode, erode), np.uint8), iterations=2) # remove noise
        if dilate > 0:
            mask = cv2.dilate(mask, np.ones((dilate, dilate), np.uint8), iterations=2) # expand area
        mask = (topK + 1 - i) * mask * (255 // topK) # set grayscale intensity so we can recolor
        combined_mask = combined_mask + mask
        i += 1
        if i > topK:
            break
    mask_size = np.count_nonzero(combined_mask)
    total_size = np.prod(combined_mask.shape)
    area_size = np.count_nonzero(combined_mask)
    shared.log.debug(f'Segment mask: size={input_image.width}x{input_image.height} input={input_mask_size}px masked={mask_size}px area={area_size/total_size:.2f}')
    colored_mask = cv2.applyColorMap(combined_mask, COLORMAP.index(colormap)) # recolor mask
    combined_image = cv2.addWeighted(np.array(input_image), 0.6, colored_mask, 0.4, 0)
    _thres, binary_mask = cv2.threshold(combined_mask, 1, 255, cv2.THRESH_BINARY_INV) # create mask
    binary_mask = np.invert(binary_mask)

    binary_mask = Image.fromarray(binary_mask)
    combined_mask = Image.fromarray(combined_mask)
    colored_mask = Image.fromarray(colored_mask)
    overlay_image = Image.fromarray(combined_image)

    # TODO return type
    binary_mask.save('/tmp/mask-binary.png')
    combined_mask.save('/tmp/mask-combined.png')
    combined_mask.save('/tmp/mask-colored.png')
    overlay_image.save('/tmp/mask-overlay.png')
    return input_image, overlay_image


# run with sam model directly needing set of points
def run_segment_points(selected_model: str, input_image: gr.Image):
    if not init(selected_model, input_image):
        return input_image
    # input_mask = input_image.get('mask', None) or Image.new('L', input_image.get('image', None).size, 0)
    input_image = input_image.get('image', None)
    with devices.inference_context():
        inputs = processor(
            input_image,
            input_points=[[[256, 256]]], # TODO calculate points based on mask
            return_tensors="pt"
        ).to(device=devices.device)
        outputs = model(
            pixel_values=inputs['pixel_values'],
            multimask_output=True,
            )
        masks = processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu()
        )
        scores = outputs.iou_scores
    mask = masks[0].squeeze(0)
    scores = scores[0].squeeze(0)
    masks = mask.unbind(0)
    output_masks = []
    for i, mask in enumerate(masks):
        mask = mask.detach().cpu().numpy()
        mask = mask.astype('uint8') * 255
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2)
        total_size = np.prod(mask.shape)
        area_size = np.count_nonzero(mask)
        shared.log.debug(f'Segment mask: area={area_size/total_size:.2f} score={scores[i].item():.2f}')
        mask = Image.fromarray(mask)
        output_masks.append(mask)


def create_segment_ui(input_image: gr.Image, preview_image: gr.Image):
    selected = gr.Dropdown(label="Segment", choices=MODELS.keys(), value='None')
    selected.change(fn=run_segment, inputs=[selected, input_image], outputs=[input_image, preview_image])
    return selected
