# Original project: https://github.com/Vishnunkumar/clipcrop/blob/main/clipcrop/clipcrop.py
import os.path
import sys

import cv2
import numpy
import numpy as np
import torch
from PIL import Image
from clip import clip
from transformers import CLIPProcessor, CLIPModel, pipeline

import modules.paths
from modules import shared, modelloader
from repositories.CodeFormer.facelib.detection.yolov5face.utils.general import xyxy2xywh, xywh2xyxy


def clip_boxes(boxes, shape):
    # Clip boxes (xyxy) to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2


def interrogate_image(image: Image):
    prev_artists = shared.opts.interrogate_use_builtin_artists
    prev_max = shared.opts.interrogate_clip_max_length
    prev_min = shared.opts.interrogate_clip_min_length
    shared.opts.interrogate_clip_min_length = 10
    shared.opts.interrogate_clip_max_length = 20
    shared.opts.interrogate_use_builtin_artists = False
    prompt = shared.interrogator.interrogate(image)
    shared.opts.interrogate_clip_min_length = prev_min
    shared.opts.interrogate_clip_max_length = prev_max
    shared.opts.interrogate_use_builtin_artists = prev_artists

    print(f"Prompt: {prompt}")
    return prompt


def find_position(parent: Image, child: Image):
    w = child.width
    h = child.height
    res = cv2.matchTemplate(np.array(parent), np.array(child), cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    top_left = max_loc
    center_x = top_left[0] + (w / 2)
    center_y = top_left[1] + (h / 2)
    return center_x, center_y


class CropClip:
    def __init__(self):
        # Model
        model_name = 'yolov5s.pt'
        model_url = 'https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5s.pt'
        model_dir = os.path.join(modules.paths.models_path, "yolo")
        model_path = modelloader.load_models(model_dir, model_url, None, '.pt', model_name)
        print(f"Model path: {model_path}")
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', model_path[0])
        # Prevent BLIP crossfire breakage
        del sys.modules['models']

    def get_center(self, image: Image, prompt: str = None):
        # If no prompt specified, interrogate the image
        if prompt is None:
            prompt = interrogate_image(image)
        # Load image into YOLO parser
        results = self.model(image)  # includes NMS
        # Crop each image result to an array
        cropped = results.crop(False)
        l = []
        for crop in cropped:
            l.append(Image.fromarray(crop["im"]))
        if len(l) == 0:
            l = [image]
        device = shared.device
        # Take out cropped YOLO images, and get the features?
        model, preprocess = clip.load("ViT-B/32", device=device)
        images = torch.stack([preprocess(im) for im in l]).to(device)
        with torch.no_grad():
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)

        image_features.cpu().numpy()
        image_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).cuda()
        image_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).cuda()

        images = [preprocess(im) for im in l]
        image_input = torch.tensor(np.stack(images)).cuda()
        image_input -= image_mean[:, None, None]
        image_input /= image_std[:, None, None]
        with torch.no_grad():
            image_features = model.encode_image(image_input).float()
        image_features /= image_features.norm(dim=-1, keepdim=True)

        def similarity_top(similarity_list, N):
            results = zip(range(len(similarity_list)), similarity_list)
            results = sorted(results, key=lambda x: x[1], reverse=True)
            top_images = []
            scores = []
            for index, score in results[:N]:
                scores.append(score)
                top_images.append(l[index])
            return scores, top_images

        # @title Crop
        with torch.no_grad():
            # Encode and normalize the description using CLIP
            text_encoded = model.encode_text(clip.tokenize(prompt).to(device))
            text_encoded /= text_encoded.norm(dim=-1, keepdim=True)

        # Retrieve the description vector and the photo vectors
        similarity = text_encoded.cpu().numpy() @ image_features.cpu().numpy().T
        similarity = similarity[0]
        scores, imgs = similarity_top(similarity, N=1)
        out = imgs[0]
        res = cv2.matchTemplate(numpy.array(image), numpy.array(out), cv2.TM_SQDIFF)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        top_left = min_loc
        bottom_right = (top_left[0] + out.width, top_left[1] + out.height)
        return [top_left[0], bottom_right[0], top_left[1], bottom_right[1]]


class ClipSeg:
    def __init__(self):
        self.num = None
        self.img = None
        self.segim = None
        self.segmentor = pipeline("image-segmentation")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    @staticmethod
    def unmask(img, mask):
        imask = cv2.bitwise_and(np.array(img), np.array(img), mask=np.array(mask))
        return imask

    def segment_image(self, image: Image, num=None):
        self.num = num
        segments = self.segmentor(image)
        text = interrogate_image(image)
        img = image
        images_list = []
        coords_list = []
        scores = []
        for x in segments:
            um = self.unmask(img, x['mask'])
            pil_mask = Image.fromarray(um)
            center = find_position(image, pil_mask)
            images_list.append(um)
            coords_list.append(center)
            scores.append(x['score'])
        inputs = self.processor(text=[text], images=images_list, return_tensors="pt", padding=True)
        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_text
        probs = logits_per_image.softmax(-1).detach().numpy()
        res_list = np.argsort(probs[0])[::-1]

        if self.num is None:
            self.num = 1

        seg_list = []
        for x in res_list[:self.num]:
            seg_dict = {"image": images_list[x], "score": scores[x], "center": coords_list[x]}
            seg_list.append(seg_dict)

        return seg_list
