import argparse
import copy

from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import box_convert

# Grounding DINO
from local_groundingdino.datasets import transforms as T
from local_groundingdino.models import build_model
from local_groundingdino.util import box_ops
from local_groundingdino.util.slconfig import SLConfig
from local_groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from local_groundingdino.util.inference import annotate, load_image, predict

import supervision as sv
import argparse
import cv2
import matplotlib.pyplot as plt
from rembg import session_factory, remove
# diffusers
import PIL
import requests
import torch

import base64
from PIL import Image, ImageOps, PngImagePlugin
from io import BytesIO

# from flask import Flask, request
import time

### face recognize
# from facenet_pytorch import MTCNN, InceptionResnetV1
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np
import pandas as pd
import os


# models_path = "/data/longcheng/stable-diffusion-webui/models"
# sd_sam_model_dir = os.path.join(models_path, "sam")
# sd_dino_model_dir = os.path.join(models_path, "grounding-dino/GroundingDINO_SwinT_OGC.py")
# sam_model_list = [f for f in os.listdir(sd_sam_model_dir) if os.path.isfile(os.path.join(sd_sam_model_dir, f)) and f.split('.')[-1] != 'txt']
# dino_model_list = ["GroundingDINO_SwinT_OGC (694MB)", "GroundingDINO_SwinB (938MB)"]

# ckpt_filenmae = os.path.join(models_path, "grounding-dino/groundingdino_swint_ogc.pth")
# device = 'cuda'

# local_image_path = 'assets/00034-1722019954.png'

# Run Grounding DINO for detection

TEXT_PROMPT = "head"
BOX_TRESHOLD = 0.05
TEXT_TRESHOLD = 0.05


def show_mask(masks, image_source, random_color=True):
    image = image_source.copy()
    # print(image.shape, masks.shape)
    
    # print(f'masks shape: {masks.shape}')
    # print(np.full(image.shape, 255, dtype=int).shape)
    total_mask =  np.sum(masks, axis=0)
    # for mask in masks:
        # print(f'mask shape: {mask.shape}')
    mask_3d = np.stack([total_mask, total_mask, total_mask], axis=-1)
        
        # print(f'mask 3d shape: {mask_3d.shape}, {mask_3d}')
    image = np.where(mask_3d,  image, np.full(image.shape, 255, dtype=np.uint8))
    return image   

def convert_image(image_source: np.array) -> torch.Tensor:
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_transformed, _ = transform(image_source, None)
    return image_transformed

def get_max_box(boxes):
    max_area = -1
    max_box = None
    for box in boxes:
        area = (box[2] - box[0]) * (box[3] - box[1])
        if area > max_area:
            max_box = box
            max_area = area
    # print(f'boxes: {boxes}, max_area: {max_area}')
    # print(max_box)
    return max_box

class MySeg:
    def __init__(self,models_path):
        self.models_path = models_path
        # sd_sam_model_dir = os.path.join(models_path, "sam")
        sd_dino_model_dir = os.path.join(models_path, "local_groundingdino/config/GroundingDINO_SwinT_OGC.py")
        # sam_model_list = [f for f in os.listdir(sd_sam_model_dir) if os.path.isfile(os.path.join(sd_sam_model_dir, f)) and f.split('.')[-1] != 'txt']
        # self.dino_model_list = ["GroundingDINO_SwinT_OGC (694MB)", "GroundingDINO_SwinB (938MB)"]
        self.device = 'cuda'

        ckpt_filenmae = os.path.join(models_path, "grounding-dino/groundingdino_swint_ogc.pth")
        
        self.groundingdino_model = self.load_model("local_groundingdino/config/GroundingDINO_SwinT_OGC.py", ckpt_filenmae, 'cuda')
        print(f'init groundingdino_model success')
        # rembg
        # rembg_model_name = 'u2net_human_seg'
        rembg_model_name = 'u2net'
        self.sess = session_factory.new_session(rembg_model_name)
        
    
    def load_model(self, model_config_path, model_checkpoint_path, device="cpu"):
        args = SLConfig.fromfile(model_config_path)
        args.device = device
        model = build_model(args)
        checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
        load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        print(load_res)
        _ = model.eval()
        return model
    
    def has_head(self, image_source, ):
        text_prompt='face'
        image = convert_image(image_source)
        image_source = np.asarray(image_source)
        
        boxes, logits, phrases = predict(
            model=self.groundingdino_model, 
            image=image, 
            caption=TEXT_PROMPT, 
            box_threshold=BOX_TRESHOLD, 
            text_threshold=TEXT_TRESHOLD
        )
        return 'face' in phrases
        
    def stop(self):
        del self.groundingdino_model
    # resize_type,0-3:0-3号Box,4:max,-1,all, -2 do not crop,box;

    
    def seg(self, image_source1, text_prompt=TEXT_PROMPT, resize_type=-3, rmbg=False, do_resize=False,resize_x=512,resize_y=512):
        if image_source1.mode != "RGB":
            image_source1 = image_source1.convert("RGB")

        image = convert_image(image_source1)
        image_source = np.asarray(image_source1)
        boxes, logits, phrases = predict(
            model=self.groundingdino_model, 
            image=image, 
            caption=text_prompt,
            box_threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD
        )
        
        H, W, _ = image_source.shape
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])
        #print(boxes_xyxy)
        if boxes_xyxy.shape[0]==0:
            return [image_source1]
        cropped_images=[]
        #crop
        if resize_type>=0 and resize_type<4: #某个号的
            box = boxes_xyxy[resize_type].cpu().detach().numpy()
            cropped_image = Image.fromarray(image_source).crop(box)
            cropped_images = [cropped_image]
            print("box,",box)
        elif resize_type == -2: #剪切扩大后的头像
            box = boxes_xyxy[0].cpu().detach().numpy()
            print("box1,",box)
            width1=box[2]-box[0]
            height1=box[3]-box[1]
            #width=max(abs(width1),abs(height1))*width1/abs(width1)
            #height=width*height1/abs(height1)
            
            if width1/height1 >=resize_x/resize_y:
                width = width1*3/2
                height = width * resize_y/resize_x

            else:   
                height = height1*3/2
                width = height * resize_x/resize_y
            if abs(width)<300 or abs(height)<300:return []
            #box[0]=max(0,box[0]-width/2)
            #box[2]=min(W,box[2]+width/2)
            #box[1]=max(0,box[1]-height/2)
            #box[3]=min(H,box[3]+height/2)
            box[0]=box[0]-(width/2-width1/2)
            box[2]=box[2]+(width/2-width1/2)
            box[1]=box[1]-(height/2-height1/2)
            box[3]=box[3]+(height/2-height1/2)
            print("box2,",box,width,height)
            cropped_image = Image.fromarray(image_source).crop(box)
            cropped_images = [cropped_image]

        elif resize_type == -3: #剪切的头像为正方形
            box = boxes_xyxy[0].cpu().detach().numpy()
            print("box1,",box)
            width1=box[2]-box[0]
            height1=box[3]-box[1]
            #width=max(abs(width1),abs(height1))*width1/abs(width1)
            #height=width*height1/abs(height1)
            
            if width1/height1 >=resize_x/resize_y:
                width = width1
                height = width * resize_y/resize_x

            else:   
                height = height1
                width = height * resize_x/resize_y
            if abs(width)<300 or abs(height)<300:return []
            #box[0]=max(0,box[0]-width/2)
            #box[2]=min(W,box[2]+width/2)
            #box[1]=max(0,box[1]-height/2)
            #box[3]=min(H,box[3]+height/2)
            box[0]=box[0]-(width/2-width1/2)
            box[2]=box[2]+(width/2-width1/2)
            box[1]=box[1]-(height/2-height1/2)
            box[3]=box[3]+(height/2-height1/2)
            print("box2,",box,width,height)
            cropped_image = Image.fromarray(image_source).crop(box)
            cropped_images = [cropped_image]
            
        elif resize_type == 4: #最大的
            box = get_max_box(boxes_xyxy.cpu().detach().numpy())
            cropped_image = Image.fromarray(image_source).crop(box)
            cropped_images = [cropped_image]
        elif resize_type == -1: #所有的
            for i in range(min(boxes_xyxy.shape[0],4)):
                box = boxes_xyxy[i].cpu().detach().numpy()
                cropped_image = Image.fromarray(image_source).crop(box)
                cropped_images += [cropped_image]
        else:
            # do not crop
            cropped_image = Image.fromarray(image_source)
            cropped_images = [cropped_image]
        if rmbg:
            rmbg=[]
            for ci in cropped_images:
                rmbg.append(remove(ci, session=self.sess))
            cropped_images = rmbg
        if do_resize:
            resize_images=[]
            for ci in cropped_images:
                ci_new = ci.resize((resize_x,resize_y),Image.ANTIALIAS)
                resize_images.append(ci_new)
                #resize_images.append(ci.resize((resize_x,resize_y)))
            print(resize_images)
            return resize_images
        return cropped_images
        # return remove(cropped_image, session=self.sess) #.resize((512, 512))
        #return remove(cropped_image, session=self.sess).resize((512,512))
        # cropped_area = get_max_box(boxes_xyxy).numpy()
#         transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(boxes_xyxy, image_source.shape[:2]).to(device)
        
#         self.sam_predictor.set_image(image_source)
#         masks, _, _ = self.sam_predictor.predict_torch(
#                     point_coords = None,
#                     point_labels = None,
#                     boxes = transformed_boxes,
#                     multimask_output = False,
#                 )
#         frame_with_mask = show_mask(masks.cpu().squeeze(1).detach().numpy(), image_source)
#         final_img = Image.fromarray(frame_with_mask)
#         # cropped_area = boxes_xyxy
#         cropped_image = final_img.crop(cropped_area)
        # return remove(Image.fromarray(image_source).crop(cropped_area))
        # return cropped_image

# app = Flask(__name__)

my_seg = MySeg("/data/qll/stable-diffusion-webui/models")

def encode_pil_to_base64(pil_image):
    with BytesIO() as output_bytes:

        # Copy any text-only metadata
        use_metadata = False
        metadata = PngImagePlugin.PngInfo()
        for key, value in pil_image.info.items():
            if isinstance(key, str) and isinstance(value, str):
                metadata.add_text(key, value)
                use_metadata = True

        pil_image.save(
            output_bytes, "PNG", pnginfo=(metadata if use_metadata else None)
        )
        bytes_data = output_bytes.getvalue()
    base64_str = str(base64.b64encode(bytes_data), "utf-8")
    return "data:image/png;base64," + base64_str

# class FaceDetector(object):
#     def __init__(self,):
#         self.mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20, thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True, device=device)
#
#     def detect(self, image):
#         return self.mtcnn(image, return_prob=True)[1] is not None
#
#     def detect_and_return(self, image):
#         return self.mtcnn(image, return_prob=True)[1]



def decode_base64_to_image(encoding):
    if encoding.startswith("data:image/"):
        encoding = encoding.split(";")[1].split(",")[1]
    try:
        image = Image.open(BytesIO(base64.b64decode(encoding)))
        return image
    except:
        return EOFError

def imageToStr(image):
    img_buffer = BytesIO()
    image.save(img_buffer, format='PNG')
    byte_data = img_buffer.getvalue()
    image_byte = base64.b64encode(byte_data)
    image_str = image_byte.decode('ascii') #byte类型转换为str
    return image_str

def strToImage(image):
    image_str= image.encode('ascii')
    image_byte = base64.b64decode(image_str)
    image = BytesIO(image_byte)
    image = Image.open(image)
    return image

# face_detector = FaceDetector()


# def detect(src,dst,txt_prompt):
#     rsp = request.get_json()
#     image = rsp['image']
#     # print(image)
#     image_pil = decode_base64_to_image(image)
#     # log image for debug
#     img_ts = int(time.time()*1000)
#     print(f'handle input_{img_ts}.png')
#     image_pil.save(f'outputs/face_input_{img_ts}.png')
#
#     image_pil = image_pil.convert('RGB')
#
#     is_face = face_detector.detect(image_pil)
#     print(f'handle input_{img_ts}.png has face: {is_face}')
#     # save img for debug
#     # final_img.save(f'outputs/face_output_{img_ts}.png')
#
#     # final_img_str = encode_pil_to_base64(final_img)
#     return {'is_face': is_face}

def seg_head(src,dst,txt_prompt, resize_type=0, rmbg=True,do_resize=False,resize_x=512,resize_y=512):
    names = os.listdir(src)
    dst=os.path.join(dst, f'{txt_prompt.replace(" ","-")}')
    if not os.path.exists(dst):
        os.makedirs(dst)
    print("seg_head,n",len(names))
    for file_name in names:
        print(file_name)
        if file_name.split(".")[-1] != 'png' and file_name.split(".")[-1] != 'jpg' and file_name.split(".")[-1] != 'JPG': continue
        f = src + rf"/{file_name}"
        with Image.open(f) as im:
            final_imgs = my_seg.seg(im,txt_prompt, resize_type, rmbg, do_resize,resize_x,resize_y)
            i=0
            for fi in final_imgs:
                #if not os.path.exists(dst):
                #    os.makedirs(dst)
                # fi.save(os.path.join(dst, '{}.png'.format(file_name.split(".")[-2])))
                fi.save(os.path.join(dst, f'{file_name.split(".")[-2]}-{i}.png'))
                i+=1

def removeBG_resize(src,dst,resize=False,rs_x=512,rs_y=512):
    rembg_model_name = 'u2net'
    # sess = session_factory.new_session(rembg_model_name)
    names = os.listdir(src)
    #dst=os.path.join(dst, f'{txt_prompt.replace(" ","-")}')
    if not os.path.exists(dst):
        os.makedirs(dst)
    print("seg_head,n",len(names))
    for file_name in names:
        print(file_name)
        if file_name.split(".")[-1] != 'png' and file_name.split(".")[-1] != 'jpg' and file_name.split(".")[-1] !='JPG' and file_name.split(".")[-1] !='PNG' : continue
        f = src + rf"/{file_name}"
        with Image.open(f) as im:
     
            # final_imgs = remove(im, session=sess)
            if resize:
                final_imgs = final_imgs.resize((rs_x, rs_y), Image.ANTIALIAS)
            print(final_imgs)
            final_imgs.save(os.path.join(dst, file_name))
            #for fi in final_imgs:
                

#seg_head("/data/qll/pics/test","/data/qll/pics/test","everything",-1,True,True,512,512)
#seg_head("/data/qll/pics/test","/data/qll/pics/test","cloth",-1,True,False,512,512)
# removeBG_resize("/data/qll/pics/xitu","/data/qll/pics/xitu/cloth",False)




# @app.route('/seg', methods=['POST'])
# def app_seg():
#     rsp = request.get_json()
#     image = rsp['image']
#     # print(image)
#     image_pil = decode_base64_to_image(image)
#     # log image for debug
#     img_ts = int(time.time()*1000)
#     print(f'handle input_{img_ts}.png')
#     image_pil.save(f'input_{img_ts}.png')
#
#     image_pil = image_pil.convert('RGB')
#
#     final_img = my_seg.seg(image_pil)
#
#     # save img for debug
#     final_img.save(f'output_{img_ts}.png')
#
#     final_img_str = encode_pil_to_base64(final_img)
#     return {'res_img': final_img_str}


# @app.route('/face_detect', methods=['POST'])
# def app_detect():
#     rsp = request.get_json()
#     image = rsp['image']
#     # print(image)
#     image_pil = decode_base64_to_image(image)
#     # log image for debug
#     img_ts = int(time.time()*1000)
#     print(f'handle input_{img_ts}.png')
#     image_pil.save(f'outputs/face_input_{img_ts}.png')
#
#     image_pil = image_pil.convert('RGB')
#
#     is_face = face_detector.detect(image_pil)
#     print(f'handle input_{img_ts}.png has face: {is_face}')
#     # save img for debug
#     # final_img.save(f'outputs/face_output_{img_ts}.png')
#
#     # final_img_str = encode_pil_to_base64(final_img)
#     return {'is_face': is_face}
#
#
# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5001)
