'''
Author: SpenserCai
Date: 2023-07-28 15:49:52
version: 
LastEditors: SpenserCai
LastEditTime: 2023-08-03 16:22:46
Description: file content
'''
from deoldify import device
from deoldify.device_id import DeviceId
from PIL import Image
import gradio as gr
import base64
#choices:  CPU, GPU0...GPU7
device.set(device=DeviceId.GPU0)

from deoldify.visualize import *

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*?Your .*? set is empty.*?")
warnings.filterwarnings("ignore", category=UserWarning, message="The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.")
warnings.filterwarnings("ignore", category=FutureWarning, message="Arguments other than a weight enum or `None`.*?")

# 图片转换为base64编码
def image_to_base64(image_path):
    with open(image_path, 'rb') as f:
        image = f.read()
    image_b64 = base64.b64encode(image).decode()
    return image_b64

def ColorizeImage(base64str, render_factor=50, artistic=False):
    vis = get_image_colorizer(root_folder=Path("models"),render_factor=render_factor, artistic=artistic)
    # 把base64转换成图片 PIL.Image
    img = Image.open(BytesIO(base64.b64decode(base64str)))
    outImg = vis.get_transformed_image_from_image(img, render_factor=render_factor)
    return outImg

with gr.Blocks(analytics_enabled=False) as ai_app_interface:
    def DeOldifyImage(image,render_factor=35,artistic=False):
        if isinstance(image,str):
            image = image_to_base64(image)
        outImg = ColorizeImage(image,render_factor,artistic)
        return outImg
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="原图",type="filepath")
            # 一个名为render_factor的滑块，范围是1-50，初始值是35，步长是1
            render_factor = gr.Slider(minimum=1, maximum=50, step=1, label="渲染因子")
            render_factor.value = 35
            # 一个名为artistic的复选框，初始值是False
            artistic = gr.Checkbox(label="艺术化")
            artistic.value = False
        with gr.Column():
            image_output = gr.Image(label="修复后的图片",type="pil")
    fix_button = gr.Button(label="上色")
    fix_button.click(inputs=[image_input,render_factor,artistic],outputs=[image_output],fn=DeOldifyImage)

ai_app_interface.launch(share=True)

    
    