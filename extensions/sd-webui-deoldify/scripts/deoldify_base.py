'''
Author: SpenserCai
Date: 2023-08-09 09:58:27
version: 
LastEditors: SpenserCai
LastEditTime: 2023-09-07 10:32:51
Description: file content
'''
from modules import shared
from deoldify import device as deoldify_device
from deoldify.device_id import DeviceId
import cv2
from PIL.Image import Image

device_id = shared.cmd_opts.device_id

if device_id is not None:
    device_id = DeviceId(int(device_id))
    deoldify_device.set(device=device_id)
else:
    deoldify_device.set(device=DeviceId.GPU0)

from deoldify.visualize import *

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*?Your .*? set is empty.*?")
warnings.filterwarnings("ignore", category=UserWarning, message="The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.")
warnings.filterwarnings("ignore", category=FutureWarning, message="Arguments other than a weight enum or `None`.*?")

def Decolorization(img:Image)->Image:
    # 通过opencv吧图片褪色成黑白
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(img)