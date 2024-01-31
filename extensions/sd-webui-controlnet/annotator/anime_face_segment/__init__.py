import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import fnmatch
import cv2

import sys

import numpy as np
from modules import devices
from einops import rearrange
from annotator.annotator_path import models_path

import torchvision
from torchvision.models import MobileNet_V2_Weights
from torchvision import transforms

COLOR_BACKGROUND = (255,255,0)
COLOR_HAIR = (0,0,255)
COLOR_EYE = (255,0,0)
COLOR_MOUTH = (255,255,255)
COLOR_FACE = (0,255,0)
COLOR_SKIN = (0,255,255)
COLOR_CLOTHES = (255,0,255)
PALETTE = [COLOR_BACKGROUND,COLOR_HAIR,COLOR_EYE,COLOR_MOUTH,COLOR_FACE,COLOR_SKIN,COLOR_CLOTHES]

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.NUM_SEG_CLASSES = 7 # Background, hair, face, eye, mouth, skin, clothes
        
        mobilenet_v2 = torchvision.models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        mob_blocks = mobilenet_v2.features
        
        # Encoder
        self.en_block0 = nn.Sequential(    # in_ch=3 out_ch=16
            mob_blocks[0],
            mob_blocks[1]
        )
        self.en_block1 = nn.Sequential(    # in_ch=16 out_ch=24
            mob_blocks[2],
            mob_blocks[3],
        )
        self.en_block2 = nn.Sequential(    # in_ch=24 out_ch=32
            mob_blocks[4],
            mob_blocks[5],
            mob_blocks[6],
        )
        self.en_block3 = nn.Sequential(    # in_ch=32 out_ch=96
            mob_blocks[7],
            mob_blocks[8],
            mob_blocks[9],
            mob_blocks[10],
            mob_blocks[11],
            mob_blocks[12],
            mob_blocks[13],
        )
        self.en_block4 = nn.Sequential(    # in_ch=96 out_ch=160
            mob_blocks[14],
            mob_blocks[15],
            mob_blocks[16],
        )
        
        # Decoder
        self.de_block4 = nn.Sequential(     # in_ch=160 out_ch=96
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(160, 96, kernel_size=3, padding=1),
            nn.InstanceNorm2d(96),
            nn.LeakyReLU(0.1),
            nn.Dropout(p=0.2)
        )
        self.de_block3 = nn.Sequential(     # in_ch=96x2 out_ch=32
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(96*2, 32, kernel_size=3, padding=1),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.Dropout(p=0.2)
        )
        self.de_block2 = nn.Sequential(     # in_ch=32x2 out_ch=24
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(32*2, 24, kernel_size=3, padding=1),
            nn.InstanceNorm2d(24),
            nn.LeakyReLU(0.1),
            nn.Dropout(p=0.2)
        )
        self.de_block1 = nn.Sequential(     # in_ch=24x2 out_ch=16
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(24*2, 16, kernel_size=3, padding=1),
            nn.InstanceNorm2d(16),
            nn.LeakyReLU(0.1),
            nn.Dropout(p=0.2)
        )
        
        self.de_block0 = nn.Sequential(     # in_ch=16x2 out_ch=7
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(16*2, self.NUM_SEG_CLASSES, kernel_size=3, padding=1),
            nn.Softmax2d()
        )
        
    def forward(self, x):
        e0 = self.en_block0(x)
        e1 = self.en_block1(e0)
        e2 = self.en_block2(e1)
        e3 = self.en_block3(e2)
        e4 = self.en_block4(e3)
        
        d4 = self.de_block4(e4)
        d4 = F.interpolate(d4, size=e3.size()[2:], mode='bilinear', align_corners=True)
        c4 = torch.cat((d4,e3),1)

        d3 = self.de_block3(c4)
        d3 = F.interpolate(d3, size=e2.size()[2:], mode='bilinear', align_corners=True)
        c3 = torch.cat((d3,e2),1)

        d2 = self.de_block2(c3)
        d2 = F.interpolate(d2, size=e1.size()[2:], mode='bilinear', align_corners=True)
        c2 =torch.cat((d2,e1),1)

        d1 = self.de_block1(c2)
        d1 = F.interpolate(d1, size=e0.size()[2:], mode='bilinear', align_corners=True)
        c1 = torch.cat((d1,e0),1)
        y = self.de_block0(c1)
        
        return y


class AnimeFaceSegment:

    model_dir = os.path.join(models_path, "anime_face_segment")

    def __init__(self):
        self.model = None
        self.device = devices.get_device_for("controlnet")

    def load_model(self):
        remote_model_path = "https://huggingface.co/bdsqlsz/qinglong_controlnet-lllite/resolve/main/Annotators/UNet.pth"
        modelpath = os.path.join(self.model_dir, "UNet.pth")
        if not os.path.exists(modelpath):
            from basicsr.utils.download_util import load_file_from_url
            load_file_from_url(remote_model_path, model_dir=self.model_dir)
        net = UNet()
        ckpt = torch.load(modelpath, map_location=self.device)
        for key in list(ckpt.keys()):
            if 'module.' in key:
                ckpt[key.replace('module.', '')] = ckpt[key]
                del ckpt[key]
        net.load_state_dict(ckpt)
        net.eval()
        self.model = net.to(self.device)

    def unload_model(self):
        if self.model is not None:
            self.model.cpu()

    def __call__(self, input_image):

        if self.model is None:
            self.load_model()
        self.model.to(self.device)
        transform = transforms.Compose([  
            transforms.Resize(512,interpolation=transforms.InterpolationMode.BICUBIC),  
            transforms.ToTensor(),])
        img = Image.fromarray(input_image)
        with torch.no_grad():
            img = transform(img).unsqueeze(dim=0).to(self.device)
            seg = self.model(img).squeeze(dim=0)
            seg = seg.cpu().detach().numpy()
            img = rearrange(seg,'h w c -> w c h')
            img = [[PALETTE[np.argmax(val)] for val in buf]for buf in img]
            return np.array(img).astype(np.uint8)