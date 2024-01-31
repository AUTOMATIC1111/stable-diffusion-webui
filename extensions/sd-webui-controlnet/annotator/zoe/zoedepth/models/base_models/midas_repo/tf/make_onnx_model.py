"""Compute depth maps for images in the input folder.
"""
import os
import ntpath
import glob
import torch
import utils
import cv2
import numpy as np
from torchvision.transforms import Compose, Normalize
from torchvision import transforms

from shutil import copyfile
import fileinput
import sys
sys.path.append(os.getcwd() + '/..')
                 
def modify_file():
    modify_filename = '../midas/blocks.py'
    copyfile(modify_filename, modify_filename+'.bak')

    with open(modify_filename, 'r') as file :
      filedata = file.read()

    filedata = filedata.replace('align_corners=True', 'align_corners=False')
    filedata = filedata.replace('import torch.nn as nn', 'import torch.nn as nn\nimport torchvision.models as models')
    filedata = filedata.replace('torch.hub.load("facebookresearch/WSL-Images", "resnext101_32x8d_wsl")', 'models.resnext101_32x8d()')

    with open(modify_filename, 'w') as file:
      file.write(filedata)
      
def restore_file():
    modify_filename = '../midas/blocks.py'
    copyfile(modify_filename+'.bak', modify_filename)

modify_file()

from midas.midas_net import MidasNet
from midas.transforms import Resize, NormalizeImage, PrepareForNet

restore_file()


class MidasNet_preprocessing(MidasNet):
    """Network for monocular depth estimation.
    """
    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input data (image)

        Returns:
            tensor: depth
        """

        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        x.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])

        return MidasNet.forward(self, x)


def run(model_path):
    """Run MonoDepthNN to compute depth maps.

    Args:
        model_path (str): path to saved model
    """
    print("initialize")

    # select device

    # load network
    #model = MidasNet(model_path, non_negative=True)
    model = MidasNet_preprocessing(model_path, non_negative=True)

    model.eval()
    
    print("start processing")

    # input
    img_input = np.zeros((3, 384, 384), np.float32)  

    # compute
    with torch.no_grad():
        sample = torch.from_numpy(img_input).unsqueeze(0)
        prediction = model.forward(sample)
        prediction = (
            torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img_input.shape[:2],
                mode="bicubic",
                align_corners=False,
            )
            .squeeze()
            .cpu()
            .numpy()
        )

    torch.onnx.export(model, sample, ntpath.basename(model_path).rsplit('.', 1)[0]+'.onnx', opset_version=9)    
    
    print("finished")


if __name__ == "__main__":
    # set paths
    # MODEL_PATH = "model.pt"
    MODEL_PATH = "../model-f6b98070.pt"
    
    # compute depth maps
    run(MODEL_PATH)
