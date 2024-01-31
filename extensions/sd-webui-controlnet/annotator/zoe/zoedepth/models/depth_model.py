# MIT License

# Copyright (c) 2022 Intelligent Systems Lab Org

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# File author: Shariq Farooq Bhat

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import PIL.Image
from PIL import Image
from typing import Union


class DepthModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = 'cpu'
    
    def to(self, device) -> nn.Module:
        self.device = device
        return super().to(device)
    
    def forward(self, x, *args, **kwargs):
        raise NotImplementedError
    
    def _infer(self, x: torch.Tensor):
        """
        Inference interface for the model
        Args:
            x (torch.Tensor): input tensor of shape (b, c, h, w)
        Returns:
            torch.Tensor: output tensor of shape (b, 1, h, w)
        """
        return self(x)['metric_depth']
    
    def _infer_with_pad_aug(self, x: torch.Tensor, pad_input: bool=True, fh: float=3, fw: float=3, upsampling_mode: str='bicubic', padding_mode="reflect", **kwargs) -> torch.Tensor:
        """
        Inference interface for the model with padding augmentation
        Padding augmentation fixes the boundary artifacts in the output depth map.
        Boundary artifacts are sometimes caused by the fact that the model is trained on NYU raw dataset which has a black or white border around the image.
        This augmentation pads the input image and crops the prediction back to the original size / view.

        Note: This augmentation is not required for the models trained with 'avoid_boundary'=True.
        Args:
            x (torch.Tensor): input tensor of shape (b, c, h, w)
            pad_input (bool, optional): whether to pad the input or not. Defaults to True.
            fh (float, optional): height padding factor. The padding is calculated as sqrt(h/2) * fh. Defaults to 3.
            fw (float, optional): width padding factor. The padding is calculated as sqrt(w/2) * fw. Defaults to 3.
            upsampling_mode (str, optional): upsampling mode. Defaults to 'bicubic'.
            padding_mode (str, optional): padding mode. Defaults to "reflect".
        Returns:
            torch.Tensor: output tensor of shape (b, 1, h, w)
        """
        # assert x is nchw and c = 3
        assert x.dim() == 4, "x must be 4 dimensional, got {}".format(x.dim())
        assert x.shape[1] == 3, "x must have 3 channels, got {}".format(x.shape[1])

        if pad_input:
            assert fh > 0 or fw > 0, "atlease one of fh and fw must be greater than 0"
            pad_h = int(np.sqrt(x.shape[2]/2) * fh)
            pad_w = int(np.sqrt(x.shape[3]/2) * fw)
            padding = [pad_w, pad_w]
            if pad_h > 0:
                padding += [pad_h, pad_h]
            
            x = F.pad(x, padding, mode=padding_mode, **kwargs)
        out = self._infer(x)
        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=(x.shape[2], x.shape[3]), mode=upsampling_mode, align_corners=False)
        if pad_input:
            # crop to the original size, handling the case where pad_h and pad_w is 0
            if pad_h > 0:
                out = out[:, :, pad_h:-pad_h,:]
            if pad_w > 0:
                out = out[:, :, :, pad_w:-pad_w]
        return out
    
    def infer_with_flip_aug(self, x, pad_input: bool=True, **kwargs) -> torch.Tensor:
        """
        Inference interface for the model with horizontal flip augmentation
        Horizontal flip augmentation improves the accuracy of the model by averaging the output of the model with and without horizontal flip.
        Args:
            x (torch.Tensor): input tensor of shape (b, c, h, w)
            pad_input (bool, optional): whether to use padding augmentation. Defaults to True.
        Returns:
            torch.Tensor: output tensor of shape (b, 1, h, w)
        """
        # infer with horizontal flip and average
        out = self._infer_with_pad_aug(x, pad_input=pad_input, **kwargs)
        out_flip = self._infer_with_pad_aug(torch.flip(x, dims=[3]), pad_input=pad_input, **kwargs)
        out = (out + torch.flip(out_flip, dims=[3])) / 2
        return out
    
    def infer(self, x, pad_input: bool=True, with_flip_aug: bool=True, **kwargs) -> torch.Tensor:
        """
        Inference interface for the model
        Args:
            x (torch.Tensor): input tensor of shape (b, c, h, w)
            pad_input (bool, optional): whether to use padding augmentation. Defaults to True.
            with_flip_aug (bool, optional): whether to use horizontal flip augmentation. Defaults to True.
        Returns:
            torch.Tensor: output tensor of shape (b, 1, h, w)
        """
        if with_flip_aug:
            return self.infer_with_flip_aug(x, pad_input=pad_input, **kwargs)
        else:
            return self._infer_with_pad_aug(x, pad_input=pad_input, **kwargs)
    
    @torch.no_grad()
    def infer_pil(self, pil_img, pad_input: bool=True, with_flip_aug: bool=True, output_type: str="numpy", **kwargs) -> Union[np.ndarray, PIL.Image.Image, torch.Tensor]:
        """
        Inference interface for the model for PIL image
        Args:
            pil_img (PIL.Image.Image): input PIL image
            pad_input (bool, optional): whether to use padding augmentation. Defaults to True.
            with_flip_aug (bool, optional): whether to use horizontal flip augmentation. Defaults to True.
            output_type (str, optional): output type. Supported values are 'numpy', 'pil' and 'tensor'. Defaults to "numpy".
        """
        x = transforms.ToTensor()(pil_img).unsqueeze(0).to(self.device)
        out_tensor = self.infer(x, pad_input=pad_input, with_flip_aug=with_flip_aug, **kwargs)
        if output_type == "numpy":
            return out_tensor.squeeze().cpu().numpy()
        elif output_type == "pil":
            # uint16 is required for depth pil image
            out_16bit_numpy = (out_tensor.squeeze().cpu().numpy()*256).astype(np.uint16)
            return Image.fromarray(out_16bit_numpy)
        elif output_type == "tensor":
            return out_tensor.squeeze().cpu()
        else:
            raise ValueError(f"output_type {output_type} not supported. Supported values are 'numpy', 'pil' and 'tensor'")
    