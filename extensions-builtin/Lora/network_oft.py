import torch

import diffusers.models.lora as diffusers_lora
import lyco_helpers
import network
from modules import devices
import math
import os
from typing import Dict, List, Optional, Tuple, Type, Union
from diffusers import AutoencoderKL
from transformers import CLIPTextModel
import numpy as np
import re
#Lot of these imports are likely redundant, will refactor and remove

#Unused regex within the original oft.py?
RE_UPDOWN = re.compile(r"(up|down)_blocks_(\d+)_(resnets|upsamplers|downsamplers|attentions)_(\d+)_")

class ModuleTypeOFT(network.ModuleType):
    def create_module(self, net: network.Network, weights: network.NetworkWeights):
        """
        weights.w.items()

        alpha  :  tensor(0.0010, dtype=torch.bfloat16)
        oft_blocks  :  tensor([[[ 0.0000e+00,  1.4400e-04,  1.7319e-03,  ..., -8.8882e-04,
           5.7373e-03, -4.4250e-03],
         [-1.4400e-04,  0.0000e+00,  8.6594e-04,  ...,  1.5945e-03,
          -8.5449e-04,  1.9684e-03], ...etc...
         , dtype=torch.bfloat16)"""
        
        if "oft_blocks" in weights.w.keys():
            module = NetworkModuleOFT(net, weights)
            return module
        else:
            return None

class NetworkModuleOFT(network.NetworkModule):
    def __init__(self,  net: network.Network, weights: network.NetworkWeights):
        super().__init__(net, weights)
        """
        dim -> num blocks
        alpha -> constraint

        alpha is equal to eps-deviation: eps
        (only with the constrained variant COFT)
        """
        self.weights = weights.w.get("oft_blocks").to(device=devices.device)
        self.net = net
        self.alpha = self.multiplier()
        self.dim = self.weights.shape[0] #num blocks
        
        # old way of calculating out_features, not technically correct:
        #self.out_dim = max(self.weights.shape[1],self.weights.shape[2])*self.dim 
        
        self.is_linear = type(self.sd_module) in [torch.nn.Linear, torch.nn.modules.linear.NonDynamicallyQuantizableLinear, torch.nn.MultiheadAttention, diffusers_lora.LoRACompatibleLinear]
        self.is_conv = type(self.sd_module) in [torch.nn.Conv2d, diffusers_lora.LoRACompatibleConv]
        if self.is_linear == True:
            self.out_dim = self.sd_module.out_features
        if self.is_conv == True:
            self.out_dim = self.sd_module.out_channels
        #The is_conv check should be redundant? I havent seen any conv layers in my testing
        
        self.block_size = self.out_dim // self.dim

        #Initialize to zeros:
        #self.oft_blocks = torch.nn.Parameter(torch.zeros(self.dim, self.block_size, self.block_size)).to(device=devices.device)
        #Load from weights
        self.oft_blocks = torch.nn.Parameter(self.weights)
        #self.oft_blocks = torch.nn.Parameter(self.weights*self.alpha) #not sure if I need to apply alpha here but I just do anyway, should weaken

        #eps constraint value, calculate by using (alpha in weights) * (out_dim)
        self.constraint = weights.w.get("alpha").to(device=devices.device)*self.out_dim

    def get_weight(self):
        try:
            self.alpha = self.multiplier() #update alpha? Not sure if necessary.
            #get_weight implementation:
            block_Q = self.weights - self.weights.transpose(1, 2)
            norm_Q = torch.norm(block_Q.flatten())
            new_norm_Q = torch.clamp(norm_Q, max=self.constraint)
            block_Q = (block_Q * ((new_norm_Q + 1e-8) / (norm_Q + 1e-8))).to(device=devices.device)
            I = torch.eye(self.block_size, device=devices.device).unsqueeze(0).repeat(self.dim, 1, 1)
            block_R = torch.matmul(I + block_Q, (I - block_Q).inverse())
            block_R_weighted =  self.alpha*block_R + (1 - self.alpha) * I
            R = torch.block_diag(*block_R_weighted)
            R = R * self.alpha #Added this line, seems to make the results better, less overbaked
            return R
        except Exception as e:
            print("ERROR:")
            print(e)
        
        
    
    def calc_updown(self, orig_weight):
        self.alpha = self.multiplier() #update alpha? Not sure if necessary.
        output_shape = self.weights.shape
        R = self.get_weight().to(device=devices.device, dtype=orig_weight.dtype)
        
        try:
            #if orig_weight.shape[0] < orig_weight.shape[1]:
                #attempt 1
                #R_expanded = torch.zeros(output_shape, device=devices.device, dtype=orig_weight.dtype)
                #R_expanded[:, :R.shape[1]] = R
                #R = R_expanded
                #temp = orig_weight[:, :R.shape[0]]
                #updown = torch.matmul(temp, R)

                #attempt 2
                #blocks = torch.split(orig_weight, split_size_or_sections=orig_weight.shape[1]//self.dim, dim=1)
                #results = [torch.matmul(block,R) for block in blocks]
                #updown = torch.cat(results, dim=1)

                #attempt 3
                #blocks = torch.split(orig_weight, split_size_or_sections=orig_weight.shape[1]//self.dim, dim=1)
                #print("R.shape:")
                #print(R.shape)
                #transformed_blocks = [torch.matmul(block.transpose(1,0),R) for block in blocks]
                #for i in range(0, len(transformed_blocks)):
                    #transformed_blocks[i] = transformed_blocks[i].transpose(1,0)
                #print("END_UPDOWN")
                #updown = torch.cat(transformed_blocks, dim=1)
            #else:
                #updown = torch.matmul(orig_weight, R)
            
            #Attempt 4:
            if self.is_linear:
                if orig_weight.shape[0] < orig_weight.shape[1]: 
                    #check for irregular linear sizes, if dim1 is larger than dim0, that means:
                    #we have dim1 composed of self.dim elements (blocks)
                    #in order to apply batched matmul, we need to view this differently, add a dimension for our blocks
                    x = orig_weight.view(self.dim, orig_weight.shape[0], orig_weight.shape[1]//self.dim)
                    #x = orig_weight.view(self.dim, orig_weight.shape[1]//self.dim, orig_weight.shape[0])
                    
                    #Since our size is irregular, I've made some assumptions here that may not be correct.
                    #I still do not fully understand what "orig_weight" represents relative to "x" in the original oft.py forward()

                    #PROBLEM EXPLANATION:
                    #We need to do a matmul between x and R
                    #That means that x columns = R rows
                    #R will always end up a square matrix of size 640x640, or 1280x1280 (something like that)
                    
                    # However, in THESE cases, where orig_weight.shape[0] < orig_weight.shape[1]:
                    # x = [640,2048] or some other similar size
                    # We would then divide 2048 into self.dim chunks (in this case 4), and get 512
                    # Thus we end up with: [4, 640, 512] where 2048 got split up into 4 channels (aka our dim)

                    # Unfortunately, we cannot apply R as a matmul on this since we have unmatched dimensions
                    # to make this calculation possible, we need to take the transpose dim(1,2) of [4, 640, 512] to get [4, 512, 640]

                    # We repeat R to fill our 4 channels, and do a batch matmul between x and R:
                    # [4, 512, 640](x) * [4, 640, 640](R)

                    # Now after that, just torch.cat the 4 channels together back into the same shape as the beginning
                    
                    # This is just an example calculation, but one like this does happen many times
                    # Well, now we can kinda "calculate" something, but im honestly not sure if this is applying R properly at all.
                    # Here is the original forward from kohya's oft.py:
                    # If we could figure out a way to apply this same operation (permute/matmul for 4 dimensional input), but to our orig_weight instead of x, that would work perfect

                    # Note: x.dim() == 4 is related to our (orig_weight.shape[0] < orig_weight.shape[1]) check
                    # If the sizes are not the same, then orig_weight.shape[1]//self.dim is the new size of our block (in that one dimension)
                    """ 
                    def forward(self, x, scale=None):
                        x = self.org_forward(x)
                        if self.multiplier == 0.0:
                            return x

                        R = self.get_weight().to(x.device, dtype=x.dtype)
                        if x.dim() == 4:
                            x = x.permute(0, 2, 3, 1)
                            x = torch.matmul(x, R)
                            x = x.permute(0, 3, 1, 2)
                        else:
                            x = torch.matmul(x, R)
                        return x
                    """

                    x = x.transpose(1,2)
                    #R_expanded = R.unsqueeze(0).expand(x.shape[0], -1, -1)
                    R_expanded = R.unsqueeze(0).repeat(x.shape[0], 1, 1)
                    #x = torch.bmm(x, R_expanded)
                    x = torch.matmul(x, R_expanded)
                    #x = x.transpose(1,2)
                    updown = torch.cat(x.unbind(0), dim=1)
                    #updown = x.view(orig_weight.shape[0], orig_weight.shape[1])
                else:
                    updown = torch.matmul(orig_weight, R)
            elif self.is_conv:
                updown = torch.matmul(orig_weight, R)
            return(self.finalize_updown(updown, orig_weight, output_shape))
            
        except Exception as e:
            print("ERROR:")
            print(e)

            