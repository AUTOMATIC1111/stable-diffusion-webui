import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image


def warp(tenInput, tenFlow, device):
    backwarp_tenGrid = {}
    k = (str(tenFlow.device), str(tenFlow.size()))
    if k not in backwarp_tenGrid:
        tenHorizontal = torch.linspace(-1.0, 1.0, tenFlow.shape[3], device=device).view(
            1, 1, 1, tenFlow.shape[3]).expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1)
        tenVertical = torch.linspace(-1.0, 1.0, tenFlow.shape[2], device=device).view(
            1, 1, tenFlow.shape[2], 1).expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3])
        backwarp_tenGrid[k] = torch.cat(
            [tenHorizontal, tenVertical], 1).to(device)

    tenFlow = torch.cat([tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0),
                         tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0)], 1)

    g = (backwarp_tenGrid[k] + tenFlow).permute(0, 2, 3, 1)
    return torch.nn.functional.grid_sample(input=tenInput, grid=g, mode='bilinear', padding_mode='border', align_corners=True)


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),        
        nn.PReLU(out_planes)
    )


class IFBlock(nn.Module):
    def __init__(self, in_planes, c=64):
        super(IFBlock, self).__init__()
        self.conv0 = nn.Sequential(conv(in_planes, c//2, 3, 2, 1), conv(c//2, c, 3, 2, 1),)
        self.convblock0 = nn.Sequential(conv(c, c), conv(c, c))
        self.convblock1 = nn.Sequential(conv(c, c), conv(c, c))
        self.convblock2 = nn.Sequential(conv(c, c), conv(c, c))
        self.convblock3 = nn.Sequential(conv(c, c), conv(c, c))
        self.conv1 = nn.Sequential(nn.ConvTranspose2d(c, c//2, 4, 2, 1), nn.PReLU(c//2), nn.ConvTranspose2d(c//2, 4, 4, 2, 1))
        self.conv2 = nn.Sequential(nn.ConvTranspose2d(c, c//2, 4, 2, 1), nn.PReLU(c//2), nn.ConvTranspose2d(c//2, 1, 4, 2, 1))

    def forward(self, x, flow, scale=1):
        x = F.interpolate(x, scale_factor= 1. / scale, mode="bilinear", align_corners=False, recompute_scale_factor=False)
        flow = F.interpolate(flow, scale_factor= 1. / scale, mode="bilinear", align_corners=False, recompute_scale_factor=False) * 1. / scale
        feat = self.conv0(torch.cat((x, flow), 1))
        feat = self.convblock0(feat) + feat
        feat = self.convblock1(feat) + feat
        feat = self.convblock2(feat) + feat
        feat = self.convblock3(feat) + feat        
        flow = self.conv1(feat)
        mask = self.conv2(feat)
        flow = F.interpolate(flow, scale_factor=scale, mode="bilinear", align_corners=False, recompute_scale_factor=False) * scale
        mask = F.interpolate(mask, scale_factor=scale, mode="bilinear", align_corners=False, recompute_scale_factor=False)
        return flow, mask


class IFNet(nn.Module):
    def __init__(self):
        super(IFNet, self).__init__()
        self.block0 = IFBlock(7+4, c=90)
        self.block1 = IFBlock(7+4, c=90)
        self.block2 = IFBlock(7+4, c=90)
        self.block_tea = IFBlock(10+4, c=90)

    def forward(self, x, scale_list=[4, 2, 1], training=False):
        if training == False:
            channel = x.shape[1] // 2
            img0 = x[:, :channel]
            img1 = x[:, channel:]
        flow_list = []
        merged = []
        mask_list = []
        warped_img0 = img0
        warped_img1 = img1
        flow = (x[:, :4]).detach() * 0
        mask = (x[:, :1]).detach() * 0
        block = [self.block0, self.block1, self.block2]
        for i in range(3):
            f0, m0 = block[i](torch.cat((warped_img0[:, :3], warped_img1[:, :3], mask), 1), flow, scale=scale_list[i])
            f1, m1 = block[i](torch.cat((warped_img1[:, :3], warped_img0[:, :3], -mask), 1), torch.cat((flow[:, 2:4], flow[:, :2]), 1), scale=scale_list[i])
            flow = flow + (f0 + torch.cat((f1[:, 2:4], f1[:, :2]), 1)) / 2
            mask = mask + (m0 + (-m1)) / 2
            mask_list.append(mask)
            flow_list.append(flow)
            warped_img0 = warp(img0, flow[:, :2], device=x.device)
            warped_img1 = warp(img1, flow[:, 2:4], device=x.device)
            merged.append((warped_img0, warped_img1))
        '''
        c0 = self.contextnet(img0, flow[:, :2])
        c1 = self.contextnet(img1, flow[:, 2:4])
        tmp = self.unet(img0, img1, warped_img0, warped_img1, mask, flow, c0, c1)
        res = tmp[:, 1:4] * 2 - 1
        '''
        for i in range(3):
            mask_list[i] = torch.sigmoid(mask_list[i])
            merged[i] = merged[i][0] * mask_list[i] + merged[i][1] * (1 - mask_list[i])    
        return flow_list, mask_list[2], merged
    
    def state_dict_converter(self):
        return IFNetStateDictConverter()


class IFNetStateDictConverter:
    def __init__(self):
        pass

    def from_diffusers(self, state_dict):
        state_dict_ = {k.replace("module.", ""): v for k, v in state_dict.items()}
        return state_dict_
    
    def from_civitai(self, state_dict):
        return self.from_diffusers(state_dict)


class RIFEInterpolater:
    def __init__(self, model, device="cuda"):
        self.model = model
        self.device = device
        # IFNet only does not support float16
        self.torch_dtype = torch.float32

    @staticmethod
    def from_model_manager(model_manager):
        return RIFEInterpolater(model_manager.RIFE, device=model_manager.device)

    def process_image(self, image):
        width, height = image.size
        if width % 32 != 0 or height % 32 != 0:
            width = (width + 31) // 32
            height = (height + 31) // 32
            image = image.resize((width, height))
        image = torch.Tensor(np.array(image, dtype=np.float32)[:, :, [2,1,0]] / 255).permute(2, 0, 1)
        return image
    
    def process_images(self, images):
        images = [self.process_image(image) for image in images]
        images = torch.stack(images)
        return images
    
    def decode_images(self, images):
        images = (images[:, [2,1,0]].permute(0, 2, 3, 1) * 255).clip(0, 255).numpy().astype(np.uint8)
        images = [Image.fromarray(image) for image in images]
        return images
    
    def add_interpolated_images(self, images, interpolated_images):
        output_images = []
        for image, interpolated_image in zip(images, interpolated_images):
            output_images.append(image)
            output_images.append(interpolated_image)
        output_images.append(images[-1])
        return output_images
    

    @torch.no_grad()
    def interpolate_(self, images, scale=1.0):
        input_tensor = self.process_images(images)
        input_tensor = torch.cat((input_tensor[:-1], input_tensor[1:]), dim=1)
        input_tensor = input_tensor.to(device=self.device, dtype=self.torch_dtype)
        flow, mask, merged = self.model(input_tensor, [4/scale, 2/scale, 1/scale])
        output_images = self.decode_images(merged[2].cpu())
        if output_images[0].size != images[0].size:
            output_images = [image.resize(images[0].size) for image in output_images]
        return output_images
    

    @torch.no_grad()
    def interpolate(self, images, scale=1.0, batch_size=4, num_iter=1):
        # Preprocess
        processed_images = self.process_images(images)

        for iter in range(num_iter):
            # Input
            input_tensor = torch.cat((processed_images[:-1], processed_images[1:]), dim=1)

            # Interpolate
            output_tensor = []
            for batch_id in range(0, input_tensor.shape[0], batch_size):
                batch_id_ = min(batch_id + batch_size, input_tensor.shape[0])
                batch_input_tensor = input_tensor[batch_id: batch_id_]
                batch_input_tensor = batch_input_tensor.to(device=self.device, dtype=self.torch_dtype)
                flow, mask, merged = self.model(batch_input_tensor, [4/scale, 2/scale, 1/scale])
                output_tensor.append(merged[2].cpu())
            
            # Output
            output_tensor = torch.concat(output_tensor, dim=0).clip(0, 1)
            processed_images = self.add_interpolated_images(processed_images, output_tensor)
            processed_images = torch.stack(processed_images)

        # To images
        output_images = self.decode_images(processed_images)
        if output_images[0].size != images[0].size:
            output_images = [image.resize(images[0].size) for image in output_images]
        return output_images


class RIFESmoother(RIFEInterpolater):
    def __init__(self, model, device="cuda"):
        super(RIFESmoother, self).__init__(model, device=device)

    @staticmethod
    def from_model_manager(model_manager):
        return RIFESmoother(model_manager.RIFE, device=model_manager.device)
    
    def process_tensors(self, input_tensor, scale=1.0, batch_size=4):
        output_tensor = []
        for batch_id in range(0, input_tensor.shape[0], batch_size):
            batch_id_ = min(batch_id + batch_size, input_tensor.shape[0])
            batch_input_tensor = input_tensor[batch_id: batch_id_]
            batch_input_tensor = batch_input_tensor.to(device=self.device, dtype=self.torch_dtype)
            flow, mask, merged = self.model(batch_input_tensor, [4/scale, 2/scale, 1/scale])
            output_tensor.append(merged[2].cpu())
        output_tensor = torch.concat(output_tensor, dim=0)
        return output_tensor

    @torch.no_grad()
    def __call__(self, rendered_frames, scale=1.0, batch_size=4, num_iter=1, **kwargs):
        # Preprocess
        processed_images = self.process_images(rendered_frames)

        for iter in range(num_iter):
            # Input
            input_tensor = torch.cat((processed_images[:-2], processed_images[2:]), dim=1)

            # Interpolate
            output_tensor = self.process_tensors(input_tensor, scale=scale, batch_size=batch_size)
            
            # Blend
            input_tensor = torch.cat((processed_images[1:-1], output_tensor), dim=1)
            output_tensor = self.process_tensors(input_tensor, scale=scale, batch_size=batch_size)

            # Add to frames
            processed_images[1:-1] = output_tensor

        # To images
        output_images = self.decode_images(processed_images)
        if output_images[0].size != rendered_frames[0].size:
            output_images = [image.resize(rendered_frames[0].size) for image in output_images]
        return output_images
