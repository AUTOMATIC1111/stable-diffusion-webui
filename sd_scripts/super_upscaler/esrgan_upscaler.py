import os,sys
from PIL import Image
import PIL
import sd_scripts.super_upscaler.arch as arch
import torch
import numpy as np
from collections import namedtuple
import math



LANCZOS = (Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS)

Grid = namedtuple("Grid", ["tiles", "tile_w", "tile_h", "image_w", "image_h", "overlap"])


def combine_grid(grid):
    def make_mask_image(r):
        r = r * 255 / grid.overlap
        r = r.astype(np.uint8)
        return Image.fromarray(r, 'L')

    mask_w = make_mask_image(
        np.arange(grid.overlap, dtype=np.float32).reshape((1, grid.overlap)).repeat(grid.tile_h, axis=0))
    mask_h = make_mask_image(
        np.arange(grid.overlap, dtype=np.float32).reshape((grid.overlap, 1)).repeat(grid.image_w, axis=1))

    combined_image = Image.new("RGB", (grid.image_w, grid.image_h))
    for y, h, row in grid.tiles:
        combined_row = Image.new("RGB", (grid.image_w, h))
        for x, w, tile in row:
            if x == 0:
                combined_row.paste(tile, (0, 0))
                continue

            combined_row.paste(tile.crop((0, 0, grid.overlap, h)), (x, 0), mask=mask_w)
            combined_row.paste(tile.crop((grid.overlap, 0, w, h)), (x + grid.overlap, 0))

        if y == 0:
            combined_image.paste(combined_row, (0, 0))
            continue

        combined_image.paste(combined_row.crop((0, 0, combined_row.width, grid.overlap)), (0, y), mask=mask_h)
        combined_image.paste(combined_row.crop((0, grid.overlap, combined_row.width, h)), (0, y + grid.overlap))

    return combined_image


def split_grid(image, tile_w=512, tile_h=512, overlap=64):
    w = image.width
    h = image.height

    non_overlap_width = tile_w - overlap
    non_overlap_height = tile_h - overlap

    cols = math.ceil((w - overlap) / non_overlap_width)
    rows = math.ceil((h - overlap) / non_overlap_height)

    dx = (w - tile_w) / (cols - 1) if cols > 1 else 0
    dy = (h - tile_h) / (rows - 1) if rows > 1 else 0

    grid = Grid([], tile_w, tile_h, w, h, overlap)
    for row in range(rows):
        row_images = []

        y = int(row * dy)

        if y + tile_h >= h:
            y = h - tile_h

        for col in range(cols):
            x = int(col * dx)

            if x + tile_w >= w:
                x = w - tile_w

            tile = image.crop((x, y, x + tile_w, y + tile_h))

            row_images.append([x, tile_w, tile])

        grid.tiles.append([y, tile_h, row_images])

    return grid


def upscale_without_tiling(model, img):
    img = np.array(img)
    img = img[:, :, ::-1]
    img = np.ascontiguousarray(np.transpose(img, (2, 0, 1))) / 255
    img = torch.from_numpy(img).float()
    img = img.unsqueeze(0).to('cuda')
    with torch.no_grad():
        output = model(img)
    output = output.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = 255. * np.moveaxis(output, 0, 2)
    output = output.astype(np.uint8)
    output = output[:, :, ::-1]
    return Image.fromarray(output, 'RGB')


def mod2normal(state_dict):
    # this code is copied from https://github.com/victorca25/iNNfer
    if 'conv_first.weight' in state_dict:
        crt_net = {}
        items = []
        for k, v in state_dict.items():
            items.append(k)

        crt_net['model.0.weight'] = state_dict['conv_first.weight']
        crt_net['model.0.bias'] = state_dict['conv_first.bias']

        for k in items.copy():
            if 'RDB' in k:
                ori_k = k.replace('RRDB_trunk.', 'model.1.sub.')
                if '.weight' in k:
                    ori_k = ori_k.replace('.weight', '.0.weight')
                elif '.bias' in k:
                    ori_k = ori_k.replace('.bias', '.0.bias')
                crt_net[ori_k] = state_dict[k]
                items.remove(k)
        crt_net['model.1.sub.23.weight'] = state_dict['trunk_conv.weight']
        crt_net['model.1.sub.23.bias'] = state_dict['trunk_conv.bias']
        crt_net['model.3.weight'] = state_dict['upconv1.weight']
        crt_net['model.3.bias'] = state_dict['upconv1.bias']
        crt_net['model.6.weight'] = state_dict['upconv2.weight']
        crt_net['model.6.bias'] = state_dict['upconv2.bias']
        crt_net['model.8.weight'] = state_dict['HRconv.weight']
        crt_net['model.8.bias'] = state_dict['HRconv.bias']
        crt_net['model.10.weight'] = state_dict['conv_last.weight']
        crt_net['model.10.bias'] = state_dict['conv_last.bias']
        state_dict = crt_net
    return state_dict


def resrgan2normal(state_dict, nb=23):
    # this code is copied from https://github.com/victorca25/iNNfer
    if "conv_first.weight" in state_dict and "body.0.rdb1.conv1.weight" in state_dict:
        re8x = 0
        crt_net = {}
        items = []
        for k, v in state_dict.items():
            items.append(k)

        crt_net['model.0.weight'] = state_dict['conv_first.weight']
        crt_net['model.0.bias'] = state_dict['conv_first.bias']

        for k in items.copy():
            if "rdb" in k:
                ori_k = k.replace('body.', 'model.1.sub.')
                ori_k = ori_k.replace('.rdb', '.RDB')
                if '.weight' in k:
                    ori_k = ori_k.replace('.weight', '.0.weight')
                elif '.bias' in k:
                    ori_k = ori_k.replace('.bias', '.0.bias')
                crt_net[ori_k] = state_dict[k]
                items.remove(k)

        crt_net[f'model.1.sub.{nb}.weight'] = state_dict['conv_body.weight']
        crt_net[f'model.1.sub.{nb}.bias'] = state_dict['conv_body.bias']
        crt_net['model.3.weight'] = state_dict['conv_up1.weight']
        crt_net['model.3.bias'] = state_dict['conv_up1.bias']
        crt_net['model.6.weight'] = state_dict['conv_up2.weight']
        crt_net['model.6.bias'] = state_dict['conv_up2.bias']

        if 'conv_up3.weight' in state_dict:
            # modification supporting: https://github.com/ai-forever/Real-ESRGAN/blob/main/RealESRGAN/rrdbnet_arch.py
            re8x = 3
            crt_net['model.9.weight'] = state_dict['conv_up3.weight']
            crt_net['model.9.bias'] = state_dict['conv_up3.bias']

        crt_net[f'model.{8 + re8x}.weight'] = state_dict['conv_hr.weight']
        crt_net[f'model.{8 + re8x}.bias'] = state_dict['conv_hr.bias']
        crt_net[f'model.{10 + re8x}.weight'] = state_dict['conv_last.weight']
        crt_net[f'model.{10 + re8x}.bias'] = state_dict['conv_last.bias']

        state_dict = crt_net
    return state_dict


def infer_params(state_dict):
    # this code is copied from https://github.com/victorca25/iNNfer
    scale2x = 0
    scalemin = 6
    n_uplayer = 0
    plus = False

    for block in list(state_dict):
        parts = block.split(".")
        n_parts = len(parts)
        if n_parts == 5 and parts[2] == "sub":
            nb = int(parts[3])
        elif n_parts == 3:
            part_num = int(parts[1])
            if (part_num > scalemin
                    and parts[0] == "model"
                    and parts[2] == "weight"):
                scale2x += 1
            if part_num > n_uplayer:
                n_uplayer = part_num
                out_nc = state_dict[block].shape[0]
        if not plus and "conv1x1" in block:
            plus = True

    nf = state_dict["model.0.weight"].shape[0]
    in_nc = state_dict["model.0.weight"].shape[1]
    out_nc = out_nc
    scale = 2 ** scale2x

    return in_nc, out_nc, nf, nb, plus, scale


def esrgan_upscale(model, img):
    ESRGAN_tile, ESRGAN_tile, ESRGAN_tile_overlap = 192, 192, 8
    if ESRGAN_tile == 0:
        return upscale_without_tiling(model, img)
    grid = split_grid(img, ESRGAN_tile, ESRGAN_tile, ESRGAN_tile_overlap)
    newtiles = []
    scale_factor = 1
    for y, h, row in grid.tiles:
        newrow = []
        for tiledata in row:
            x, w, tile = tiledata
            output = upscale_without_tiling(model, tile)
            scale_factor = output.width // tile.width
            newrow.append([x * scale_factor, w * scale_factor, output])
        newtiles.append([y * scale_factor, h * scale_factor, newrow])
    newgrid = Grid(newtiles, grid.tile_w * scale_factor, grid.tile_h * scale_factor, grid.image_w * scale_factor,
                   grid.image_h * scale_factor, grid.overlap * scale_factor)
    output = combine_grid(newgrid)
    return output


def load_model(path: str):
    filename = path
    if not os.path.exists(filename) or filename is None:
        print("模型不存在")
        return None
    state_dict = torch.load(filename, map_location=None)
    if "params_ema" in state_dict:
        state_dict = state_dict["params_ema"]
    elif "params" in state_dict:
        state_dict = state_dict["params"]
        num_conv = 16 if "realesr-animevideov3" in filename else 32
        model = arch.SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=num_conv, upscale=4,
                                     act_type='prelu')
        model.load_state_dict(state_dict)
        model.eval()
        return model

    if "body.0.rdb1.conv1.weight" in state_dict and "conv_first.weight" in state_dict:
        nb = 6 if "RealESRGAN_x4plus_anime_6B" in filename else 23
        state_dict = resrgan2normal(state_dict, nb)
    elif "conv_first.weight" in state_dict:
        state_dict = mod2normal(state_dict)
    elif "model.0.weight" not in state_dict:
        raise Exception("The file is not a recognized ESRGAN model.")

    in_nc, out_nc, nf, nb, plus, mscale = infer_params(state_dict)

    model = arch.RRDBNet(in_nc=in_nc, out_nc=out_nc, nf=nf, nb=nb, upscale=mscale, plus=plus)
    model.load_state_dict(state_dict)
    model.eval()

    return model


def do_upscale(img, selected_model):
    model = load_model(selected_model)
    if model is None:
        return img
    model.to('cuda:0')
    img = esrgan_upscale(model, img)
    return img


def upscale(img: PIL.Image, scale, selected_model: str = None):
    dest_w = int(img.width * scale)
    dest_h = int(img.height * scale)
    for i in range(3):
        shape = (img.width, img.height)
        img = do_upscale(img, selected_model)
        if shape == (img.width, img.height):
            break
        if img.width >= dest_w and img.height >= dest_h:
            break
    if img.width != dest_w or img.height != dest_h:
        img = img.resize((int(dest_w), int(dest_h)), resample=LANCZOS)
    return img
