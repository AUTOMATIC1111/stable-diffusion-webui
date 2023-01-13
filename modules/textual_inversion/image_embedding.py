import base64
import json
import numpy as np
import zlib
from PIL import Image, PngImagePlugin, ImageDraw, ImageFont
from fonts.ttf import Roboto
import torch
from modules.shared import opts


class EmbeddingEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return {'TORCHTENSOR': obj.cpu().detach().numpy().tolist()}
        return json.JSONEncoder.default(self, obj)


class EmbeddingDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, d):
        if 'TORCHTENSOR' in d:
            return torch.from_numpy(np.array(d['TORCHTENSOR']))
        return d


def embedding_to_b64(data):
    d = json.dumps(data, cls=EmbeddingEncoder)
    return base64.b64encode(d.encode())


def embedding_from_b64(data):
    d = base64.b64decode(data)
    return json.loads(d, cls=EmbeddingDecoder)


def lcg(m=2**32, a=1664525, c=1013904223, seed=0):
    while True:
        seed = (a * seed + c) % m
        yield seed % 255


def xor_block(block):
    g = lcg()
    randblock = np.array([next(g) for _ in range(np.product(block.shape))]).astype(np.uint8).reshape(block.shape)
    return np.bitwise_xor(block.astype(np.uint8), randblock & 0x0F)


def style_block(block, sequence):
    im = Image.new('RGB', (block.shape[1], block.shape[0]))
    draw = ImageDraw.Draw(im)
    i = 0
    for x in range(-6, im.size[0], 8):
        for yi, y in enumerate(range(-6, im.size[1], 8)):
            offset = 0
            if yi % 2 == 0:
                offset = 4
            shade = sequence[i % len(sequence)]
            i += 1
            draw.ellipse((x+offset, y, x+6+offset, y+6), fill=(shade, shade, shade))

    fg = np.array(im).astype(np.uint8) & 0xF0

    return block ^ fg


def insert_image_data_embed(image, data):
    d = 3
    data_compressed = zlib.compress(json.dumps(data, cls=EmbeddingEncoder).encode(), level=9)
    data_np_ = np.frombuffer(data_compressed, np.uint8).copy()
    data_np_high = data_np_ >> 4
    data_np_low = data_np_ & 0x0F

    h = image.size[1]
    next_size = data_np_low.shape[0] + (h-(data_np_low.shape[0] % h))
    next_size = next_size + ((h*d)-(next_size % (h*d)))

    data_np_low = np.resize(data_np_low, next_size)
    data_np_low = data_np_low.reshape((h, -1, d))

    data_np_high = np.resize(data_np_high, next_size)
    data_np_high = data_np_high.reshape((h, -1, d))

    edge_style = list(data['string_to_param'].values())[0].cpu().detach().numpy().tolist()[0][:1024]
    edge_style = (np.abs(edge_style)/np.max(np.abs(edge_style))*255).astype(np.uint8)

    data_np_low = style_block(data_np_low, sequence=edge_style)
    data_np_low = xor_block(data_np_low)
    data_np_high = style_block(data_np_high, sequence=edge_style[::-1])
    data_np_high = xor_block(data_np_high)

    im_low = Image.fromarray(data_np_low, mode='RGB')
    im_high = Image.fromarray(data_np_high, mode='RGB')

    background = Image.new('RGB', (image.size[0]+im_low.size[0]+im_high.size[0]+2, image.size[1]), (0, 0, 0))
    background.paste(im_low, (0, 0))
    background.paste(image, (im_low.size[0]+1, 0))
    background.paste(im_high, (im_low.size[0]+1+image.size[0]+1, 0))

    return background


def crop_black(img, tol=0):
    mask = (img > tol).all(2)
    mask0, mask1 = mask.any(0), mask.any(1)
    col_start, col_end = mask0.argmax(), mask.shape[1]-mask0[::-1].argmax()
    row_start, row_end = mask1.argmax(), mask.shape[0]-mask1[::-1].argmax()
    return img[row_start:row_end, col_start:col_end]


def extract_image_data_embed(image):
    d = 3
    outarr = crop_black(np.array(image.convert('RGB').getdata()).reshape(image.size[1], image.size[0], d).astype(np.uint8)) & 0x0F
    black_cols = np.where(np.sum(outarr, axis=(0, 2)) == 0)
    if black_cols[0].shape[0] < 2:
        print('No Image data blocks found.')
        return None

    data_block_lower = outarr[:, :black_cols[0].min(), :].astype(np.uint8)
    data_block_upper = outarr[:, black_cols[0].max()+1:, :].astype(np.uint8)

    data_block_lower = xor_block(data_block_lower)
    data_block_upper = xor_block(data_block_upper)

    data_block = (data_block_upper << 4) | (data_block_lower)
    data_block = data_block.flatten().tobytes()

    data = zlib.decompress(data_block)
    return json.loads(data, cls=EmbeddingDecoder)


def caption_image_overlay(srcimage, title, footerLeft, footerMid, footerRight, textfont=None):
    from math import cos

    image = srcimage.copy()
    fontsize = 32
    if textfont is None:
        try:
            textfont = ImageFont.truetype(opts.font or Roboto, fontsize)
            textfont = opts.font or Roboto
        except Exception:
            textfont = Roboto

    factor = 1.5
    gradient = Image.new('RGBA', (1, image.size[1]), color=(0, 0, 0, 0))
    for y in range(image.size[1]):
        mag = 1-cos(y/image.size[1]*factor)
        mag = max(mag, 1-cos((image.size[1]-y)/image.size[1]*factor*1.1))
        gradient.putpixel((0, y), (0, 0, 0, int(mag*255)))
    image = Image.alpha_composite(image.convert('RGBA'), gradient.resize(image.size))

    draw = ImageDraw.Draw(image)

    font = ImageFont.truetype(textfont, fontsize)
    padding = 10

    _, _, w, h = draw.textbbox((0, 0), title, font=font)
    fontsize = min(int(fontsize * (((image.size[0]*0.75)-(padding*4))/w)), 72)
    font = ImageFont.truetype(textfont, fontsize)
    _, _, w, h = draw.textbbox((0, 0), title, font=font)
    draw.text((padding, padding), title, anchor='lt', font=font, fill=(255, 255, 255, 230))

    _, _, w, h = draw.textbbox((0, 0), footerLeft, font=font)
    fontsize_left = min(int(fontsize * (((image.size[0]/3)-(padding))/w)), 72)
    _, _, w, h = draw.textbbox((0, 0), footerMid, font=font)
    fontsize_mid = min(int(fontsize * (((image.size[0]/3)-(padding))/w)), 72)
    _, _, w, h = draw.textbbox((0, 0), footerRight, font=font)
    fontsize_right = min(int(fontsize * (((image.size[0]/3)-(padding))/w)), 72)

    font = ImageFont.truetype(textfont, min(fontsize_left, fontsize_mid, fontsize_right))

    draw.text((padding, image.size[1]-padding),               footerLeft, anchor='ls', font=font, fill=(255, 255, 255, 230))
    draw.text((image.size[0]/2, image.size[1]-padding),       footerMid, anchor='ms', font=font, fill=(255, 255, 255, 230))
    draw.text((image.size[0]-padding, image.size[1]-padding), footerRight, anchor='rs', font=font, fill=(255, 255, 255, 230))

    return image


if __name__ == '__main__':

    testEmbed = Image.open('test_embedding.png')
    data = extract_image_data_embed(testEmbed)
    assert data is not None

    data = embedding_from_b64(testEmbed.text['sd-ti-embedding'])
    assert data is not None

    image = Image.new('RGBA', (512, 512), (255, 255, 200, 255))
    cap_image = caption_image_overlay(image, 'title', 'footerLeft', 'footerMid', 'footerRight')

    test_embed = {'string_to_param': {'*': torch.from_numpy(np.random.random((2, 4096)))}}

    embedded_image = insert_image_data_embed(cap_image, test_embed)

    retrived_embed = extract_image_data_embed(embedded_image)

    assert str(retrived_embed) == str(test_embed)

    embedded_image2 = insert_image_data_embed(cap_image, retrived_embed)

    assert embedded_image == embedded_image2

    g = lcg()
    shared_random = np.array([next(g) for _ in range(100)]).astype(np.uint8).tolist()

    reference_random = [253, 242, 127,  44, 157,  27, 239, 133,  38,  79, 167,   4, 177,
                         95, 130,  79,  78,  14,  52, 215, 220, 194, 126,  28, 240, 179,
                        160, 153, 149,  50, 105,  14,  21, 218, 199,  18,  54, 198, 193,
                         38, 128,  19,  53, 195, 124,  75, 205,  12,   6, 145,   0,  28,
                         30, 148,   8,  45, 218, 171,  55, 249,  97, 166,  12,  35,   0,
                         41, 221, 122, 215, 170,  31, 113, 186,  97, 119,  31,  23, 185,
                         66, 140,  30,  41,  37,  63, 137, 109, 216,  55, 159, 145,  82,
                         204, 86,  73, 222,  44, 198, 118, 240,  97]

    assert shared_random == reference_random

    hunna_kay_random_sum = sum(np.array([next(g) for _ in range(100000)]).astype(np.uint8).tolist())

    assert 12731374 == hunna_kay_random_sum
