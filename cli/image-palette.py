#!/usr/bin/env python
# based on <https://towardsdatascience.com/image-color-extraction-with-python-in-4-steps-8d9370d9216e>

import os
import io
import pathlib
import argparse
import importlib
import pandas as pd
import numpy as np
import extcolors
import filetype
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from colormap import rgb2hex
from PIL import Image
from util import log
grid = importlib.import_module('image-grid').grid

def color_to_df(param):
    colors_pre_list = str(param).replace('([(','').split(', (')[0:-1]
    df_rgb = [i.split('), ')[0] + ')' for i in colors_pre_list]
    df_percent = [i.split('), ')[1].replace(')','') for i in colors_pre_list]
    #convert RGB to HEX code
    df_color_up = [rgb2hex(int(i.split(", ")[0].replace("(","")),
                           int(i.split(", ")[1]),
                           int(i.split(", ")[2].replace(")",""))) for i in df_rgb]
    df = pd.DataFrame(zip(df_color_up, df_percent), columns = ['c_code','occurence'])
    return df


def palette(img, params, output):
    size = 1024
    img.thumbnail((size, size), Image.HAMMING)

    #crate dataframe
    colors_x = extcolors.extract_from_image(img, tolerance = params.color, limit = 13)
    df_color = color_to_df(colors_x)

    #annotate text
    list_color = list(df_color['c_code'])
    list_precent = [int(i) for i in list(df_color['occurence'])]
    text_c = [c + ' ' + str(round(p * 100 / sum(list_precent), 1)) +'%' for c, p in zip(list_color, list_precent)]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(120,60), dpi=10)
    fig.set_facecolor('black')

    #donut plot
    wedges, _text = ax1.pie(list_precent, labels= text_c, labeldistance= 1.05, colors = list_color, textprops={'fontsize': 100, 'color':'white'})
    plt.setp(wedges, width=0.3)

    #add image in the center of donut plot
    data = np.asarray(img)
    imagebox = OffsetImage(data, zoom=2.5)
    ab = AnnotationBbox(imagebox, (0, 0))
    ax1.add_artist(ab)

    #color palette
    x_posi, y_posi, y_posi2 = 160, -260, -260
    for c in list_color:
        if list_color.index(c) <= 5:
            y_posi += 240
            rect = patches.Rectangle((x_posi, y_posi), 540, 230, facecolor = c)
            ax2.add_patch(rect)
            ax2.text(x = x_posi + 100, y = y_posi + 140, s = c, fontdict={'fontsize': 140}, color = 'white')
        else:
            y_posi2 += 240
            rect = patches.Rectangle((x_posi + 600, y_posi2), 540, 230, facecolor = c)
            ax2.add_artist(rect)
            ax2.text(x = x_posi + 700, y = y_posi2 + 140, s = c, fontdict={'fontsize': 140}, color = 'white')

    # add background to force layout
    fig.set_facecolor('black')
    ax2.axis('off')
    tmp = Image.new('RGB', (2000, 1400), (0, 0, 0))
    plt.imshow(tmp)
    plt.tight_layout(rect = (-0.08, -0.2, 1.18, 1.05))

    # save image
    if output is not None:
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        pltimg = Image.open(buf)
        pltimg = pltimg.convert('RGB')
        pltimg.save(output)
        buf.close()
        log.info({ 'palette created': output })

    plt.close()
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'extract image color palette')
    parser.add_argument('--color', type=int, default=20, help="color tolerance threshdold")
    parser.add_argument('--output', type=str, required=False, default='', help='folder to store images')
    parser.add_argument('--suffix', type=str, required=False, default='pallete', help='add suffix to image name')
    parser.add_argument('--grid', default=False, action='store_true', help = "create grid of images before processing")
    parser.add_argument('input', type=str, nargs='*')
    args = parser.parse_args()
    log.info({ 'palette args': vars(args) })
    if args.output != '':
        pathlib.Path(args.output).mkdir(parents = True, exist_ok = True)
    if not args.grid:
        for arg in args.input:
            if os.path.isfile(arg) and filetype.is_image(arg):
                image = Image.open(arg)
                fn = os.path.join(args.output, pathlib.Path(arg).stem + '-' + args.suffix + '.jpg')
                palette(image, args, fn)
            elif os.path.isdir(arg):
                for root, _dirs, files in os.walk(arg):
                    for f in files:
                        if filetype.is_image(os.path.join(root, f)):
                            image = Image.open(os.path.join(root, f))
                            fn = os.path.join(args.output, pathlib.Path(f).stem + '-' + args.suffix + '.jpg')
                            palette(image, args, fn)
    else:
        images = []
        for arg in args.input:
            if os.path.isfile(arg) and filetype.is_image(arg):
                images.append(Image.open(arg))
            elif os.path.isdir(arg):
                for root, _dirs, files in os.walk(arg):
                    for f in files:
                        if filetype.is_image(os.path.join(root, f)):
                            images.append(Image.open(os.path.join(root, f)))
        image = grid(images)
        fn = os.path.join(args.output, args.suffix + '.jpg')
        palette(image, args, fn)
