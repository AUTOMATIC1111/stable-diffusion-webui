#!/bin/env python
# based on <https://towardsdatascience.com/image-color-extraction-with-python-in-4-steps-8d9370d9216e>

import os
import sys
import pandas as pd
import numpy as np
import extcolors
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from colormap import rgb2hex
from PIL import Image


def color_to_df(input):
    colors_pre_list = str(input).replace('([(','').split(', (')[0:-1]
    df_rgb = [i.split('), ')[0] + ')' for i in colors_pre_list]
    df_percent = [i.split('), ')[1].replace(')','') for i in colors_pre_list]  
    #convert RGB to HEX code
    df_color_up = [rgb2hex(int(i.split(", ")[0].replace("(","")),
                           int(i.split(", ")[1]),
                           int(i.split(", ")[2].replace(")",""))) for i in df_rgb]
    df = pd.DataFrame(zip(df_color_up, df_percent), columns = ['c_code','occurence'])
    return df


def color_wheel(input_image, resize, tolerance, zoom):   
    #resize
    img = Image.open(input_image)
    if img.size[0] >= resize:
        wpercent = (resize / float(img.size[0]))
        hsize = int((float(img.size[1]) * float(wpercent)))
        img = img.resize((resize, hsize))
    
    #crate dataframe
    colors_x = extcolors.extract_from_image(img, tolerance = tolerance, limit = 13)
    df_color = color_to_df(colors_x)
    
    #annotate text
    list_color = list(df_color['c_code'])
    list_precent = [int(i) for i in list(df_color['occurence'])]
    text_c = [c + ' ' + str(round(p * 100 / sum(list_precent), 1)) +'%' for c, p in zip(list_color, list_precent)]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(120,60), dpi=10)
    
    #donut plot
    wedges, _text = ax1.pie(list_precent, labels= text_c, labeldistance= 1.05, colors = list_color, textprops={'fontsize': 140, 'color':'black'})
    plt.setp(wedges, width=0.3)

    #add image in the center of donut plot
    data = np.asarray(img)
    imagebox = OffsetImage(data, zoom=zoom)
    ab = AnnotationBbox(imagebox, (0, 0))
    ax1.add_artist(ab)
    
    #color palette
    x_posi, y_posi, y_posi2 = 160, -200, -200
    for c in list_color:
        if list_color.index(c) <= 5:
            y_posi += 220
            rect = patches.Rectangle((x_posi, y_posi), 480, 200, facecolor = c)
            ax2.add_patch(rect)
            ax2.text(x = x_posi + 40, y = y_posi + 120, s = c, fontdict={'fontsize': 140})
        else:
            y_posi2 += 220
            rect = patches.Rectangle((x_posi + 600, y_posi2), 480, 200, facecolor = c)
            ax2.add_artist(rect)
            ax2.text(x = x_posi + 640, y = y_posi2 + 120, s = c, fontdict={'fontsize': 140})

    #background
    tmp_file = 'tmp.png'
    fig, _ax = plt.subplots(figsize=(200,140),dpi=10)
    fig.set_facecolor('white')
    plt.savefig(tmp_file)
    plt.close(fig)

    fig.set_facecolor('white')
    ax2.axis('off')
    tmp = plt.imread(tmp_file)
    plt.imshow(tmp)
    plt.tight_layout()
    plt.savefig('palette.jpg')
    plt.close()
    os.remove(tmp_file)
    return


if __name__ == '__main__':
    sys.argv.pop(0)
    for arg in sys.argv:
        color_wheel(arg, 512, 10, 2)
