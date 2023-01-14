#!/bin/env python

import io
import os
import sys
import json
import pathlib
import logging
import numpy as np
import scipy as sp
from PIL import Image, ImageFont, ImageDraw
from matplotlib import pyplot as plt
from util import log, Map

def settings(logdir: str, name: str):
    filename = os.path.join(logdir, name, 'settings.json')
    # shutil.copyfile(filename, os.path.join(logdir, f"{name}.json"))
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
        data = Map(data)
        log.debug({ 'settings': data })
    return data


def plot(logdir: str, name: str):
    f = os.path.join(logdir, name, 'train.csv')
    if not os.path.isfile(f):
        log.debug({ 'train log missing': f })
        return
    name = pathlib.Path(f).parent.name
    # shutil.copyfile(f, os.path.join(logdir, f"{name}.csv"))
    img = os.path.join(logdir, f"{name}.png")

    step, loss, rate = plt.np.loadtxt(f, delimiter = ',', skiprows = 1, usecols = [0, 3, 4], unpack = True)
    d = settings(logdir, name)
    try:
        log.debug({ 'loss plot': name, 'output': img, 'data': f, 'records': len(step) })
    except:
        return # no data
    if len(step) < 5:
        return

    plt.rcParams.update({'font.variant':'small-caps'})
    plt.rc('axes', edgecolor='gray')
    plt.rc('font', size=10)
    plt.rc('font', variant='small-caps')
    plt.grid(color='gray', linewidth=1, axis='both', alpha=0.5)
    plt.rcParams['figure.figsize'] = [14, 6]
    plt.subplots_adjust(right=1000)

    fig, ax1 = plt.subplots()
    fig.set_facecolor('black')

    ax1.set_facecolor(color = (0.1, 0.1, 0.1, 0.5))
    ax1.tick_params(axis='x', labelcolor='white')
    ax1.set_xlabel('step'.upper(), color='white')
    ax1.set_xlim(0, d.steps)
    ax1.set_ylim(0, 0.5)
    ax1.set_axisbelow(True)
    ax1.xaxis.grid(color='gray', linestyle='dashed')
    ax1.yaxis.grid(color='gray', linestyle='dashed')

    # loss with additional interpolated values to smooth out the curve
    ax1.set_ylabel('loss'.upper(), color='gray')
    ax1.plot(step, loss, 'go')
    spline = sp.interpolate.make_interp_spline(step, loss)
    x_ = np.linspace(min(step), max(step), num = len(step * 3), endpoint = True, retstep = False, dtype = int, axis = 0)
    y_ = spline(x_)
    ax1.plot(x_, y_, color='gray')
    ax1.tick_params(axis='y', labelcolor='gray')

    # moving average
    window = 10
    if len(loss) > window:
        ma = []
        for ind in range(window - 1):
            ma.insert(0, np.nan)
        for ind in range(len(loss) - window + 1):
            ma.append(np.mean(loss[ind:ind+window]))
        ax1.plot(step, ma, color="maroon", linewidth=5)

    # learning rate
    ax2 = ax1.twinx()
    ax2.set_ylabel('learning rate'.upper(), color='cyan')
    ax2.plot(step, rate, color='cyan', linewidth=3, linestyle='dashed')
    ax2.tick_params(axis='y', labelcolor='cyan')

    # create chart and convert to pil
    fig.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    pltimg = Image.open(buf)
    size = (pltimg.size[0], pltimg.size[1] + 240)
    image = Image.new('RGB', size = size, color = (206, 100, 0))
    font = ImageFont.truetype('DejaVuSansMono', 18)
    image.paste(pltimg, box=(0, 240))
    buf.close()

    # text
    textl = f"""NAME: {d.embedding_name.upper()}
IMAGES: {d.num_of_dataset_images}
VECTORS: {d.num_vectors_per_token}
STEPS: {d.steps}
BATCH-SIZE: {d.batch_size}
GRADIENT-STEP: {d.gradient_step}
LEARN-RATE: {d.learn_rate}
SAMPLING-METHOD: {d.latent_sampling_method}
MODEL: {d.model_name.upper()}
"""

    minval = f"{round(np.min(loss), 4)} @ {round(step[np.argmin(loss)])}"
    maxval = f"{round(np.max(loss), 4)} @ {round(step[np.argmax(loss)])}"
    textr = f"""{d.datetime}
LOSS: {round(loss[-1], 4)}
MIN: {minval}
MAX: {maxval}
"""
    ctx = ImageDraw.Draw(image)
    ctx.text((8, 8), textl, font = font, fill = (255, 255, 255), spacing = 8)
    ctx.text((image.size[0] - 220, 8), textr, font = font, fill = (255, 255, 255))

    image.save(img)


if __name__ == "__main__":
    log.setLevel(logging.DEBUG)
    if len(sys.argv) == 2:
        arg = sys.argv[1]
        log.debug({ 'args': arg })
        plot(os.path.dirname(arg), os.path.basename(arg))
    else:
        log.debug({ 'error': 'specify embedding name'})
