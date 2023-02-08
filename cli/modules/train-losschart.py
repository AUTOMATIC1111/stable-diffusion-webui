#!/bin/env python

import io
import os
import sys
import json
import pathlib
import logging
import torch
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from matplotlib import pyplot as plt
from util import log, Map


def settings(logdir: str, name: str):
    filename = os.path.join(logdir, name, 'settings.json')
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
    img = os.path.join(logdir, f"{name}.train.png")

    step, loss, rate = plt.np.loadtxt(f, delimiter = ',', skiprows = 1, usecols = [0, 3, 4], unpack = True)
    d = settings(logdir, name)
    # window = d.get('gradient_step', 1) * d.get('batch_size', 1)
    window = d.get('save_embedding_every', 1)
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
    plt.rcParams['figure.figsize'] = [14, 14]
    plt.rcParams['figure.facecolor'] = 'black'
    figure, axis = plt.subplots(2, 1)

    # create top graph
    ax0 = axis[0]
    ax0.set_facecolor(color = (0.1, 0.1, 0.1, 0.5))
    ax0.tick_params(axis='x', labelcolor='white')
    ax0.set_xlabel('step'.upper(), color='white')
    ax0.set_xlim(0, d.steps)
    ax0.set_ylim(0, 0.5)
    ax0.set_axisbelow(True)
    ax0.xaxis.grid(color='gray', linestyle='dashed')
    ax0.yaxis.grid(color='gray', linestyle='dashed')

    # loss values
    ax0.plot(step, loss, color='gray', label='loss value')
    ax0.set_ylabel('loss value'.upper(), color='#CE6400')
    ax0.tick_params(axis='y', labelcolor='#CE6400')

    # trendline
    z = np.polyfit(step, loss, 1)
    p = np.poly1d(z)
    ax0.plot(step, p(loss), color='#5020F0', linewidth=3, label='loss trendline')

    # moving average
    if len(loss) > window:
        maval = []
        minval = []
        maxval = []
        for ind in range(window - 1):
            maval.insert(0, np.nan)
            minval.insert(0, np.nan)
            maxval.insert(0, np.nan)
        for ind in range(len(loss) - window + 1):
            maval.append(np.mean(loss[ind:ind+window]))
            minval.append(np.min(loss[ind:ind+window]))
            maxval.append(np.max(loss[ind:ind+window]))
        ax0.plot(step, maval, color='#CE6400', linewidth=5, label='average loss value')
        ax0.plot(step, minval, color='#500010', linewidth=5, label='min loss per epoch')
        ax0.plot(step, maxval, color='#005010', linewidth=5, label='max loss per epoch')

    # learning rate
    ax0_right = ax0.twinx()
    ax0_right.set_ylabel('learn rate'.upper(), color='cyan')
    ax0_right.plot(step, rate, color='cyan', linewidth=3, linestyle='dashed', label='learn rate')
    ax0_right.tick_params(axis='y', labelcolor='cyan')

    # axis legend
    handles0, labels0 = ax0.get_legend_handles_labels() # because ax2 is twin, both are included
    ax0.legend(handles0, labels0, loc="best")

    # embeddings
    ax1 = axis[1]
    ax1.set_facecolor(color = (0.1, 0.1, 0.1, 0.5))
    ax1.set_ylabel('vector average'.upper(), color=(1, 0.2, 0.5))
    ax1.tick_params(axis='y', labelcolor=(1, 0.2, 0.5))
    ax1.set_xlim(0, d.steps)
    ax1_right = ax1.twinx()
    ax1_right.set_ylabel('vector norm'.upper(), color=(0.2, 1.0, 0.5))
    ax1_right.tick_params(axis='y', labelcolor=(0.2, 1.0, 0.5))
    ax1.xaxis.grid(color='gray', linestyle='dashed')
    ax1.yaxis.grid(color='gray', linestyle='dashed')
    x = []
    avg = []
    norm = []
    avg_v = [[] for y in range(d.num_vectors_per_token)]
    norm_v = [[] for y in range(d.num_vectors_per_token)]
    embedding_files = sorted(pathlib.Path(os.path.join(logdir, name, 'embeddings')).glob('*.pt'), key=os.path.getmtime)
    for f in embedding_files:
        embed = torch.load(f, map_location=torch.device("cpu")) # pylint: disable=no-member
        x.append(embed["step"] + 1)
        token = list(embed["string_to_token"].keys())[0]
        tensors = embed["string_to_param"][token]
        val = tensors.detach().numpy()
        data = val.flatten()
        avg.append(np.average(np.abs(data)))
        norm.append(np.linalg.norm(data))
        for i in range(val.shape[0]):
            avg_v[i].append(np.average(np.abs(val[i])))
            norm_v[i].append(np.linalg.norm(val[i]))
    ax1.plot(x, avg, color=(1, 0.2, 0.5), linewidth=3, label='all vectors average value')
    ax1_right.plot(x, norm, color=(0.2, 1, 0.5), linewidth=3, label='all vectors norm value', linestyle='dashed')
    for i in range(d.num_vectors_per_token):
        ax1.plot(x, avg_v[i], color= (i / (d.num_vectors_per_token + 1), 0.2, 0.5), linewidth=1, label=f'vector={i} average value')
        ax1_right.plot(x, norm_v[i], color= (0.2, i / (d.num_vectors_per_token + 1), 0.5), linewidth=1, label=f'vector={i} norm value', linestyle='dashed')

    # axis legend
    handles1, labels1 = ax1.get_legend_handles_labels() # because ax2 is twin, both are included
    ax1.legend(handles1, labels1, loc="upper left")
    ax1_right.legend(loc="upper right")

    # create chart and convert to pil
    figure.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    pltimg = Image.open(buf)
    size = (pltimg.size[0], pltimg.size[1] + 240)
    image = Image.new('RGB', size = size, color = (206, 100, 0))
    font = ImageFont.truetype('DejaVuSansMono', 18)
    image.paste(pltimg, box=(0, 240))
    buf.close()
    plt.close()

    # text
    textl = f"""NAME: {d.embedding_name.upper()}
IMAGES: {d.num_of_dataset_images}
VECTORS: {d.num_vectors_per_token}
STEPS: {d.steps}
BATCH-SIZE: {d.batch_size}
GRADIENT-STEP: {d.gradient_step}
SAMPLING-METHOD: {d.latent_sampling_method}
MODEL: {d.model_name.upper()}
LEARN-RATE: {d.learn_rate.replace(' ', '')}
"""

    minval = f"{round(np.min(loss), 4)} @ {round(step[np.argmin(loss)])}"
    maxval = f"{round(np.max(loss), 4)} @ {round(step[np.argmax(loss)])}"
    textr = f"""{d.datetime}
LOSS: {round(loss[-1], 4)}
MIN: {minval}
MAX: {maxval}
TREND: {z[0]:.5f}
"""
    if len(avg) > 0:
        textr += f"VECTOR AVG: {avg[-1]:.3f}\n"
    if len(norm) > 0:
        textr += f"VECTOR NORM: {norm[-1]:.3f}\n"
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
        log.error({ 'loss chart': 'specify embedding name'})
