#!/bin/env python
"""
auto-generate learn-rate
"""
import io
import math
import logging
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from matplotlib import pyplot as plt
from util import log, Map


loss_types = ['linear', 'log', 'linalg', 'power']


def gen_steps(steps, step):
    return [x for x in range(1, steps + step) if x % step == 0]


def gen_loss_rate(steps: int, step: int, loss_start: float, loss_end: float, loss_type: loss_types, power: int = 3):
    def norm(val):
        return ((loss_start - loss_end) * val) / val.max() + loss_end

    steps_val = gen_steps(steps, step)

    if loss_type == 'linear':
        loss_val = np.interp(steps_val, [steps_val[0], steps_val[-1]], [loss_start, loss_end])

    elif loss_type == 'log':
        loss_val = np.logspace(loss_start, 0, num=len(steps_val), base=math.e)
        loss_val = norm(loss_val - loss_val.min())

    elif loss_type == 'linalg':
        loss_val = np.array(steps_val[::-1], dtype='float')
        loss_val = norm(loss_val / np.linalg.norm(loss_val))

    elif loss_type == 'power':
        loss_val = np.array([math.pow(x, power) for x in range(len(steps_val))][::-1])
        loss_val = norm(loss_val)

    else:
        return []

    return loss_val


def gen_loss_rate_str(steps: int, step: int, loss_start: float, loss_end: float, loss_type: loss_types, power: int = 3):
    steps_val = gen_steps(steps, step)
    loss_val = gen_loss_rate(steps, step, loss_start, loss_end, loss_type, power)
    loss_rate = [f"{loss_val[i]:.4f}:{steps_val[i]}" for i in range(len(steps_val))]
    loss_rate = ', '.join(loss_rate)
    log.debug({ 'loss_rate': loss_rate, 'function': loss_type, 'power': power })
    return loss_rate


def example_plot(steps: int, step: int, loss_start: float, loss_end: float):
    plt.rcParams.update({'font.variant':'small-caps'})
    plt.rc('axes', edgecolor='gray')
    plt.rc('font', size=10)
    plt.rc('font', variant='small-caps')
    plt.grid(color='gray', linewidth=1, axis='both', alpha=0.5)
    plt.rcParams['figure.figsize'] = [14, 6]
    plt.figure(facecolor='black')

    loss_rates = []

    ax1 = plt.subplot(1, 2, 1)
    ax1.set_facecolor('grey')
    ax1.set_xlabel('step'.upper(), color='white')
    ax1.set_ylabel('loss value'.upper(), color='white')
    ax1.xaxis.grid(color='gray', linestyle='dashed')
    ax1.tick_params(axis='x', labelcolor='white')
    ax1.tick_params(axis='y', labelcolor='white')
    ax1.legend(loc="best")
    for loss_type in [x for x in loss_types if x != 'power']:
        col = (np.random.random(), np.random.random(), np.random.random())
        x = gen_steps(steps, step)
        y = gen_loss_rate(loss_type = loss_type, steps = steps, step = step, loss_start = loss_start, loss_end = loss_end)
        ax1.plot(x, y, label=loss_type, color = col)
        loss = gen_loss_rate_str(loss_type = loss_type, steps = steps, step = step, loss_start = loss_start, loss_end = loss_end)
        loss_rates.append(f"LOSS {loss} TYPE {loss_type}")
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels)

    ax2 = plt.subplot(1, 2, 2)
    ax2.set_facecolor('grey')
    ax2.set_xlabel('step'.upper(), color='white')
    ax2.set_ylabel('loss value'.upper(), color='white')
    ax2.xaxis.grid(color='gray', linestyle='dashed')
    ax2.tick_params(axis='x', labelcolor='white')
    ax2.tick_params(axis='y', labelcolor='white')
    ax2.legend(loc="best")
    for power in range(1, 11):
        col = (np.random.random(), np.random.random(), np.random.random())
        x = gen_steps(steps, step)
        y = gen_loss_rate(loss_type = 'power', power = power, steps = steps, step = step, loss_start = loss_start, loss_end = loss_end)
        ax2.plot(x, y, label=f"power={pow}", color = col)
        loss = gen_loss_rate_str(loss_type = 'power', power = power, steps = steps, step = step, loss_start = loss_start, loss_end = loss_end)
        loss_rates.append(f"LOSS {loss} TYPE power={power}")
    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(handles, labels)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    pltimg = Image.open(buf)
    size = (pltimg.size[0], pltimg.size[1] + 300)
    image = Image.new('RGB', size = size, color = (206, 100, 0))
    font = ImageFont.truetype('DejaVuSansMono', 14)
    image.paste(pltimg, box=(0, 300))
    buf.close()

    # text
    rates = "\n".join(loss_rates)
    text = f"STEPS {steps} STEP {step} LOSS-START {loss_start} LOSS-END {loss_end}\n" + rates

    ctx = ImageDraw.Draw(image)
    ctx.text((8, 8), text, font = font, fill = (255, 255, 255), spacing = 8)
    image.save('lossrate.jpg')


if __name__ == "__main__":
    log.setLevel(logging.DEBUG)
    arg = Map({
        "steps": 500,
        "step": 50,
        "loss_start": 0.01,
        "loss_end": 0.001
    })
    log.debug({ 'options': arg })
    example_plot(**arg)
