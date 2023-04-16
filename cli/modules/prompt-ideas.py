#!/bin/env python
"""
generate prompt ideas
model from: <https://huggingface.co/FredZhang7/distilgpt2-stable-diffusion-v2>
"""

import logging
import argparse
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from util import log


tokenizer = None
model = None


def prompt(text: str, temp: float = 0.9, top: int = 8, penalty: float = 1.2, alpha: float = 0.6, num: int = 5, length: int = 80):
    global tokenizer, model # pylint: disable=global-statement
    if tokenizer is None:
        tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    if model is None:
        model = GPT2LMHeadModel.from_pretrained('FredZhang7/distilgpt2-stable-diffusion-v2')
    input_ids = tokenizer(text, return_tensors='pt').input_ids
    output = model.generate(input_ids,
        do_sample = True,
        temperature = temp,
        top_k = top,
        max_length = length,
        num_return_sequences = num,
        repetition_penalty = penalty,
        penalty_alpha = alpha,
        no_repeat_ngram_size = 1,
        early_stopping = True
    )
    outputs = []
    for i in range(len(output)):
        outputs.append(tokenizer.decode(output[i], skip_special_tokens=True))
    return outputs


if __name__ == "__main__": # create & train test embedding when used from cli
    log.info({ 'idea': 'generate prompts' })
    parser = argparse.ArgumentParser(description='idea: generate prompts')
    parser.add_argument("--temp", type = float, default = 0.9, required = False, help = "higher temperature produces more diverse results with a higher risk of less coherent text, default: %(default)s")
    parser.add_argument("--top", type = int, default = 8, required = False, help = "number of tokens to sample from at each step, default: %(default)s")
    parser.add_argument("--penalty", type = float, default = 1.2, required = False, help = "penalty value for each repetition of a token, default: %(default)s")
    parser.add_argument("--alpha", type = float, default = 0.6, required = False, help = "penalty alpha value, default: %(default)s")
    parser.add_argument("--num", type = int, default = 10, required = False, help = "number of results to generate, default: %(default)s")
    parser.add_argument("--length", type = int, default = 85, required = False, help = "maximum number of output tokens, default: %(default)s")
    parser.add_argument('--debug', default = False, action='store_true', help = "print extra debug information, default: %(default)s")
    parser.add_argument('text', type = str, nargs = '*')
    params = parser.parse_args()
    if params.debug:
        log.setLevel(logging.DEBUG)
        log.debug({ 'debug': True })
    log.debug({ 'args': params.__dict__ })
    sentence = ' '.join(params.text)
    res = prompt(text = sentence, temp = params.temp, top = params.top, penalty = params.penalty, alpha = params.alpha, num = params.num, length = params.length)
    log.info({ 'ideas for':  sentence })
    for line in res:
        log.info(line)
