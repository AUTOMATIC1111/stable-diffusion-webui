#!/bin/env python
"""
use microsoft promptist to beautify prompt
- <https://huggingface.co/spaces/microsoft/Promptist>
"""

import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
from util import log


def load_prompter():
    model = AutoModelForCausalLM.from_pretrained("microsoft/Promptist") # pylint: disable=redefined-outer-name
    tokenizer = AutoTokenizer.from_pretrained("gpt2") # pylint: disable=redefined-outer-name
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return model, tokenizer


model, tokenizer = load_prompter()


def beautify(plain_text):
    input_ids = tokenizer(plain_text.strip() + " Rephrase:", return_tensors = "pt").input_ids
    eos_id = tokenizer.eos_token_id
    outputs = model.generate(input_ids, do_sample = False, max_new_tokens = 75, num_beams = 8, num_return_sequences = 8, eos_token_id = eos_id, pad_token_id = eos_id, length_penalty = -1.0)
    output_texts = tokenizer.batch_decode(outputs, skip_special_tokens = True)
    texts = []
    for output_text in output_texts:
        texts.append(output_text.replace(plain_text+" Rephrase:", "").strip())
    longest = max(texts, key = len)
    log.info({ 'beautified': longest })
    return longest

if __name__ == "__main__": # create & train test embedding when used from cli
    sys.argv.pop(0)
    text = ' '.join(sys.argv)
    log.info({ 'prompt': text })
    output = beautify(text)
