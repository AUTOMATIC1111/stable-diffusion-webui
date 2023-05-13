#!/usr/bin/env python
"""
use microsoft promptist to beautify prompt
- <https://huggingface.co/spaces/microsoft/Promptist>
"""

import sys
from util import log

def load_model():
    log.info({ 'loading': 'model' })
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained("microsoft/Promptist") # pylint: disable=redefined-outer-name
    return model

def load_tokenizer():
    log.info({ 'loading': 'tokenizer' })
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2") # pylint: disable=redefined-outer-name
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer

def beautify(plain_text):
    tokenizer = load_tokenizer()
    input_ids = tokenizer(plain_text.strip() + " Rephrase:", return_tensors = "pt").input_ids
    eos_id = tokenizer.eos_token_id
    model = load_model()
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
