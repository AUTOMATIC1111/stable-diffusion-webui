import os
import pandas as pd


def split_weighted_subprompts(text):
    """
    grabs all text up to the first occurrence of ':' 
    uses the grabbed text as a sub-prompt, and takes the value following ':' as weight
    if ':' has no value defined, defaults to 1.0
    repeats until no text remaining
    """
    remaining = len(text)
    prompts = []
    weights = []
    while remaining > 0:
        if ":" in text:
            idx = text.index(":") # first occurrence from start
            # grab up to index as sub-prompt
            prompt = text[:idx]
            remaining -= idx
            # remove from main text
            text = text[idx+1:]
            # find value for weight 
            if " " in text:
                idx = text.index(" ") # first occurence
            else: # no space, read to end
                idx = len(text)
            if idx != 0:
                try:
                    weight = float(text[:idx])
                except: # couldn't treat as float
                    print(f"Warning: '{text[:idx]}' is not a value, are you missing a space?")
                    weight = 1.0
            else: # no value found
                weight = 1.0
            # remove from main text
            remaining -= idx
            text = text[idx+1:]
            # append the sub-prompt and its weight
            prompts.append(prompt)
            weights.append(weight)
        else: # no : found
            if len(text) > 0: # there is still text though
                # take remainder as weight 1
                prompts.append(text)
                weights.append(1.0)
            remaining = 0
    return prompts, weights

def logger(params, log_csv):
    os.makedirs('logs', exist_ok=True)
    cols = [arg for arg, _ in params.items()]
    if not os.path.exists(log_csv):
        df = pd.DataFrame(columns=cols) 
        df.to_csv(log_csv, index=False)

    df = pd.read_csv(log_csv)
    for arg in cols:
        if arg not in df.columns:
            df[arg] = ""
    df.to_csv(log_csv, index = False)

    li = {}
    cols = [col for col in df.columns]
    data = {arg:value for arg, value in params.items()}
    for col in cols:
        if col in data:
            li[col] = data[col]
        else:
            li[col] = ''

    df = pd.DataFrame(li,index = [0])
    df.to_csv(log_csv,index=False, mode='a', header=False)