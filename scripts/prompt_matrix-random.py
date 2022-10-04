import math
from collections import namedtuple
from copy import copy
import random

import modules.scripts as scripts
import gradio as gr

from modules import images
from modules.processing import process_images, Processed
from modules.shared import opts, cmd_opts, state
import modules.sd_samplers
import re
import os
import sys
import traceback
from PIL import Image
import modules.shared as shared
import glob


def replace_wildcard(chunk):
    if " " not in chunk:
        file_dir = os.path.dirname(os.path.realpath("__file__"))
        replacement_file = os.path.join(file_dir, f"cfg/promptgen/{chunk}.csv")
        if os.path.exists(replacement_file):
            with open(replacement_file, "r", encoding="utf8", newline='') as f:
                lines = f.readlines()
                stripped = []
                for line in lines:
                    stripped.append(line.strip())                        
                stripped.remove('name,blank,blank2')
                #print(stripped)
                return random.choice(stripped).replace(",,","")
    return chunk

def oxlamon_matrix(prompt, seed, n_iter, batch_size):
    pattern = re.compile(r'(,\s){2,}')

    class PromptItem:
        def __init__(self, text, parts, item):
            self.text = text
            self.parts = parts
            if item:
                self.parts.append( item )

    def clean(txt):
        return re.sub(pattern, ', ', txt)

    def getrowcount( txt ):
        for data in re.finditer( ".*?\\((.*?)\\).*", txt ):
            if data:
                return len(data.group(1).split("|"))
            break
        return None

    def repliter( txt ):
        for data in re.finditer( ".*?\\((.*?)\\).*", txt ):
            if data:
                r = data.span(1)
                for item in data.group(1).split("|"):
                    yield (clean(txt[:r[0]-1] + item.strip() + txt[r[1]+1:]), item.strip())
            break

    def iterlist( items ):
        outitems = []
        for item in items:
            for newitem, newpart in repliter(item.text):
                outitems.append( PromptItem(newitem, item.parts.copy(), newpart) )

        return outitems

    def getmatrix( prompt ):
        dataitems = [ PromptItem( prompt[1:].strip(), [], None ) ]
        while True:
            newdataitems = iterlist( dataitems )
            if len( newdataitems ) == 0:
                return dataitems
            dataitems = newdataitems
    def classToArrays( items, seed, n_iter ):
        texts = []
        parts = []
        seeds = []

        for item in items:
            itemseed = seed
            for i in range(n_iter):
                replace_prompt = item.text
                texts.append( item.text )
                parts.append( f"Seed: {itemseed}\n" + "\n".join(item.parts) )
                seeds.append( itemseed )
                itemseed += 1
        return seeds, texts, parts
    all_seeds, all_prompts, prompt_matrix_parts = classToArrays(getmatrix( prompt ), seed, n_iter)
    n_iter = math.ceil(len(all_prompts) / batch_size)

    needrows = getrowcount(prompt)
    if needrows:
        xrows = math.sqrt(len(all_prompts))
        xrows = round(xrows)
        # if columns is to much
        cols = math.ceil(len(all_prompts) / xrows)
        if cols > needrows*4:
            needrows *= 2

    return all_seeds, n_iter, prompt_matrix_parts, all_prompts, needrows

class Script(scripts.Script):
    def title(self):
        return "Prompt Matrix V + Random"

    def ui(self, is_img2img):
        same_seed = gr.Checkbox(label='Use same seed (off = random)', value=False)

        return [same_seed]

    def run(self, p, same_seed):
        modules.processing.fix_seed(p)
        original_prompt = p.prompt[0] if type(p.prompt) == list else p.prompt

        all_prompts = []
        prompt_matrix_parts = []
        simple_templating = False
        add_original_image = True
        if p.prompt.startswith("@"):
            simple_templating = True
            all_seeds, n_iter, prompt_matrix_parts, all_prompts, frows = oxlamon_matrix(p.prompt, p.seed, p.n_iter, p.batch_size)
        else:
            all_prompts = []
            prompt_matrix_parts = p.prompt.split("|")
            combination_count = 2 ** (len(prompt_matrix_parts) - 1)
            for combination_num in range(combination_count):
                current = prompt_matrix_parts[0]
                for n, text in enumerate(prompt_matrix_parts[1:]):
                    if combination_num & (2 ** n) > 0:
                        current += ("" if text.strip().startswith(",") else ", ") + text
                all_prompts.append(current)
            all_seeds = len(all_prompts) * [p.seed]

        p.n_iter = math.ceil(len(all_prompts) / p.batch_size)
        p.do_not_save_grid = True

        print(f"Prompt matrix will create {len(all_prompts)} images using a total of {n_iter} batches.")
        new_prompt = []
        for string in all_prompts:
            replace_prompt = "".join(replace_wildcard(chunk) for chunk in str(string).split("__"))
            new_prompt.append(replace_prompt)

        #p.prompt = all_prompts
        p.prompt = new_prompt
        if same_seed:
            p.seed = [p.seed for _ in all_prompts]
        else:
            p.seed = [int(random.randrange(4294967294)) for _ in all_prompts]
        p.prompt_for_display = original_prompt
        processed = process_images(p)

        return processed

