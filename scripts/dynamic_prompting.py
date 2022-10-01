import re, random

import math

import modules.scripts as scripts

from modules.processing import process_images, fix_seed
from modules.shared import opts


def pick_variant(template):
    """
    Generate random prompts given a template 
    This function was copied from the following colab, but I think it may have originated somewhere else: https://colab.research.google.com/drive/1P5MEMtLM3RGCqGfSQWs1cMntrMgSnKDe?usp=sharing#scrollTo=PAsdW6XqxVO_

    Template syntax

        Variations
            {opt1|opt2|opt3} : will randomly pick 1 of the options for every batch item.

            In this case, "opt1" or "opt2" or "opt3"

        Combinations
            [2$$opt1|opt2|opt3] : will randomly combine 2 of the options for every batch, separated with a comma

            In this case, "opt1, opt2" or "opt2, opt3", or "opt1, opt3" or the same pairs in the reverse order.

            The prefix (2$$) can use any number between 1 and the total number of options you defined

            NB : if you omit the size prefix, the number of options combined will be defined randomly

        Nesting
            You can have variations inside combinations but not the other way round (for now)

            Example:

            I love[ {red|white} wine | {layered|chocolate} cake | {german|belgian} beer]
    """
    if template is None:
        return None

    out = template
    variants = re.findall(r"\{[^{}]*?}", out)

    for v in variants:
        opts = [s.strip() for s in v.strip("{}").split("|")]
        out = out.replace(v, random.choice(opts))

    combinations = re.findall(r"\[[^\[\]]*?]", out)
    for c in combinations:
        sc = c.strip("[]")
        parts = sc.split("$$")
        n_pick = None

        if len(parts) > 2:
            raise ValueError(" we do not support more than 1 $$ in a combination")
        if len(parts) == 2:
            sc = parts[1]
            n_pick = int(parts[0])
        opts = [s.strip() for s in sc.split("|")]
        if not n_pick:
            n_pick = random.randint(1,len(opts))

        sample = random.sample(opts, n_pick)
        out = out.replace(c, ", ".join(sample))

    if len(variants + combinations) > 0:
        return pick_variant(out)
    return out


class Script(scripts.Script):
    def title(self):
        return "Dynamic Prompting"

    def run(self, p):
        fix_seed(p)

        original_prompt = p.prompt[0] if type(p.prompt) == list else p.prompt
        

        all_prompts = [
            pick_variant(original_prompt) for _ in range(p.n_iter)
        ]
        all_seeds = [int(p.seed) + (x if p.subseed_strength == 0 else 0) for x in range(len(all_prompts))]

        p.n_iter = math.ceil(len(all_prompts) / p.batch_size)
        p.do_not_save_grid = True

        print(f"Prompt matrix will create {len(all_prompts)} images using a total of {p.n_iter} batches.")

        p.prompt = all_prompts
        p.seed = all_seeds
        p.prompt_for_display = original_prompt
        processed = process_images(p)

        return processed
