import re
from collections import namedtuple
import torch

import modules.shared as shared


weighted_prompt_regex = re.compile("""
    (?P<prompt>     # capture group for 'prompt'
    (?:\\\@|[^\@])+   # match one or more non '@' characters or escaped at symbols '\@'
    )               # end 'prompt'
    (?:             # non-capture group
    \@+              # match one or more '@' characters
    (?P<weight>     # capture group for 'weight'
    -?\d*\.{0,1}\d+ # match positive or negative integer or decimal number
    )?              # end weight capture group, make optional
    \s*             # strip spaces after weight
    |               # OR
    $               # else, if no '@' then match end of line
    )               # end non-capture group
""", re.VERBOSE)

WeightedPrompt = namedtuple("WeightedPrompt", ["text", "weight"])

def split_weighted_subprompts(input_string, normalize=True):
    parsed_prompts = [WeightedPrompt(match.group("prompt").replace("\\@", "@"), float(match.group("weight") or 1)) for match in re.finditer(weighted_prompt_regex, input_string)]
    
    if not normalize:
        return parsed_prompts
    
    weight_sum = sum([x.weight for x in parsed_prompts])

    if weight_sum == 0:
        equal_weight = 1 / (len(parsed_prompts) or 1)
        return [WeightedPrompt(text, equal_weight) for text,weight in parsed_prompts]

    return [WeightedPrompt(text, weight / weight_sum) for text,weight in parsed_prompts]

re_prompt = re.compile(r'''
(.*?)
\[
    ([^]:]+):
    (?:([^]:]*):)?
    ([0-9]*\.?[0-9]+)
]
|
(.+)
''', re.X)

# a prompt like this: "fantasy landscape with a [mountain:lake:0.25] and [an oak:a christmas tree:0.75][ in foreground::0.6][ in background:0.25] [shoddy:masterful:0.5]"
# will be represented with prompt_schedule like this (assuming steps=100):
# [25, 'fantasy landscape with a mountain and an oak in foreground shoddy']
# [50, 'fantasy landscape with a lake and an oak in foreground in background shoddy']
# [60, 'fantasy landscape with a lake and an oak in foreground in background masterful']
# [75, 'fantasy landscape with a lake and an oak in background masterful']
# [100, 'fantasy landscape with a lake and a christmas tree in background masterful']


def get_learned_conditioning_prompt_schedules(prompts, steps):
    res = []
    cache = {}

    for prompt in prompts:
        prompt_schedule: list[list[str | int]] = [[steps, ""]]

        cached = cache.get(prompt, None)
        if cached is not None:
            res.append(cached)
            continue

        for m in re_prompt.finditer(prompt):
            plaintext = m.group(1) if m.group(5) is None else m.group(5)
            concept_from = m.group(2)
            concept_to = m.group(3)
            if concept_to is None:
                concept_to = concept_from
                concept_from = ""
            swap_position = float(m.group(4)) if m.group(4) is not None else None

            if swap_position is not None:
                if swap_position < 1:
                    swap_position = swap_position * steps
                swap_position = int(min(swap_position, steps))

            swap_index = None
            found_exact_index = False
            for i in range(len(prompt_schedule)):
                end_step = prompt_schedule[i][0]
                prompt_schedule[i][1] += plaintext

                if swap_position is not None and swap_index is None:
                    if swap_position == end_step:
                        swap_index = i
                        found_exact_index = True

                    if swap_position < end_step:
                        swap_index = i

            if swap_index is not None:
                if not found_exact_index:
                    prompt_schedule.insert(swap_index, [swap_position, prompt_schedule[swap_index][1]])

                for i in range(len(prompt_schedule)):
                    end_step = prompt_schedule[i][0]
                    must_replace = swap_position < end_step

                    prompt_schedule[i][1] += concept_to if must_replace else concept_from

        res.append(prompt_schedule)
        cache[prompt] = prompt_schedule
        #for t in prompt_schedule:
        #    print(t)

    return res


ScheduledPromptConditioning = namedtuple("ScheduledPromptConditioning", ["end_at_step", "cond"])
ScheduledPromptBatch = namedtuple("ScheduledPromptBatch", ["shape", "schedules"])


def get_learned_conditioning(prompts, steps):

    res = []

    prompt_schedules = get_learned_conditioning_prompt_schedules(prompts, steps)
    cache = {}

    for prompt, prompt_schedule in zip(prompts, prompt_schedules):

        cached = cache.get(prompt, None)
        if cached is not None:
            res.append(cached)
            continue

        cond_schedule = []
        promptKeys =  [promptText for _,promptText in prompt_schedule]

        if shared.opts.prompt_blending_enable:
          for promptText in set(promptKeys):
            weighted_subprompts = split_weighted_subprompts(promptText, shared.opts.prompt_blending_normalize)
            promptKeys += [p.text for p in weighted_subprompts]

        conds = shared.sd_model.get_learned_conditioning(promptKeys)
        condLookup = dict(zip(promptKeys,conds))

        for end_at_step,promptText in prompt_schedule:
            
            weighted_subprompts = []
            if shared.opts.prompt_blending_enable:
                weighted_subprompts = split_weighted_subprompts(promptText, shared.opts.prompt_blending_normalize)

            if len(weighted_subprompts) > 1:
                texts = [p.text for p in weighted_subprompts]
                cond = [condLookup[x] for x in texts]
                c = torch.zeros_like(cond[0])
                for i, (text, weight) in enumerate(weighted_subprompts):
                    c = torch.add(c, cond[i], alpha=weight)
                cond = [c]
            else:
                cond = [condLookup[promptText]]

            cond_schedule.append(ScheduledPromptConditioning(end_at_step, cond[0]))

        cache[prompt] = cond_schedule
        res.append(cond_schedule)

    return ScheduledPromptBatch((len(prompts),) + res[0][0].cond.shape, res)


def reconstruct_cond_batch(c: ScheduledPromptBatch, current_step):
    res = torch.zeros(c.shape, device=shared.device, dtype=next(shared.sd_model.parameters()).dtype)
    for i, cond_schedule in enumerate(c.schedules):
        target_index = 0
        for curret_index, (end_at, cond) in enumerate(cond_schedule):
            if current_step <= end_at:
                target_index = curret_index
                break
        res[i] = cond_schedule[target_index].cond

    return res



#get_learned_conditioning_prompt_schedules(["fantasy landscape with a [mountain:lake:0.25] and [an oak:a christmas tree:0.75][ in foreground::0.6][ in background:0.25] [shoddy:masterful:0.5]"], 100)
